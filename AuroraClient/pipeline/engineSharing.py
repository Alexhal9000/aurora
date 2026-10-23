"""Share this computer: the engine side of the Settings panel (Aurora 1.9.4+).

GET  /engine-sharing/            status for the panel
POST /engine-sharing/ {action}   one of the actions below

Who may act: the person at this computer (signed in over loopback: they already have
the OS account), and the owner and managers of this server from other computers. Members
only see the status.

Nothing here widens what the shared service can reach on its own. The service runs under
its own account (Linux aurora-tools, macOS _auroratools, Windows Local Service), which
cannot grant itself folders, open firewall ports or restart itself. The admin side is
deployment/linux/aurora-tools-network on Linux and deployment/shared/aurora_sharing_admin.py
on Windows and macOS (beta). A change asked for in the panel goes through one of these:

* turn_on (Aurora running as the desktop user, not shared yet): the admin side runs with
  the system's own admin prompt on this computer's screen (Linux pkexec password window,
  Windows UAC, macOS administrator password). Aurora then quits and the shared service
  takes over the same port and identity.
* request (Aurora running as the shared service): folders and port are written to
  sharing_request.json in the service's config folder and wait until someone with admin
  rights on this computer confirms them there: the "Confirm on this computer" button
  (aurora-confirm: link, which opens the confirm window after the admin prompt), the app
  menu / Start menu entry, or `aurora-tools confirm` over SSH. One request at a time, a
  new one replaces it, it expires after REQUEST_TTL_SECONDS. The admin side checks it
  again and shows it in full before applying, and a port that does not come up is put back.
* turn_off: only ever narrows access, so it needs no password: the engine leaves
  turn_off.request in its config folder; a root job turns sharing off (Linux systemd path
  unit, macOS launchd WatchPaths job), or on Windows the service runner stops for good.
* Direct sharing (no service, no admin side): developer runs only (run_aurora.sh sets
  AURORA_DEV=1). turn_on, apply and turn_off edit this engine's own engine.json and
  restart it (run_aurora.sh starts it again on RESTART_EXIT_CODE).
* pair_start / pair_confirm / pair_cancel: pairing with the lab from Settings, for the
  person at this computer (same steps and owner confirmation as `aurora-tools pair`).
* pause / resume: members (not the owner or managers) cannot open the server from other
  computers until resumed; engine.json "members_paused", enforced here and on the website.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from AuroraClient import engineConfig
from pipeline import engineJobs, networkFolders, networkMode

logger = logging.getLogger(__name__)

HELPER = "/usr/lib/aurora-tools/aurora-tools-network"
LINUX_CONFIRM_ENTRY = "/usr/share/applications/aurora-tools-confirm.desktop"
CONFIRM_LINK = "aurora-confirm:changes"
# Windows and macOS (beta): aurora_sharing_admin.py keeps its settings here (admins only).
WINDOWS_STATE_DIR = Path(os.environ.get("ProgramData") or r"C:\ProgramData") / "Aurora"
MAC_STATE_DIR = Path("/Library/Application Support/Aurora Sharing")
MAC_DAEMON_PLIST = Path("/Library/LaunchDaemons/ca.ucalgary.aurora.sharing.plist")
MAC_ADMIN_TOOL = Path("/Applications/Aurora.app/Contents/Resources/aurora_sharing_admin.py")
REQUEST_NAME = "sharing_request.json"
TURN_OFF_NAME = "turn_off.request"
REQUEST_TTL_SECONDS = 24 * 3600
MAX_FOLDERS = 20
MAX_PATH_LENGTH = 1024
RESTART_EXIT_CODE = 75   # run_aurora.sh starts the engine again on this exit code
PAUSED_MESSAGE = "The owner has paused access to this computer from other computers for now."


# ---------------------------------------------------------------------------
# Who is asking
# ---------------------------------------------------------------------------

def session_role(record: dict) -> str:
    """'local' (at this computer), or this server's group role: owner, manager, member."""
    if not record.get("remote"):
        return "local"
    group = networkMode.current_group()
    for member in group.get("members", []):
        if isinstance(member, dict) and str(member.get("user_id")) == str(record.get("user_id")):
            return str(member.get("role") or "member")
    return "member"


def can_manage(record: dict) -> bool:
    return session_role(record) in ("local", "owner", "manager")


def members_paused() -> bool:
    return bool(engineConfig._read_raw().get("members_paused")) and engineConfig.is_network_mode()


def refuses_member(record: dict) -> bool:
    """A network session that the pause keeps out (members only)."""
    return bool(record.get("remote")) and members_paused() and session_role(record) == "member"


# ---------------------------------------------------------------------------
# Pending change request (service only)
# ---------------------------------------------------------------------------

def _request_path() -> Path:
    return engineConfig.config_dir() / REQUEST_NAME


def _turn_off_path() -> Path:
    return engineConfig.config_dir() / TURN_OFF_NAME


def pending_request():
    path = _request_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    done = _admin_settings().get("done_requests") or []
    if not isinstance(data, dict) or float(data.get("expires_at") or 0) <= time.time() or data.get("id") in done:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    return data


def _clean_folders(raw) -> list:
    """Syntax only: the service may not even be able to see a folder it has not been given
    yet, so existence and the off-limits list are checked by the root helper (and shown to
    the person confirming)."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("Choose at least one folder to share.")
    if len(raw) > MAX_FOLDERS:
        raise ValueError(f"Share at most {MAX_FOLDERS} folders.")
    folders = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("Folder names must be text.")
        folder = item.strip()
        if not os.path.isabs(folder) or len(folder) > MAX_PATH_LENGTH:
            raise ValueError(f"Use a full folder path (for example {os.path.expanduser('~')}{os.sep}Documents): {folder[:80]}")
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in folder) or (sys.platform == "win32" and '"' in folder):
            raise ValueError("Folder names cannot contain control characters.")
        folder = os.path.normpath(folder)
        if os.path.dirname(folder) == folder:
            raise ValueError("The whole disk cannot be shared; choose a folder.")
        if folder not in folders:
            folders.append(folder)
    return folders


def _clean_port(raw) -> int:
    try:
        port = int(raw)
    except (TypeError, ValueError):
        raise ValueError("The port must be a number.")
    if not 1024 <= port <= 65535:
        raise ValueError("Use a port between 1024 and 65535.")
    if port == engineConfig.engine_port():
        raise ValueError(f"Port {port} is Aurora's own port on this computer; choose another.")
    return port


def current_folders() -> list:
    return [str(folder) for folder in networkFolders.shared_folders()]


MAC_DEFAULT_SHARE = "/Users/Shared/Aurora"


def _suggested_folders(shared: bool) -> list:
    """macOS keeps Documents away from background processes, so a first share there starts
    with /Users/Shared/Aurora (created when sharing turns on)."""
    if sys.platform == "darwin" and not shared and not engineConfig._read_raw().get("shared_folders"):
        return [MAC_DEFAULT_SHARE]
    return current_folders()


def dev_mode() -> bool:
    """Developer run from the source folder (run_aurora.sh), never the shared service."""
    return os.environ.get("AURORA_DEV") == "1" and not networkFolders.running_as_service()


def direct_mode() -> bool:
    """Sharing changed by this engine itself (developer runs, see module docstring)."""
    return dev_mode()


# ---------------------------------------------------------------------------
# The admin side on each system
# ---------------------------------------------------------------------------

def _admin_state_dir():
    if sys.platform == "win32":
        return WINDOWS_STATE_DIR
    if sys.platform == "darwin":
        return MAC_STATE_DIR
    return None


def _admin_settings() -> dict:
    folder = _admin_state_dir()
    if folder is None:
        return {}
    try:
        data = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _admin_tool():
    """aurora_sharing_admin.py of this installation, or None (Linux, developer runs)."""
    if sys.platform == "win32":
        # %LOCALAPPDATA%\Aurora\venv\Scripts\python.exe -> %LOCALAPPDATA%\Aurora\resources\...
        tool = Path(sys.executable).parents[2] / "resources" / "aurora_sharing_admin.py"
    elif sys.platform == "darwin":
        tool = MAC_ADMIN_TOOL
    else:
        return None
    return tool if tool.is_file() else None


def sharing_supported() -> bool:
    if sys.platform.startswith("linux"):
        return os.access(HELPER, os.X_OK)
    return _admin_tool() is not None


def confirm_how() -> dict:
    """How someone with admin rights confirms a request, for the panel."""
    if sys.platform == "win32":
        tools = Path(sys.executable).parents[2] / "aurora-tools.cmd"
        return {"link": CONFIRM_LINK, "menu": "Aurora - confirm sharing changes", "where": "Start menu",
                "command": f'"{tools}" confirm', "enable": f'"{tools}" network enable',
                "shell": "an administrator terminal (or SSH as an administrator)"}
    if sys.platform == "darwin":
        return {"link": CONFIRM_LINK, "menu": "", "where": "",
                "command": "sudo aurora-tools confirm",
                "enable": "sudo /Applications/Aurora.app/Contents/Resources/aurora-tools network enable",
                "shell": "Terminal (or SSH)"}
    return {"link": CONFIRM_LINK if os.path.exists(LINUX_CONFIRM_ENTRY) else "",
            "menu": "Aurora: confirm sharing changes", "where": "app menu",
            "command": "sudo aurora-tools confirm", "enable": "sudo aurora-tools network enable",
            "shell": "a terminal (or SSH)"}


_SYSTEM_FOLDERS = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/proc",
                   "/root", "/run", "/sbin", "/snap", "/sys", "/usr", "/opt", "/tmp", "/var/tmp",
                   "/var/lib", "/var/log", "/var/cache", "/var/spool", "/var/mail", "/var/backups",
                   "/var/run", "/lost+found",
                   # macOS
                   "/System", "/Library", "/Applications", "/private", "/cores")
_PRIVATE_NAMES = {".ssh", ".gnupg", ".aurora", ".config", ".local", ".mozilla", ".pki", ".cache",
                  "Library", "AppData"}


def _system_folders() -> list:
    if sys.platform != "win32":
        return list(_SYSTEM_FOLDERS)
    env = os.environ
    names = [env.get("SystemRoot") or r"C:\Windows", env.get("ProgramFiles") or r"C:\Program Files",
             env.get("ProgramFiles(x86)") or r"C:\Program Files (x86)", env.get("ProgramData") or r"C:\ProgramData"]
    return [os.path.normcase(os.path.realpath(name)) for name in names if name]


def _home_folders() -> set:
    homes = {os.path.realpath(os.path.expanduser("~"))}
    if sys.platform == "win32":
        users = os.path.dirname(os.path.realpath(os.path.expanduser("~")))
        try:
            homes |= {os.path.realpath(os.path.join(users, name)) for name in os.listdir(users)}
        except OSError:
            pass
    else:
        import pwd
        homes |= {os.path.realpath(p.pw_dir) for p in pwd.getpwall()
                  if (p.pw_uid >= 500 or p.pw_uid == 0) and p.pw_dir not in ("/", "/nonexistent", "/var/empty")}
    return {os.path.normcase(home) for home in homes}


def check_folders_here(folders: list) -> list:
    """The root helper's folder rules (aurora-tools-network check_folders), for direct
    sharing that changes folders without it. Returns resolved folders or raises ValueError."""
    system = _system_folders()
    homes = _home_folders()
    resolved = []
    for folder in folders:
        real = os.path.realpath(os.path.expanduser(folder))
        if not os.path.isdir(real):
            raise ValueError(f"Not a folder on this computer: {folder}")
        key = os.path.normcase(real)
        sep = os.sep

        def inside(base: str) -> bool:
            base = os.path.normcase(base)
            return key == base or key.startswith(base.rstrip(sep) + sep)

        if os.path.dirname(real) == real or key in ("/var", "/volumes", "/users", "/home") \
                or any(inside(d) for d in system):
            raise ValueError(f"{real} is a system folder and cannot be shared.")
        if any(home == key or home.startswith(key.rstrip(sep) + sep) for home in homes):
            raise ValueError(f"{real} holds a whole home folder; share a folder inside it, such as Documents.")
        parts = set(real.replace("\\", "/").split("/"))
        if _PRIVATE_NAMES & parts:
            raise ValueError(f"{real} is a private settings folder and cannot be shared.")
        if real not in resolved:
            resolved.append(real)
    return resolved


def _restart_soon(seconds: float = 1.5) -> None:
    """Developer run: exit so run_aurora.sh starts the engine again with the new listeners."""
    def stop():
        time.sleep(seconds)
        logger.warning("Sharing settings changed: restarting this developer Aurora.")
        os._exit(RESTART_EXIT_CODE)
    threading.Thread(target=stop, name="aurora-dev-restart", daemon=True).start()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _service_installed() -> bool:
    """The shared service is set up (and not turned off) on this computer."""
    if sys.platform == "win32":
        settings = _admin_settings()
        if not settings.get("enabled"):
            return False
        try:
            marker = WINDOWS_STATE_DIR / "service" / "sharing_disabled"
            return marker.stat().st_mtime < float(settings.get("enabled_at") or 0)
        except FileNotFoundError:
            return True
        except (OSError, TypeError, ValueError):
            return True
    if sys.platform == "darwin":
        return MAC_DAEMON_PLIST.exists()
    if not os.access(HELPER, os.X_OK):
        return False
    try:
        return subprocess.run([HELPER, "is-shared"], timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def status_payload(record: dict) -> dict:
    config = engineConfig.load_config()
    service = networkFolders.running_as_service()
    shared = service or config["mode"] == engineConfig.MODE_NETWORK
    linux_helper = sys.platform.startswith("linux") and os.access(HELPER, os.X_OK)
    pairing = networkMode.pairing_state() or {}
    raw = engineConfig._read_raw()
    pending = pending_request() if service else None
    payload = {
        "platform": sys.platform,
        "running_as_service": service,
        "shared": shared,
        "network_ready": networkMode.network_ready() if shared else False,
        "address": networkMode._lan_address() if shared else "",
        "port": config["network_port"],
        "folders": _suggested_folders(service or shared),
        "members_paused": bool(raw.get("members_paused")),
        "turning_off": service and _turn_off_path().exists(),
        "pending": pending,
        "server_name": pairing.get("server_name") or "",
        "paired": bool(pairing),
        "lab_origin": record.get("lab_origin") or pairing.get("lab_origin") or "",
        "role": session_role(record),
        "can_manage": can_manage(record),
        "is_local": not record.get("remote"),
        # Turning sharing on needs the admin prompt on this computer's screen.
        "can_turn_on": (not service and not shared and not record.get("remote")
                        and (sharing_supported() or direct_mode())),
        "sharing_supported": sharing_supported(),
        "confirm": confirm_how() if service and not record.get("remote") else
                   {k: v for k, v in confirm_how().items() if k != "link"},
        "dev_mode": dev_mode(),
        "direct_mode": direct_mode(),
        # Pairing from Settings: only the person at this computer may make an account its owner.
        "pairing_flow": networkMode.pair_poll() if not record.get("remote") else {},
        "helper_installed": linux_helper,
        "jobs_running": len(engineJobs.running()),
        "request_ttl_hours": REQUEST_TTL_SECONDS // 3600,
    }
    if not service and not shared and payload["can_turn_on"] and not payload["direct_mode"] and _service_installed():
        # The shared service is set up but this desktop Aurora is running anyway
        # (developer run, or the service failed to start): do not offer a second one.
        payload["can_turn_on"] = False
        payload["service_conflict"] = True
    return payload


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _end_paused_sessions() -> None:
    from pipeline import engineAuth  # engineAuth imports this module
    engineAuth.end_sessions_where(refuses_member, PAUSED_MESSAGE)


def _check_in_soon() -> None:
    threading.Thread(target=networkMode.check_in_once, name="aurora-sharing-check-in", daemon=True).start()


def _quit_soon(seconds: float) -> None:
    """Let the response reach the browser, then stop this desktop engine so the shared
    service (which retries every few seconds) can take the port."""
    def stop():
        time.sleep(seconds)
        logger.warning("Network sharing is on: this Aurora stops so the shared service takes over.")
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=stop, name="aurora-handover", daemon=True).start()


def turn_on(folders: list, port: int) -> tuple:
    """Run the admin side with the system's admin prompt on this screen."""
    if sys.platform == "win32":
        return _turn_on_windows(folders, port)
    if sys.platform == "darwin":
        return _turn_on_mac(folders, port)
    command = ["pkexec", HELPER, "enable"]
    for folder in folders:
        command += ["--folder", folder]
    command += ["--port", str(port)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        return False, "pkexec is missing on this computer. In a terminal: sudo aurora-tools network enable"
    except subprocess.TimeoutExpired:
        return False, "The password window was not answered in time."
    if result.returncode in (126, 127) and "Error:" not in (result.stderr or ""):
        return False, "Sharing was not turned on: the password was not entered."
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, (detail[-1] if detail else "The helper failed.").replace("Error: ", "")
    logger.info("Network sharing turned on:\n%s", result.stdout)
    return True, result.stdout


def _folder_args(folders: list) -> list:
    return [item for folder in folders for item in ("--folder", folder)]


def _turn_on_windows(folders: list, port: int) -> tuple:
    """UAC prompt, then the admin tool in its own (hidden) window; it reports back through
    a result file because an elevated process cannot be read from here."""
    result_file = Path(tempfile.gettempdir()) / f"aurora-sharing-{secrets.token_hex(6)}.json"
    args = ["-I", str(_admin_tool()), "network", "enable", *_folder_args(folders), "--port", str(port),
            "--owner-home", str(Path.home()), "--result-file", str(result_file)]
    arg_line = " ".join(f'"{arg}"' for arg in args)
    quote = lambda text: "'" + str(text).replace("'", "''") + "'"  # noqa: E731 - PowerShell literal
    script = (f"$p = Start-Process -FilePath {quote(sys.executable)} -ArgumentList {quote(arg_line)} "
              "-Verb RunAs -WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode")
    try:
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-Command", script], capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return False, "Turning sharing on took too long; see Aurora's log."
    except OSError as exc:
        return False, f"Could not ask Windows for admin rights: {exc}"
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "Sharing was not turned on: Windows did not get permission to make the change."
    finally:
        try:
            result_file.unlink()
        except OSError:
            pass
    output = "\n".join(str(line) for line in result.get("output") or [])
    if result.get("code") != 0:
        return False, str(result.get("error") or "The change did not work.")
    logger.info("Network sharing turned on:\n%s", output)
    return True, output


def _turn_on_mac(folders: list, port: int) -> tuple:
    """macOS administrator password window (osascript), then the admin tool as root."""
    python = os.path.realpath(sys.executable)   # the pinned, root-owned CPython behind the venv
    command = " ".join(shlex.quote(str(part)) for part in [
        python, "-I", MAC_ADMIN_TOOL, "network", "enable", *_folder_args(folders), "--port", str(port),
        "--owner", getpass.getuser(), "--owner-home", str(Path.home())])
    escaped = command.replace("\\", "\\\\").replace('"', '\\"')
    script = (f'do shell script "{escaped} 2>&1" with administrator privileges '
              'with prompt "Aurora needs an administrator to share this computer on the network."')
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return False, "The password window was not answered in time."
    except OSError as exc:
        return False, f"Could not ask macOS for the administrator password: {exc}"
    if result.returncode != 0:
        if "-128" in (result.stderr or ""):
            return False, "Sharing was not turned on: the password was not entered."
        lines = [line for line in (result.stderr or "").replace("\r", "\n").splitlines() if line.strip()]
        detail = next((line for line in reversed(lines) if "Error:" in line), lines[-1] if lines else "")
        detail = detail.split("Error:", 1)[-1].strip().rstrip(".") if detail else "The change did not work"
        return False, detail[:300]
    logger.info("Network sharing turned on:\n%s", result.stdout)
    return True, result.stdout


class EngineSharingView(APIView):
    """GET/POST /engine-sharing/ — see the module docstring."""

    authentication_classes = []
    permission_classes = []

    def get(self, request, *args, **kwargs):
        record = getattr(request, "aurora_session", None)
        if record is None:
            return Response({"code": "engine_session_required"}, status=status.HTTP_401_UNAUTHORIZED)
        return Response(status_payload(record), status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        record = getattr(request, "aurora_session", None)
        if record is None:
            return Response({"code": "engine_session_required"}, status=status.HTTP_401_UNAUTHORIZED)
        if not can_manage(record):
            return Response({"error": "Only the owner or a manager of this computer can change sharing."},
                            status=status.HTTP_403_FORBIDDEN)
        data = request.data if hasattr(request.data, "get") else {}
        action = data.get("action")
        service = networkFolders.running_as_service()
        shared = service or engineConfig.is_network_mode()

        def refuse(message, code=status.HTTP_400_BAD_REQUEST):
            return Response({"error": message}, status=code)

        def done(**extra):
            return Response({**status_payload(record), **extra}, status=status.HTTP_200_OK)

        if action in ("pause", "resume"):
            if not shared:
                return refuse("This computer is not shared on the network.")
            engineConfig.write_config(members_paused=action == "pause")
            logger.info("Network access for members %s by %s", "paused" if action == "pause" else "resumed",
                        record.get("name"))
            if action == "pause":
                _end_paused_sessions()
            _check_in_soon()
            return done()

        if action == "turn_off":
            if direct_mode() and shared:
                engineConfig.write_config(mode=engineConfig.MODE_LOCAL)
                logger.warning("Sharing turned off by %s", record.get("name"))
                _restart_soon()
                return done(restarting=True)
            if not service:
                return refuse("Sharing is not running as a service here; nothing to turn off.")
            try:
                path = _turn_off_path()
                path.write_text(json.dumps({"by": record.get("name"), "at": time.time()}) + "\n", encoding="utf-8")
            except OSError as exc:
                return refuse(f"Could not ask to turn sharing off: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)
            logger.warning("Turning network sharing off, asked by %s", record.get("name"))
            return done()

        if action in ("pair_start", "pair_confirm", "pair_cancel"):
            if record.get("remote"):
                return refuse("Pair this computer from Aurora at the computer itself.")
            try:
                if action == "pair_start":
                    if networkMode.pairing_state():
                        return refuse("This computer is already paired.")
                    networkMode.pair_start(record.get("lab_origin") or "", data.get("name") or "")
                elif action == "pair_confirm":
                    networkMode.pair_confirm(bool(data.get("accept")))
                else:
                    networkMode.pair_cancel()
            except ValueError as exc:
                return refuse(str(exc))
            return done()

        if action == "cancel_request":
            try:
                _request_path().unlink()
            except OSError:
                pass
            return done()

        try:
            folders = _clean_folders(data.get("folders"))
            port = _clean_port(data.get("port"))
        except ValueError as exc:
            return refuse(str(exc))

        if direct_mode() and action in ("apply", "turn_on"):
            if action == "turn_on" and (record.get("remote") or shared):
                return refuse("Turn sharing on from Aurora at the computer itself.")
            if action == "apply" and not shared:
                return refuse("Sharing is off.")
            if engineJobs.running():
                return refuse("Wait for the running job to finish: Aurora restarts for this change.")
            try:
                folders = check_folders_here(folders)
            except ValueError as exc:
                return refuse(str(exc))
            restart = action == "turn_on" or port != engineConfig.load_config()["network_port"]
            engineConfig.write_config(mode=engineConfig.MODE_NETWORK, shared_folders=folders, network_port=port)
            networkMode.ensure_tls_certificate()
            logger.warning("Sharing %s by %s: folders %s, port %s",
                           "turned on" if action == "turn_on" else "changed", record.get("name"), folders, port)
            if restart:
                _restart_soon()
            else:
                _check_in_soon()
            return done(restarting=restart)

        if action == "request":
            if not service:
                return refuse("Change requests are for the shared service; turn sharing on instead.")
            request_data = {
                "id": secrets.token_hex(8),
                "requested_at": time.time(),
                "expires_at": time.time() + REQUEST_TTL_SECONDS,
                "by": {"name": str(record.get("name") or "")[:120], "role": session_role(record),
                       "user_id": str(record.get("user_id") or "")},
                "folders": folders,
                "port": port,
            }
            path = _request_path()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(request_data, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, path)
            logger.info("Sharing change requested by %s: folders %s, port %s", record.get("name"), folders, port)
            return done()

        if action == "turn_on":
            if record.get("remote") or service or shared:
                return refuse("Turn sharing on from Aurora at the computer itself.")
            if not sharing_supported():
                return refuse("Turning sharing on from Settings needs the installed Aurora.")
            if _service_installed():
                return refuse("Sharing is already set up on this computer; open Aurora from the app menu to use it.")
            if engineJobs.running():
                return refuse("Wait for the running job to finish: Aurora restarts as a shared service.")
            ok, detail = turn_on(folders, port)
            if not ok:
                return refuse(detail)
            # Remember the choice for next time (this desktop config stays in local mode).
            engineConfig.write_config(shared_folders=folders, network_port=port)
            _quit_soon(3.0)
            return done(handover=True, helper_output=detail)

        return refuse("Unknown action.")
