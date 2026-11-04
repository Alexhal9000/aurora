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
* Stray share: a desktop Aurora (not the service, not a developer run) that finds
  "mode": "network" in engine.json, left there by a developer run (both read the same
  ~/.aurora). It has no admin side to ask, so turn_off writes local mode itself (it only
  narrows access), ends the network sessions, and the network port closes when Aurora
  is opened again (no launcher here restarts it on RESTART_EXIT_CODE).
* pair_start / pair_confirm / pair_cancel: pairing with the lab from Settings, for the
  person at this computer (same steps and owner confirmation as `aurora-tools pair`).
* pause / resume: members (not the owner or managers) cannot open the server from other
  computers until resumed; engine.json "members_paused", enforced here and on the website.
* Folder access (Linux, since 2026-11-04): Turn on gives the service each shared folder
  itself only, never every file inside (fast, and private items stay private).
  status "folder_access" says per folder what the service cannot open or change
  (sharedFolderAccess.py, checked as the service account in the background);
  recheck_access runs it again; request_full_access asks, through the same confirm step
  as a request, to give the service access to everything in one shared folder.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import re
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
from pipeline import engineJobs, networkFolders, networkMode, sharedFolderAccess

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


def stray_share(service: bool, shared: bool) -> bool:
    """Shared through engine.json by a developer run, but this is neither the service nor
    a developer run (see "Stray share" in the module docstring)."""
    return shared and not service and not dev_mode()


def direct_mode() -> bool:
    """Sharing changed by this engine itself (developer runs, see module docstring).
    On Linux with the installed helper a developer run is shared like an install instead:
    the real password prompt, folder grants and systemd service, which then runs this
    source folder as the aurora-tools account (dev_source_args). AURORA_DEV_DIRECT=1 keeps
    the old direct way (no service, no root)."""
    if not dev_mode():
        return False
    if os.environ.get("AURORA_DEV_DIRECT") == "1":
        return True
    if engineConfig.is_network_mode():
        # Already shared the direct way (engine.json "mode": "network", from before the
        # service path or with AURORA_DEV_DIRECT): changes and Turn off stay direct too.
        return True
    return not (sys.platform.startswith("linux") and _helper_runs_dev_source())


def _helper_runs_dev_source() -> bool:
    """The installed helper knows --source (an older one refuses it: share directly then)."""
    try:
        with open(HELPER, encoding="utf-8", errors="replace") as handle:
            return os.access(HELPER, os.X_OK) and "--source)" in handle.read()
    except OSError:
        return False


def dev_source_args() -> list:
    """Developer run shared as the service: the checkout and virtualenv the service runs
    (aurora-tools-network enable --source/--venv); [] for installed Aurora."""
    if not dev_mode() or direct_mode():
        return []
    source = Path(__file__).resolve().parents[2]      # <checkout>/AuroraClient/pipeline
    return ["--source", str(source), "--venv", sys.prefix]


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


# ---------------------------------------------------------------------------
# Which folders may be shared (the same rules as the admin side: aurora-tools-network
# check_folders on Linux, aurora_sharing_admin.check_folders on Windows and macOS).
#
# Refused, because sharing them would hand out the computer itself: the system disk as a
# whole, system folders, a whole home folder (ssh keys, logins), private settings folders,
# and the folders that only hold other drives (/media, /run/media/<you>, /mnt, /Volumes).
# Everything else may be shared. Folders on another drive (USB, external, network) are
# allowed with warnings that say what can go wrong there, shown in Settings before turning
# sharing on and printed again when the change is confirmed.
# ---------------------------------------------------------------------------

_SYSTEM_FOLDERS = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/libx32", "/proc",
                   "/root", "/sbin", "/snap", "/sys", "/usr", "/opt", "/tmp", "/var/tmp",
                   "/var/lib", "/var/log", "/var/cache", "/var/spool", "/var/mail", "/var/backups",
                   "/var/run", "/lost+found",
                   # macOS
                   "/System", "/Library", "/Applications", "/private", "/cores")
_PRIVATE_NAMES = {".ssh", ".gnupg", ".aurora", ".aurora-dev", ".config", ".local", ".mozilla", ".pki", ".cache",
                  "Library", "AppData"}
# Where Linux and macOS mount drives. /run is a system folder except /run/media/<you>/<drive>.
_DRIVE_ROOTS = ("/run/media", "/media", "/mnt", "/Volumes")
# Filesystems without per-account permissions (USB sticks, Windows disks on Linux).
_NO_PERMISSION_FS = {"vfat", "msdos", "fat", "fat32", "exfat", "ntfs", "ntfs3", "fuseblk"}
_NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs", "sshfs", "afpfs", "davfs", "fuse.rclone"}


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


def _drive_of(path: str) -> tuple:
    """(where the drive holding path is mounted, its filesystem type, kind) with kind
    'removable', 'network', 'fixed' or ''. Empty strings when this cannot be told."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            root = ctypes.create_unicode_buffer(300)
            if not kernel.GetVolumePathNameW(str(path), root, 300):
                return "", "", ""
            fs_name = ctypes.create_unicode_buffer(64)
            kernel.GetVolumeInformationW(root.value, None, 0, None, None, None, fs_name, 64)
            kind = {2: "removable", 3: "fixed", 4: "network"}.get(kernel.GetDriveTypeW(root.value), "")
            return root.value, fs_name.value.lower(), kind
        except (AttributeError, OSError, ValueError):
            return "", "", ""
    best, fstype = "", ""
    try:
        with open("/proc/self/mounts", encoding="utf-8", errors="replace") as mounts:
            for line in mounts:
                fields = line.split()
                if len(fields) < 3:
                    continue
                # Spaces and tabs in mount points are written as octal escapes (\040).
                where = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[1])
                inside = path == where or path.startswith(where.rstrip("/") + "/")
                if inside and len(where) >= len(best):
                    best, fstype = where, fields[2]
    except OSError:
        pass   # macOS has no /proc: the drive is told from where it is mounted
    kind = "network" if fstype in _NETWORK_FS else ""
    if not kind and any(path == r or path.startswith(r + "/") for r in _DRIVE_ROOTS):
        kind = "removable"
    return best, fstype, kind


def _mounts_inside(real: str) -> list:
    """Mount points (network drives, USB disks) strictly inside a folder. Network sessions
    never see them through the folder: networkFolders.crosses_mount."""
    if not sys.platform.startswith("linux"):
        return []
    root = Path(real)
    return sorted(str(point) for point in networkFolders.mount_points() if root in point.parents)


def review_folder(folder: str, must_exist: bool = True) -> dict:
    """{"folder", "real", "error", "warnings"} for one folder under the rules above.
    must_exist=False: the shared service cannot see folders it was not given yet, so a
    folder it cannot see is not an error there (the admin side checks it)."""
    result = {"folder": folder, "real": folder, "error": "", "warnings": []}
    try:
        real = os.path.realpath(os.path.expanduser(folder))
    except (OSError, ValueError):
        result["error"] = f"Not a valid folder name: {folder}"
        return result
    result["real"] = real
    visible = os.path.isdir(real)
    if not visible and must_exist:
        result["error"] = f"Not a folder on this computer: {folder}"
        return result
    windows = sys.platform == "win32"
    key = os.path.normcase(real)
    sep = os.sep

    def inside(base: str) -> bool:
        base = os.path.normcase(base)
        return key == base or key.startswith(base.rstrip(sep) + sep)

    def refuse(message: str) -> dict:
        result["error"] = message
        return result

    system_drive = os.path.normcase((os.environ.get("SystemDrive") or "C:") + "\\") if windows else "/"
    whole_disk = os.path.dirname(real) == real
    if whole_disk and (not windows or key == system_drive):
        return refuse(f"{real} is the whole system disk and cannot be shared; choose a folder on it.")
    if any(inside(d) for d in _system_folders()) or key in ("/var", "/users", "/home") \
            or (inside("/run") and not inside("/run/media")):
        return refuse(f"{real} is a system folder and cannot be shared.")
    parent = os.path.dirname(key)
    holds_drives = key in ("/run/media", "/media", "/mnt", "/volumes") or parent == "/run/media" \
        or (parent == "/media" and visible and not os.path.ismount(real))
    if holds_drives:
        return refuse(f"{real} holds the drives plugged into this computer, not files; share a folder on "
                      "one of the drives instead (or the drive itself).")
    if any(home == key or home.startswith(key.rstrip(sep) + sep) for home in _home_folders()):
        return refuse(f"{real} holds a whole home folder; share a folder inside it, such as Documents.")
    if _PRIVATE_NAMES & set(real.replace("\\", "/").split("/")):
        return refuse(f"{real} is a private settings folder and cannot be shared.")
    if not visible:
        # The shared account cannot see this folder. Usually it has not been given access
        # yet (the admin side does that), but a network or FUSE mount (sshfs, CIFS, NFS) can
        # never be given any: only the mounter's own account may open it. Say so now,
        # instead of at the confirmation in a terminal.
        mount, fstype, kind = _drive_of(real)
        if not direct_mode() and mount and mount != "/" and (kind == "network" or fstype.startswith("fuse.")):
            # One short line; the steps (mount_fix) sit behind a "How to fix" button in Settings.
            result["fix"] = mount_fix(real)
            return refuse("Only your own account can open this drive, so it cannot be shared yet.")
        return result

    # Short notes, one line each. The account caveats are about the background service,
    # which runs as its own account: a developer run shares as you, so they do not apply.
    warnings = result["warnings"]
    own_account = direct_mode()
    mount, fstype, kind = _drive_of(real)
    if whole_disk or (mount and os.path.normcase(mount.rstrip("\\/") or mount) == key and key not in ("/", "/home")):
        warnings.append("Whole drive: everything on it is shared.")
    inner = _mounts_inside(real)
    if inner:
        shown = ", ".join(os.path.basename(m) or m for m in inner[:3]) + (" and more" if len(inner) > 3 else "")
        warnings.append(f"Mounted drives inside ({shown}) are not shared with it.")
    if own_account:
        return result
    if kind == "network":
        warnings.append("Network drive: its server decides whether the shared Aurora may open it.")
    elif kind == "removable":
        warnings.append("Separate drive: shared only while it is plugged in.")
    if fstype in _NO_PERMISSION_FS and not windows:
        # No Linux permissions (exFAT, FAT, NTFS): the mount options decide for every
        # account at once. Turning sharing on makes the desktop mount such drives with the
        # aurora-drives group allowed to write (aurora-tools-network setup_drive_access);
        # a drive mounted before that needs to be plugged in again once.
        try:
            info = os.stat(real)
            import grp
            drives_gid = grp.getgrnam("aurora-drives").gr_gid
        except (OSError, KeyError, ImportError):
            info, drives_gid = None, None
        bits = 0
        if info is not None:
            bits = info.st_mode & 0o007
            if info.st_gid == drives_gid:
                bits |= (info.st_mode >> 3) & 0o007
        if bits & 0o007 != 0o007:
            warnings.append(f"{fstype} drive: remounted when sharing turns on.")
    if sys.platform == "darwin" and inside("/Volumes"):
        warnings.append("macOS may keep background apps off drives; if so, give Aurora's Python Full Disk Access.")
    return result


def check_folders_here(folders: list) -> list:
    """The admin side's folder rules, for direct sharing that changes folders without it.
    Returns resolved folders or raises ValueError (warnings are only logged here)."""
    resolved = []
    for folder in folders:
        review = review_folder(folder)
        if review["error"]:
            raise ValueError(review["error"])
        for warning in review["warnings"]:
            logger.warning("Shared folder %s: %s", review["real"], warning)
        if review["real"] not in resolved:
            resolved.append(review["real"])
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


def _paired_elsewhere(record: dict, pairing: dict) -> bool:
    here = (record.get("lab_origin") or "").rstrip("/")
    there = (pairing.get("lab_origin") or "").rstrip("/")
    return bool(here and there and here != there)


def _pairing_unusable(pairing: dict) -> bool:
    """Paired with a lab website this Aurora no longer trusts (e.g. a dev site, after the
    shared service took over a developer's pairing): it can never check in there."""
    there = (pairing.get("lab_origin") or "").rstrip("/")
    return bool(there) and there not in engineConfig.trusted_lab_origins()


def member_folders_payload() -> dict:
    """For the owner and managers: each member (by the lab's last list) and the folders
    they are limited to (None = all shared folders, or none while restrict_new is on)."""
    settings = networkFolders.member_folder_settings()
    members = []
    for member in networkMode.current_group().get("members", []):
        if not isinstance(member, dict) or str(member.get("role") or "member") != "member":
            continue
        user_id = str(member.get("user_id") or "")
        members.append({
            "user_id": user_id,
            "name": str(member.get("name") or ""),
            "blocked": bool(member.get("blocked")),
            "folders": settings["members"].get(user_id),
        })
    return {"restrict_new": settings["restrict_new"], "members": members}


def _in_use_by_other(record: dict) -> bool:
    """Someone other than the person asking holds the computer (the lease) or has a job
    running on it. The same person on another device or browser does not count."""
    from pipeline import engineAuth  # engineAuth imports this module
    user_id = str(record.get("user_id", ""))
    holder = engineAuth.lease_holder()
    if holder is not None and holder["user_id"] != user_id:
        return True
    return any(str(job.get("user_id") or "") not in ("", user_id) for job in engineJobs.running())


# Id of the last waiting request that asked for full folder access: once it is gone
# (confirmed or discarded at the computer) the folder access is checked again.
_watched_full_access_request = None


def _recheck_after_full_access(pending) -> None:
    global _watched_full_access_request
    if pending and pending.get("full_access"):
        _watched_full_access_request = pending.get("id")
    elif _watched_full_access_request and not pending:
        _watched_full_access_request = None
        sharedFolderAccess.start_checks(current_folders(), force=True)


def status_payload(record: dict) -> dict:
    config = engineConfig.load_config()
    service = networkFolders.running_as_service()
    shared = service or config["mode"] == engineConfig.MODE_NETWORK
    linux_helper = sys.platform.startswith("linux") and os.access(HELPER, os.X_OK)
    pairing = networkMode.pairing_state() or {}
    raw = engineConfig._read_raw()
    pending = pending_request() if service else None
    if service:
        _recheck_after_full_access(pending)
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
        # Paired with a different lab website than the one this session came from (e.g. a
        # dev site): check-ins go there, so the website you use never sees this computer.
        "paired_lab_origin": pairing.get("lab_origin") or "",
        "paired_elsewhere": _paired_elsewhere(record, pairing),
        # Re-pairing from Settings replaces a pairing that cannot work anyway; a working one
        # on the shared service changes owner only with admin rights (sudo aurora-tools pair --reset).
        "can_repair": bool(pairing) and _paired_elsewhere(record, pairing)
                      and (not service or _pairing_unusable(pairing)),
        "paired_owner": (pairing.get("owner") or {}).get("name") or "",
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
        "stray_share": stray_share(service, shared),
        # Pairing from Settings: only the person at this computer may make an account its owner.
        "pairing_flow": networkMode.pair_poll() if not record.get("remote") else {},
        "helper_installed": linux_helper,
        "jobs_running": len(engineJobs.running()),
        # Header servers icon: yellow while someone else is using this shared computer.
        "in_use_by_other": shared and _in_use_by_other(record),
        "request_ttl_hours": REQUEST_TTL_SECONDS // 3600,
        # The person at this computer: their home folder, so Settings can turn the open
        # project's "~/..." path into the full path it suggests sharing.
        "home": str(Path.home()) if not record.get("remote") else "",
    }
    if record.get("remote") and not payload["can_manage"]:
        # A member sees only the folders given to them, not the names of everyone else's.
        payload["folders"] = [str(folder) for folder in
                              networkFolders.member_folders(record.get("user_id"))]
    if payload["can_manage"] and shared:
        payload["member_folders"] = member_folders_payload()
    # Only the service's own account gives a true answer (a desktop or direct run would
    # check the owner's rights); "give access to everything" exists in the Linux helper.
    payload["full_access_supported"] = service and linux_helper
    if payload["can_manage"] and service and linux_helper:
        payload["folder_access"] = [
            # For someone on another computer: what to paste in a terminal on this one.
            {**entry, "grant_command": "sudo aurora-tools network grant-all " + shlex.quote(entry["folder"])}
            for entry in sharedFolderAccess.status(current_folders())
        ]
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


# Turning sharing on can take minutes after the password (folder permissions over large
# shared folders), so the helper gets 30 min and its lines go to the Aurora terminal as
# they come instead of all at the end: the terminal used to look frozen (2026-11-04).
HELPER_TIMEOUT_SECONDS = 30 * 60


def _run_helper_streaming(command: list, timeout: float) -> subprocess.CompletedProcess:
    """subprocess.run(capture_output=True, text=True, timeout=...) that also logs each
    stdout line of the helper while it runs. Raises TimeoutExpired after killing it."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    out_lines, err_lines = [], []

    def drain_stderr():
        for line in proc.stderr:
            err_lines.append(line)
            logger.warning("Sharing helper: %s", line.rstrip())

    err_thread = threading.Thread(target=drain_stderr, name="aurora-sharing-helper-stderr", daemon=True)
    err_thread.start()
    timed_out = threading.Event()

    def on_timeout():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, on_timeout)
    timer.daemon = True
    timer.start()
    try:
        for line in proc.stdout:
            out_lines.append(line)
            logger.info("Sharing helper: %s", line.rstrip())
        proc.wait()
    finally:
        timer.cancel()
    err_thread.join(timeout=5)
    if timed_out.is_set():
        raise subprocess.TimeoutExpired(command, timeout, output="".join(out_lines), stderr="".join(err_lines))
    return subprocess.CompletedProcess(command, proc.returncode, "".join(out_lines), "".join(err_lines))


def turn_on(folders: list, port: int, pair_owner: str = "") -> tuple:
    """Run the admin side with the system's admin prompt on this screen. pair_owner: the
    lab account of the person turning it on; the shared service then pairs only with it."""
    if sys.platform == "win32":
        return _turn_on_windows(folders, port, pair_owner)
    if sys.platform == "darwin":
        return _turn_on_mac(folders, port, pair_owner)
    command = ["pkexec", HELPER, "enable"]
    for folder in folders:
        command += ["--folder", folder]
    command += ["--port", str(port), *_owner_args(pair_owner), *dev_source_args()]
    try:
        result = _run_helper_streaming(command, HELPER_TIMEOUT_SECONDS)
    except FileNotFoundError:
        return False, "pkexec is missing on this computer. In a terminal: sudo aurora-tools network enable"
    except subprocess.TimeoutExpired:
        return False, ("Sharing did not finish within 30 minutes (the password window was not answered, or "
                       "the shared folders are very large). See the Aurora terminal for the last step.")
    if result.returncode in (126, 127) and "Error:" not in (result.stderr or ""):
        return False, "Sharing was not turned on: the password was not entered."
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, (detail[-1] if detail else "The helper failed.").replace("Error: ", "")
    logger.info("Network sharing turned on:\n%s", result.stdout)
    notes = remount_shared_drives(folders)
    return True, "\n".join([result.stdout.rstrip(), *notes]).strip() + "\n"


def _mount_table() -> list:
    """(device, mount point, filesystem, options) of every mount; /proc/mounts writes a
    space in a path as \\040 (drive labels such as "STORE N GO")."""
    unescape = lambda text: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), text)  # noqa: E731
    table = []
    try:
        with open("/proc/self/mounts", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 4:
                    table.append((unescape(parts[0]), unescape(parts[1]), parts[2], parts[3].split(",")))
    except OSError:
        pass
    return table


def _programs_using(mount_point: str) -> list:
    """Names of this person's programs holding files on a drive (fuser -vm), such as a Files
    window (nautilus) left open on it, so the note can say what to close."""
    try:
        result = subprocess.run(["fuser", "-vm", mount_point], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    names = []
    for line in (result.stderr or "").splitlines():   # fuser -v prints its table on stderr
        parts = line.split()
        if len(parts) >= 4 and parts[-3].isdigit() and parts[-1] not in names:
            names.append(parts[-1])
    return names


RESTRICTED_NOTE = "Aurora cannot open this folder."


def write_note(path: str):
    """Why Aurora (this engine's account) cannot save in a folder it lists, or None when it
    can. Shown as a warning icon beside the Working Directory path (ListDirectoriesView),
    so it appears only where saving would really fail: access(2) answers for this very
    process, ACLs, read-only mounts and drive mount options included. Checks the folder
    and its extracted/ folder, where Aurora writes everything it makes."""
    if not (networkFolders.running_as_service() or engineConfig.is_network_mode()):
        return None   # sharing is off: Aurora runs as the person themselves, nothing to warn about
    if not networkFolders.is_inside_shared(path):
        return None   # only inside the shared folders; elsewhere nobody works through this Aurora
    real = os.path.realpath(path)
    targets = [real] + [t for t in (os.path.join(real, "extracted"),) if os.path.isdir(t)]
    if all(os.access(target, os.W_OK | os.X_OK) for target in targets):
        return None
    mount = None
    for device, mount_point, fstype, options in _mount_table():
        if real == mount_point or real.startswith(mount_point.rstrip("/") + "/"):
            if mount is None or len(mount_point) > len(mount[1]):
                mount = (device, mount_point, fstype, options)
    if mount and "ro" in mount[3]:
        return "Read-only: Aurora cannot save here."
    if (mount and mount[2] in _NO_PERMISSION_FS and mount[1].startswith(("/run/media/", "/media/"))
            and networkFolders.running_as_service()):
        # exFAT/FAT/NTFS: the mount options decide for every account at once; a drive
        # mounted before sharing was turned on lacks the aurora-drives group (see
        # remount_shared_drives), the one case a replug fixes.
        try:
            import grp
            drives_gid = grp.getgrnam("aurora-drives").gr_gid
        except (ImportError, KeyError):
            drives_gid = None
        if drives_gid is None or f"gid={drives_gid}" not in mount[3]:
            return "Aurora cannot save on this drive yet. Turn sharing off and on again to remount it."
    return "Aurora cannot save in this folder."


def mount_fix(path: str):
    """What can be done when Aurora cannot save in a folder that sits on a mount of its own
    (a network drive mounted by hand, a read-only mount): the mount's details and the
    commands that fix it, for the dialog behind the warning icon beside the Working
    Directory path. None when the folder is not on a separate Linux mount.
      command  sudo aurora-tools network fix-mount ...: changes the mount in place
               (mount -o remount), which keeps its login, so no password for the file
               server is asked here or sent through Aurora.
      manual   unmount + mount again with the same source and user, for kernels that
               refuse the in-place change; the person types the password themselves."""
    if not sys.platform.startswith("linux"):
        return None
    real = os.path.realpath(path)
    best = None
    for device, mount_point, fstype, options in _mount_table():
        if real == mount_point or real.startswith(mount_point.rstrip("/") + "/"):
            if best is None or len(mount_point) > len(best[1]):
                best = (device, mount_point, fstype, options)
    if best is None or best[1] == "/" or (best[2] not in ("cifs", "smb3", "nfs", "nfs4", "fuseblk")
                                          and not best[2].startswith("fuse.") and "ro" not in best[3]):
        return None
    device, mount_point, fstype, options = best
    quote = shlex.quote
    fix = {"mount_point": mount_point, "type": fstype, "source": device, "read_only": "ro" in options,
           "can_fix": fstype in ("cifs", "smb3") or fstype.startswith("fuse.") or "ro" in options,
           # "Fix it" in the dialog: only what root alone can change; a FUSE mount needs its owner
           "run_here": fstype in ("cifs", "smb3") or "ro" in options,
           "command": f"sudo aurora-tools network fix-mount {quote(mount_point)}", "manual": ""}
    if fstype in ("cifs", "smb3"):
        user = next((o.split("=", 1)[1] for o in options if o.startswith("username=")), "")
        extra = f"uid={os.getuid()},gid=aurora-drives,file_mode=0664,dir_mode=0775"
        fix["manual"] = (f"sudo umount {quote(mount_point)} && sudo mount -t cifs {quote(device)} {quote(mount_point)} "
                         f"-o {('user=' + user + ',') if user else ''}{extra}")
        fix["manual_note"] = "Put the same options in /etc/fstab to keep them after a reboot."
    elif fstype.startswith("fuse."):
        tool = "sshfs" if fstype == "fuse.sshfs" else "<the program that mounted it>"
        # gid/umask make the files look like they belong to group aurora-drives, which the shared
        # service is in; the server still decides what this login may change. -uz detaches a
        # mount something still has open (the new mount takes its place).
        fix["manual"] = (f"fusermount3 -uz {quote(mount_point)} && {tool} {quote(device)} {quote(mount_point)} "
                         "-o allow_other,default_permissions,uid=$(id -u),gid=$(getent group aurora-drives | cut -d: -f3),umask=0002")
        fix["manual_note"] = "Run step 2 as yourself, without sudo, so the mount keeps your login."
    elif fstype in ("nfs", "nfs4"):
        fix["manual_note"] = "The file server decides: give the aurora-tools user access on it."
    return fix


def run_fix_mount(path: str) -> tuple:
    """Run the helper's fix-mount with the system's password window (desktop Aurora only;
    the shared service has no screen, so there the dialog shows the command to type)."""
    fix = mount_fix(path)
    if not fix or not fix["run_here"]:
        return False, "This folder is not on a mount Aurora can change from here."
    try:
        result = subprocess.run(["pkexec", HELPER, "fix-mount", fix["mount_point"]],
                                capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        return False, "pkexec is missing. In a terminal: " + fix["command"]
    except subprocess.TimeoutExpired:
        return False, "The password window was not answered in time."
    if result.returncode in (126, 127) and "Error:" not in (result.stderr or ""):
        return False, "Nothing was changed: the password was not entered."
    text = (result.stdout or "").strip().splitlines()
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, (detail[-1] if detail else "The change failed.").replace("Error: ", "")
    logger.info("Mount %s changed for the shared Aurora: %s", fix["mount_point"], text[-1] if text else "")
    return True, text[-1] if text else "Done."


def remount_shared_drives(folders: list) -> list:
    """Linux: unmount and mount again the USB / Windows-format drives (exFAT, FAT, NTFS)
    holding shared folders, so they pick up the mount options aurora-tools-network enable
    just wrote (group aurora-drives may write; setup_drive_access). Those drives have no
    file permissions of their own: the options are fixed when a drive is mounted, so one
    that was already plugged in would stay read-only for the shared Aurora.

    Runs here, as the person at the computer in their own desktop session, through udisks
    (udisksctl), exactly like ejecting and plugging the drive in again: same device, same
    /run/media/<user>/<label> path, and udisks refuses to unmount while any file on the
    drive is open, so nothing is cut off mid-write. Returns one note per drive for the
    Turn on result; a drive that could not be remounted keeps the eject-and-replug advice."""
    if not sys.platform.startswith("linux"):
        return []
    try:
        import grp
        drives_gid = grp.getgrnam("aurora-drives").gr_gid
    except (ImportError, KeyError):
        return []   # helper wrote no drive settings (no folder on /run/media or /media)
    table = _mount_table()
    todo = {}
    for folder in folders:
        real = os.path.realpath(os.path.expanduser(str(folder)))
        best = None
        for device, mount_point, fstype, options in table:
            if real == mount_point or real.startswith(mount_point.rstrip("/") + "/"):
                if best is None or len(mount_point) > len(best[1]):
                    best = (device, mount_point, fstype, options)
        if best is None:
            continue
        device, mount_point, fstype, options = best
        if fstype not in _NO_PERMISSION_FS:
            continue
        if not mount_point.startswith(("/run/media/", "/media/")) or not device.startswith("/dev/"):
            continue
        if f"gid={drives_gid}" in options:
            continue   # already mounted with the shared Aurora's group
        todo[device] = mount_point
    notes = []
    for device, mount_point in todo.items():
        name = os.path.basename(mount_point)
        replug = f"  {name}: the shared Aurora can only open files there; turn sharing off and on again to retry"
        try:
            off = subprocess.run(["udisksctl", "unmount", "-b", device, "--no-user-interaction"],
                                 capture_output=True, text=True, timeout=60)
            if off.returncode != 0:
                reason = (off.stderr or off.stdout or "").strip().splitlines()
                logger.warning("Could not unmount %s (%s): %s", mount_point, device, reason[-1] if reason else "")
                users = _programs_using(mount_point)
                notes.append(replug + (f" (in use now by {', '.join(users)}; close them first)." if users
                                       else " (it is in use now; close files open on it first)."))
                continue
            # The options are passed explicitly: desktop automounters (KDE, GNOME) send
            # their own gid= with every mount, which wins over udisks' mount_options.conf
            # (seen on Alejandro's computer). udisks allows gid= of a group the caller is in,
            # which is why the helper adds the owner to aurora-drives.
            on = subprocess.run(["udisksctl", "mount", "-b", device, "--no-user-interaction",
                                 "-o", f"gid={drives_gid},umask=0002"],
                                capture_output=True, text=True, timeout=60)
            if on.returncode != 0:
                logger.warning("Mounting %s with the shared Aurora's group failed (%s); mounting it as before.",
                               device, (on.stderr or on.stdout or "").strip())
                on = subprocess.run(["udisksctl", "mount", "-b", device, "--no-user-interaction"],
                                    capture_output=True, text=True, timeout=60)
                if on.returncode == 0:
                    notes.append(f"  {name}: could not be mounted for the shared Aurora to save there.")
                    continue
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("Could not remount %s (%s): %s", mount_point, device, exc)
            notes.append(replug + ".")
            continue
        if on.returncode != 0:
            reason = (on.stderr or on.stdout or "").strip().splitlines()
            logger.warning("Could not mount %s (%s) again: %s", mount_point, device, reason[-1] if reason else "")
            notes.append(f"  {name}: was unmounted but could not be mounted again; plug it in again.")
            continue
        now = next((mp for dev, mp, _fs, _opts in _mount_table() if dev == device), None)
        if now and now != mount_point:
            notes.append(f"  {name}: mounted again at {now} (not {mount_point}); share that folder instead.")
        else:
            notes.append(f"  {name}: remounted so the shared Aurora can save there.")
        logger.info("Remounted %s (%s) for the shared Aurora", mount_point, device)
    return notes


def _folder_args(folders: list) -> list:
    return [item for folder in folders for item in ("--folder", folder)]


def _owner_args(pair_owner: str) -> list:
    owner = str(pair_owner or "")
    return ["--pair-owner", owner] if owner.isdigit() else []


def _turn_on_windows(folders: list, port: int, pair_owner: str = "") -> tuple:
    """UAC prompt, then the admin tool in its own (hidden) window; it reports back through
    a result file because an elevated process cannot be read from here."""
    result_file = Path(tempfile.gettempdir()) / f"aurora-sharing-{secrets.token_hex(6)}.json"
    args = ["-I", str(_admin_tool()), "network", "enable", *_folder_args(folders), "--port", str(port),
            *_owner_args(pair_owner), "--owner-home", str(Path.home()), "--result-file", str(result_file)]
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


def _turn_on_mac(folders: list, port: int, pair_owner: str = "") -> tuple:
    """macOS administrator password window (osascript), then the admin tool as root."""
    python = os.path.realpath(sys.executable)   # the pinned, root-owned CPython behind the venv
    command = " ".join(shlex.quote(str(part)) for part in [
        python, "-I", MAC_ADMIN_TOOL, "network", "enable", *_folder_args(folders), "--port", str(port),
        *_owner_args(pair_owner),
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
            # Said in the log too: the access log alone only shows "Bad Request", which
            # tells nothing when a report or the terminal is all there is to go on.
            logger.warning("Sharing action %r by %s refused: %s", action, record.get("name") or "someone", message)
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
            if stray_share(service, shared):
                engineConfig.write_config(mode=engineConfig.MODE_LOCAL)
                logger.warning("Sharing (left on by a developer run) turned off by %s; the network port "
                               "closes when Aurora is opened again.", record.get("name"))
                from pipeline import engineAuth  # engineAuth imports this module
                engineAuth.end_sessions_where(lambda r: bool(r.get("remote")), "Sharing was turned off on this computer.")
                return done(reopen_needed=True)
            if not service:
                return refuse("Sharing is not running as a service here; nothing to turn off.")
            try:
                path = _turn_off_path()
                path.write_text(json.dumps({"by": record.get("name"), "at": time.time()}) + "\n", encoding="utf-8")
            except OSError as exc:
                return refuse(f"Could not ask to turn sharing off: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)
            logger.warning("Turning network sharing off, asked by %s", record.get("name"))
            return done()

        if action in ("pair_start", "pair_confirm", "pair_cancel", "pair_auto_confirm"):
            if record.get("remote"):
                return refuse("Pair this computer from Aurora at the computer itself.")
            try:
                owner_id = networkMode.pair_owner_id() if service else ""
                if owner_id and str(record.get("user_id") or "") != owner_id and action != "pair_cancel":
                    # Sharing was turned on from Settings by someone else: only they pair it
                    # from Aurora (an admin can still pair by code: sudo aurora-tools pair).
                    return refuse("Only the person who turned sharing on can pair this computer from Aurora.")
                if (action == "pair_confirm" and owner_id and data.get("accept")
                        and networkMode.claimed_by_id() not in ("", owner_id)):
                    return refuse("This code was used by another lab account; cancel and pair again.")
                if action == "pair_start":
                    pairing = networkMode.pairing_state() or {}
                    if pairing and not _paired_elsewhere(record, pairing):
                        return refuse("This computer is already paired.")
                    # The shared service's working pairing with another lab website changes
                    # owner only with admin rights; restoring one this lab already lists for
                    # the signed-in person is not a change of owner, so that still goes ahead.
                    restore_only = bool(pairing and service and not _pairing_unusable(pairing))
                    # Desktop Aurora paired with another lab website (e.g. a test site): the old
                    # pairing stays until the new one is confirmed. Either way the pairing is
                    # restored instead when the lab already lists this computer with the
                    # signed-in person as its owner (networkMode.pair_start).
                    networkMode.pair_start(record.get("lab_origin") or "", data.get("name") or "",
                                           user_id=str(record.get("user_id") or ""),
                                           restore_only=restore_only)
                elif action == "pair_confirm":
                    networkMode.pair_confirm(bool(data.get("accept")))
                elif action == "pair_auto_confirm":
                    networkMode.pair_auto_confirm(record.get("user_id") or "", record.get("lab_origin") or "")
                else:
                    networkMode.pair_cancel()
            except ValueError as exc:
                return refuse(str(exc))
            return done()

        if action == "member_folders":
            # Narrows what members see inside the shared folders; no OS permission changes.
            if not shared:
                return refuse("Turn sharing on first.")
            members = data.get("members") if isinstance(data.get("members"), dict) else {}
            restrict_new = data.get("restrict_new")
            if restrict_new is None:
                restrict_new = networkFolders.member_folder_settings()["restrict_new"]
            try:
                saved = networkFolders.save_member_folders(bool(restrict_new), members)
            except ValueError as exc:
                return refuse(str(exc))
            except OSError as exc:
                return refuse(f"Could not save: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)
            logger.info("Member folders changed by %s: %s (new members %s)", record.get("name"),
                        {k: v for k, v in saved["members"].items() if k in members},
                        "see nothing until given folders" if saved["restrict_new"] else "see all shared folders")
            return done()

        if action == "cancel_request":
            try:
                _request_path().unlink()
            except OSError:
                pass
            return done()

        if action == "check_folders":
            # Settings shows these before sharing is turned on or a change is asked for:
            # refused folders (with the reason) and the risks of the others. Changes nothing.
            raw = data.get("folders") if isinstance(data.get("folders"), list) else []
            checks = [review_folder(f.strip(), must_exist=not service)
                      for f in raw[:MAX_FOLDERS] if isinstance(f, str) and f.strip()]
            return Response({"checks": checks}, status=status.HTTP_200_OK)

        if action == "fix_mount":
            # Warning icon beside the Working Directory path: change the mount holding the
            # open folder so Aurora can save there (network drive mounted by hand, read-only).
            if record.get("remote") or service:
                return refuse("Do this at the computer: run the command shown in the dialog in a terminal.")
            if not sys.platform.startswith("linux"):
                return refuse("Changing a mount from here is only available on Linux.")
            ok, detail = run_fix_mount(str(data.get("path") or ""))
            if not ok:
                return refuse(detail)
            return done(message=detail)

        if action == "remember_folders":
            # Sharing off: the folders chosen in Settings are kept for when it is turned on
            # (nothing is shared or granted yet; only "Turn on" asks for admin rights).
            if shared or record.get("remote"):
                return refuse("Change the shared folders with a request while sharing is on.")
            try:
                folders = _clean_folders(data.get("folders"))
            except ValueError as exc:
                return refuse(str(exc))
            engineConfig.write_config(shared_folders=folders)
            return done()

        if action == "recheck_access":
            if not service:
                return refuse("Folder access is checked by the shared service.")
            sharedFolderAccess.start_checks(current_folders(), force=True)
            return done()

        if action == "browse_access":
            # Settings' folder view of what members cannot open (one level at a time).
            if not service:
                return refuse("Folder access is checked by the shared service.")
            folder = str(data.get("folder") or "")
            if folder not in current_folders():
                return refuse("That folder is not one of the shared folders.")
            try:
                return Response(sharedFolderAccess.browse(folder, str(data.get("path") or "")),
                                status=status.HTTP_200_OK)
            except ValueError as exc:
                return refuse(str(exc))

        if action == "request_full_access":
            # "Give members access to everything in this folder": the owner's explicit choice
            # per folder, applied with admin rights at the computer (aurora-tools confirm).
            if not service or not (sys.platform.startswith("linux") and os.access(HELPER, os.X_OK)):
                return refuse("Giving access to everything in a folder is done by the shared service on Linux.")
            if record.get("remote"):
                # Confirmed without a y/N question once the admin password is given, so only
                # the person at this computer may ask; others get the grant-all command.
                return refuse("Do this at the computer itself, or run the command shown in a terminal there.")
            folder = str(data.get("folder") or "")
            now_folders = current_folders()
            if folder not in now_folders:
                return refuse("That folder is not one of the shared folders.")
            # Optional "path": a folder or file inside it, chosen in Settings' folder view, so
            # only that part is opened up (Alejandro, 2026-11-04). The helper checks it again.
            try:
                target, rel = sharedFolderAccess.resolve_inside(folder, str(data.get("path") or ""))
            except ValueError as exc:
                return refuse(str(exc))
            last = next((e for e in sharedFolderAccess.status(now_folders) if e.get("folder") == folder), {})
            if rel == "." and last.get("checked_at") and last.get("complete") and not last.get("fixable", 1):
                # Everything left belongs to other accounts: the helper would change nothing.
                return refuse("Nothing in that folder that members cannot open belongs to its owner; "
                              "the accounts that own those items decide.")
            pending = pending_request()
            full = [target]
            if pending:
                same = (pending.get("folders") == now_folders
                        and int(pending.get("port") or 0) == engineConfig.load_config()["network_port"])
                if not same:
                    return refuse("A change is already waiting for confirmation. Confirm or cancel it first.")
                full = sorted(set(pending.get("full_access") or []) | {target})
            request_data = {
                "id": secrets.token_hex(8),
                "requested_at": time.time(),
                "expires_at": time.time() + REQUEST_TTL_SECONDS,
                "by": {"name": str(record.get("name") or "")[:120], "role": session_role(record),
                       "user_id": str(record.get("user_id") or "")},
                "folders": now_folders,
                "port": engineConfig.load_config()["network_port"],
                "full_access": full,
            }
            path = _request_path()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(request_data, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, path)
            logger.info("Full access to %s requested by %s", ", ".join(full), record.get("name"))
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
            ok, detail = turn_on(folders, port, record.get("user_id") or "")
            if not ok:
                return refuse(detail)
            # Remember the choice for next time (this desktop config stays in local mode).
            engineConfig.write_config(shared_folders=folders, network_port=port)
            if dev_mode():
                # run_aurora.sh starts again, waits while the service runs this checkout,
                # and comes back here once sharing is turned off.
                _restart_soon(3.0)
            else:
                _quit_soon(3.0)
            return done(handover=True, helper_output=detail)

        return refuse("Unknown action.")
