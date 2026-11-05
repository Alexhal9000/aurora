"""Cancel a running task by restarting this Aurora engine (universal cancel, 2026-11).

Aurora's tools run inside the engine process (the request thread, worker threads, and
worker subprocesses such as antsWorker, the ALPACA RANSAC pool, the grid preview pool or
fine-tuning). Stopping one tool cleanly from the outside would need cancel points in
every tool. Instead the cross on the progress bar ends the whole engine at once, and
whatever started the engine brings it back:

* POST /engine-cancel/ checks who asks (may_cancel), marks the running jobs "cancelled"
  in the job history, answers, and a second later kills every child process of the
  engine and the engine itself with exit code CANCEL_EXIT_CODE (os._exit: no clean-up
  code runs, exactly like a crash).
* The crash-safe protocol (pipeline/crashSafety.py) makes that safe: every pipeline
  write is atomic, every subject change is a transaction, and the next request naming a
  project (RecoverInterruptedWork) or opening it (GetProjectSettingsView) rolls back the
  subject that was being worked on. Subjects a batch already finished stay finished.
  Leftover ANTs temp folders are swept by engine pid (antsWorker.sweep_stale_temp).
* Who brings the engine back, per way of running (each sets AURORA_CANCEL_RESTART=1, so
  an engine started any other way refuses to cancel instead of staying dead):
    - Linux shared service (deployment/linux/aurora-service): systemd unit
      Restart=on-failure, RestartSec=5 (aurora-tools-network write_unit). Exit 76 is a
      failure, so systemd starts it again and kills leftovers of the old cgroup.
    - macOS shared service: launchd KeepAlive {SuccessfulExit: false}, ThrottleInterval 10
      (aurora_sharing_admin.py) restarts any non-zero exit.
    - Windows shared service: the startup task's RestartCount/RestartInterval does not
      restart a task whose program exited, so aurora_service_runner.py supervises its own
      engine process and starts it again on exit 76.
    - Desktop Aurora (Linux /usr/bin/aurora-tools, Windows/macOS launcher.py): the
      launcher sees exit 76, opens Aurora again in a new window (the desktop app), and
      its own window closes.
    - Developer run (run_aurora.sh): restarts in the same terminal (like exit 75).
* The browser shows a full-page "Cancelling" screen meanwhile, blocks its own engine
  requests (browser-side loops must not reach the new engine), waits for /get-version/
  to answer with a new boot_id (at least CANCEL_MIN_WAIT in the page) and reloads.

Who may cancel (may_cancel): anyone at this computer (local mode, or the person at the
shared computer), the shared computer's owner and managers; a network member only when
no running job belongs to someone else, and they hold the computer or a job of theirs
is running (a restart ends everyone's work on this engine).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from pipeline import engineJobs, engineSharing

logger = logging.getLogger(__name__)

CANCEL_EXIT_CODE = 76          # launchers and service runners start Aurora again on it
RESTART_ENV = "AURORA_CANCEL_RESTART"
EXIT_DELAY_SECONDS = 1.0       # lets the answer reach the browser first
CHILD_TERM_SECONDS = 2.0       # children get SIGTERM, then SIGKILL after this

_lock = threading.Lock()
_cancelling = False


def boot_id() -> str:
    """Changes with every engine start (/get-version/ "boot_id"): the page reloads once it
    sees a new one, i.e. the restarted engine answers."""
    from pipeline import crashSafety
    return crashSafety.ENGINE_BOOT_ID


def restart_kind() -> str:
    """How this engine comes back after CANCEL_EXIT_CODE, or "" when nothing would."""
    if os.environ.get(RESTART_ENV) != "1":
        return ""
    if os.environ.get("AURORA_SERVICE") == "1" or os.environ.get("AURORA_DEV_SERVICE") == "1":
        return "service"
    if engineSharing.dev_mode():
        return "developer"
    return "desktop"


def may_cancel(record: dict) -> str:
    """"" when this session may restart the engine, otherwise the reason it may not."""
    role = engineSharing.session_role(record)
    if role in ("local", "owner", "manager"):
        return ""
    user_id = str(record.get("user_id", ""))
    jobs = engineJobs.running_owners()
    others = [name for owner, name in jobs if owner != user_id]
    if others:
        return (f"A task of {others[0] or 'another person'} is running on this computer; only its "
                "owner or a manager can cancel it.")
    from pipeline import engineAuth  # engineAuth imports engineSharing; keep this lazy
    holder = engineAuth.lease_holder()
    if jobs or (holder is not None and holder.get("user_id") == user_id):
        return ""
    return "Only the person using this computer, its owner or a manager can cancel a task."


def _kill_children() -> None:
    """Every process this engine started (worker subprocesses, spawn pools, their
    children), so none keeps working or holds files after the engine is gone. systemd
    would kill them anyway; Windows and the desktop launchers would not."""
    try:
        import psutil
    except ImportError:
        logger.warning("Cancel: psutil missing, worker processes are not stopped.")
        return
    try:
        children = psutil.Process(os.getpid()).children(recursive=True)
    except Exception:  # noqa: BLE001 - the engine still exits
        return
    for child in children:
        try:
            child.terminate()
        except Exception:  # noqa: BLE001 - already gone, or not ours
            pass
    try:
        _gone, alive = psutil.wait_procs(children, timeout=CHILD_TERM_SECONDS)
    except Exception:  # noqa: BLE001
        alive = children
    for child in alive:
        try:
            child.kill()
        except Exception:  # noqa: BLE001
            pass


def _exit_now(name: str) -> None:
    time.sleep(EXIT_DELAY_SECONDS)
    print(f"Task cancelled by {name}: Aurora stops now and starts again.", flush=True)
    _kill_children()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(CANCEL_EXIT_CODE)


def cancel_engine(record: dict) -> None:
    """Mark the running jobs cancelled, then end this engine in a second (see module doc)."""
    global _cancelling
    with _lock:
        if _cancelling:
            return
        _cancelling = True
    name = record.get("name") or ("the person at this computer" if not record.get("remote") else "a lab user")
    engineJobs.cancel_running(f"Cancelled by {name}.")
    logger.warning("Cancel requested by %s (%s): restarting Aurora.", name, restart_kind())
    threading.Thread(target=_exit_now, args=(name,), name="aurora-cancel", daemon=True).start()


class EngineCancelView(APIView):
    """POST /engine-cancel/ -> {"ok": true, "boot_id", "restart"} and the engine restarts.

    Lease free (engineAuth.LEASE_FREE_PATH_PREFIXES): the owner or a manager can cancel a
    task while a member holds the computer. 403 with "error" when this session may not,
    409 when nothing would start Aurora again."""

    authentication_classes = []
    permission_classes = []

    def post(self, request, *args, **kwargs):
        record = getattr(request, "aurora_session", None)
        if record is None:
            return Response({"code": "engine_session_required"}, status=status.HTTP_401_UNAUTHORIZED)
        kind = restart_kind()
        if not kind:
            return Response({
                "error": "This Aurora was not started from the Aurora app, so it could not start again "
                         "by itself. Stop it in its terminal (Ctrl+C) and start it again instead.",
                "code": "cancel_no_restart"}, status=status.HTTP_409_CONFLICT)
        refusal = may_cancel(record)
        if refusal:
            return Response({"error": refusal, "code": "cancel_forbidden"}, status=status.HTTP_403_FORBIDDEN)
        cancel_engine(record)
        return Response({"ok": True, "boot_id": boot_id(), "restart": kind}, status=status.HTTP_200_OK)
