"""Who is using this Aurora engine right now, and what is it running (Aurora 1.9.4+).

Feeds the "In use by …" line on the lab's My Aurora servers page:

    Ana Gómez has Aurora open · nothing running
    Ana Gómez is running "Denoising scan_12 using NLM..." · 42% · about 12 min left
    Ana Gómez's job is still running (Aurora is closed)

Two inputs, both in memory and cheap:

* Presence: the packaged SPA posts /engine-auth/heartbeat/ every 30 s while a tab is
  open (and once with leaving=1 when the tab closes). A session is "present" if it beat
  within PRESENCE_SECONDS; background tabs are throttled by browsers to ~1/min, hence
  the margin.
* Jobs: engineJobs keeps the list of running jobs (long requests, progress messages)
  and their history. This module passes the middleware's and asgi.py's reports on to it.

Stdlib only (networkMode imports this before Django is configured). Thread safe: heartbeats
arrive on Django worker threads, progress from tool threads, snapshots from the check-in
thread.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

from pipeline import engineJobs

PRESENCE_SECONDS = 150
LABEL_MAX = 160

_lock = threading.Lock()
_presence: Dict[str, dict] = {}     # session key -> {user_id, name, last_beat, first_beat}
_last_activity: Dict[str, float] = {}  # session key -> last heartbeat or request (lease grace)
_listeners: list = []


def _now() -> float:
    return time.time()


def on_change(callback: Callable[[], None]) -> None:
    """Call `callback` (no args, must be quick) whenever someone arrives/leaves or a job
    starts/ends. networkMode uses it to check in right away instead of in 5 minutes."""
    _listeners.append(callback)


def _notify() -> None:
    for callback in list(_listeners):
        try:
            callback()
        except Exception:  # noqa: BLE001 - a listener must never break a request
            pass


engineJobs.on_change(_notify)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def heartbeat(session_key: str, record: dict, leaving: bool = False) -> None:
    """A tab of this session is open (or, with leaving=True, was just closed)."""
    now = _now()
    with _lock:
        was_present = _present_users_locked(now)
        _last_activity[session_key] = now
        if leaving:
            _presence.pop(session_key, None)
        else:
            entry = _presence.setdefault(session_key, {"first_beat": now})
            entry.update({
                "user_id": str(record.get("user_id", "")),
                "name": str(record.get("name", "")),
                "last_beat": now,
            })
        changed = was_present != _present_users_locked(now)
    if changed:
        _notify()


def forget_session(session_key: str) -> None:
    """Session ended (sign-out, kick): it is no longer present."""
    with _lock:
        removed = _presence.pop(session_key, None) is not None
        _last_activity.pop(session_key, None)
    if removed:
        _notify()


def request_started(record: dict, request, session_key: str = "") -> Optional[str]:
    if session_key:
        touch(session_key)
    return engineJobs.request_started(record, request, session_key)


def request_finished(token: Optional[str], status_code: int = 200, error: str = "") -> None:
    # Only jobs were shown on the lab page; their end is worth an immediate update.
    if engineJobs.request_finished(token, status_code, error):
        _notify()


def observe_progress(message: dict) -> None:
    """Called with every progress_group message (see asgi.py)."""
    if engineJobs.observe_progress(message):
        _notify()


def touch(session_key: str) -> None:
    """Note activity for a session (lease claimed or refreshed)."""
    with _lock:
        _last_activity[session_key] = _now()


def session_idle_seconds(session_key: str) -> float:
    """Seconds since this session last beat or made a request; 0 while a request runs.
    The lease (engineAuth) frees a holder that has been idle longer than its grace period."""
    if engineJobs.session_busy(session_key):
        return 0.0
    now = _now()
    with _lock:
        last = _last_activity.get(session_key)
    return float("inf") if last is None else max(0.0, now - last)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def _present_users_locked(now: float) -> frozenset:
    return frozenset(
        entry["user_id"] for entry in _presence.values()
        if now - entry["last_beat"] <= PRESENCE_SECONDS
    )


def snapshot() -> dict:
    """Current state, shaped for the lab check-in.

    {"user": {"user_id", "name", "present", "since"} | None,
     "job": {"label", "percent", "elapsed", "eta_seconds"} | None}
    """
    now = _now()
    with _lock:
        present = [
            entry for entry in _presence.values()
            if now - entry["last_beat"] <= PRESENCE_SECONDS
        ]
    jobs = engineJobs.running()
    current = jobs[0] if jobs else None  # oldest

    job = None
    if current is not None:
        fraction = current["progress"]
        job = {
            "label": str(current["label"])[:LABEL_MAX],
            "percent": None if fraction is None else round(fraction * 100.0, 1),
            "elapsed": current["elapsed"],
            "eta_seconds": current["eta_seconds"],
        }

    user = None
    if present:
        latest = max(present, key=lambda entry: entry["last_beat"])
        user = {
            "user_id": latest["user_id"],
            "name": latest["name"],
            "present": True,
            "since": min(e["first_beat"] for e in present if e["user_id"] == latest["user_id"]),
        }
    elif current is not None and current["user_id"]:
        user = {
            "user_id": current["user_id"],
            "name": current["name"],
            "present": False,
            "since": current["started"],
        }
    return {"user": user, "job": job}


def state_key() -> tuple:
    """Coarse state used to decide whether the lab needs an update now."""
    snap = snapshot()
    user, job = snap["user"], snap["job"]
    return (
        user["user_id"] if user else None,
        bool(user and user["present"]),
        engineJobs.label_stem(job["label"]) if job else None,
        # Each further 10% of progress is worth an update (still at most every
        # MIN_EVENT_GAP_SECONDS, see networkMode).
        int(job["percent"] // 10) if job and job["percent"] is not None else None,
    )
