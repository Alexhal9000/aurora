"""What this Aurora engine is running, and what it ran before (Aurora 1.9.4+).

The engine is the source of truth for jobs, not the browser tab that started them. A tab
can close, a laptop can sleep: the job keeps running here, and any Aurora tab opened
later asks GET /engine-jobs/ and shows it again ("reattaches"), locally or over the
network alike.

What counts as a job
* A signed-in request that changes something (not GET, not a read-only route in
  _READ_ONLY_PREFIXES) and is still running after JOB_AFTER_SECONDS, or any request that
  sends progress messages (Aurora shows a progress bar for it). Most tools run inside
  their HTTP request; EngineSessionMiddleware reports when each one starts and ends.
* Progress messages from a thread that keeps working after its request returned (no
  request is running when they arrive): a "background" job that ends at 100% or when its
  messages stop for PROGRESS_FRESH_SECONDS.
Progress (label, current/total, fraction 0..1) comes from the progress_group channel
messages every tool already sends (asgi.py wraps group_send); no tool code changes.
Staged tools (messages with timing "elapsed", see views._send_step_progress) also get
their steps recorded: the end line lists each step's label and seconds, so step-time
ratios can be learned from the log later.

History log
Every job writes one "start" line once it counts as a job and one "end" line when it
finishes, to ~/.aurora/job_history.jsonl (about 300-600 bytes each; progress is never
written). When the file would pass LOG_MAX_BYTES it becomes job_history.1.jsonl and a new
file starts, so the two files never hold more than about 5 MB. A start without an end
means the engine stopped mid-job (crash, power cut, update): the next start marks it
"interrupted". Writing is best effort: a full disk or a read-only folder skips the line
and never touches the job.

Stdlib only (networkMode imports engineActivity, which imports this, before Django is
configured). Thread safe: requests start and end on Django worker threads, progress
arrives from tool threads, the watcher runs on its own thread.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

from AuroraClient import engineConfig

JOB_AFTER_SECONDS = 5
PROGRESS_FRESH_SECONDS = 120     # a background job silent this long is over
WATCH_SECONDS = 5
RECENT_MAX = 50                  # finished jobs kept in memory for /engine-jobs/?history=1
LOG_NAME = "job_history.jsonl"
LOG_MAX_BYTES = 2_621_440        # 2.5 MB per file, two files at most (5 MB)
BODY_PEEK_MAX_BYTES = 256 * 1024 # request bodies read for directory / scans, JSON only
LABEL_MAX = 160
ERROR_MAX = 500
PATH_MAX = 300
SCAN_NAMES_MAX = 5
STEPS_MAX = 30                   # staged-tool steps kept per job for the end line

# Never jobs, even when slow: sign-in, polling, files streamed to the browser, previews and
# other reads that only show something in the tab that asked.
_IGNORED_PREFIXES = ("/engine-auth/", "/engine-jobs/", "/engine-sharing/", "/get-version/", "/local_aurora/",
                     "/static/", "/docs/")
_READ_ONLY_PREFIXES = (
    "/upload", "/list-", "/get-", "/load-", "/display-", "/preview-", "/download-",
    "/quick-access-paths/", "/projection-png/", "/ensure-quick-grid-projections/",
    "/extract-scans-preflight/", "/nifti-header-info/", "/full-resolution-value-histogram/",
    "/elastic-slice-ncc-heatmap/", "/edge-detection-preview/", "/registration-heatmap/",
    "/registration-subject-data/", "/memory-monitor/", "/ai-models/foundation/status/",
    "/ai-finetune/discover/", "/ai-finetune/validate/", "/ai-finetune/gpu/",
)

# Finished jobs reported to the lab's Activity list with the next check-in (networkMode):
# tool, who, scan count, duration, result. Never folders or scan names; error text has
# paths removed. Only jobs that ran REPORT_MIN_SECONDS or were interrupted.
REPORT_MIN_SECONDS = 60
REPORT_QUEUE_MAX = 50
REPORT_ERROR_MAX = 140

_lock = threading.Lock()
_unreported: deque = deque(maxlen=REPORT_QUEUE_MAX)
_running: Dict[str, dict] = {}       # job id -> job (requests not yet jobs included)
_recent: deque = deque(maxlen=RECENT_MAX)
_loaded = False
_watcher_started = False


def _now() -> float:
    return time.time()


def _log_path() -> Path:
    return engineConfig.config_dir() / LOG_NAME


def _clip(value, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _humanize_path(path: str) -> str:
    words = re.sub(r"[-_/]+", " ", path).strip()
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", words)
    return words[:1].upper() + words[1:].lower() if words else "Working"


def label_stem(label: str) -> str:
    # "Denoising scan_12 using NLM..." and "Denoising scan_13 using NLM..." are one job.
    return re.sub(r"\S*\d\S*", "", label or "").strip().lower()[:40]


# ---------------------------------------------------------------------------
# History file
# ---------------------------------------------------------------------------

def _append(entry: dict) -> None:
    """Best effort: one JSON line, rotating at LOG_MAX_BYTES. Never raises."""
    try:
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        data = line.encode("utf-8")
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size and size + len(data) > LOG_MAX_BYTES:
            os.replace(path, path.with_name(LOG_NAME.replace(".jsonl", ".1.jsonl")))
        with open(path, "ab") as handle:
            handle.write(data)
    except Exception:  # noqa: BLE001 - history must never break a job
        pass


def _read_lines(path: Path) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue  # a line cut short by a crash
        if isinstance(entry, dict) and entry.get("id"):
            entries.append(entry)
    return entries


def load() -> None:
    """Read the history once at startup: fill the recent list and mark jobs the engine was
    running when it last stopped as interrupted. Safe to call more than once."""
    global _loaded
    with _lock:
        if _loaded:
            return
        _loaded = True
    path = _log_path()
    entries = _read_lines(path.with_name(LOG_NAME.replace(".jsonl", ".1.jsonl"))) + _read_lines(path)
    starts: Dict[str, dict] = {}
    finished: List[dict] = []
    for entry in entries:
        if entry.get("event") == "start":
            starts[entry["id"]] = entry
        elif entry.get("event") == "end":
            start = starts.pop(entry["id"], {})
            finished.append({**start, **entry})
    last_seen = entries[-1].get("at") if entries else None
    for job_id, start in starts.items():
        end = {
            "event": "end", "id": job_id, "at": last_seen, "status": "interrupted",
            "error": "Aurora stopped while this was running (closed, crashed, restarted or the "
                     "computer turned off). Its results may be incomplete.",
        }
        _append(end)
        finished.append({**start, **end})
        _queue_report(_public_fields({**start, **end}))
    finished.sort(key=lambda job: job.get("started") or 0)
    with _lock:
        for job in finished[-RECENT_MAX:]:
            _recent.append(_public_fields(job))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _request_details(request) -> dict:
    """Directory and scans from a JSON request body, when small enough to peek at."""
    details: dict = {}
    try:
        if not str(request.META.get("CONTENT_TYPE", "")).startswith("application/json"):
            return details
        length = int(request.META.get("CONTENT_LENGTH") or 0)
        if length <= 0 or length > BODY_PEEK_MAX_BYTES:
            return details
        body = json.loads(request.body or b"{}")  # Django keeps the body for the view
    except Exception:  # noqa: BLE001 - details are optional
        return details
    if not isinstance(body, dict):
        return details
    for key in ("directory", "selectedDirectory", "target_folder"):
        if isinstance(body.get(key), str) and body[key]:
            details["directory"] = _clip(body[key], PATH_MAX)
            break
    for key in ("scans", "scan_names", "scanNames", "subjects"):
        scans = body.get(key)
        if isinstance(scans, list) and scans:
            names = []
            for scan in scans[:SCAN_NAMES_MAX]:
                name = scan.get("name") if isinstance(scan, dict) else scan
                if isinstance(name, str):
                    names.append(_clip(name, 80))
            details["scan_count"] = len(scans)
            details["scans"] = names
            break
    return details


def _is_job_request(method: str, path: str) -> bool:
    if method in ("GET", "HEAD", "OPTIONS"):
        return False
    return not any(path.startswith(prefix) for prefix in _READ_ONLY_PREFIXES)


def request_started(record: dict, request, session_key: str = "") -> Optional[str]:
    """EngineSessionMiddleware, before the view. Returns the id to pass to
    request_finished, or None for routes that are never tracked."""
    path = getattr(request, "path", "") or ""
    if any(path.startswith(prefix) for prefix in _IGNORED_PREFIXES):
        return None
    _ensure_started()
    method = getattr(request, "method", "GET") or "GET"
    candidate = _is_job_request(method, path)
    job = {
        "id": secrets.token_hex(6),
        "kind": "request",
        "candidate": candidate,          # may become a job (see _is_job_request)
        "visible": False,                # counts as a job now (logged, shown)
        "session": session_key,
        "user_id": str(record.get("user_id", "")),
        "name": _clip(record.get("name", ""), 120),
        "remote": bool(record.get("remote")),
        "path": path,
        "method": method,
        "tool": _humanize_path(path),
        "label": "",
        "started": _now(),
        "progress": None, "current": None, "total": None, "scan_name": "",
        "progress_at": None,
    }
    if candidate:
        job.update(_request_details(request))
    with _lock:
        _running[job["id"]] = job
    return job["id"]


def request_finished(job_id: Optional[str], status_code: int = 200, error: str = "") -> bool:
    """EngineSessionMiddleware, after the view (also after the browser went away: the view
    thread always finishes). Returns True if this request had been shown as a job."""
    if job_id is None:
        return False
    with _lock:
        job = _running.pop(job_id, None)
        if job is None:
            return False
        promoted_now = _maybe_promote_locked(job, _now())
        was_job = job["visible"]
    if promoted_now:
        _log_start(job)  # promoted and finished in the same breath
    if was_job:
        failed = status_code >= 400 or bool(error)
        _finish(job, "failed" if failed else "done",
                error or (f"Request failed ({status_code})." if failed else ""))
    return was_job


def observe_progress(message: dict) -> bool:
    """Every progress_group message (asgi.py). Returns True if a new job appeared."""
    if not isinstance(message, dict):
        return False
    try:
        fraction = float(message.get("progress"))
        if fraction > 1.0:          # tolerate a tool that reports a percentage
            fraction /= 100.0
        fraction = max(0.0, min(1.0, fraction))
    except (TypeError, ValueError):
        fraction = None
    label = _clip(message.get("custom_message"), LABEL_MAX)
    now = _now()
    finished = None
    new_job = None
    with _lock:
        requests = [job for job in _running.values() if job["kind"] == "request"]
        tools = [job for job in requests if job["candidate"]]
        # Read routes that may be computing: any POST, or a GET running for a while
        # (a quick GET is a preview or poll, not the sender of this progress).
        slow_reads = [job for job in requests if not job["candidate"]
                      and (job.get("method") != "GET" or now - job["started"] >= 1.0)]
        if tools:
            # Progress belongs to the newest running tool request (usually the only one);
            # previews and listings the tab makes meanwhile are not candidates.
            target = max(tools, key=lambda job: job["started"])
        elif slow_reads:
            # A computation on a "read" route that reports progress (e.g. the registration
            # error heatmap): Aurora shows a progress bar for it, so it is a job too.
            # Prefer POSTs (computations) over GETs (previews), then the longest running.
            target = min(slow_reads, key=lambda job: (job.get("method") == "GET", job["started"]))
            target["candidate"] = True
        else:
            target = next(
                (job for job in _running.values()
                 if job["kind"] == "background" and (not label or label_stem(label) == label_stem(job["label"]))),
                None,
            )
            if target is None:
                if fraction is not None and fraction >= 1.0:
                    return False  # a final "done" message after its request returned
                target = {
                    "id": secrets.token_hex(6), "kind": "background", "candidate": True,
                    "visible": False, "session": "", "user_id": "", "name": "", "remote": False,
                    "path": "", "tool": "Background task", "label": label, "started": now,
                    "progress": None, "current": None, "total": None, "scan_name": "",
                    "progress_at": None,
                }
                _running[target["id"]] = target
        target.update({
            "label": label or target["label"],
            "progress": fraction if fraction is not None else target["progress"],
            "current": message.get("current") if isinstance(message.get("current"), (int, float)) else target["current"],
            "total": message.get("total") if isinstance(message.get("total"), (int, float)) else target["total"],
            "scan_name": _clip(message.get("scan_name"), 120) or target["scan_name"],
            "progress_at": now,
        })
        if message.get("timing") == "elapsed" and isinstance(message.get("current"), (int, float)):
            steps = target.setdefault("steps", [])
            step = int(message["current"])
            if step > 0 and (not steps or steps[-1]["n"] != step) and len(steps) < STEPS_MAX:
                steps.append({"n": step, "label": label, "t": now})
        if target["kind"] == "background":
            if not target["visible"]:
                target["visible"] = True
                new_job = dict(target)
            if fraction is not None and fraction >= 1.0:
                _running.pop(target["id"], None)
                finished = target
        else:
            if _maybe_promote_locked(target, now):
                new_job = dict(target)
    if new_job:
        _log_start(new_job)
    if finished is not None:
        _finish(finished, "done", "")
    return new_job is not None


def _maybe_promote_locked(job: dict, now: float) -> bool:
    """A candidate request becomes a job once it has run JOB_AFTER_SECONDS."""
    if job["visible"] or not job["candidate"] or now - job["started"] < JOB_AFTER_SECONDS:
        return False
    job["visible"] = True
    return True


def _log_start(job: dict) -> None:
    _append({
        "event": "start", "id": job["id"], "at": round(_now(), 1), "started": round(job["started"], 1),
        "tool": job["tool"], "label": job["label"], "path": job["path"], "kind": job["kind"],
        "user_id": job["user_id"], "name": job["name"], "remote": job["remote"],
        "directory": job.get("directory", ""), "scan_count": job.get("scan_count"),
        "scans": job.get("scans", []),
    })


def _finish(job: dict, status: str, error: str) -> None:
    ended = _now()
    entry = {
        "event": "end", "id": job["id"], "at": round(ended, 1), "status": status,
        "error": _clip(error, ERROR_MAX), "label": job["label"],
        "progress": job["progress"], "current": job["current"], "total": job["total"],
    }
    steps = job.get("steps")
    if steps:
        # Seconds per step: until the next step started, the last one until the job ended.
        entry["steps"] = [
            {"n": step["n"], "label": step["label"],
             "seconds": round((steps[i + 1]["t"] if i + 1 < len(steps) else ended) - step["t"], 1)}
            for i, step in enumerate(steps)
        ]
    _append(entry)
    record = _public_fields({**job, **entry})
    with _lock:
        _recent.append(record)
    _queue_report(record)


_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|~?/)[^\s'\",;:)]+")


def _queue_report(record: dict) -> None:
    if not record.get("user_id") or record.get("status") not in ("done", "failed", "interrupted"):
        return
    started, ended = record.get("started"), record.get("ended")
    seconds = (ended - started) if isinstance(started, (int, float)) and isinstance(ended, (int, float)) else None
    if record["status"] != "interrupted" and (seconds is None or seconds < REPORT_MIN_SECONDS):
        return
    error = _PATH_RE.sub("…", str(record.get("error") or "")) if record["status"] != "done" else ""
    with _lock:
        _unreported.append({
            "id": record.get("id"),
            "user_id": record.get("user_id"),
            "tool": _clip(record.get("tool"), 60),
            "status": record["status"],
            "seconds": None if seconds is None else int(max(0, seconds)),
            "scan_count": record.get("scan_count"),
            "error": _clip(error, REPORT_ERROR_MAX),
            "ended": ended,
        })


def take_reports() -> List[dict]:
    """Finished jobs not yet sent to the lab (removed from the queue; see return_reports).
    "ended_ago" (seconds) instead of a timestamp, so the lab's clock decides."""
    now = _now()
    with _lock:
        items = list(_unreported)
        _unreported.clear()
    out = []
    for item in items:
        item = dict(item)
        ended = item.pop("ended", None)
        item["ended_ago"] = int(max(0, now - ended)) if isinstance(ended, (int, float)) else None
        out.append((item, ended))
    return out


def return_reports(items: List[tuple]) -> None:
    """Put reports back after a failed check-in (oldest first, queue limit applies)."""
    with _lock:
        for item, ended in reversed(items):
            if len(_unreported) < REPORT_QUEUE_MAX:
                _unreported.appendleft({**item, "ended": ended})


# ---------------------------------------------------------------------------
# Watcher: promotes long requests (writes their start line, so a crash mid-job is
# visible later) and ends background jobs that went quiet.
# ---------------------------------------------------------------------------

def _ensure_started() -> None:
    global _watcher_started
    if _watcher_started:
        return
    with _lock:
        if _watcher_started:
            return
        _watcher_started = True
    load()
    threading.Thread(target=_watch, name="aurora-jobs", daemon=True).start()


def _watch() -> None:
    while True:
        time.sleep(WATCH_SECONDS)
        try:
            tick()
        except Exception:  # noqa: BLE001 - keep watching
            pass


def tick() -> List[dict]:
    """One watcher pass; returns jobs that started counting (for tests and listeners)."""
    now = _now()
    promoted, stale = [], []
    with _lock:
        for job in list(_running.values()):
            if job["kind"] == "request":
                if _maybe_promote_locked(job, now):
                    promoted.append(dict(job))
            elif now - (job["progress_at"] or job["started"]) > PROGRESS_FRESH_SECONDS:
                _running.pop(job["id"], None)
                stale.append(job)
    for job in promoted:
        _log_start(job)
    for job in stale:
        _finish(job, "done", "")
    if promoted or stale:
        _notify()
    return promoted


_listeners: list = []


def on_change(callback) -> None:
    """Called (no args, must be quick) when a job starts counting or ends."""
    _listeners.append(callback)


def _notify() -> None:
    for callback in list(_listeners):
        try:
            callback()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _public_fields(job: dict) -> dict:
    return {
        "id": job.get("id"),
        "tool": job.get("tool") or "Working",
        "label": job.get("label") or job.get("tool") or "Working",
        "user_id": job.get("user_id", ""),
        "name": job.get("name", ""),
        "remote": bool(job.get("remote")),
        "directory": job.get("directory", ""),
        "scan_count": job.get("scan_count"),
        "scans": job.get("scans") or [],
        "started": job.get("started"),
        "ended": job.get("at") if job.get("event") == "end" else None,
        "status": job.get("status", "running"),
        "error": job.get("error", ""),
        "progress": job.get("progress"),
        "current": job.get("current"),
        "total": job.get("total"),
    }


def eta_seconds(job: dict, now: Optional[float] = None) -> Optional[int]:
    """Linear estimate from the reported fraction; None until there is enough to go on."""
    now = _now() if now is None else now
    fraction = job.get("progress")
    elapsed = now - (job.get("started") or now)
    if fraction is None or not (0.03 <= fraction < 1.0) or elapsed < 20:
        return None
    return int(elapsed * (1.0 - fraction) / fraction)


def running() -> List[dict]:
    """Jobs running now, oldest first."""
    now = _now()
    with _lock:
        jobs = [dict(job) for job in _running.values() if job["visible"] or _would_show(job, now)]
    jobs.sort(key=lambda job: job["started"])
    result = []
    for job in jobs:
        public = _public_fields(job)
        public.update({
            "status": "running",
            "elapsed": int(now - job["started"]),
            "eta_seconds": eta_seconds(job, now),
            "scan_name": job.get("scan_name", ""),
        })
        result.append(public)
    return result


def _would_show(job: dict, now: float) -> bool:
    return job["candidate"] and job["kind"] == "request" and now - job["started"] >= JOB_AFTER_SECONDS


def recent() -> List[dict]:
    """Finished jobs, newest first (memory; filled from the log at startup)."""
    load()
    with _lock:
        return list(reversed(_recent))


def session_busy(session_key: str) -> bool:
    """A request of this session is running (any request, job or not)."""
    if not session_key:
        return False
    with _lock:
        return any(job.get("session") == session_key for job in _running.values()
                   if job["kind"] == "request")
