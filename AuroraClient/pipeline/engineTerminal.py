"""Read-only view of this engine's terminal (Aurora 1.9.4, network sharing beta).

When Aurora runs as a background service nobody sees its terminal, so the engine keeps
the last TERMINAL_LINES lines it printed (requests, tool output, errors) in memory and
shows them to the people allowed to look:

* GET /engine-terminal/?after=N  -> {"lines": [[seq, time, text], ...], "last": seq}
  Signed-in sessions only, and only the roles in TERMINAL_ROLES (at this computer, or the
  server's owner and managers). Lines are read-only: nothing can be typed into the engine.
* GET /terminal/  -> a small page that signs in with a lab pass (?enginePass=, from the
  lab's My Aurora servers page) or reuses this browser's session, then follows the lines.

* POST /engine-report/ {kind: "problem"|"suggestion", comment}  -> the report button.
  Anyone signed in may send one. A problem report carries the last REPORT_LOG_LINES lines,
  added here and never shown to a network member. A paired computer sends it to the lab
  itself, signed with its key; Aurora not paired with the lab hands it back to the
  browser of the person at this computer, who sends it with their own lab sign-in (their
  own computer's log). Where it is emailed is only known to the lab website.
  Since 1.9.6 that log is fuller than the terminal (report_log): every error kept whole
  with its request and traceback, the reporting tab's console and failed calls, then the
  terminal; the browser may add a picture of the page behind the report window
  ("screenshot", a JPEG data URL), which only the lab website turns into an attachment.

* Failed requests print their details (ErrorDetailMiddleware, see "Error details"):
  the message the browser got, the request's parameters (secrets hidden), who asked and
  the traceback of the exception behind it. Successful requests print nothing more.

All three paths skip the one-person lease (engineAuth), so watching the terminal never takes
the computer from whoever is using it. Everything goes over the engine's own port and
sign-in; there is no second listener.
"""

from __future__ import annotations

import collections
import contextvars
import io
import json
import logging
import os
import platform
import re
import sys
import sysconfig
import threading
import time
import traceback

from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

TERMINAL_LINES = 3000
MAX_LINE_CHARS = 2000
# Who may watch: "local" = signed in at this computer; owner / manager of the shared server.
TERMINAL_ROLES = ("local", "owner", "manager")

REPORT_LOG_LINES = 600   # recent_log(); reports use report_log() below
# A problem report's log (report_log): the whole stays under what the lab keeps
# (REPORT_LOG_MAX = 200 000 in frontend/auroraServers.py); errors and the browser's lines
# have their own share so a long terminal never pushes them out.
REPORT_LOG_CHARS = 190_000
REPORT_ERRORS_CHARS = 70_000
REPORT_BROWSER_CHARS = 30_000
# The picture of the page (ReportIssueDialog.js keeps it under this; the lab checks it again).
SCREENSHOT_MAX_CHARS = 600_000
SCREENSHOT_PREFIXES = ("data:image/jpeg;base64,", "data:image/png;base64,")
REPORT_COMMENT_MAX = 5000
REPORTS_PER_HOUR = 5
_report_times: dict = {}   # lab user id -> recent report times (in memory)
logger = logging.getLogger(__name__)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_lock = threading.Lock()
_lines: collections.deque = collections.deque(maxlen=TERMINAL_LINES)
_seq = 0
_installed = False
_started = time.time()


def _add(text: str) -> None:
    global _seq
    text = _ANSI.sub("", text).rstrip("\r")
    if not text.strip():
        return
    with _lock:
        _seq += 1
        _lines.append((_seq, round(time.time(), 3), text[:MAX_LINE_CHARS]))


class _Tee(io.TextIOBase):
    """Passes everything to the real stream and keeps whole lines for the viewer."""

    def __init__(self, stream):
        super().__init__()
        self._stream = stream
        self._partial = ""
        self._partial_lock = threading.Lock()

    def write(self, text):
        written = self._stream.write(text)
        try:
            with self._partial_lock:
                data = self._partial + str(text)
                parts = data.replace("\r\n", "\n").split("\n")
                self._partial = parts.pop()[-MAX_LINE_CHARS:]
            for part in parts:
                _add(part)
        except Exception:  # noqa: BLE001 - the viewer must never break printing
            pass
        return written

    def flush(self):
        return self._stream.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    def writable(self):
        return True


def install() -> None:
    """Start keeping lines. Safe to call more than once."""
    global _installed
    if _installed:
        return
    _installed = True
    _install_error_records()
    originals = {}
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None or isinstance(stream, _Tee):
            continue
        tee = _Tee(stream)
        originals[id(stream)] = tee
        setattr(sys, name, tee)
    # Log handlers made before this point (daphne's) hold the real streams: point them at
    # the tees so their lines are kept too, once.
    loggers = [logging.getLogger()] + [logger for logger in logging.Logger.manager.loggerDict.values()
                                       if isinstance(logger, logging.Logger)]
    for logger in loggers:
        for handler in logger.handlers:
            stream = getattr(handler, "stream", None)
            if isinstance(handler, logging.StreamHandler) and id(stream) in originals:
                handler.setStream(originals[id(stream)])


# ---------------------------------------------------------------------------
# Error details (1.9.6): a failed request prints everything needed to fix it
# ---------------------------------------------------------------------------
# Normal operation stays as short as compactAccessLog makes it. A request that fails
# (status >= 400) also prints one block with what the person saw and why:
#
#   [15:49:26]  ✗ Error details  POST /engine-sharing/  →  400 Bad Request
#       message : C:\Users\...\Documents contains links to other places (...)
#       request : {"action": "add_folder", "path": "C:\\Users\\...\\Documents"}
#       by      : Bassel Arnaout (local)
#       caught  : ValueError in pipeline/engineSharing.py:412 (_check_folder)
#         Traceback (most recent call last): ...
#
# Most views catch their own exceptions and answer {"error": str(e)}, which used to leave
# only "Bad Request: /x/" in the terminal. Python 3.12+ reports every caught exception
# (sys.monitoring EXCEPTION_HANDLED); while a request runs, the last few are kept with
# their tracebacks, and the one whose text is the answered error is printed with it
# (for a 5xx answer the last one caught is shown anyway, marked as such). Nothing is kept
# outside a request, and nothing is printed for a request that succeeds.
#
# The same blocks, and every ERROR logged with a traceback, are also kept whole (not cut
# into terminal lines) for problem reports: see report_log().
# AURORA_ERROR_DETAILS=0 turns the blocks and the exception watch off.

ERROR_DETAILS_ENABLED = os.environ.get("AURORA_ERROR_DETAILS", "1").strip() not in ("0", "off", "false")
ERROR_DETAILS_KEPT = 40          # whole error blocks kept for reports
ERROR_DETAIL_MAX_CHARS = 16000   # one block (a long traceback is cut in the middle)
REQUEST_BODY_MAX_BYTES = 64 * 1024   # JSON bodies up to this size are shown (redacted)
REQUEST_SHOWN_CHARS = 1500
CAUGHT_PER_REQUEST = 8
# Answers that are part of normal operation, not errors: "in use by someone else"
# (one person at a time, see asgi.py) and a signed-out tab's checks.
QUIET_STATUSES = (409,)
QUIET_CODES = ("engine_session_required",)
# Values never printed or reported, whatever the endpoint (matched on the key's name).
_SECRET_KEY = re.compile(r"(pass(word)?|token|secret|signature|private_?key|api_?key|authorization|cookie|"
                         r"screenshot|browser_log)$", re.IGNORECASE)

_errors: collections.deque = collections.deque(maxlen=ERROR_DETAILS_KEPT)
_caught = contextvars.ContextVar("aurora_caught_exceptions", default=None)
_watch_installed = False


def _keep_error(text: str) -> None:
    if len(text) > ERROR_DETAIL_MAX_CHARS:
        half = ERROR_DETAIL_MAX_CHARS // 2
        text = text[:half] + "\n    ... (cut) ...\n" + text[-half:]
    with _lock:
        _errors.append((round(time.time(), 3), text))


def _on_exception_handled(code, instruction_offset, exception):
    # Called for every exception caught anywhere; only requests in flight keep it.
    box = _caught.get()
    if box is not None:
        box.append(exception)
        if len(box) > CAUGHT_PER_REQUEST:
            del box[0]


def _install_exception_watch() -> None:
    """Start sys.monitoring's caught-exception events (Python 3.12+). Once, best effort."""
    global _watch_installed
    if _watch_installed or not ERROR_DETAILS_ENABLED:
        return
    _watch_installed = True
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        return
    # Tool ids 0 (debugger), 1 (coverage), 2 (profiler) and 5 (optimizer) belong to other
    # tools by convention; take a free one so a debugger attached by a developer still works.
    for tool_id in (4, 3):
        try:
            if monitoring.get_tool(tool_id) is not None:
                continue
            monitoring.use_tool_id(tool_id, "aurora-error-details")
            monitoring.register_callback(tool_id, monitoring.events.EXCEPTION_HANDLED, _on_exception_handled)
            monitoring.set_events(tool_id, monitoring.events.EXCEPTION_HANDLED)
            return
        except Exception:  # noqa: BLE001 - error details are optional
            continue


def _redact(value, depth: int = 0):
    if depth > 6:
        return "..."
    if isinstance(value, dict):
        return {key: ("(hidden)" if _SECRET_KEY.search(str(key)) else _redact(item, depth + 1))
                for key, item in value.items()}
    if isinstance(value, list):
        shown = [_redact(item, depth + 1) for item in value[:50]]
        return shown + [f"... {len(value) - 50} more"] if len(value) > 50 else shown
    if isinstance(value, str) and len(value) > 400:
        return value[:400] + f"... ({len(value)} characters)"
    return value


def _request_summary(request, body) -> str:
    """The request's parameters as one line: query string and JSON body, secrets hidden."""
    parts = []
    query = {key: (values if len(values) > 1 else values[0]) for key, values in request.GET.lists()}
    if query:
        parts.append("query " + json.dumps(_redact(query), ensure_ascii=False, default=str))
    if body is not None:
        try:
            parsed = json.loads(body.decode("utf-8") or "null")
            if parsed is not None:
                parts.append("body " + json.dumps(_redact(parsed), ensure_ascii=False, default=str))
        except (ValueError, UnicodeDecodeError):
            parts.append(f"body ({len(body)} bytes, not JSON)")
    elif request.method not in ("GET", "HEAD", "OPTIONS"):
        length = request.META.get("CONTENT_LENGTH") or "?"
        parts.append(f"body ({request.META.get('CONTENT_TYPE') or 'no type'}, {length} bytes, not shown)")
    text = "  ".join(parts) or "(no parameters)"
    return text if len(text) <= REQUEST_SHOWN_CHARS else text[:REQUEST_SHOWN_CHARS] + "…"


def _answered_error(response) -> tuple:
    """(message, code) from the JSON the browser received, or ("", "")."""
    data = getattr(response, "data", None)
    if data is None and "json" in str(response.get("Content-Type", "")):
        try:
            data = json.loads(response.content[:200000].decode("utf-8"))
        except (ValueError, UnicodeDecodeError, AttributeError):
            data = None
    if isinstance(data, dict):
        message = data.get("error") or data.get("message") or data.get("detail") or ""
        if not isinstance(message, str):
            message = json.dumps(_redact(message), ensure_ascii=False, default=str)
        return message, str(data.get("code") or "")
    if isinstance(data, list) and data:
        return json.dumps(_redact(data), ensure_ascii=False, default=str)[:1000], ""
    return "", ""


_LIBRARY_PATHS = tuple(path.replace("\\", "/") for path in
                       {sysconfig.get_paths().get("stdlib", ""), sysconfig.get_paths().get("purelib", ""),
                        sysconfig.get_paths().get("platlib", "")} if path)


def _raised_in_aurora(exc) -> bool:
    """True when a frame of the traceback is Aurora's own code (not Python's or a library's),
    so a library's own caught-and-handled exception (DRF's renderer, Django's
    checks) is not shown as the cause of a 5xx."""
    for frame in traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []:
        path = frame.filename.replace("\\", "/")
        if "site-packages/" not in path and not path.startswith(_LIBRARY_PATHS) and not path.startswith("<"):
            return True
    return False


def _matching_exception(caught, message: str, status_code: int):
    """(exception, how) for the traceback to show: the caught exception whose text is the
    answered message, else for a 5xx the last one caught in Aurora's code; (None, "")."""
    message = (message or "").strip()
    for exc in reversed(caught or []):
        text = str(exc).strip()
        if message and text and (text in message or message in text):
            return exc, "caught"
    if status_code >= 500:
        for exc in reversed(caught or []):
            if _raised_in_aurora(exc):
                return exc, "last caught"
    return None, ""


def _where(exc) -> str:
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    if not frames:
        return type(exc).__name__
    last = frames[-1]
    path = last.filename.replace("\\", "/")
    for marker in ("/AuroraClient/", "/site-packages/"):
        if marker in path:
            path = path.rsplit(marker, 1)[1]   # pipeline/views.py, rest_framework/views.py
            break
    else:
        path = os.path.basename(path)
    return f"{type(exc).__name__} in {path}:{last.lineno} ({last.name})"


def _print_block(text: str) -> None:
    # One write, so lines of two failing requests never interleave.
    try:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(text.replace("✗", "x").replace("→", "->").encode("ascii", "replace").decode("ascii") + "\n")
    except Exception:  # noqa: BLE001 - printing must never break a request
        pass


class ErrorDetailMiddleware:
    """Django middleware: prints (and keeps for reports) the details of a failed request.
    See "Error details" above. Placed after EngineSessionMiddleware, so it knows who asked."""

    def __init__(self, get_response):
        self.get_response = get_response
        _install_exception_watch()

    def __call__(self, request):
        if not ERROR_DETAILS_ENABLED:
            return self.get_response(request)
        body = None
        content_type = str(request.META.get("CONTENT_TYPE") or "")
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        if "json" in content_type and 0 < length <= REQUEST_BODY_MAX_BYTES:
            try:
                # Read now: Django keeps it, and the view reads the same bytes afterwards.
                body = request.body
            except Exception:  # noqa: BLE001
                body = None
        box = []
        token = _caught.set(box)
        try:
            response = self.get_response(request)
        finally:
            _caught.reset(token)
        try:
            if response.status_code >= 400:
                self._report(request, response, body, box)
        except Exception:  # noqa: BLE001 - details are best effort
            pass
        return response

    def _report(self, request, response, body, caught) -> None:
        status_code = response.status_code
        message, code = _answered_error(response)
        if status_code in QUIET_STATUSES or code in QUIET_CODES:
            return
        record = getattr(request, "aurora_session", None) or {}
        who = "(not signed in)"
        if record:
            try:
                from pipeline import engineSharing
                who = f"{record.get('name') or 'lab user ' + str(record.get('user_id'))} ({engineSharing.session_role(record)})"
            except Exception:  # noqa: BLE001
                who = str(record.get("name") or "")
        reason = getattr(response, "reason_phrase", "") or ""
        marker = "✗" if status_code >= 500 else "!"
        lines = [f"[{time.strftime('%H:%M:%S')}]  {marker} Error details  {request.method} {request.path}"
                 f"  →  {status_code} {reason}".rstrip(),
                 f"    message : {message or '(no message in the answer)'}"]
        if code:
            lines.append(f"    code    : {code}")
        lines.append(f"    request : {_request_summary(request, body)}")
        lines.append(f"    by      : {who}")
        exc, how = _matching_exception(caught, message, status_code)
        if exc is not None:
            label = "caught  " if how == "caught" else "last caught (may be unrelated)"
            lines.append(f"    {label}: {_where(exc)}")
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()
            lines.extend("      " + line for line in trace.splitlines())
        text = "\n".join(lines)
        _print_block(text)
        _keep_error(time.strftime("[%Y-%m-%d ") + text[1:])   # the day too, for reports


class _ErrorRecordHandler(logging.Handler):
    """Keeps every ERROR logged with a traceback (background jobs, threads) for reports,
    whole. It prints nothing: the log's own handler already did."""

    def emit(self, record):
        try:
            if record.name == "django.request":
                return  # failed requests: ErrorDetailMiddleware keeps a fuller block
            text = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))}  " \
                   f"{record.levelname} {record.name}: {record.getMessage()}"
            if record.exc_info:
                text += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
            _keep_error(text)
        except Exception:  # noqa: BLE001
            pass


def _install_error_records() -> None:
    root = logging.getLogger()
    if not any(isinstance(handler, _ErrorRecordHandler) for handler in root.handlers):
        root.addHandler(_ErrorRecordHandler(level=logging.ERROR))


def lines_after(after: int) -> tuple:
    with _lock:
        last = _seq
        if after >= last:
            return [], last
        return [list(item) for item in _lines if item[0] > after], last


def may_watch(record: dict) -> bool:
    from pipeline import engineSharing
    return engineSharing.session_role(record) in TERMINAL_ROLES


class EngineTerminalView(APIView):
    """GET /engine-terminal/?after=N (see the module docstring)."""

    authentication_classes = []
    permission_classes = []

    def get(self, request, *args, **kwargs):
        record = getattr(request, "aurora_session", None)
        if record is None:
            return Response({"code": "engine_session_required"}, status=status.HTTP_401_UNAUTHORIZED)
        if not may_watch(record):
            return Response({"error": "Only the owner or a manager of this computer can see its terminal.",
                             "code": "terminal_forbidden"}, status=status.HTTP_403_FORBIDDEN)
        try:
            after = max(0, int(request.query_params.get("after") or 0))
        except (TypeError, ValueError):
            after = 0
        lines, last = lines_after(after)
        return Response({"lines": lines, "last": last}, status=status.HTTP_200_OK)


def recent_log(count: int = REPORT_LOG_LINES) -> str:
    with _lock:
        items = list(_lines)[-count:]
    return "\n".join(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(when))}  {text}"
                     for _seq_no, when, text in items)


def _environment() -> list:
    """A few facts about this computer that often explain a problem (no personal data)."""
    facts = [f"python {sys.version.split()[0]} ({sys.executable})",
             f"system {platform.platform()} ({platform.machine()})",
             f"cpus {os.cpu_count()}"]
    try:
        import psutil
        memory = psutil.virtual_memory()
        facts.append(f"memory {memory.total / 2**30:.1f} GB, {memory.available / 2**30:.1f} GB free")
    except Exception:  # noqa: BLE001
        pass
    facts.append(f"engine up {time.time() - _started:.0f} s; terminal lines kept {len(_lines)} of {TERMINAL_LINES}")
    return facts


def _fit_newest(blocks: list, budget: int) -> tuple:
    """The newest blocks that fit in budget characters, oldest first, and how many were left out."""
    kept, used = [], 0
    for block in reversed(blocks):
        if used + len(block) + 1 > budget:
            break
        kept.append(block)
        used += len(block) + 1
    return list(reversed(kept)), len(blocks) - len(kept)


def report_log(browser_log: str = "") -> str:
    """The log attached to a problem report, fuller than the terminal shows:
    this computer, every error kept whole with its request and traceback (see "Error
    details"), the reporting tab's own console and failed calls (sent by the browser), and
    then as much of the terminal as fits. Stays under REPORT_LOG_CHARS, which the lab keeps
    whole (frontend/auroraServers.py REPORT_LOG_MAX)."""
    with _lock:
        errors = [f"{text}" for _when, text in _errors]
        terminal = [f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(when))}  {text}"
                    for _seq_no, when, text in _lines]
    sections = ["== This computer ==", *_environment(), ""]

    shown_errors, left_out = _fit_newest(errors, REPORT_ERRORS_CHARS)
    sections.append(f"== Errors with full details ({len(shown_errors)}, newest last"
                    + (f"; {left_out} older not shown" if left_out else "") + ") ==")
    sections += [block + "\n" for block in shown_errors] or ["(no errors since Aurora started)", ""]

    browser_log = (browser_log or "").strip()
    if browser_log:
        if len(browser_log) > REPORT_BROWSER_CHARS:
            browser_log = "... (older lines cut)\n" + browser_log[-REPORT_BROWSER_CHARS:]
        sections += ["== This browser tab: console errors, warnings and failed calls ==", browser_log, ""]

    head = "\n".join(sections)
    shown_lines, cut = _fit_newest(terminal, max(0, REPORT_LOG_CHARS - len(head) - 200))
    tail = [f"== Terminal (last {len(shown_lines)} lines"
            + (f"; {cut} older not shown" if cut else "") + ") ==", *shown_lines]
    return head + "\n" + "\n".join(tail)


def _report_allowed(user_id: str) -> bool:
    now = time.time()
    with _lock:
        recent = [when for when in _report_times.get(user_id, []) if now - when < 3600]
        if len(recent) >= REPORTS_PER_HOUR:
            _report_times[user_id] = recent
            return False
        _report_times[user_id] = recent + [now]
        return True


class EngineReportView(APIView):
    """POST /engine-report/ (see the module docstring)."""

    authentication_classes = []
    permission_classes = []

    def post(self, request, *args, **kwargs):
        from AuroraClient import engineConfig
        from pipeline import engineSharing, networkMode

        record = getattr(request, "aurora_session", None)
        if record is None:
            return Response({"code": "engine_session_required"}, status=status.HTTP_401_UNAUTHORIZED)
        data = request.data if hasattr(request.data, "get") else {}
        kind = data.get("kind")
        comment = data.get("comment") if isinstance(data.get("comment"), str) else ""
        if kind not in ("problem", "suggestion"):
            return Response({"error": "Choose a problem or a suggestion."}, status=status.HTTP_400_BAD_REQUEST)
        if not comment.strip():
            return Response({"error": "Write a few words about it."}, status=status.HTTP_400_BAD_REQUEST)
        if not _report_allowed(str(record.get("user_id") or "")):
            return Response({"error": "You sent several reports in the last hour; try again later."},
                            status=status.HTTP_429_TOO_MANY_REQUESTS)
        page = data.get("page") if isinstance(data.get("page"), str) else ""
        # From the reporting tab (ReportIssueDialog.js): its console lines and a picture of
        # the page behind the report window. Anything not shaped as expected is dropped.
        browser_log = data.get("browser_log") if isinstance(data.get("browser_log"), str) else ""
        screenshot = data.get("screenshot") if isinstance(data.get("screenshot"), str) else ""
        if len(screenshot) > SCREENSHOT_MAX_CHARS or not screenshot.startswith(SCREENSHOT_PREFIXES):
            screenshot = ""
        report = {
            "kind": kind,
            "comment": comment[:REPORT_COMMENT_MAX],
            "log": report_log(browser_log[-2 * REPORT_BROWSER_CHARS:]) if kind == "problem" else "",
            "details": {
                "version": networkMode._installed_version(),
                "platform": sys.platform,
                "mode": engineConfig.load_config()["mode"],
                "role": engineSharing.session_role(record),
                "service": os.environ.get("AURORA_SERVICE") == "1",
                "engine_id": engineConfig.engine_id(),
                "page": page[:200],
                "browser": str(request.META.get("HTTP_USER_AGENT", ""))[:200],
            },
        }
        pairing = networkMode.pairing_state() or {}
        lab = (pairing.get("lab_origin") or "").rstrip("/")
        if lab and lab in engineConfig.trusted_lab_origins():
            try:
                response = networkMode.signed_post(lab, "/auroraReport", {
                    **report,
                    **({"screenshot": screenshot} if screenshot else {}),
                    "reporter_user_id": str(record.get("user_id") or ""),
                    "reporter_name": str(record.get("name") or "")[:120],
                }, timeout=30)
            except Exception as exc:  # noqa: BLE001 - lab unreachable
                logger.warning("Report not sent: the lab website %s cannot be reached (%s: %s).",
                               lab, type(exc).__name__, exc)
                response = None
            if response is not None and response.status_code == 201:
                logger.info("%s report sent by %s.", kind.capitalize(), record.get("name"))
                return Response({"sent": True}, status=status.HTTP_200_OK)
            if response is not None and response.status_code not in (404,):
                try:
                    message = response.json().get("error") or ""
                except ValueError:
                    message = ""
                logger.warning("Report not taken by the lab website %s: %s %s", lab, response.status_code,
                               message or response.text[:300])
                return Response({"error": message or f"The lab website answered {response.status_code}."},
                                status=status.HTTP_502_BAD_GATEWAY)
            if record.get("remote"):
                return Response({"error": "Could not reach the lab website; try again in a while."},
                                status=status.HTTP_502_BAD_GATEWAY)
        elif record.get("remote"):
            return Response({"error": "This computer is not paired with the lab website."},
                            status=status.HTTP_409_CONFLICT)
        # The person at this computer, Aurora not paired (or the lab did not take it):
        # their browser sends it with their lab sign-in, to the lab this session came from.
        # The picture is not sent back: the browser still has it and adds it itself.
        return Response({"sent": False, "lab_origin": record.get("lab_origin") or "", "report": report},
                        status=status.HTTP_200_OK)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Aurora terminal</title>
<style>
:root { color-scheme: dark; }
body { margin: 0; background: #0b0f14; color: #d6dde6; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
header { position: sticky; top: 0; display: flex; gap: 12px; align-items: center; padding: 10px 16px;
         background: #111821; border-bottom: 1px solid #1f2a36; font-family: system-ui, sans-serif; }
header b { font-weight: 600; } #state { color: #8a99a8; font-size: 12px; flex: 1; }
button { background: #1d2a38; color: #d6dde6; border: 1px solid #2c3d50; border-radius: 6px; padding: 4px 10px; cursor: pointer; }
#out { padding: 10px 16px 40px; white-space: pre-wrap; word-break: break-word; }
.t { color: #5d6c7b; } .err { color: #ff8f8f; } .warn { color: #ffd27a; }
</style></head><body>
<header><b>Aurora terminal</b><span id="state">Connecting...</span>
<button id="pause" type="button">Pause</button></header>
<div id="out"></div>
<script>
(function () {
  var out = document.getElementById('out'), state = document.getElementById('state');
  var pauseBtn = document.getElementById('pause'), last = 0, paused = false, stopped = false;
  pauseBtn.onclick = function () { paused = !paused; pauseBtn.textContent = paused ? 'Follow' : 'Pause'; };
  function say(text) { state.textContent = text; }
  function cls(text) { return /error|traceback|exception|failed|\\u2717/i.test(text) ? 'err' : /warning|\\u26a0/i.test(text) ? 'warn' : ''; }
  function add(lines) {
    var nearBottom = window.innerHeight + window.scrollY >= document.body.scrollHeight - 40;
    var frag = document.createDocumentFragment();
    lines.forEach(function (l) {
      var row = document.createElement('div'), t = document.createElement('span');
      t.className = 't'; t.textContent = new Date(l[1] * 1000).toLocaleTimeString() + '  ';
      row.appendChild(t); row.appendChild(document.createTextNode(l[2]));
      var c = cls(l[2]); if (c) row.className = c;
      frag.appendChild(row);
    });
    out.appendChild(frag);
    while (out.childNodes.length > 5000) out.removeChild(out.firstChild);
    if (!paused && nearBottom) window.scrollTo(0, document.body.scrollHeight);
  }
  function poll() {
    if (stopped) return;
    fetch('/engine-terminal/?after=' + last, { cache: 'no-store', credentials: 'same-origin' })
      .then(function (r) { return r.json().then(function (d) { return [r.status, d]; }); })
      .then(function (res) {
        var code = res[0], d = res[1];
        if (code === 401) { stopped = true; say('Signed out. Open the terminal again from My Aurora servers.'); return; }
        if (code === 403) { stopped = true; say(d.error || 'Not allowed.'); return; }
        if (code !== 200) { say('Aurora answered ' + code + '; retrying...'); return; }
        if (d.lines && d.lines.length) add(d.lines);
        last = d.last || last;
        say('Live, read-only' + (paused ? ' (paused scrolling)' : ''));
      })
      .catch(function () { say('Cannot reach Aurora; retrying...'); })
      .then(function () { if (!stopped) setTimeout(poll, document.hidden ? 5000 : 1500); });
  }
  function start() {
    var params = new URLSearchParams(location.search), pass = params.get('enginePass');
    if (pass) history.replaceState(null, '', location.pathname);
    fetch('/engine-auth/status/', { cache: 'no-store', credentials: 'same-origin' })
      .then(function (r) { return r.json(); })
      .then(function (s) {
        if (s.authenticated || !pass) return null;
        return fetch('/engine-auth/start/', { method: 'POST', credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pass: pass }) })
          .then(function (r) { if (!r.ok) return r.json().then(function (d) { throw new Error(d.error || 'Sign-in refused'); }); });
      })
      .then(poll)
      .catch(function (e) { say(e.message || 'Sign-in failed'); });
  }
  start();
})();
</script></body></html>
"""


class EngineTerminalPageView(APIView):
    """GET /terminal/ — the viewer page (no data in it; the lines come from /engine-terminal/)."""

    authentication_classes = []
    permission_classes = []

    def get(self, request, *args, **kwargs):
        response = HttpResponse(_PAGE, content_type="text/html; charset=utf-8")
        response["Cache-Control"] = "no-store"
        response["Content-Security-Policy"] = ("default-src 'none'; script-src 'unsafe-inline'; "
                                               "style-src 'unsafe-inline'; connect-src 'self'; "
                                               "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        response["X-Content-Type-Options"] = "nosniff"
        return response
