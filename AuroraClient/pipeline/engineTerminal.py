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

All three paths skip the one-person lease (engineAuth), so watching the terminal never takes
the computer from whoever is using it. Everything goes over the engine's own port and
sign-in; there is no second listener.
"""

from __future__ import annotations

import collections
import io
import logging
import os
import re
import sys
import threading
import time

from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

TERMINAL_LINES = 3000
MAX_LINE_CHARS = 2000
# Who may watch: "local" = signed in at this computer; owner / manager of the shared server.
TERMINAL_ROLES = ("local", "owner", "manager")

REPORT_LOG_LINES = 600
REPORT_COMMENT_MAX = 5000
REPORTS_PER_HOUR = 5
_report_times: dict = {}   # lab user id -> recent report times (in memory)
logger = logging.getLogger(__name__)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_lock = threading.Lock()
_lines: collections.deque = collections.deque(maxlen=TERMINAL_LINES)
_seq = 0
_installed = False


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
        report = {
            "kind": kind,
            "comment": comment[:REPORT_COMMENT_MAX],
            "log": recent_log() if kind == "problem" else "",
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
                    "reporter_user_id": str(record.get("user_id") or ""),
                    "reporter_name": str(record.get("name") or "")[:120],
                }, timeout=20)
            except Exception as exc:  # noqa: BLE001 - lab unreachable
                logger.warning("Report not sent: the lab website cannot be reached (%s).", type(exc).__name__)
                response = None
            if response is not None and response.status_code == 201:
                logger.info("%s report sent by %s.", kind.capitalize(), record.get("name"))
                return Response({"sent": True}, status=status.HTTP_200_OK)
            if response is not None and response.status_code not in (404,):
                try:
                    message = response.json().get("error") or ""
                except ValueError:
                    message = ""
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
