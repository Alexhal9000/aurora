"""
Compact terminal access log for the Aurora engine (daphne).

Daphne's default access log writes one NCSA line per HTTP request. Opening a folder
or a grid with hundreds of scans fires hundreds of near-identical requests
(/extract-json/ per scan on folder load, /get-grid-view/ per tile, two
/projection-png/ GETs per Quick Grid tile), which buries the lines that matter
(errors, slow actions) and makes the terminal hard to use for debugging.

This module replaces that log with a compact one:

* Failed requests (status >= 400) are printed immediately, one line each, with
  the full path + query so the failing scan / parameters are visible.
* Successful requests are grouped by method + endpoint (query string dropped).
  A group is printed once when its burst goes quiet (GROUP_IDLE_FLUSH_S); long
  bursts also print a "so far" progress line every GROUP_MAX_SPAN_S, so a 1000-scan
  load shows a handful of lines instead of 1000+. Counts and timings are cumulative
  over the whole burst.
  A group with a single request prints as a normal one-line entry with its duration.
* Front-end assets (.js/.css/.png/fonts...) are folded into one "static files" group.
* Background checks that every open Aurora tab or lab page repeats on its own (the
  presence heartbeat every 30 s, sign-in status, job and version polls) are not printed
  one by one: they are counted, and one "background checks" line is printed just before
  the next real activity, or once an hour while nothing else happens. Over a long idle
  session the finished tasks above therefore stay close. Failed checks still print at once.
* WebSocket connect / disconnect are printed as one line each (handshake noise dropped).
* Long-running requests (N4, registration, AI...) get a heartbeat: once a request has
  been running for HEARTBEAT_AFTER_S, a "still running" line is printed every
  HEARTBEAT_EVERY_S until it completes. Daphne only logs a request when it finishes, so
  without this a long method that prints little of its own would look stalled.
  Method-level step prints from the pipeline are NOT touched by this module; they keep
  printing in order, which is what makes a copied terminal useful for debugging.

Mode is chosen with the AURORA_ACCESS_LOG environment variable:
    compact (default)  - behaviour above
    full               - daphne's original per-request NCSA log (unchanged)
    errors             - only failed requests are printed

Installed from asgi.py via install(); daphne's CLI loads the ASGI app before it
builds its access logger, so swapping daphne.cli.AccessLogGenerator there is enough
for every launcher (run_aurora.sh/.command, deploy scripts, macOS/Windows bundles).
Anything unexpected falls back to daphne's own logger, so logging can never break
request handling.
"""

import datetime
import os
import re
import sys
import threading
import time

# A burst for one endpoint is reported once no new request arrived for this long.
GROUP_IDLE_FLUSH_S = 1.0
# Long bursts also report a running count at least this often (progress feedback).
GROUP_MAX_SPAN_S = 5.0
# Background flush cadence.
FLUSH_TICK_S = 0.5
# Endpoint column width, so counts/durations line up.
PATH_COLUMN = 34
# Heartbeat for requests still in flight (see RequestHeartbeatMiddleware).
HEARTBEAT_AFTER_S = 20.0
HEARTBEAT_EVERY_S = 30.0
# Successful background polls (see module docstring): counted, summarised on one line.
BACKGROUND_ENDPOINTS = {
    ('POST', '/engine-auth/heartbeat/'): 'heartbeat',
    ('GET', '/engine-auth/status/'): 'sign-in status',
    ('GET', '/engine-jobs/'): 'job checks',
    ('GET', '/get-version/'): 'version checks',
    ('GET', '/engine-sharing/'): 'sharing status',
}
BACKGROUND_SUMMARY_EVERY_S = 3600.0
# Error lines keep the query string (useful to see the scan), but long ones are clipped.
ERROR_PATH_MAX = 180

_STATIC_FILE_RE = re.compile(
    r'\.(js|mjs|css|map|png|jpe?g|gif|svg|ico|webp|woff2?|ttf|otf|json|wasm|txt)$',
    re.IGNORECASE,
)


def _now_label():
    return datetime.datetime.now().strftime('%H:%M:%S')


# In-flight HTTP requests: id -> {'method', 'path', 'start', 'last_beat'}.
# Filled by RequestHeartbeatMiddleware, read by the access logger's flush thread.
_inflight_requests = {}
_inflight_lock = threading.Lock()


def _format_elapsed(seconds):
    minutes, secs = divmod(int(seconds), 60)
    return f'{minutes}m {secs:02d}s' if minutes else f'{secs}s'


class RequestHeartbeatMiddleware:
    """
    ASGI middleware that records when each HTTP request starts and ends, so the access
    logger can print "still running" heartbeats for long requests. Pure bookkeeping:
    it never changes the request or response.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http':
            return await self.app(scope, receive, send)
        request_id = object()
        now = time.monotonic()
        with _inflight_lock:
            _inflight_requests[request_id] = {
                'method': scope.get('method', '-'),
                'path': scope.get('path', ''),
                'start': now,
                'last_beat': now,
            }
        try:
            return await self.app(scope, receive, send)
        finally:
            with _inflight_lock:
                _inflight_requests.pop(request_id, None)


def _format_duration(seconds):
    if seconds is None:
        return ''
    if seconds < 1:
        return f'{seconds * 1000:.0f} ms'
    return f'{seconds:.2f} s'


class CompactAccessLogGenerator:
    """
    Drop-in replacement for daphne.access.AccessLogGenerator (same constructor and
    __call__(protocol, action, details) interface), writing the compact log above.
    """

    def __init__(self, stream, mode='compact'):
        self.stream = stream
        self.mode = mode
        self._lock = threading.Lock()
        # (method, endpoint label) -> running group stats
        self._groups = {}
        # Background polls since the last summary line: label -> count, plus when it began.
        self._background = {}
        self._background_since = None
        self._background_started = None
        if mode == 'compact':
            threading.Thread(target=self._flush_loop, name='aurora-access-log', daemon=True).start()

    # ---- daphne action-logger interface -------------------------------------------

    def __call__(self, protocol, action, details):
        try:
            if protocol == 'http' and action == 'complete':
                self._log_http(details)
            elif protocol == 'websocket' and self.mode == 'compact':
                if action == 'connected':
                    self._write(f'[{_now_label()}]  ⇄ WS connected     {details.get("path", "")}')
                elif action == 'disconnected':
                    self._write(f'[{_now_label()}]  ⇄ WS disconnected  {details.get("path", "")}')
                elif action == 'rejected':
                    self._write(f'[{_now_label()}]  ✗ WS rejected      {details.get("path", "")}')
                # 'connecting' is handshake noise; the outcome above is what matters.
        except Exception:
            # Never let logging interfere with serving requests.
            pass

    # ---- HTTP ---------------------------------------------------------------------

    def _log_http(self, details):
        method = details.get('method') or '-'
        full_path = details.get('path') or ''
        status = details.get('status')
        duration = details.get('time_taken')

        if status is not None and int(status) >= 400:
            marker = '✗' if int(status) >= 500 else '!'
            shown = full_path if len(full_path) <= ERROR_PATH_MAX else full_path[:ERROR_PATH_MAX] + '…'
            self._write(
                f'[{_now_label()}]  {marker} {method:<4} {shown}  →  {status}  {_format_duration(duration)}'
            )
            return

        if self.mode == 'errors':
            return

        endpoint = full_path.split('?', 1)[0]
        background = BACKGROUND_ENDPOINTS.get((method, endpoint))
        if background:
            with self._lock:
                if not self._background:
                    self._background_since = _now_label()
                    self._background_started = time.monotonic()
                self._background[background] = self._background.get(background, 0) + 1
            return
        label = 'static files' if _STATIC_FILE_RE.search(endpoint) else endpoint
        key = (method, label)
        now = time.monotonic()
        with self._lock:
            group = self._groups.get(key)
            if group is None:
                group = {
                    'count': 0, 'total_time': 0.0, 'max_time': 0.0, 'statuses': set(),
                    'last': now, 'last_report_time': now, 'last_report_count': 0,
                }
                self._groups[key] = group
            group['count'] += 1
            group['total_time'] += duration or 0.0
            group['max_time'] = max(group['max_time'], duration or 0.0)
            group['statuses'].add(status)
            group['last'] = now

    def _flush_loop(self):
        while True:
            time.sleep(FLUSH_TICK_S)
            try:
                self._flush(force=False)
                self._heartbeat()
                if (self._background_started is not None
                        and time.monotonic() - self._background_started >= BACKGROUND_SUMMARY_EVERY_S):
                    self._flush_background()
            except Exception:
                pass

    def _heartbeat(self):
        now = time.monotonic()
        lines = []
        with _inflight_lock:
            for request in _inflight_requests.values():
                elapsed = now - request['start']
                if elapsed < HEARTBEAT_AFTER_S:
                    continue
                first_beat = request['last_beat'] == request['start']
                if not first_beat and now - request['last_beat'] < HEARTBEAT_EVERY_S:
                    continue
                request['last_beat'] = now
                lines.append(
                    f'[{_now_label()}]  ⏳ {request["method"]:<4} {request["path"]:<{PATH_COLUMN}}  '
                    f'still running ({_format_elapsed(elapsed)})'
                )
        for line in lines:
            self._write(line)

    def _flush(self, force):
        now = time.monotonic()
        lines = []
        with self._lock:
            for key in list(self._groups):
                group = self._groups[key]
                if force or now - group['last'] >= GROUP_IDLE_FLUSH_S:
                    # Burst finished: one final line with the cumulative totals.
                    lines.append(self._format_group(key, group, finished=True))
                    del self._groups[key]
                elif (now - group['last_report_time'] >= GROUP_MAX_SPAN_S
                      and group['count'] > group['last_report_count']):
                    # Still busy: progress line, then keep accumulating.
                    lines.append(self._format_group(key, group, finished=False))
                    group['last_report_time'] = now
                    group['last_report_count'] = group['count']
        for line in lines:
            self._write(line)

    def _format_group(self, key, group, finished):
        method, label = key
        count = group['count']
        statuses = sorted(s for s in group['statuses'] if s is not None)
        status_text = '/'.join(str(s) for s in statuses) or '-'
        marker = '✓' if finished else '…'
        prefix = f'[{_now_label()}]  {marker} {method:<4} {label:<{PATH_COLUMN}}'

        if count == 1:
            return f'{prefix}  {status_text:<7}  {_format_duration(group["total_time"])}'

        avg = group['total_time'] / count
        progress = f'×{count}' if finished else f'×{count} so far'
        return (
            f'{prefix}  {status_text:<7}  {progress:<12} '
            f'avg {_format_duration(avg)} · max {_format_duration(group["max_time"])}'
        )

    def _flush_background(self):
        with self._lock:
            counts, since = self._background, self._background_since
            self._background, self._background_since, self._background_started = {}, None, None
        if not counts:
            return
        total = sum(counts.values())
        parts = ', '.join(f'{label} ×{count}' for label, count in sorted(counts.items(), key=lambda item: -item[1]))
        self._write(f'[{_now_label()}]  · background checks ×{total} since {since} ({parts})', background=True)

    # ---- output -------------------------------------------------------------------

    def _write(self, line, background=False):
        if not background and self._background:
            # Summarise the quiet stretch first, so the log reads in order.
            self._flush_background()
        text = line + '\n'
        try:
            self.stream.write(text)
        except UnicodeEncodeError:
            # Legacy Windows consoles (cp1252) cannot print the markers.
            ascii_text = (
                text.replace('✓', '+').replace('✗', 'x').replace('⇄', '<>')
                .replace('→', '->').replace('×', 'x').replace('⏳', '..').replace('·', '|').replace('…', '...')
            )
            self.stream.write(ascii_text.encode('ascii', 'replace').decode('ascii'))
        try:
            self.stream.flush()
        except Exception:
            pass


def install():
    """
    Swap daphne's access logger for the compact one (see module docstring).
    Safe to call when not running under daphne's CLI (e.g. manage.py): it is a no-op.
    """
    mode = os.environ.get('AURORA_ACCESS_LOG', 'compact').strip().lower()
    if mode not in ('compact', 'errors'):
        return  # 'full' (or anything unknown): keep daphne's original access log
    try:
        import daphne.cli as daphne_cli
    except Exception:
        return

    original = daphne_cli.AccessLogGenerator
    if getattr(original, '_aurora_compact', False):
        return  # already installed (module re-import)

    def compact_factory(stream):
        try:
            return CompactAccessLogGenerator(stream, mode=mode)
        except Exception:
            return original(stream)

    compact_factory._aurora_compact = True
    daphne_cli.AccessLogGenerator = compact_factory
