"""Engine sessions: who may call this Aurora engine (1.9.4+).

Before 1.9.4 every engine route was open to anything that could reach the port. Now each
HTTP request and WebSocket must carry an engine session cookie, created only from a
handoff pass signed by the lab website.

Flow (one lab round trip per sign-in, nothing per request):

1. Lab /MainAurora reads ``engine_id`` from this engine's public ``/get-version/``.
2. Lab ``/auroraHandoff`` signs a short pass for that engine (Ed25519):
   ``base64url(json claims) + "." + base64url(signature over the first segment)``.
   Claims: v, iss (lab origin), aud (engine_id), sub (lab user id), name, level, iat,
   exp, nonce, kid (signing key id).
3. The packaged SPA posts the pass to ``/engine-auth/start/``. We verify it against the
   lab's public key (fetched once from ``<iss>/auroraEngineKeys`` over HTTPS, then cached
   on disk), check audience, time window and single use, and set an HttpOnly,
   SameSite=Strict cookie.
4. ``EngineSessionMiddleware`` and ``EngineSessionWebsocketMiddleware`` check that
   cookie in memory. The lab is not contacted again until the session ends, so lab
   outages or slow links never interrupt work in progress.

The lab DRF token is still exchanged separately in local mode for lab-side features
(theme, model downloads); the engine itself never relies on it for access.

Network mode (engine.json mode "network", shared on the LAN) adds, for requests that do
not come from this computer itself:

* Passes must be network passes (claim net=1) naming this computer's pairing key (claim
  skf). The lab signs those only for the owner and active, unblocked group members.
* An offline second check against the last member list (networkMode.group_access), and
  after every check-in sessions of removed, blocked or kicked people end.
* One user at a time (the lease, for every session in network mode): the first signed-in
  session holds it while it keeps beating or working, and frees it after
  LEASE_GRACE_SECONDS idle. Others get 409 engine_in_use; the same person on another
  device may take over (/engine-auth/take-over/).
* Sessions end after a time without real use (requests; the tab heartbeat and background
  reads never count): NETWORK_SESSION_SECONDS (12 h) for every session while network mode
  is on, whether opened here or from another computer, and LOCAL_SESSION_SECONDS (7 d) with
  sharing off. A session ended for idling in network mode tells its tab it was signed off.
  Removals still end sessions at the next check-in.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import os
import secrets
import sys
import threading
import time
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Dict, Optional

from django.core.signals import got_request_exception
from django.http import JsonResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from AuroraClient import engineConfig
from pipeline import engineActivity, engineJobs, engineSharing, networkMode

logger = logging.getLogger(__name__)

PASS_VERSION = 1
# Lab passes live 60 s; allow some clock drift between this machine and the lab server.
CLOCK_SKEW_SECONDS = 300
MAX_PASS_LIFETIME_SECONDS = 300
# Sessions slide: they end after this long without real use (see renew_session), not after
# a fixed time. Network mode on: NETWORK_SESSION_SECONDS for all; off: LOCAL_SESSION_SECONDS.
LOCAL_SESSION_SECONDS = 7 * 24 * 3600
NETWORK_SESSION_SECONDS = 12 * 3600
# Renewal is written to disk (and the cookie re-sent) at most this often, not per request.
SESSION_RENEW_INTERVAL_SECONDS = 3600
# Shown by the tab whose network session ended for lack of use (engine_session_ended).
NETWORK_IDLE_MESSAGE = ("Signed off from this computer after 12 hours without use. "
                        "Your lab account is still signed in; choose Open again to continue.")
# A lease holder that has neither beaten nor made a request for this long (tab closed,
# laptop asleep, connection lost) frees the computer for the next person.
LEASE_GRACE_SECONDS = 10 * 60

SESSION_COOKIE = "aurora_engine_session"


# Routes that stay open without a session. Everything else needs one.
PUBLIC_PATH_PREFIXES = (
    "/local_aurora/",   # packaged SPA shell and its assets
    "/static/",         # bundled runtime media (textures, icons)
    "/get-version/",    # lab splash detection: version + engine_id only
    "/engine-auth/",    # sign-in / status / sign-out
    "/docs/",           # read-only bundled documentation (lab docs page reads it)
    "/terminal/",       # terminal viewer page shell; its lines need a session (engineTerminal)
)

# Signed-in routes that do not take the one-person lease: watching the terminal must not
# take the computer from whoever is using it, nor count as using it.
LEASE_FREE_PATH_PREFIXES = (
    "/engine-terminal/",
    "/engine-report/",   # the report button works even while someone else uses Aurora
)
# Reads that do not use the computer: they neither need the lease nor refresh it (the
# header's servers icon asks /engine-sharing/ every minute, which must not keep an idle
# tab holding the computer, and must still answer while someone else holds it).
LEASE_FREE_READ_PREFIXES = (
    "/engine-sharing/",
)

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class PassError(Exception):
    """A handoff pass that must not create a session; message is safe to show."""


# Production + (local mode only) dev lab sites; see engineConfig.
trusted_lab_origins = engineConfig.trusted_lab_origins


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


# ---------------------------------------------------------------------------
# Lab public keys (fetched once per lab origin, cached in memory and on disk)
# ---------------------------------------------------------------------------

_keys_lock = threading.Lock()
_keys: Dict[str, Dict[str, str]] = {}      # iss -> {kid: base64 raw public key}
_keys_loaded_from_disk = False
_last_fetch: Dict[str, float] = {}
_FETCH_COOLDOWN_SECONDS = 30


def _keys_cache_path() -> Path:
    return engineConfig.config_dir() / "lab_keys.json"


def _load_keys_from_disk() -> None:
    global _keys_loaded_from_disk
    if _keys_loaded_from_disk:
        return
    _keys_loaded_from_disk = True
    try:
        data = json.loads(_keys_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        for iss, kid_map in data.items():
            if iss in trusted_lab_origins() and isinstance(kid_map, dict):
                _keys.setdefault(iss, {}).update(
                    {str(k): str(v) for k, v in kid_map.items()}
                )


def _save_keys_to_disk() -> None:
    path = _keys_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_keys, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Could not cache lab keys: %s", exc)


def _fetch_lab_keys(iss: str) -> None:
    now = time.time()
    if now - _last_fetch.get(iss, 0.0) < _FETCH_COOLDOWN_SECONDS:
        return
    _last_fetch[iss] = now
    import requests  # local import: only needed on a cache miss

    try:
        response = requests.get(f"{iss}/auroraEngineKeys", timeout=8)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - any failure means "no new keys"
        logger.warning("Could not fetch lab signing keys from %s: %s", iss, exc)
        return
    found = {}
    for entry in payload.get("keys", []) if isinstance(payload, dict) else []:
        if isinstance(entry, dict) and entry.get("alg") == "Ed25519":
            kid, public_key = entry.get("kid"), entry.get("public_key")
            if isinstance(kid, str) and isinstance(public_key, str):
                found[kid] = public_key
    if found:
        _keys.setdefault(iss, {}).update(found)
        _save_keys_to_disk()


def _public_key_for(iss: str, kid: str):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    with _keys_lock:
        _load_keys_from_disk()
        encoded = _keys.get(iss, {}).get(kid)
        if encoded is None:
            _fetch_lab_keys(iss)
            encoded = _keys.get(iss, {}).get(kid)
    if encoded is None:
        raise PassError("The lab website's signing key could not be loaded. Check the internet connection and try again.")
    try:
        return Ed25519PublicKey.from_public_bytes(_b64url_decode(encoded))
    except Exception as exc:  # noqa: BLE001
        raise PassError("The lab website's signing key is invalid.") from exc


# ---------------------------------------------------------------------------
# Pass verification
# ---------------------------------------------------------------------------

_used_nonces: Dict[str, float] = {}   # nonce -> forget after this time
_nonce_lock = threading.Lock()


def verify_pass(token: str, remote: bool = False) -> dict:
    """Return the pass claims, or raise PassError. Consumes the pass nonce.

    remote: the request did not come from this computer (only possible in network mode).
    """
    from cryptography.exceptions import InvalidSignature

    if not isinstance(token, str) or token.count(".") != 1 or len(token) > 4096:
        raise PassError("Malformed sign-in pass.")
    payload_segment, signature_segment = token.split(".")
    try:
        claims = json.loads(_b64url_decode(payload_segment))
        signature = _b64url_decode(signature_segment)
    except (ValueError, TypeError) as exc:
        raise PassError("Malformed sign-in pass.") from exc
    if not isinstance(claims, dict) or claims.get("v") != PASS_VERSION:
        raise PassError("Unsupported sign-in pass. Update Aurora or reload the lab page.")

    iss = claims.get("iss")
    if iss not in trusted_lab_origins():
        raise PassError("The sign-in pass did not come from a trusted lab website.")
    if claims.get("aud") != engineConfig.engine_id():
        raise PassError("The sign-in pass was issued for a different Aurora installation.")

    public_key = _public_key_for(iss, str(claims.get("kid", "")))
    try:
        public_key.verify(signature, payload_segment.encode("ascii"))
    except InvalidSignature as exc:
        raise PassError("The sign-in pass signature is invalid.") from exc

    now = time.time()
    try:
        iat, exp = float(claims["iat"]), float(claims["exp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PassError("Malformed sign-in pass.") from exc
    if exp - iat > MAX_PASS_LIFETIME_SECONDS:
        raise PassError("Malformed sign-in pass.")
    if now < iat - CLOCK_SKEW_SECONDS or now > exp + CLOCK_SKEW_SECONDS:
        raise PassError("The sign-in pass has expired (or this computer's clock is wrong). Reload the lab page.")

    if remote:
        _check_network_pass(claims, iat)

    nonce = claims.get("nonce")
    if not isinstance(nonce, str) or not nonce:
        raise PassError("Malformed sign-in pass.")
    with _nonce_lock:
        for old, forget_at in list(_used_nonces.items()):
            if forget_at < now:
                del _used_nonces[old]
        if nonce in _used_nonces:
            raise PassError("This sign-in pass was already used. Reload the lab page.")
        _used_nonces[nonce] = exp + CLOCK_SKEW_SECONDS
    return claims


def _check_network_pass(claims: dict, issued_at: float) -> None:
    """Extra rules for a pass used from another computer on the LAN."""
    if claims.get("net") != 1:
        raise PassError("Open this computer from My Aurora servers on the lab website.")
    own_fingerprint = networkMode.own_key_fingerprint()
    if not own_fingerprint or claims.get("skf") != own_fingerprint:
        raise PassError("The sign-in pass names a different Aurora server. Reload My Aurora servers and try again.")
    allowed, reason = networkMode.group_access(str(claims.get("sub", "")), issued_at)
    if not allowed:
        raise PassError(reason)


def is_remote_address(address: Optional[str]) -> bool:
    """True unless the address is this computer (loopback). Unknown counts as remote."""
    try:
        return not ipaddress.ip_address((address or "").split("%")[0]).is_loopback
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Sessions (in memory, persisted as hashes so an engine restart keeps people signed in)
# ---------------------------------------------------------------------------

_sessions_lock = threading.Lock()
_sessions: Dict[str, dict] = {}   # sha256(session id) -> record
_sessions_loaded = False


def _sessions_path() -> Path:
    return engineConfig.config_dir() / "engine_sessions.json"


def _hash_session_id(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _load_sessions() -> None:
    global _sessions_loaded
    if _sessions_loaded:
        return
    _sessions_loaded = True
    try:
        data = json.loads(_sessions_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        now = time.time()
        _sessions.update({
            key: record for key, record in data.items()
            if isinstance(record, dict) and record.get("expires_at", 0) > now
        })


def _save_sessions() -> None:
    path = _sessions_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_sessions), encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Could not persist engine sessions: %s", exc)


def create_session(claims: dict, remote: bool = False) -> tuple:
    """Start a session for verified pass claims. Returns (session_id, record)."""
    session_id = secrets.token_urlsafe(32)
    now = time.time()
    try:
        level = int(claims.get("level", 0))
    except (TypeError, ValueError):
        level = 0
    record = {
        "user_id": str(claims.get("sub", "")),
        "name": str(claims.get("name", ""))[:200],
        "level": level,
        "lab_origin": claims.get("iss"),
        "remote": bool(remote),
        "created_at": now,
        "last_used": now,
        "expires_at": now + _idle_limit(),
    }
    with _sessions_lock:
        _load_sessions()
        for key in [k for k, r in _sessions.items() if r.get("expires_at", 0) <= now]:
            del _sessions[key]
        _sessions[_hash_session_id(session_id)] = record
        _save_sessions()
    logger.info("Engine session started for lab user %s (%s)", record["user_id"], record["name"])
    return session_id, record


def get_session(session_id: Optional[str]) -> Optional[dict]:
    if not session_id:
        return None
    key = _hash_session_id(session_id)
    with _sessions_lock:
        _load_sessions()
        record = _sessions.get(key)
        if record is None:
            return None
        now = time.time()
        network = engineConfig.is_network_mode()
        # A session from before sharing was turned on may hold a 7 day expiry; in network
        # mode its last use decides, like every other session.
        idle = network and now - record.get("last_used", record.get("created_at", 0)) > NETWORK_SESSION_SECONDS
        if record.get("expires_at", 0) <= now or idle:
            del _sessions[key]
            _save_sessions()
            if network:
                with _ended_lock:
                    _ended_reasons[key] = NETWORK_IDLE_MESSAGE
            return None
        return record


def _idle_limit() -> int:
    """How long a session may go without real use: 12 h in network mode, else 7 days."""
    return NETWORK_SESSION_SECONDS if engineConfig.is_network_mode() else LOCAL_SESSION_SECONDS


def renew_session(session_id: Optional[str], heartbeat: bool = False) -> bool:
    """Sliding expiry: real use pushes a session's end to _idle_limit() from now.

    The tab heartbeat never counts (heartbeat=True does nothing): an Aurora page left open
    but unused ends like a closed one. Written at most every SESSION_RENEW_INTERVAL_SECONDS,
    so a session ends between (limit - 1 h) and limit after its last use. Returns True when
    the record was renewed, so the caller re-sends the cookie with the new max_age.
    """
    if not session_id or heartbeat:
        return False
    key = _hash_session_id(session_id)
    now = time.time()
    with _sessions_lock:
        _load_sessions()
        record = _sessions.get(key)
        if record is None or record.get("expires_at", 0) <= now:
            return False
        if now - record.get("last_used", record.get("created_at", 0)) < SESSION_RENEW_INTERVAL_SECONDS:
            return False
        record["last_used"] = now
        record["expires_at"] = now + _idle_limit()
        _save_sessions()
    return True


def end_session(session_id: Optional[str]) -> None:
    if not session_id:
        return
    _end_session_key(_hash_session_id(session_id))


def _end_session_key(key: str, reason: str = "") -> None:
    """End a session by its stored hash. A reason is shown once to that browser instead
    of silently re-signing it in (a kick must not bounce straight back in)."""
    with _sessions_lock:
        _load_sessions()
        if _sessions.pop(key, None) is not None:
            _save_sessions()
    if reason:
        with _ended_lock:
            if len(_ended_reasons) > 200:
                _ended_reasons.clear()
            _ended_reasons[key] = reason
    release_lease(key)
    engineActivity.forget_session(key)


_ended_lock = threading.Lock()
_ended_reasons: Dict[str, str] = {}   # session hash -> why it ended (kick, removal)


def _ended_reason(session_id: Optional[str]) -> str:
    if not session_id:
        return ""
    with _ended_lock:
        return _ended_reasons.get(_hash_session_id(session_id), "")


def apply_group_update(group: dict) -> None:
    """After a check-in (network mode): end sessions of people the owner removed, blocked
    or kicked. Registered with networkMode below."""
    if not engineConfig.is_network_mode():
        return
    members = {str(m.get("user_id")): m for m in group.get("members", []) if isinstance(m, dict)}
    kick = group.get("kick") if isinstance(group.get("kick"), dict) else None
    with _sessions_lock:
        _load_sessions()
        records = list(_sessions.items())
    for key, record in records:
        user_id = str(record.get("user_id", ""))
        entry = members.get(user_id)
        remote = bool(record.get("remote"))
        reason = ""
        # Membership governs sessions opened from other computers; someone signed in at
        # this computer itself already has its OS account, so only a kick ends theirs.
        if remote and entry is None:
            reason = "You were removed from this computer's group."
        elif remote and entry.get("blocked"):
            reason = "The owner of this computer has blocked your access."
        elif kick and str(kick.get("user_id")) == user_id:
            # The lab hands each kick over once, so it ends whatever sessions exist now.
            reason = "The owner or a manager of this computer ended your session."
        if reason:
            logger.info("Ending engine session of lab user %s: %s", user_id, reason)
            _end_session_key(key, reason)


networkMode.on_group_update(apply_group_update)


def end_sessions_where(predicate, reason: str) -> None:
    """End every session whose record matches (engineSharing: pausing members)."""
    with _sessions_lock:
        _load_sessions()
        records = list(_sessions.items())
    for key, record in records:
        if predicate(record):
            logger.info("Ending engine session of lab user %s: %s", record.get("user_id"), reason)
            _end_session_key(key, reason)


# ---------------------------------------------------------------------------
# Lease: one person at a time in network mode
# ---------------------------------------------------------------------------

_lease_lock = threading.Lock()
_lease: Optional[dict] = None   # {"session", "user_id", "name", "since"}


def release_lease(session_key: str) -> None:
    global _lease
    with _lease_lock:
        if _lease is not None and _lease["session"] == session_key:
            _lease = None


def lease_holder() -> Optional[dict]:
    """Who holds the computer now ({"user_id", "name", "since"}), or None when the lease is
    free: nobody holds it, or the holder has been idle past the grace period (the next
    claim_lease would take it)."""
    with _lease_lock:
        holder = _lease
    if holder is None or engineActivity.session_idle_seconds(holder["session"]) > LEASE_GRACE_SECONDS:
        return None
    return {"user_id": holder["user_id"], "name": holder["name"], "since": holder["since"]}


def claim_lease(session_key: str, record: dict, take_over: bool = False) -> Optional[dict]:
    """Hold or take the lease for this session. Returns None when it may proceed,
    otherwise who holds the computer (for the 409 response)."""
    global _lease
    if not engineConfig.is_network_mode():
        return None
    user_id = str(record.get("user_id", ""))
    with _lease_lock:
        holder = _lease
        free = (
            holder is None
            or holder["session"] == session_key
            or engineActivity.session_idle_seconds(holder["session"]) > LEASE_GRACE_SECONDS
            or (take_over and holder["user_id"] == user_id)
        )
        if free:
            engineActivity.touch(session_key)
            if holder is None or holder["session"] != session_key:
                _lease = {"session": session_key, "user_id": user_id,
                          "name": record.get("name", ""), "since": time.time()}
                if holder is not None:
                    logger.info("Lease moved from %s to %s", holder["name"], _lease["name"])
            return None
        return {
            "holder_name": holder["name"],
            "since": holder["since"],
            "same_user": holder["user_id"] == user_id,
        }


def _in_use_response(conflict: dict) -> JsonResponse:
    if conflict["same_user"]:
        message = "You are using Aurora on this computer from another device or browser."
    else:
        message = f"{conflict['holder_name']} is using this computer right now."
    return JsonResponse({"error": message, "code": "engine_in_use", **conflict}, status=409)


# ---------------------------------------------------------------------------
# Sessions (in memory, persisted as hashes so an engine restart keeps people signed in)
# ---------------------------------------------------------------------------


def acceptable_host(host_header: str) -> bool:
    """Network mode Host check (settings.py allows "*" then): IP literals, localhost and
    this computer's own name. Any other name could be a DNS-rebinding domain."""
    import platform

    host = (host_header or "").strip().lower()
    if host.startswith("["):
        host = host[1:host.find("]")] if "]" in host else host
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    node = (platform.node() or "").lower()
    return host in ("localhost", node, f"{node}.local") and bool(host)


def _is_public_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in PUBLIC_PATH_PREFIXES)


def _session_payload(record: Optional[dict]) -> dict:
    payload = {
        "authenticated": record is not None,
        "engine_id": engineConfig.engine_id(),
        "mode": engineConfig.load_config()["mode"],
    }
    if record is not None:
        payload.update({
            "name": record["name"],
            "level": record["level"],
            "expires_at": record["expires_at"],
            # Opened from another computer: the SPA has no lab token then and takes the
            # name and level from here (the lab token never leaves the lab for a server).
            "network": bool(record.get("remote")),
            # The lab website this session was signed by (developer runs move to the dev one).
            "lab_origin": record.get("lab_origin") or "",
        })
    return payload


def engine_identity() -> dict:
    """Public fields merged into /get-version/ so the lab knows whom to sign for. "service":
    the shared background Aurora (1.9.4 network sharing), which launchers and the Settings
    hand-over wait for."""
    return {"engine_id": engineConfig.engine_id(), "engine_auth": PASS_VERSION,
            "service": os.environ.get("AURORA_SERVICE") == "1",
            "dev_lab": dev_lab_origin()}


def dev_lab_origin() -> str:
    """Developer run (run_aurora.sh): the local dev lab site the page should use for
    everything (sign-in, pairing, reports, My Aurora servers), or "" (installed Aurora, or
    AURORA_DEV_LAB_URL=off)."""
    if not engineSharing.dev_mode():
        return ""
    origin = os.environ.get("AURORA_DEV_LAB_URL", "").strip().rstrip("/")
    if not origin or origin == "off":
        return ""
    origin = origin.replace("://0.0.0.0:", "://127.0.0.1:")
    return origin if origin in engineConfig.trusted_lab_origins() else ""


# ---------------------------------------------------------------------------
# HTTP middleware
# ---------------------------------------------------------------------------

class EngineSessionMiddleware:
    """Reject engine API calls that have no valid engine session.

    Also refuses state-changing requests whose Origin is another site, as a second
    guard next to the SameSite=Strict cookie.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if engineConfig.is_network_mode() and not acceptable_host(request.META.get("HTTP_HOST", "")):
            return JsonResponse({"error": "Unknown host name.", "code": "engine_bad_host"}, status=400)
        if request.method == "OPTIONS":
            return self.get_response(request)

        if request.method not in SAFE_METHODS:
            origin = request.headers.get("Origin")
            if origin and origin != f"{request.scheme}://{request.get_host()}":
                return JsonResponse(
                    {"error": "Cross-site request refused.", "code": "engine_cross_site"},
                    status=403,
                )

        if _is_public_path(request.path):
            return self.get_response(request)

        session_id = request.COOKIES.get(SESSION_COOKIE)
        record = get_session(session_id)
        if record is None:
            reason = _ended_reason(session_id)
            if reason:
                return JsonResponse({"error": reason, "code": "engine_session_ended"}, status=401)
            return JsonResponse(
                {
                    "error": "Sign in through the lab website to use this Aurora engine.",
                    "code": "engine_session_required",
                },
                status=401,
            )
        session_key = _hash_session_id(session_id)
        if engineSharing.refuses_member(record):
            _end_session_key(session_key, engineSharing.PAUSED_MESSAGE)
            return JsonResponse({"error": engineSharing.PAUSED_MESSAGE, "code": "engine_session_ended"}, status=401)
        if any(request.path.startswith(prefix) for prefix in LEASE_FREE_PATH_PREFIXES) or (
            request.method in SAFE_METHODS
            and any(request.path.startswith(prefix) for prefix in LEASE_FREE_READ_PREFIXES)
        ):
            request.aurora_session = record
            return self.get_response(request)
        conflict = claim_lease(session_key, record)
        if conflict is not None:
            return _in_use_response(conflict)
        request.aurora_session = record
        renewed = renew_session(session_id)
        # Long requests are how most tools run; engineJobs keeps them as jobs (history,
        # shown again in a tab opened later, the lab's "In use by" line).
        token = engineActivity.request_started(record, request, session_key)
        status_code, error = 500, ""
        try:
            response = self.get_response(request)
            status_code = getattr(response, "status_code", 200)
            if status_code >= 400:
                error = _response_error(response)
            if renewed:
                _set_session_cookie(request, response, session_id, record)
            return response
        except Exception as exc:  # noqa: BLE001 - recorded, then raised as before
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            engineActivity.request_finished(
                token, status_code, error or getattr(request, "_aurora_exception", ""))


def _response_error(response) -> str:
    """The message a failed tool sent back ({"error": ...} etc.), for the job history."""
    data = getattr(response, "data", None)
    if data is None and not getattr(response, "streaming", False):
        try:
            if "json" in response.get("Content-Type", ""):
                data = json.loads(response.content[:20000])
        except Exception:  # noqa: BLE001 - optional detail
            data = None
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            if data.get(key):
                return str(data[key])
    return ""


def _remember_exception(sender, request=None, **kwargs):
    """Uncaught view errors become a plain 500 page; keep the exception for the history."""
    exc = sys.exc_info()[1]
    if request is not None and exc is not None:
        try:
            request._aurora_exception = f"{type(exc).__name__}: {exc}"
        except Exception:  # noqa: BLE001
            pass


got_request_exception.connect(_remember_exception, dispatch_uid="aurora_engine_job_errors")


# ---------------------------------------------------------------------------
# WebSocket middleware
# ---------------------------------------------------------------------------

class EngineSessionWebsocketMiddleware:
    """Close WebSocket handshakes that carry no valid engine session cookie."""

    def __init__(self, inner):
        self.inner = inner

    @staticmethod
    def _same_origin(scope) -> bool:
        # settings.ALLOWED_HOSTS is "*" in network mode, which also opens Channels'
        # AllowedHostsOriginValidator; require Origin == this Host ourselves.
        headers = {name: value.decode("latin-1") for name, value in scope.get("headers", [])}
        host = headers.get(b"host", "")
        origin = headers.get(b"origin", "")
        if not acceptable_host(host):
            return False
        return origin in (f"https://{host}", f"http://{host}")

    async def __call__(self, scope, receive, send):
        cookie_header = ""
        for name, value in scope.get("headers", []):
            if name == b"cookie":
                cookie_header = value.decode("latin-1")
                break
        cookies = SimpleCookie()
        try:
            cookies.load(cookie_header)
        except Exception:  # noqa: BLE001 - malformed cookie header = no session
            cookies = SimpleCookie()
        if engineConfig.is_network_mode() and not self._same_origin(scope):
            from channels.security.websocket import WebsocketDenier

            return await WebsocketDenier.as_asgi()(scope, receive, send)
        morsel = cookies.get(SESSION_COOKIE)
        session_id = morsel.value if morsel else None
        record = get_session(session_id)
        # Same one-person rule as HTTP: only the lease holder gets progress sockets.
        if record is None or claim_lease(_hash_session_id(session_id), record) is not None:
            from channels.security.websocket import WebsocketDenier

            return await WebsocketDenier.as_asgi()(scope, receive, send)
        return await self.inner(scope, receive, send)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def _set_session_cookie(request, response, session_id: str, record: dict) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=max(0, int(record["expires_at"] - time.time())),
        httponly=True,
        samesite="Strict",
        secure=request.is_secure(),
        path="/",
    )


class EngineAuthStartView(APIView):
    """POST /engine-auth/start/ {pass} — exchange a lab-signed pass for an engine session."""

    authentication_classes = []
    permission_classes = []

    def post(self, request, *args, **kwargs):
        token = request.data.get("pass") if hasattr(request.data, "get") else None
        remote = is_remote_address(request.META.get("REMOTE_ADDR"))
        if remote and not engineConfig.is_network_mode():
            # Cannot happen while we listen on loopback only; refuse if it ever does.
            return Response({"error": "This Aurora is not shared on the network.", "code": "engine_not_shared"},
                            status=status.HTTP_403_FORBIDDEN)
        try:
            claims = verify_pass(token, remote=remote)
        except PassError as exc:
            logger.warning("Engine sign-in refused: %s", exc)
            return Response(
                {"error": str(exc), "code": "engine_pass_invalid"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        if remote and engineSharing.refuses_member({"remote": True, "user_id": str(claims.get("sub", ""))}):
            return Response({"error": engineSharing.PAUSED_MESSAGE, "code": "engine_paused"},
                            status=status.HTTP_403_FORBIDDEN)
        # A new sign-in replaces whatever session this browser held.
        end_session(request.COOKIES.get(SESSION_COOKIE))
        session_id, record = create_session(claims, remote=remote)
        if remote:
            logger.info("Network session for %s from %s", record["name"], request.META.get("REMOTE_ADDR"))
        response = Response(_session_payload(record), status=status.HTTP_200_OK)
        _set_session_cookie(request, response, session_id, record)
        return response


class EngineAuthStatusView(APIView):
    """GET /engine-auth/status/ — is this browser signed in to the engine?"""

    authentication_classes = []
    permission_classes = []

    def get(self, request, *args, **kwargs):
        session_id = request.COOKIES.get(SESSION_COOKIE)
        record = get_session(session_id)
        payload = _session_payload(record)
        if record is None:
            reason = _ended_reason(session_id)
            if reason:
                payload.update({"code": "engine_session_ended", "error": reason})
        return Response(payload, status=status.HTTP_200_OK)


class EngineAuthEndView(APIView):
    """POST /engine-auth/end/ — sign this browser out of the engine."""

    authentication_classes = []
    permission_classes = []

    def post(self, request, *args, **kwargs):
        end_session(request.COOKIES.get(SESSION_COOKIE))
        response = Response({"authenticated": False}, status=status.HTTP_200_OK)
        response.delete_cookie(SESSION_COOKIE, path="/", samesite="Strict")
        return response


class EngineAuthHeartbeatView(APIView):
    """POST /engine-auth/heartbeat/[?leaving=1] — an Aurora tab of this session is open.

    The packaged SPA calls it every 30 s, and once with leaving=1 (sendBeacon) when the tab
    closes. Feeds engineActivity, which tells the lab who has Aurora open.
    """

    authentication_classes = []
    permission_classes = []

    def post(self, request, *args, **kwargs):
        session_id = request.COOKIES.get(SESSION_COOKIE)
        record = get_session(session_id)
        if record is None:
            # Same answer as the middleware, so the tab says why (kick, removal) when known.
            reason = _ended_reason(session_id)
            if reason:
                return Response({"authenticated": False, "code": "engine_session_ended", "error": reason},
                                status=status.HTTP_401_UNAUTHORIZED)
            return Response(
                {"authenticated": False, "code": "engine_session_required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        leaving = request.query_params.get("leaving") == "1"
        session_key = _hash_session_id(session_id)
        if not leaving:
            # A waiting tab (someone else holds the computer) is not "using" it.
            conflict = claim_lease(session_key, record)
            if conflict is not None:
                return _in_use_response(conflict)
        engineActivity.heartbeat(session_key, record, leaving=leaving)
        response = Response(_session_payload(record), status=status.HTTP_200_OK)
        # An open tab is not use: renew_session ignores heartbeats (sessions end after
        # 12 h / 7 d without real requests).
        if not leaving and renew_session(session_id, heartbeat=True):
            _set_session_cookie(request, response, session_id, record)
        return response


class EngineAuthTakeOverView(APIView):
    """POST /engine-auth/take-over/ — the same person moves to this device or browser.

    Only for a lease held by the same lab user (or one that has gone idle); someone else's
    session is ended by the owner or a manager from the website instead.
    """

    authentication_classes = []
    permission_classes = []

    def post(self, request, *args, **kwargs):
        session_id = request.COOKIES.get(SESSION_COOKIE)
        record = get_session(session_id)
        if record is None:
            return Response({"authenticated": False, "code": "engine_session_required"},
                            status=status.HTTP_401_UNAUTHORIZED)
        conflict = claim_lease(_hash_session_id(session_id), record, take_over=True)
        if conflict is not None:
            return _in_use_response(conflict)
        return Response(_session_payload(record), status=status.HTTP_200_OK)


class EngineJobsView(APIView):
    """GET /engine-jobs/[?history=1] — jobs running on this engine, and recent ones.

    Aurora asks on load and when a tab becomes visible again, so a job started in a tab
    that has since closed (or on a laptop that went to sleep) shows up again until it ends.
    Running jobs show to everyone signed in (the lab's servers page shows them to the group
    anyway); people on other computers see only their own history and folders.
    """

    authentication_classes = []
    permission_classes = []

    def get(self, request, *args, **kwargs):
        record = getattr(request, "aurora_session", None)
        if record is None:
            return Response({"authenticated": False, "code": "engine_session_required"},
                            status=status.HTTP_401_UNAUTHORIZED)
        user_id = str(record.get("user_id", ""))
        limited = bool(record.get("remote")) and engineConfig.is_network_mode()

        def shape(job: dict) -> dict:
            job = dict(job)
            job["mine"] = bool(job["user_id"]) and job["user_id"] == user_id
            if limited and not job["mine"]:
                job.update(directory="", scans=[])
            job.pop("user_id", None)
            return job

        payload = {"running": [shape(job) for job in engineJobs.running()], "now": time.time()}
        if request.query_params.get("history") == "1":
            payload["recent"] = [
                shape(job) for job in engineJobs.recent()
                if not limited or job["user_id"] == user_id
            ]
        return Response(payload, status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Workspace: where each person was working, kept by the engine
# ---------------------------------------------------------------------------

_workspace_lock = threading.Lock()


def _workspace_path() -> Path:
    return engineConfig.config_dir() / "workspace.json"


def _read_workspace() -> dict:
    try:
        data = json.loads(_workspace_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


class EngineWorkspaceView(APIView):
    """GET/POST /engine-workspace/ {current_path} — the folder each lab account last had
    open on this engine. The browser keeps its own copy too, but on a shared computer the
    engine's copy wins when it is newer: after a take-over from another computer, Aurora
    opens where that person left off. Only a path string is kept; listing it is still
    subject to the shared-folder rules of every other view."""

    authentication_classes = []
    permission_classes = []

    def get(self, request, *args, **kwargs):
        record = getattr(request, "aurora_session", None)
        if record is None:
            return Response({"code": "engine_session_required"}, status=status.HTTP_401_UNAUTHORIZED)
        with _workspace_lock:
            entry = _read_workspace().get(str(record.get("user_id", ""))) or {}
        return Response({"current_path": entry.get("current_path") or "",
                         "updated_at": entry.get("updated_at") or 0}, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        record = getattr(request, "aurora_session", None)
        if record is None:
            return Response({"code": "engine_session_required"}, status=status.HTTP_401_UNAUTHORIZED)
        data = request.data if hasattr(request.data, "get") else {}
        path = data.get("current_path")
        if not isinstance(path, str) or not path or len(path) > 4096 or "\x00" in path:
            return Response({"error": "current_path must be a folder path."}, status=status.HTTP_400_BAD_REQUEST)
        user_id = str(record.get("user_id", ""))
        if not user_id:
            return Response({"error": "No lab account on this session."}, status=status.HTTP_400_BAD_REQUEST)
        now = time.time()
        with _workspace_lock:
            workspace = _read_workspace()
            workspace[user_id] = {"current_path": path, "updated_at": now}
            # Bounded: the 200 most recently active accounts.
            if len(workspace) > 200:
                keep = sorted(workspace.items(), key=lambda item: item[1].get("updated_at", 0))[-200:]
                workspace = dict(keep)
            target = _workspace_path()
            tmp = target.with_suffix(".tmp")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(workspace) + "\n", encoding="utf-8")
                os.replace(tmp, target)
            except OSError as exc:
                return Response({"error": f"Could not save: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({"current_path": path, "updated_at": now}, status=status.HTTP_200_OK)
