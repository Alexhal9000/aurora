"""Pairing and check-in with the lab website (Aurora 1.9.4+).

Pairing makes a lab member the owner of this computer's Aurora, so they can share it with
their group (frontend/auroraServers.py on the lab site). Nobody types a password here:

    aurora-tools pair                     # Linux package
    python -m pipeline.networkMode pair   # from AuroraClient/ in a dev checkout
        --lab http://127.0.0.1:8004       # dev lab site instead of hallgrimssonlab.ca
        --reset                           # new identity (pair with someone else)

1. We ask the lab for a short code and show it.
2. The person types it at <lab>/MainAurora/pair while signed in.
3. We show who claimed it and ask the person at this computer to confirm.

This engine's identity is an Ed25519 key in ~/.aurora/engine_key.pem (0600). Every
request below is signed with it; the lab never receives a secret it could leak.

Once paired, a background thread checks in every few minutes (start_check_in_thread,
called from asgi.py): it reports status and receives the group's member list and any
pending kick. Check-ins are never on a request path, so a slow or unreachable lab only
delays membership changes; it never interrupts work.

Stdlib + cryptography + requests only, so the CLI runs before Django is configured.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import threading
import time
from pathlib import Path

from AuroraClient import engineConfig
from pipeline import engineActivity, engineJobs

logger = logging.getLogger(__name__)

LAB_ORIGIN_PRODUCTION = "https://hallgrimssonlab.ca"
CHECK_IN_SECONDS = 5 * 60
CHECK_IN_BUSY_SECONDS = 20    # while a job runs and someone has My Aurora servers open (lab says "watched")
CHECK_IN_BUSY_UNWATCHED_SECONDS = 120  # while a job runs and nobody is looking: spares the lab server
MIN_EVENT_GAP_SECONDS = 15    # someone arriving/leaving or a job starting/ending checks in at once, at most this often
POLL_SECONDS = 30  # how often the engine notices a new pairing or retries a failed check-in
PAIR_POLL_SECONDS = 4


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _key_path() -> Path:
    return engineConfig.config_dir() / "engine_key.pem"


def _pairing_path() -> Path:
    return engineConfig.config_dir() / "pairing.json"


def _group_path() -> Path:
    return engineConfig.config_dir() / "network_group.json"


def _write_private_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Engine key
# ---------------------------------------------------------------------------

def load_or_create_key(reset: bool = False):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = _key_path()
    if reset and path.exists():
        path.unlink()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = Ed25519PrivateKey.generate().private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(pem)
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def public_key_b64(private_key) -> str:
    from cryptography.hazmat.primitives import serialization

    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return _b64url(raw)


def key_fingerprint(private_key) -> str:
    """Same fingerprint the lab stores (sha256 of the raw public key). Handoff passes for
    network sessions carry it, so a lab record made with another key cannot sign people in."""
    from cryptography.hazmat.primitives import serialization

    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def signed_post(lab_origin: str, path: str, payload: dict, private_key=None, timeout: float = 15):
    """POST JSON to the lab, signed with this engine's key (see frontend/auroraServers.py)."""
    import requests

    private_key = private_key or load_or_create_key()
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    timestamp = str(int(time.time()))
    message = f"{timestamp}\nPOST\n{path}\n{hashlib.sha256(body).hexdigest()}".encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Aurora-Engine-Key": public_key_b64(private_key),
        "X-Aurora-Timestamp": timestamp,
        "X-Aurora-Signature": _b64url(private_key.sign(message)),
    }
    return requests.post(f"{lab_origin}{path}", data=body, headers=headers, timeout=timeout)


def _trusted_lab_origin(origin: str) -> str:
    # Same allow-list as handoff passes: production, plus local dev sites in local mode.
    origin = (origin or LAB_ORIGIN_PRODUCTION).rstrip("/")
    if origin not in engineConfig.trusted_lab_origins():
        raise SystemExit(f"Refusing to pair with {origin}: not a lab website Aurora trusts.")
    return origin


def _installed_version() -> str:
    for candidate in ("/opt/aurora-tools/version.txt",
                      os.path.join(os.environ.get("LOCALAPPDATA", ""), "Aurora", "version.txt"),
                      "/Library/Application Support/Aurora/version.txt"):
        try:
            return Path(candidate).read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


# ---------------------------------------------------------------------------
# Pairing (interactive; runs in a terminal or behind the Settings panel in step 4)
# ---------------------------------------------------------------------------

def pairing_state() -> dict:
    return _read_json(_pairing_path())


def run_pairing_cli(lab_origin: str, name: str, reset: bool) -> int:
    lab_origin = _trusted_lab_origin(lab_origin)
    if reset:
        for path in (_pairing_path(), _group_path()):
            try:
                path.unlink()
            except OSError:
                pass
    key = load_or_create_key(reset=reset)

    import requests

    try:
        response = signed_post(lab_origin, "/auroraPairStart", {
            "engine_id": engineConfig.engine_id(),
            "name": name,
            "version": _installed_version(),
        }, private_key=key)
    except requests.RequestException:
        print(f"\nCould not reach the lab website at {lab_origin}.")
        print("Check this computer's internet connection (or that the dev site is running) and try again.")
        return 1
    data = response.json() if response.headers.get("Content-Type", "").startswith("application/json") else {}
    known = _known_pairing(response, data)
    if known:
        # Run by an administrator at this computer, with this computer's own key: the lab
        # already lists it, so only the local record was missing.
        _restore_pairing(lab_origin, key, known)
        print(f"\n  This computer was already paired: \"{known['server_name']}\", owned by "
              f"{known['owner']['name']}. Restored that pairing.")
        print("  To give it to someone else instead: sudo aurora-tools pair --reset\n")
        return 0
    if response.status_code != 200:
        print(f"\nCould not start pairing: {data.get('error') or response.status_code}")
        return 1

    print("\n" + "=" * 60)
    print("  Pair this computer with your lab account".center(60))
    print("=" * 60)
    print(f"\n  1. On any device, open:  {data['pair_url']}")
    print("  2. Sign in, and type this code:\n")
    print(f"        {data['code']}\n")
    print(f"  The code works once and expires in {data['expires_in'] // 60} minutes.")
    print("  Waiting for the code to be used... (Ctrl+C to cancel)\n")

    deadline = time.time() + data["expires_in"] + 5
    while time.time() < deadline:
        time.sleep(PAIR_POLL_SECONDS)
        try:
            status_response = signed_post(lab_origin, "/auroraPairStatus", {}, private_key=key)
            status_data = status_response.json()
        except Exception as exc:  # noqa: BLE001 - keep waiting through network blips
            logger.debug("pair status: %s", exc)
            continue
        state = status_data.get("state")
        if state == "expired":
            break
        if state != "claimed":
            continue
        who = status_data.get("claimed_by") or {}
        print(f"  The code was used by: {who.get('name')} ({who.get('email')})\n")
        answer = input("  Is this you, or someone you trust to own this computer? [y/N] ").strip().lower()
        try:
            confirm = signed_post(lab_origin, "/auroraPairConfirm", {"accept": answer in ("y", "yes")}, private_key=key)
            result = confirm.json()
        except (requests.RequestException, ValueError):
            print(f"\n  Could not reach the lab website at {lab_origin}. Run pairing again.")
            return 1
        if answer not in ("y", "yes"):
            print("\n  Pairing cancelled. Nothing was shared. Run pairing again to get a new code.")
            return 1
        if confirm.status_code != 200:
            print(f"\n  Pairing failed: {result.get('error')}")
            return 1
        _write_private_json(_pairing_path(), {
            "lab_origin": lab_origin,
            "server_name": result.get("server_name"),
            "owner": result.get("owner"),
            "key_fingerprint": key_fingerprint(key),
            "paired_at": time.time(),
        })
        print(f"\n  Paired. {result['owner']['name']} owns \"{result['server_name']}\" and can add")
        print(f"  people at {lab_origin}/MainAurora/servers\n")
        check_in_once()
        return 0

    print("  The code expired before it was used. Run pairing again for a new one.")
    return 1


# ---------------------------------------------------------------------------
# Pairing from Aurora Settings (person at this computer; same steps as the CLI)
# ---------------------------------------------------------------------------

_pair_lock = threading.Lock()
_pair_flow: dict = {}   # {"lab_origin", "code", "pair_url", "expires_at", "state", "claimed_by", "checked_at", "error"}


def _pair_public(flow: dict) -> dict:
    if not flow:
        return {}
    return {key: flow.get(key) for key in ("code", "pair_url", "expires_at", "state", "claimed_by", "error")}


def _known_pairing(response, data: dict) -> dict:
    """The lab's record of this computer when pairing start says it is already paired
    (409 already_paired with its server name and owner; older labs send no record), else {}."""
    if response.status_code != 409 or data.get("code") != "already_paired":
        return {}
    server = data.get("server") if isinstance(data.get("server"), dict) else {}
    owner = server.get("owner") if isinstance(server.get("owner"), dict) else {}
    if not owner.get("id"):
        return {}
    return {"server_name": str(server.get("name") or "")[:100],
            "owner": {"id": str(owner["id"]), "name": str(owner.get("name") or "")[:120],
                      "email": str(owner.get("email") or "")[:200]}}


def _restore_pairing(lab_origin: str, key, known: dict) -> dict:
    """Write the local pairing record from the lab's own record of this key (see
    _known_pairing). Nothing changes on the lab: the computer just learns again that it
    is the server the lab already knows, so its group and members stay as they were."""
    if (pairing_state() or {}).get("lab_origin") not in (None, lab_origin):
        # Switching lab website: the old group belongs to the old one.
        try:
            _group_path().unlink()
        except OSError:
            pass
    _write_private_json(_pairing_path(), {
        "lab_origin": lab_origin,
        "server_name": known["server_name"],
        "owner": {"name": known["owner"]["name"], "email": known["owner"]["email"]},
        "key_fingerprint": key_fingerprint(key),
        "paired_at": time.time(),
        "restored": True,
    })
    logger.warning("Pairing with %s restored: this computer is already \"%s\" owned by %s.",
                   lab_origin, known["server_name"], known["owner"]["name"])
    threading.Thread(target=check_in_once, name="aurora-pair-check-in", daemon=True).start()
    return {"state": "paired", "server_name": known["server_name"], "restored": True}


def pair_start(lab_origin: str, name: str, user_id: str = "", restore_only: bool = False) -> dict:
    """Ask the lab for a pairing code. The person types it on the website, then confirms
    here who claimed it (pair_confirm). Raises ValueError with a readable message.

    If the lab already knows this computer's key and its owner is `user_id` (the person
    signed in at this computer), the local record is restored instead and no code is
    needed (returns state "paired"). Someone else still cannot take the computer over.
    restore_only: refuse to hand out a code (the shared service's working pairing with
    another lab website changes owner only through sudo aurora-tools pair --reset)."""
    lab_origin = (lab_origin or LAB_ORIGIN_PRODUCTION).rstrip("/")
    if lab_origin not in engineConfig.trusted_lab_origins():
        raise ValueError(f"{lab_origin} is not a lab website Aurora trusts.")
    key = load_or_create_key()
    try:
        response = signed_post(lab_origin, "/auroraPairStart", {
            "engine_id": engineConfig.engine_id(),
            "name": (name or platform.node() or "Aurora server")[:100],
            "version": _installed_version(),
        }, private_key=key)
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - lab unreachable or not JSON
        raise ValueError(f"Could not reach the lab website at {lab_origin}.") from exc
    known = _known_pairing(response, data)
    if known and user_id and known["owner"]["id"] == str(user_id):
        with _pair_lock:
            _pair_flow.clear()
        return _restore_pairing(lab_origin, key, known)
    if known:
        raise ValueError(
            f"This computer is already paired with {known['owner']['name']} on this lab website. "
            f"{known['owner']['name']} can remove it under My Aurora servers so you can pair it, "
            "or an administrator of this computer can run: sudo aurora-tools pair --reset")
    if restore_only:
        raise ValueError("This computer is paired with another lab website. To pair it with "
                         "this one, in a terminal: sudo aurora-tools pair --reset")
    if response.status_code != 200:
        raise ValueError(f"Could not start pairing: {data.get('error') or response.status_code}")
    with _pair_lock:
        _pair_flow.clear()
        _pair_flow.update({
            "lab_origin": lab_origin, "code": data["code"], "pair_url": data["pair_url"],
            "expires_at": time.time() + int(data.get("expires_in") or 600), "state": "waiting",
            "claimed_by": None, "checked_at": 0.0, "error": "",
        })
        return _pair_public(_pair_flow)


def pair_poll() -> dict:
    """Current pairing flow; asks the lab whether the code was used (at most every 3 s)."""
    with _pair_lock:
        flow = dict(_pair_flow)
    if not flow or flow.get("state") != "waiting":
        return _pair_public(flow)
    if time.time() > flow["expires_at"]:
        with _pair_lock:
            _pair_flow["state"] = "expired"
            return _pair_public(_pair_flow)
    if time.time() - flow["checked_at"] < 3:
        return _pair_public(flow)
    try:
        status_data = signed_post(flow["lab_origin"], "/auroraPairStatus", {},
                                  private_key=load_or_create_key(), timeout=8).json()
    except Exception:  # noqa: BLE001 - keep waiting through network blips
        status_data = {}
    with _pair_lock:
        _pair_flow["checked_at"] = time.time()
        state = status_data.get("state")
        if state == "expired":
            _pair_flow["state"] = "expired"
        elif state == "claimed":
            _pair_flow["state"] = "claimed"
            who = status_data.get("claimed_by") or {}
            _pair_flow["claimed_by"] = {"name": str(who.get("name") or "")[:120], "email": str(who.get("email") or "")[:200]}
            # Lab account id (not shown): lets the signed-in person's own claim confirm itself.
            _pair_flow["claimed_by_id"] = str(who.get("id") or "")
        return _pair_public(_pair_flow)


def pair_owner_id() -> str:
    """Lab account that turned sharing on from Settings (see set_network_mode), or ''."""
    return str(engineConfig._read_raw().get("pair_owner_id") or "")


def claimed_by_id() -> str:
    with _pair_lock:
        return _pair_flow.get("claimed_by_id") or ""


def pair_auto_confirm(user_id: str, lab_origin: str) -> dict:
    """Pairing from Aurora Settings by the person signed in at this computer: their browser
    claimed the code with their own lab sign-in, so when the lab says the claim came from
    that same account, confirm it without asking "is this you?". Anyone else's claim still
    waits for the manual yes/no (pair_confirm)."""
    flow = pair_poll()
    with _pair_lock:
        claimed_id = _pair_flow.get("claimed_by_id") or ""
        flow_lab = (_pair_flow.get("lab_origin") or "").rstrip("/")
    if flow.get("state") != "claimed":
        return flow
    if not (user_id and claimed_id == str(user_id) and flow_lab == (lab_origin or "").rstrip("/")):
        return flow
    return pair_confirm(True)


def pair_confirm(accept: bool) -> dict:
    """The person at this computer says whether the account that used the code may own it."""
    with _pair_lock:
        flow = dict(_pair_flow)
    if flow.get("state") != "claimed":
        raise ValueError("Nobody has used the pairing code yet.")
    key = load_or_create_key()
    try:
        confirm = signed_post(flow["lab_origin"], "/auroraPairConfirm", {"accept": bool(accept)}, private_key=key)
        result = confirm.json()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not reach the lab website at {flow['lab_origin']}.") from exc
    with _pair_lock:
        if not accept:
            _pair_flow.clear()
            return {"state": "cancelled"}
        if confirm.status_code != 200:
            _pair_flow.update(state="failed", error=str(result.get("error") or confirm.status_code))
            return _pair_public(_pair_flow)
        if (pairing_state() or {}).get("lab_origin") not in (None, flow["lab_origin"]):
            # Re-paired with another lab website: its group belongs to the old one.
            try:
                _group_path().unlink()
            except OSError:
                pass
        _write_private_json(_pairing_path(), {
            "lab_origin": flow["lab_origin"],
            "server_name": result.get("server_name"),
            "owner": result.get("owner"),
            "key_fingerprint": key_fingerprint(key),
            "paired_at": time.time(),
        })
        _pair_flow.clear()
    threading.Thread(target=check_in_once, name="aurora-pair-check-in", daemon=True).start()
    return {"state": "paired", "server_name": result.get("server_name")}


def pair_cancel() -> None:
    with _pair_lock:
        _pair_flow.clear()


# ---------------------------------------------------------------------------
# Check-in
# ---------------------------------------------------------------------------

def _lan_address() -> str:
    """Best guess at this computer's LAN address (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        return ""


def current_group() -> dict:
    """Last member list from the lab (used offline to check network sessions)."""
    return _read_json(_group_path())


def own_key_fingerprint() -> str:
    """Fingerprint of the key this computer was paired with ("" if not paired). Network
    passes must name it (claim skf), so a lab record made with another key is useless."""
    return str(pairing_state().get("key_fingerprint") or "")


def group_access(user_id: str, issued_at: float) -> tuple:
    """May this lab user open a network session, by the last member list we have?

    The lab checks membership before signing a pass, so this is the offline second check:
    refuse anyone the list marks blocked, and anyone missing from a list fetched after the
    pass was issued (removed since). A list older than the pass cannot know about people
    added in the last few minutes, so absence alone is not enough then.
    Returns (allowed, reason).
    """
    group = current_group()
    members = {str(m.get("user_id")): m for m in group.get("members", []) if isinstance(m, dict)}
    entry = members.get(str(user_id))
    if entry is not None:
        if entry.get("blocked"):
            return False, "The owner of this computer has blocked your access."
        return True, ""
    if not group or float(group.get("updated_at") or 0) < issued_at:
        return True, ""
    return False, "You are not in this computer's group. Ask its owner to add you."


# ---------------------------------------------------------------------------
# TLS certificate for the network listener
# ---------------------------------------------------------------------------

TLS_VALID_DAYS = 5 * 365


def ensure_tls_certificate() -> Path:
    """Create the self-signed certificate for the HTTPS listener if it is missing.

    Made on this computer, never leaves it except as its SHA-256 fingerprint (sent at
    check-in so the website can show it next to the browser's warning). Covers this
    computer's name, localhost and its current LAN address.
    """
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_path, key_path = engineConfig.tls_paths()
    if cert_path.exists() and key_path.exists():
        return cert_path
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(cert_path.parent, 0o700)

    key = ec.generate_private_key(ec.SECP256R1())
    host = platform.node() or "aurora"
    names = [x509.DNSName(host), x509.DNSName("localhost")]
    if not host.endswith(".local"):
        names.append(x509.DNSName(f"{host}.local"))
    addresses = {"127.0.0.1"}
    if _lan_address():
        addresses.add(_lan_address())
    names += [x509.IPAddress(ipaddress.ip_address(a)) for a in sorted(addresses)]
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"Aurora on {host}"[:64]),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Aurora (self-signed)"),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=TLS_VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path


def tls_fingerprint() -> str:
    """SHA-256 of the certificate (colon-separated hex, as browsers show it), or ""."""
    cert_path, _ = engineConfig.tls_paths()
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes

        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        return cert.fingerprint(hashes.SHA256()).hex(":").upper()
    except Exception:  # noqa: BLE001 - no certificate yet
        return ""


def network_ready() -> bool:
    cert_path, key_path = engineConfig.tls_paths()
    return engineConfig.is_network_mode() and cert_path.exists() and key_path.exists()


_group_listeners: list = []


def on_group_update(callback) -> None:
    """callback(group) after each successful check-in (engineAuth ends sessions of people
    who were removed, blocked or kicked)."""
    _group_listeners.append(callback)


def _activity_payload() -> tuple:
    """(lease, activity) for the check-in: who is using this engine and what is running."""
    snap = engineActivity.snapshot()
    user, job = snap["user"], snap["job"]
    lease = None
    if user and user["user_id"]:
        lease = {"user_id": user["user_id"], "seconds": int(max(0, time.time() - user["since"]))}
    return lease, {"present": bool(user and user["present"]), "job": job}


def check_in_once(lease: dict | None = None) -> bool:
    state = pairing_state()
    if not state.get("lab_origin"):
        return False
    config = engineConfig.load_config()
    activity = None
    if lease is None:
        lease, activity = _activity_payload()
    # Finished jobs for the website's Activity list (sent once; put back if this fails).
    reports = engineJobs.take_reports()
    if not _check_in_post(state, config, lease, activity, reports):
        engineJobs.return_reports(reports)
        return False
    return True


_lab_unreachable: dict = {}   # lab origin -> True while its check-ins fail to connect
_lab_watching = {"value": True}   # last check-in answer; a lab without the flag counts as watching


def _check_in_post(state: dict, config: dict, lease, activity, reports: list) -> bool:
    try:
        response = signed_post(state["lab_origin"], "/auroraServerCheckIn", {
            "version": _installed_version(),
            # Shared only when the HTTPS listener can exist (network mode + certificate).
            "network_enabled": network_ready(),
            "address": _lan_address(),
            "port": config["network_port"],
            "tls_fingerprint": tls_fingerprint(),
            # Settings "Pause members": the website stops signing passes for members.
            "members_paused": bool(engineConfig._read_raw().get("members_paused")),
            "lease": lease,
            "activity": activity,
            "jobs": [item for item, _ended in reports],
        })
    except Exception as exc:  # noqa: BLE001 - lab unreachable: keep last known group
        # Said once per outage (the loop retries every POLL_SECONDS * 2), not every time.
        if not _lab_unreachable.get(state["lab_origin"]):
            _lab_unreachable[state["lab_origin"]] = True
            logger.warning("Aurora check-in: the lab website at %s cannot be reached (%s). This "
                           "computer is paired with it; retrying quietly every minute.",
                           state["lab_origin"], type(exc).__name__)
        return False
    if _lab_unreachable.pop(state["lab_origin"], None):
        logger.info("Aurora check-in: the lab website at %s answers again.", state["lab_origin"])
    if response.status_code == 404:
        # Unpaired on the website: forget the pairing; sharing stops.
        logger.warning("This computer was unpaired on the lab website.")
        for path in (_pairing_path(), _group_path()):
            try:
                path.unlink()
            except OSError:
                pass
        return False
    if response.status_code != 200:
        logger.info("Aurora check-in refused: %s", response.status_code)
        return False
    data = response.json()
    _lab_watching["value"] = bool(data.get("watched", True))
    group = {
        "name": data.get("name"),
        "owner": data.get("owner"),
        "members": data.get("members", []),
        "kick": data.get("kick"),
        "updated_at": time.time(),
    }
    _write_private_json(_group_path(), group)
    for callback in list(_group_listeners):
        try:
            callback(group)
        except Exception as exc:  # noqa: BLE001 - never break the check-in loop
            logger.warning("Applying the group update failed: %s", exc)
    if state.get("server_name") != data.get("name"):
        state["server_name"] = data.get("name")
        _write_private_json(_pairing_path(), state)
    return True


_check_in_started = False


def start_check_in_thread() -> None:
    """Background check-ins while paired. Safe to call more than once."""
    global _check_in_started
    if _check_in_started:
        return
    _check_in_started = True

    wake = threading.Event()
    # Someone opening/closing Aurora or a job starting/ending wakes the loop early.
    engineActivity.on_change(wake.set)

    def loop():
        # Pairing happens in a separate process (aurora-tools pair), so poll the pairing
        # file every POLL_SECONDS: a newly paired engine shows as online within seconds.
        # After that: every CHECK_IN_SECONDS, every CHECK_IN_BUSY_SECONDS while a job runs,
        # and soon after any change in who is using it (at most every MIN_EVENT_GAP_SECONDS).
        last_ok = 0.0
        last_attempt = 0.0
        last_failed = False
        reported_key = None
        while True:
            wake.wait(CHECK_IN_BUSY_SECONDS if reported_key and reported_key[2] else POLL_SECONDS)
            wake.clear()
            if not pairing_state().get("lab_origin"):
                last_ok, reported_key = 0.0, None
                continue
            now = time.time()
            key = engineActivity.state_key()
            if key[2]:
                interval = CHECK_IN_BUSY_SECONDS if _lab_watching["value"] else CHECK_IN_BUSY_UNWATCHED_SECONDS
            else:
                interval = CHECK_IN_SECONDS
            due = now - last_ok >= interval
            # After a failed check-in (lab unreachable) wait longer before trying again.
            gap = POLL_SECONDS * 2 if last_failed else MIN_EVENT_GAP_SECONDS
            changed = key != reported_key and now - last_attempt >= gap
            if not (due or changed):
                if key != reported_key:
                    wake.set()  # changed but too soon: look again shortly
                    time.sleep(gap)
                continue
            last_attempt = now
            last_failed = not check_in_once()
            if not last_failed:
                last_ok, reported_key = time.time(), key

    threading.Thread(target=loop, name="aurora-check-in", daemon=True).start()


def print_status() -> int:
    config = engineConfig.load_config()
    state = pairing_state()
    group = current_group()
    print(f"Mode:      {config['mode']}  (this computer: http://127.0.0.1:{config['port']})")
    if config["mode"] == engineConfig.MODE_NETWORK:
        ready = "ready" if network_ready() else "certificate missing: restart Aurora"
        print(f"Network:   https://{_lan_address() or '<this computer>'}:{config['network_port']}  ({ready})")
        print(f"Shared:    {', '.join(str(f) for f in _shared_folders_for_status())}")
        if tls_fingerprint():
            print(f"TLS:       {tls_fingerprint()}")
    if not state:
        print("Pairing:   not paired  (run: aurora-tools pair)")
        return 0
    owner = state.get("owner") or {}
    print(f"Pairing:   \"{state.get('server_name')}\" owned by {owner.get('name')} ({owner.get('email')})")
    print(f"Lab:       {state.get('lab_origin')}")
    if group:
        names = [f"{m['name']} ({m['role']}{', blocked' if m.get('blocked') else ''})" for m in group.get("members", [])]
        print(f"Group:     {', '.join(names) or 'owner only'}")
        print(f"Updated:   {time.strftime('%Y-%m-%d %H:%M', time.localtime(group.get('updated_at', 0)))}")
    return 0


def _shared_folders_for_status() -> list:
    raw = engineConfig._read_raw().get("shared_folders")
    return raw if isinstance(raw, list) and raw else [str(Path.home() / "Documents") + " (default)"]


def set_network_mode(enable: bool, shared_folders=None, network_port=None, pair_owner=None) -> int:
    """aurora-tools network enable|disable: edit engine.json and make the certificate.

    Takes effect when Aurora restarts (the HTTPS listener starts with Daphne). The Linux
    package wraps this with the service, the aurora-tools account and the firewall rule.
    """
    if enable and not engineConfig.NETWORK_MODE_AVAILABLE:
        print("Network sharing is not available in this Aurora version.")
        return 1
    changes = {"mode": engineConfig.MODE_NETWORK if enable else engineConfig.MODE_LOCAL}
    if shared_folders:
        folders = [os.path.realpath(os.path.expanduser(f)) for f in shared_folders]
        missing = [f for f in folders if not os.path.isdir(f)]
        if missing:
            print(f"Not a folder: {', '.join(missing)}")
            return 1
        changes["shared_folders"] = folders
    if network_port:
        changes["network_port"] = int(network_port)
    if enable and pair_owner:
        # Lab account of whoever turned sharing on from Settings: only they may pair it
        # from Aurora (engineSharing). Headless setups pair with the code (aurora-tools pair).
        changes["pair_owner_id"] = str(pair_owner)
    engineConfig.write_config(**changes)
    if enable:
        ensure_tls_certificate()
        print("Network sharing is on. Restart Aurora to start the HTTPS listener.")
        if not pairing_state():
            print("Then pair this computer with your lab account:  aurora-tools pair")
    else:
        print("Network sharing is off. Restart Aurora; it will listen on this computer only.")
    check_in_once()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aurora-tools", description="Aurora network sharing")
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("pair", help="pair this computer with your lab account")
    pair.add_argument("--lab", default=LAB_ORIGIN_PRODUCTION, help=argparse.SUPPRESS)
    pair.add_argument("--name", default=platform.node() or "Aurora server", help="name shown on the website")
    pair.add_argument("--reset", action="store_true", help="forget the current pairing and start over")
    sub.add_parser("status", help="show mode, pairing and group")
    sub.add_parser("check-in", help=argparse.SUPPRESS)
    # Launchers run this before Daphne: makes the certificate when network mode is on.
    sub.add_parser("tls", help=argparse.SUPPRESS)
    network = sub.add_parser("network", help="share this computer on the local network, or stop")
    network.add_argument("action", choices=("enable", "disable"))
    network.add_argument("--folder", action="append", help="folder to share (repeatable; default ~/Documents)")
    network.add_argument("--port", type=int, help="HTTPS port for other computers (default 8021)")
    network.add_argument("--pair-owner", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    if args.command == "pair":
        try:
            return run_pairing_cli(args.lab, args.name, args.reset)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return 1
    if args.command == "check-in":
        return 0 if check_in_once() else 1
    if args.command == "network":
        return set_network_mode(args.action == "enable", args.folder, args.port, args.pair_owner)
    if args.command == "tls":
        if engineConfig.is_network_mode():
            ensure_tls_certificate()
        return 0
    return print_status()


if __name__ == "__main__":
    sys.exit(main())
