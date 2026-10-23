"""Engine configuration shared by Django settings and every launcher.

One small JSON file decides how this Aurora engine is exposed:

    ~/.aurora/engine.json   (AURORA_CONFIG_DIR overrides the folder)
    {
        "mode": "local",      # "local" = this computer only; "network" = LAN sharing (1.9.4+)
        "port": 8020,         # plain HTTP for this computer, always on 127.0.0.1
        "network_port": 8021  # network mode only: HTTPS for other computers on the LAN
    }

Missing file or bad values fall back to local mode on 8020, so a fresh install is never
exposed to the network by accident. This computer always uses http://127.0.0.1:<port>,
in both modes; network mode adds a second, HTTPS listener on all interfaces with the
self-signed certificate in ~/.aurora/tls/ (networkMode.ensure_tls_certificate), so
visitors' traffic is encrypted and the person at the computer sees no certificate warning.

Stdlib only: launchers import this file by path (Windows/macOS) or run
``python -m AuroraClient.engineConfig port|bind`` (Linux shell launchers) before Django
is configured.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path

DEFAULT_PORT = 8020
DEFAULT_NETWORK_PORT = 8021
MODE_LOCAL = "local"
MODE_NETWORK = "network"
LOOPBACK_BIND_HOST = "127.0.0.1"
NETWORK_BIND_HOST = "0.0.0.0"

# Host header values Django accepts in local mode (DNS-rebinding guard). 0.0.0.0 is
# kept because older lab-site "Server Connection" settings may still point at it.
LOCAL_ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]", "0.0.0.0"]

# Network mode (LAN sharing) relies on TLS (networkMode.ensure_tls_certificate), the
# one-person lease and member checks (engineAuth) and the shared-folder limits
# (networkFolders). Set False to make every install ignore "network" in engine.json.
NETWORK_MODE_AVAILABLE = True


# Lab websites whose handoff passes and pairing this engine accepts. Local BHTools dev
# servers are only trusted in local mode: a shared engine trusts production alone.
PRODUCTION_LAB_ORIGINS = frozenset({
    "https://hallgrimssonlab.ca",
    "https://www.hallgrimssonlab.ca",
})
DEV_LAB_ORIGINS = frozenset(
    f"http://{host}:{port}"
    for host in ("127.0.0.1", "localhost")
    for port in range(8000, 8005)
)


def trusted_lab_origins() -> frozenset:
    # AURORA_DEV_LAB=1 lets a developer test network mode against a local BHTools site
    # (the browser on the same machine reaches the engine through its LAN address).
    dev = os.environ.get("AURORA_DEV_LAB", "").strip() == "1"
    if is_network_mode() and not dev:
        return PRODUCTION_LAB_ORIGINS
    # LAN tests against a dev lab site opened from another computer, e.g.
    # AURORA_DEV_LAB_ORIGINS=http://10.0.0.5:8000 (only together with AURORA_DEV_LAB=1).
    extra = frozenset(
        origin.strip().rstrip("/")
        for origin in os.environ.get("AURORA_DEV_LAB_ORIGINS", "").split(",")
        if dev and origin.strip().startswith(("http://", "https://"))
    )
    return PRODUCTION_LAB_ORIGINS | DEV_LAB_ORIGINS | extra


def config_dir() -> Path:
    env_dir = os.environ.get("AURORA_CONFIG_DIR", "").strip()
    return Path(env_dir) if env_dir else Path.home() / ".aurora"


def config_path() -> Path:
    return config_dir() / "engine.json"


_raw_cache = {"key": None, "data": {}}


def _read_raw() -> dict:
    # Called on every engine request (lease, network checks): re-read only when the file
    # changed, so editing engine.json still takes effect without a restart.
    path = config_path()
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return {}
    if _raw_cache["key"] == key:
        return dict(_raw_cache["data"])
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    _raw_cache.update(key=key, data=data)
    return dict(data)


def write_config(**changes) -> dict:
    """Update engine.json (mode, port, network_port, shared_folders); returns the raw file."""
    data = _read_raw()
    data.update(changes)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return data


def _valid_port(value) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return DEFAULT_PORT
    # Unprivileged range only: no root needed, no clash with well-known services.
    return port if 1024 <= port <= 65535 else DEFAULT_PORT


def load_config() -> dict:
    """Return the effective config with defaults applied (never raises)."""
    raw = _read_raw()
    mode = MODE_NETWORK if (NETWORK_MODE_AVAILABLE and raw.get("mode") == MODE_NETWORK) else MODE_LOCAL
    env_port = os.environ.get("AURORA_PORT", "").strip()
    port = _valid_port(env_port or raw.get("port", DEFAULT_PORT))
    network_port = _valid_port(raw.get("network_port", DEFAULT_NETWORK_PORT))
    if network_port == port:
        network_port = DEFAULT_NETWORK_PORT if port != DEFAULT_NETWORK_PORT else DEFAULT_PORT
    return {"mode": mode, "port": port, "network_port": network_port}


def is_network_mode() -> bool:
    return load_config()["mode"] == MODE_NETWORK


def bind_host() -> str:
    """Address of the plain-HTTP listener: always loopback. Launchers that only know
    `-b/-p` therefore never expose the engine; the network (HTTPS) listener comes only
    from daphne_endpoints()."""
    return LOOPBACK_BIND_HOST


def engine_port() -> int:
    return load_config()["port"]


def tls_dir() -> Path:
    return config_dir() / "tls"


def tls_paths() -> tuple:
    """(certificate, private key) for the network listener."""
    return tls_dir() / "cert.pem", tls_dir() / "key.pem"


def _endpoint_escape(path: Path) -> str:
    # Twisted endpoint strings separate fields with ":" (also in Windows drive letters).
    return str(path).replace("\\", "\\\\").replace(":", "\\:")


def daphne_endpoints() -> list:
    """Twisted endpoint strings for `daphne -e ...`.

    Always plain HTTP on loopback for this computer. In network mode, plus HTTPS on all
    interfaces when the certificate exists (launchers run `networkMode tls` first); without
    a certificate we stay loopback-only rather than serve the LAN unencrypted.
    """
    config = load_config()
    endpoints = [f"tcp:port={config['port']}:interface={LOOPBACK_BIND_HOST}"]
    if config["mode"] == MODE_NETWORK:
        cert, key = tls_paths()
        if cert.exists() and key.exists():
            endpoints.append(
                f"ssl:port={config['network_port']}:interface={NETWORK_BIND_HOST}"
                f":privateKey={_endpoint_escape(key)}:certKey={_endpoint_escape(cert)}"
            )
    return endpoints


def engine_id() -> str:
    """Stable random id for this install; the lab signs handoff passes for this audience."""
    path = config_dir() / "engine_id"
    try:
        stored = path.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    except OSError:
        pass
    generated = secrets.token_hex(16)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated + "\n", encoding="utf-8")
        os.chmod(path, 0o600)
    except OSError:
        pass
    return generated


if __name__ == "__main__":
    # Shell launchers: PORT=$(python -m AuroraClient.engineConfig port)
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "port":
        print(engine_port())
    elif what == "bind":
        print(bind_host())
    elif what == "mode":
        print(load_config()["mode"])
    elif what == "endpoints":
        # One per line: mapfile -t EPS < <(python -m AuroraClient.engineConfig endpoints)
        print("\n".join(daphne_endpoints()))
    else:
        print("usage: python -m AuroraClient.engineConfig port|bind|mode|endpoints", file=sys.stderr)
        sys.exit(2)
