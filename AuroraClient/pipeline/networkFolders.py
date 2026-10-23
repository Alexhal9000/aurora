"""Folders a shared Aurora shows to people on other computers (Aurora 1.9.4 network mode).

Someone who opens a shared server from another computer may only read and write inside
the shared folders listed in ~/.aurora/engine.json:

    "shared_folders": ["/home/ana/Documents"]

Default (key missing or empty): the Documents folder of the account running Aurora. The
owner or a manager changes the list in Aurora Settings (step 4). The person at this
computer itself (loopback) is not limited: they already have the OS account.

Enforced in one place, NetworkPathPermission (DRF default permission, settings.py),
before any view runs: every request argument that names a file or folder is resolved
(~ expanded, symlinks followed) and must lie inside a shared folder. Views that search the
whole home folder are refused outright. The dedicated `aurora` OS account the Linux
service runs as is the second layer for anything this list misses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import BasePermission

from AuroraClient import engineConfig

# Request arguments that carry a path (absolute, or ~-relative) on their own.
PATH_KEYS = ("directory", "path", "target_folder", "mask_folder")
# Arguments that name something inside a folder argument (directory, path, target_folder):
# checked joined to it, which also catches "../" tricks.
CHILD_KEYS = ("filename", "filenames", "source_filename", "folderName", "scan_name", "subject_name")
# Views that look outside any folder they are given (FileCopyView searches the whole home
# folder for relative paths); not available from other computers.
REMOTE_BLOCKED_VIEWS = frozenset({"FileCopyView"})
# Views that fall back to the working directory when a folder argument is missing.
REQUIRED_PATH_KEYS = {"FileUploadView": "target_folder"}
PARENT_KEYS = ("directory", "path", "target_folder")

OUTSIDE_MESSAGE = (
    "That folder is not shared on this computer. Ask its owner to add it to the shared "
    "folders in Aurora Settings."
)


def shared_folders() -> List[Path]:
    raw = engineConfig._read_raw().get("shared_folders")
    folders = [str(f) for f in raw if isinstance(f, str) and f.strip()] if isinstance(raw, list) else []
    if not folders:
        folders = [str(Path.home() / "Documents")]
    resolved = []
    for folder in folders:
        try:
            resolved.append(Path(os.path.realpath(os.path.expanduser(folder))))
        except (OSError, ValueError):
            continue
    return resolved


def is_inside_shared(path: str, roots: Iterable[Path] = None) -> bool:
    if not isinstance(path, str) or not path.strip() or "\x00" in path:
        return False
    try:
        real = Path(os.path.realpath(os.path.expanduser(path)))
    except (OSError, ValueError):
        return False
    for root in roots if roots is not None else shared_folders():
        if real == root or root in real.parents:
            return True
    return False


SERVICE_HOME = "/var/lib/aurora-tools"


def running_as_service() -> bool:
    """Running as the Linux background service (aurora-tools account), whose own home
    folder holds nothing of the user's: file shortcuts then point at the shared folders."""
    if os.environ.get("AURORA_SERVICE") == "1":
        return True
    try:
        return str(Path.home()) == SERVICE_HOME
    except (RuntimeError, OSError):
        return False


def is_remote_request(request) -> bool:
    """Signed in from another computer (network session)? See engineAuth.create_session."""
    record = getattr(request, "aurora_session", None)
    return bool(record and record.get("remote")) and engineConfig.is_network_mode()


def _values(container, key) -> list:
    if container is None:
        return []
    try:
        if hasattr(container, "getlist"):
            values = container.getlist(key)
        else:
            value = container.get(key) if hasattr(container, "get") else None
            values = value if isinstance(value, list) else ([] if value is None else [value])
    except Exception:  # noqa: BLE001 - unparsable body: the view rejects it anyway
        return []
    return [v for v in values if isinstance(v, str) and v != ""]


class NetworkPathPermission(BasePermission):
    """Keep network sessions inside the shared folders (no-op for everyone else)."""

    message = OUTSIDE_MESSAGE

    def has_permission(self, request, view) -> bool:
        if not is_remote_request(request):
            return True
        if type(view).__name__ in REMOTE_BLOCKED_VIEWS:
            raise PermissionDenied("This tool is not available when using a computer over the network.")
        roots = shared_folders()
        sources = (request.query_params, getattr(request, "data", None))
        required = REQUIRED_PATH_KEYS.get(type(view).__name__)
        if required and not any(_values(source, required) for source in sources):
            raise PermissionDenied(OUTSIDE_MESSAGE)
        directories = []
        for source in sources:
            for key in PATH_KEYS:
                for value in _values(source, key):
                    # The file browser's listing clamps itself to the shared folders
                    # (listing_start) instead of failing, so going "up" stays usable.
                    if key == "path" and type(view).__name__ == "ListDirectoriesView":
                        continue
                    if not is_inside_shared(value, roots):
                        raise PermissionDenied(OUTSIDE_MESSAGE)
                    if key in PARENT_KEYS:
                        directories.append(value)
        for directory in directories:
            for source in sources:
                for key in CHILD_KEYS:
                    for value in _values(source, key):
                        if not is_inside_shared(os.path.join(os.path.expanduser(directory), value), roots):
                            raise PermissionDenied(OUTSIDE_MESSAGE)
        return True


def listing_start(path: str) -> str:
    """ListDirectoriesView: where a network session's file browser may look. "~", "root"
    and anything outside the shared folders open the first shared folder instead."""
    roots = shared_folders()
    if path in ("~", "root") or not is_inside_shared(path, roots):
        return str(roots[0]) if roots else str(Path.home() / "Documents")
    return path
