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

import json
import os
import threading
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


# ---------------------------------------------------------------------------
# Per-member folders: the owner and managers can limit a member to some folders inside
# the shared ones (lab with several people's data side by side). Kept next to engine.json
# in member_folders.json, which only this engine writes:
#     {"restrict_new": false, "members": {"<lab user id>": ["/home/ana/Documents/LabA"]}}
# A member with an entry sees only those folders (each must lie inside a shared folder;
# others are ignored). A member without one sees all shared folders, or none while
# restrict_new is on. Owner and managers always see all shared folders.
# ---------------------------------------------------------------------------

MEMBER_FOLDERS_FILE = "member_folders.json"
MAX_FOLDERS_PER_MEMBER = 20
MAX_MEMBER_ENTRIES = 500
NO_FOLDERS_MESSAGE = (
    "No folders on this computer are shared with you yet. Ask its owner or a manager to "
    "give you a folder in Aurora Settings."
)
_member_lock = threading.Lock()


def _member_folders_path() -> Path:
    return engineConfig.config_dir() / MEMBER_FOLDERS_FILE


def member_folder_settings() -> dict:
    try:
        data = json.loads(_member_folders_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    raw = data.get("members") if isinstance(data.get("members"), dict) else {}
    members = {}
    for user_id, folders in raw.items():
        if isinstance(folders, list):
            members[str(user_id)] = [f for f in folders if isinstance(f, str) and f.strip()]
    return {"restrict_new": bool(data.get("restrict_new")), "members": members}


def save_member_folders(restrict_new: bool, members: dict) -> dict:
    """Validate and store the per-member folders. Raises ValueError with a message to show.
    `members` maps lab user id -> list of folders, or None to drop the entry (all shared)."""
    if not isinstance(members, dict):
        raise ValueError("members must be an object.")
    roots = shared_folders()
    with _member_lock:
        current = member_folder_settings()["members"]
        for user_id, folders in members.items():
            user_id = str(user_id)
            if not user_id.isdigit() or len(user_id) > 12:
                raise ValueError("Unknown member.")
            if folders is None:
                current.pop(user_id, None)
                continue
            if not isinstance(folders, list) or len(folders) > MAX_FOLDERS_PER_MEMBER:
                raise ValueError(f"Give a member at most {MAX_FOLDERS_PER_MEMBER} folders.")
            clean = []
            for folder in folders:
                if not isinstance(folder, str) or not folder.strip() or "\x00" in folder or len(folder) > 4096:
                    raise ValueError("Each folder must be a path.")
                real = os.path.realpath(os.path.expanduser(folder.strip()))
                if not is_inside_shared(real, roots):
                    raise ValueError(f"{folder} is not inside a shared folder.")
                if not os.path.isdir(real):
                    raise ValueError(f"{folder} is not a folder on this computer.")
                if real not in clean:
                    clean.append(real)
            current[user_id] = clean
        if len(current) > MAX_MEMBER_ENTRIES:
            raise ValueError("Too many members.")
        target = _member_folders_path()
        tmp = target.with_suffix(".tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"restrict_new": bool(restrict_new), "members": current}, indent=1) + "\n",
                       encoding="utf-8")
        os.replace(tmp, target)
    return member_folder_settings()


def _group_role(user_id: str) -> str:
    from pipeline import networkMode  # networkMode imports more of the engine
    for member in networkMode.current_group().get("members", []):
        if isinstance(member, dict) and str(member.get("user_id")) == str(user_id):
            return str(member.get("role") or "member")
    return "member"


def member_folders(user_id: str, roots: List[Path] = None) -> List[Path]:
    """The folders a member (not owner/manager) may use; [] = none."""
    roots = shared_folders() if roots is None else roots
    settings = member_folder_settings()
    own = settings["members"].get(str(user_id))
    if own is None:
        return [] if settings["restrict_new"] else roots
    allowed = []
    for folder in own:
        try:
            real = Path(os.path.realpath(os.path.expanduser(folder)))
        except (OSError, ValueError):
            continue
        if is_inside_shared(str(real), roots) and real not in allowed:
            allowed.append(real)
    return allowed


def folders_for(request) -> List[Path]:
    """Folders this request may use: the shared folders, narrowed for a network session of
    a member (see member_folders). Local sessions get the shared folders as before."""
    roots = shared_folders()
    if not is_remote_request(request):
        return roots
    record = getattr(request, "aurora_session", None) or {}
    if _group_role(record.get("user_id")) in ("owner", "manager"):
        return roots
    return member_folders(record.get("user_id"), roots)


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
        roots = folders_for(request)
        if not roots:
            raise PermissionDenied(NO_FOLDERS_MESSAGE)
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


def listing_start(path: str, roots: List[Path] = None) -> str:
    """ListDirectoriesView: where a network session's file browser may look. "~", "root"
    and anything outside the shared folders open the first shared folder instead."""
    roots = shared_folders() if roots is None else roots
    if not roots:
        raise PermissionDenied(NO_FOLDERS_MESSAGE)
    if path in ("~", "root") or not is_inside_shared(path, roots):
        return str(roots[0]) if roots else str(Path.home() / "Documents")
    return path
