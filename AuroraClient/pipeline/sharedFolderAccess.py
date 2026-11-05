"""
What the shared service can really open and change in its shared folders (Aurora 1.9.8+).

Since 2026-11-04 (Alejandro's decision, "fast on + pending folders") turning sharing on
gives the service account access to each shared folder itself only, plus default entries
so that new files there stay usable by both the owner and the service. Nothing already
inside is changed: an item someone kept private stays private, and Turn on takes seconds
instead of walking every file. This module tells the owner what that leaves out:

* cannot_open: files the service cannot read, and folders it cannot list (counted once;
  what is inside them is not looked at).
* read_only: files and folders it can open but not change. Members can view them, but
  saving into them (e.g. an existing project's extracted/ folder) fails.

The check runs inside the shared service, as its own account, so os.access gives the true
answer (ACLs included). It walks each folder in a background thread with a time and size
budget; a very large folder reports an incomplete count instead of making Settings wait.
The owner can then choose, per folder, "Give members access to everything in this
folder", which asks for admin rights on the computer (engineSharing action
request_full_access -> sudo aurora-tools confirm, full_access).

Items of other accounts are counted apart and never opened up from here; private paths
are only shown on request, folder by folder: browse() lists one level of a shared folder
with each item's owner and permissions, marking what members cannot open or change and
which folders hold such items further down (Settings' "Browse what members cannot open",
Alejandro 2026-11-04: a folder view instead of a long list of paths).

Symbolic links are never followed (same rule as the folder grants: links are ignored).
"""

from __future__ import annotations

import logging
import os
import stat
import threading
import time

# grp and pwd exist only on Linux/macOS. This module is imported at startup on every
# platform (engineAuth -> engineSharing), so a top-level import crashed Aurora 1.9.9 on
# Windows (ModuleNotFoundError: grp). Without them, owner and group show as numbers.
try:
    import grp
    import pwd
except ImportError:   # Windows
    grp = pwd = None

logger = logging.getLogger(__name__)

# Per folder: stop after this long or this many items and report the count as partial.
CHECK_SECONDS = 20.0
CHECK_MAX_ITEMS = 400_000
# Results older than this are refreshed the next time Settings asks.
RESULT_MAX_AGE_SECONDS = 10 * 60
# Restricted paths listed per folder (the first item on each path that cannot be opened).
RESTRICTED_LIST_MAX = 200

_lock = threading.Lock()
_results: dict = {}       # folder -> result dict
_running: set = set()     # folders being checked now
_inside: dict = {}        # folder -> {relative folder: [cannot open, read only] further down}
# Items listed per folder level in browse(); the rest is counted.
BROWSE_MAX_ENTRIES = 1000


def _access_changed_at() -> float:
    """When the root helper last finished changing permissions (aurora-tools-network
    mark_access_changed), or 0."""
    try:
        from AuroraClient import engineConfig
        return os.stat(engineConfig.config_dir() / "access_changed").st_mtime
    except (OSError, ImportError):
        return 0.0


_SERVICE_UID = os.getuid() if hasattr(os, "getuid") else -1


def _passage(path: str) -> bool:
    """A folder opened up only "on the way" to a shared item (aurora-tools-network
    grant_path_full: the service account's own ACL entry is exactly r-x). Members may enter
    it and see the names there by the owner's choice, so it is not a problem to report."""
    try:
        raw = os.getxattr(path, "system.posix_acl_access", follow_symlinks=False)
    except (OSError, AttributeError):
        return False
    # Linux POSIX ACL xattr: 4-byte version, then 8-byte entries (tag u16, perm u16, id u32).
    for offset in range(4, len(raw) - 7, 8):
        tag = int.from_bytes(raw[offset:offset + 2], "little")
        perm = int.from_bytes(raw[offset + 2:offset + 4], "little")
        ident = int.from_bytes(raw[offset + 4:offset + 8], "little")
        if tag == 0x02 and ident == _SERVICE_UID:   # ACL_USER for this account
            return perm == 0b101
    return False


def _relative(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    return "." if rel == "." else rel


_names: dict = {}


def _account(uid: int) -> str:
    key = ("u", uid)
    if key not in _names:
        try:
            _names[key] = pwd.getpwuid(uid).pw_name
        except (KeyError, OverflowError, AttributeError):   # AttributeError: no pwd (Windows)
            _names[key] = str(uid)
    return _names[key]


def _group(gid: int) -> str:
    key = ("g", gid)
    if key not in _names:
        try:
            _names[key] = grp.getgrgid(gid).gr_name
        except (KeyError, OverflowError, AttributeError):   # AttributeError: no grp (Windows)
            _names[key] = str(gid)
    return _names[key]


def check_folder(folder: str, seconds: float = CHECK_SECONDS, max_items: int = CHECK_MAX_ITEMS) -> dict:
    """Walk one shared folder as this process's account.

    Owners matter (Alejandro, 2026-11-04): an item that belongs to someone other than the
    shared folder's own owner (another lab member, or set up by a system administrator
    before the drive was attached) is that person's to open up, so it is counted apart
    and "give access to everything" leaves it alone (aurora-tools-network
    grant_folder_full only changes items of the folder's owner).

    Private items are not listed by default: "restricted" holds only the boundaries, the
    first item on each path the service cannot open (what is below it is never looked
    at), with its kind, owner, group and permissions, for Settings' "Show list of
    restricted paths". Read-only items are only counted."""
    started = time.monotonic()
    result = {
        "folder": folder,
        "complete": True,
        "items_checked": 0,
        "folder_owner": "",
        "cannot_open": 0, "cannot_open_own": 0, "cannot_open_other": 0,
        "read_only": 0, "read_only_own": 0, "read_only_other": 0,
        "other_owners": {},          # account -> items of theirs members cannot open or change
        "restricted": [],            # boundaries members cannot open (capped)
        "restricted_more": 0,
        "checked_at": time.time(),
        "error": "",
        # Not sent to Settings (status() drops "_" keys): for each folder, how many items
        # further down members cannot open / change. browse() marks folders with it.
        "_inside": {},
    }
    try:
        info = os.lstat(folder)
    except OSError:
        result["error"] = "not found, or the shared Aurora cannot reach it"
        return result
    owner_uid = info.st_uid
    result["folder_owner"] = _account(owner_uid)

    def note(kind: str, path: str, st, is_dir: bool) -> None:
        own = st is None or st.st_uid == owner_uid
        result[kind] += 1
        slot = 0 if kind == "cannot_open" else 1
        rel = _relative(folder, path)
        while rel != ".":
            rel = os.path.dirname(rel) or "."
            counts = result["_inside"].setdefault(rel, [0, 0])
            counts[slot] += 1
        result[f"{kind}_{'own' if own else 'other'}"] += 1
        if not own:
            name = _account(st.st_uid)
            result["other_owners"][name] = result["other_owners"].get(name, 0) + 1
        if kind == "cannot_open":
            if len(result["restricted"]) < RESTRICTED_LIST_MAX:
                result["restricted"].append({
                    "path": _relative(folder, path),
                    "kind": "folder" if is_dir else "file",
                    "owner": _account(st.st_uid) if st is not None else "",
                    "group": _group(st.st_gid) if st is not None else "",
                    "mode": stat.filemode(st.st_mode) if st is not None else "",
                })
            else:
                result["restricted_more"] += 1

    if not os.path.isdir(folder):
        result["error"] = "not a folder"
        return result
    if not os.access(folder, os.R_OK | os.X_OK):
        note("cannot_open", folder, info, True)
        return result
    if not os.access(folder, os.W_OK | os.X_OK):
        note("read_only", folder, info, True)

    stack = [folder]
    while stack:
        if result["items_checked"] >= max_items or time.monotonic() - started > seconds:
            result["complete"] = False
            break
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    result["items_checked"] += 1
                    try:
                        if entry.is_symlink():
                            continue
                        st = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(st.st_mode):
                            if not os.access(entry.path, os.R_OK | os.X_OK):
                                note("cannot_open", entry.path, st, True)   # not listed: contents unknown
                                continue
                            if not os.access(entry.path, os.W_OK | os.X_OK) and not _passage(entry.path):
                                note("read_only", entry.path, st, True)
                            stack.append(entry.path)
                        elif stat.S_ISREG(st.st_mode):
                            if not os.access(entry.path, os.R_OK):
                                note("cannot_open", entry.path, st, False)
                            elif not os.access(entry.path, os.W_OK):
                                note("read_only", entry.path, st, False)
                    except OSError:
                        note("cannot_open", entry.path, None, False)
        except PermissionError:
            try:
                note("cannot_open", current, os.lstat(current), True)
            except OSError:
                note("cannot_open", current, None, True)
        except OSError as exc:
            # Vanished while walking, or an unreadable mount: say it, keep going.
            logger.debug("Shared folder check skipped %s: %s", current, exc)
    result["seconds"] = round(time.monotonic() - started, 1)
    # What "give access to everything" can change: the folder owner's own items only.
    result["fixable"] = result["cannot_open_own"] + result["read_only_own"]
    return result


def _run(folder: str) -> None:
    try:
        result = check_folder(folder)
    except Exception as exc:   # never let a check take the engine down
        logger.warning("Shared folder check of %s failed: %s", folder, exc)
        result = {"folder": folder, "complete": False, "items_checked": 0, "cannot_open": 0, "read_only": 0,
                  "fixable": 0, "restricted": [], "other_owners": {}, "checked_at": time.time(),
                  "error": str(exc)[:200]}
    inside = result.pop("_inside", None) or {}
    with _lock:
        _results[folder] = result
        _inside[folder] = inside
        _running.discard(folder)
    if result["cannot_open"] or result["read_only"]:
        logger.info("Shared folder %s: %s item(s) members cannot open, %s they cannot change%s",
                    folder, result["cannot_open"], result["read_only"],
                    "" if result["complete"] else " (partial count)")


def start_checks(folders: list, force: bool = False) -> None:
    """Check folders in the background: those never checked, older than
    RESULT_MAX_AGE_SECONDS, or all of them with force."""
    now = time.time()
    changed_at = _access_changed_at()
    with _lock:
        todo = [f for f in folders if f not in _running and (
            force or f not in _results or now - _results[f]["checked_at"] > RESULT_MAX_AGE_SECONDS
            or _results[f]["checked_at"] < changed_at)]
        _running.update(todo)
    for folder in todo:
        threading.Thread(target=_run, args=(folder,), name="aurora-share-access-check", daemon=True).start()


def status(folders: list) -> list:
    """One entry per shared folder for Settings: state "checking" or the last result.
    Starts the checks that are due."""
    start_checks(folders)
    with _lock:
        out = []
        for folder in folders:
            entry = {k: v for k, v in (_results.get(folder) or {"folder": folder}).items() if not k.startswith("_")}
            entry["state"] = "checking" if folder in _running else ("done" if folder in _results else "checking")
            out.append(entry)
        return out


def resolve_inside(folder: str, rel: str = ""):
    """(absolute path, normalised relative path or ".") of an item inside a shared folder.
    ValueError when rel leaves the folder, passes through a link or is not there."""
    rel = (rel or "").strip().strip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts) or os.path.isabs(rel) or any("\n" in p for p in parts):
        raise ValueError("That path is not inside the shared folder.")
    here = folder
    for part in parts:
        here = os.path.join(here, part)
        try:
            if stat.S_ISLNK(os.lstat(here).st_mode):
                raise ValueError("Links are not followed in shared folders.")
        except OSError as exc:
            raise ValueError("That item is no longer there.") from exc
    return here, "/".join(parts) or "."


def browse(folder: str, rel: str = "") -> dict:
    """One level of a shared folder, as the shared service sees it (Settings' folder view).

    Each item: name, kind (folder, file, link, other), owner, group, access (rwx as members
    get it through the shared service's account) and state: "blocked" (members cannot open it), "read_only" (they can open but not
    change it), "passage" (a folder opened on the way to a shared item: enter and see names
    only, by choice), "link" (never followed, so not shared), or "ok". Folders also carry
    inside_blocked / inside_read_only: items further down, from the last check (the item's
    own state is read now). Problem items come first. rel is relative to the shared folder;
    leaving it, or passing through a link, is refused (ValueError)."""
    here, rel = resolve_inside(folder, rel)
    with _lock:
        last = dict(_results.get(folder) or {})
        inside = _inside.get(folder) or {}
    try:
        owner_uid = os.lstat(folder).st_uid
    except OSError as exc:
        raise ValueError("The shared folder is not there, or the shared Aurora cannot reach it.") from exc

    def describe(path: str, name: str, st) -> dict:
        # "access": what members get, i.e. the shared service's own account (owner, group,
        # others and ACL entries all counted by os.access), not the owner's rwx bits
        # (Alejandro, 2026-11-04). "mode" (ls style) stays for reference.
        access = "" if stat.S_ISLNK(st.st_mode) else "".join(
            letter if os.access(path, flag) else "-"
            for letter, flag in (("r", os.R_OK), ("w", os.W_OK), ("x", os.X_OK)))
        item = {"name": name, "owner": _account(st.st_uid), "group": _group(st.st_gid),
                "mode": stat.filemode(st.st_mode), "access": access, "other_owner": st.st_uid != owner_uid}
        if stat.S_ISLNK(st.st_mode):
            item.update(kind="link", state="link")
        elif stat.S_ISDIR(st.st_mode):
            item["kind"] = "folder"
            if not os.access(path, os.R_OK | os.X_OK):
                item["state"] = "blocked"
            else:
                if os.access(path, os.W_OK | os.X_OK):
                    item["state"] = "ok"
                else:
                    item["state"] = "passage" if _passage(path) else "read_only"
                child = os.path.relpath(path, folder)
                counts = inside.get(child)
                if counts:
                    item["inside_blocked"], item["inside_read_only"] = counts
        elif stat.S_ISREG(st.st_mode):
            item["kind"] = "file"
            if not os.access(path, os.R_OK):
                item["state"] = "blocked"
            else:
                item["state"] = "ok" if os.access(path, os.W_OK) else "read_only"
        else:
            item.update(kind="other", state="ok")
        return item

    try:
        current = describe(here, os.path.basename(here) or here, os.lstat(here))
    except OSError as exc:
        raise ValueError("That folder is no longer there.") from exc
    items = []
    error = ""
    if current["state"] == "blocked":
        error = "Members cannot open this folder, so what is inside is not shown."
    else:
        try:
            with os.scandir(here) as entries:
                for entry in entries:
                    try:
                        items.append(describe(entry.path, entry.name, entry.stat(follow_symlinks=False)))
                    except OSError:
                        items.append({"name": entry.name, "kind": "file", "state": "blocked",
                                      "owner": "", "group": "", "mode": "", "access": "---", "other_owner": False})
        except OSError:
            error = "The shared Aurora could not list this folder."

    rank = {"blocked": 0, "read_only": 1}

    def order(item):
        state_rank = rank.get(item["state"], 3 if item["state"] == "link" else 2)
        if state_rank == 2 and (item.get("inside_blocked") or item.get("inside_read_only")):
            state_rank = 1.5
        return (state_rank, item["kind"] != "folder", item["name"].lower())

    items.sort(key=order)
    return {
        "folder": folder,
        "path": "" if rel == "." else rel,
        "here": current,
        "folder_owner": _account(owner_uid),
        "items": items[:BROWSE_MAX_ENTRIES],
        "more": max(0, len(items) - BROWSE_MAX_ENTRIES),
        "error": error,
        # Folder marks come from the last check; a partial check may have missed some.
        "checked_at": last.get("checked_at"),
        "complete": bool(last.get("complete", False)),
        "checking": folder in _running,
    }


def api_exception_handler(exc, context):
    """REST framework exception handler (settings.REST_FRAMEWORK EXCEPTION_HANDLER).

    On the shared service an item the owner has not opened up raises PermissionError deep
    inside a tool; uncaught, that was a bare server error. Say what it is and who can
    change it instead. Everything else goes to REST framework's own handler unchanged
    (views that catch errors themselves still report them their own way)."""
    from rest_framework import status as http_status
    from rest_framework.response import Response
    from rest_framework.views import exception_handler

    response = exception_handler(exc, context)
    if response is not None or not isinstance(exc, PermissionError):
        return response
    from pipeline import networkFolders   # late: networkFolders is imported by the settings chain
    if not networkFolders.running_as_service():
        return None
    item = getattr(exc, "filename", None) or "this item"
    logger.warning("Shared Aurora was refused access to %s", item)
    return Response({
        "error": (f"The shared Aurora is not allowed to open or change {item}. The computer's owner can give "
                  "members access in Aurora Settings > Share this computer."),
        "code": "shared_item_not_opened",
    }, status=http_status.HTTP_403_FORBIDDEN)
