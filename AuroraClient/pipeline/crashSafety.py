"""Crash-safe protocol (CSP) for everything Aurora writes to a project (Aurora 1.9.4+).

Goal
----
The engine can be killed at any instant (crash, kill -9, power button, the user
restarting it from the menu) and started again without stale, half-written or
malformed data in the project. After a restart every subject is either exactly as it
was before the interrupted method started, or exactly as the method left it on
success. Never something in between.

The four rules (every method that writes project data follows them)
-------------------------------------------------------------------
1. ATOMIC FILES. Never write straight to a final path (nib.save(img, path),
   open(path, 'w'), np.savez(path), mesh.export(path), shutil.copy(src, path)). Use
   the atomic_* helpers below: they write a hidden temp file next to the target
   (".<name>.csp-tmp-<id><ext>", which glob('*') never matches), fsync it, then
   os.replace it over the target, so readers see the old file or the new one, never a
   truncated one.

2. NEVER RENAME A SOURCE TO READ IT. Masks are stored as ".nii.mask.gz" (and
   ".nii.removal_mask.gz"), which nibabel will not open by name. The old pattern
   renamed the source to "_mask.nii.gz", loaded it and renamed it back; a crash in
   between silently lost the previous edit's mask. Use load_nifti_readonly(path).

3. ONE TRANSACTION PER SUBJECT. A method's work on one subject (new edit volume, its
   lossy copy, masks, landmarks, distances, sidecars and the subject JSON) runs inside

       with SubjectTransaction(scan_dir, 'crop'):
           ...

   On entry the subject folder is journalled in "<scan_dir>/.aurora_txn/": a listing
   of its files, a real copy of every JSON file, and a hard link to every other file
   (nothing on filesystems without hard links, e.g. exFAT). Because rule 1 always
   replaces files rather than rewriting them, a hard link keeps the original content.
   A file that must be changed in place, or any existing non-JSON file on exFAT, needs
   txn.preserve(path) before it is touched. Deleting the journal
   file is the commit. If the method raises, or the engine dies before the commit, the
   rollback deletes every file that was not there before and puts back every file that
   changed, so the subject is byte-for-byte as before. Deletes and renames that are part
   of the operation go through txn.remove(path) / txn.rename(src, dst) so they can be
   undone too.

4. LINKED SUBJECTS COMMIT TOGETHER. When one operation changes several subjects that
   must stay in step (linked scans replaying an edit on their siblings), wrap them in
   LinkedTransaction(directory, operation) and add each subject with
   group.subject(scan_dir). A single marker file (<project>/.aurora_txn_groups/) decides the outcome for all of them
   after a crash: present = roll every member forward, absent = roll every member back.

Recovery
--------
recover_project(directory) runs when a project is opened (GetProjectSettingsView) and
SubjectTransaction recovers a subject itself before starting on it. A journal belongs
to a live run only when it carries this process's ENGINE_BOOT_ID and is registered as
active here; anything else is left over from a dead engine and is rolled back (or
forward, for a committed linked group). Leftover hidden temp files and orphaned
".nii.removal_mask.gz" files are swept as well.

Adding a new derivative file to a method? Write it inside that method's transaction,
through an atomic_* helper. If it is a regenerable cache outside any transaction, an
atomic_* helper is enough. Each writing method carries a "CRASH-SAFE PROTOCOL" comment
pointing here.

Stdlib + numpy/nibabel only (imported lazily) so engine modules can import it early.
"""

from __future__ import annotations

import contextlib
import functools
import gzip
import json
import os
import shutil
import threading
import time
import uuid

# One id per engine process. A journal stamped with another id was left by a dead engine.
ENGINE_BOOT_ID = uuid.uuid4().hex

TEMP_MARKER = '.csp-tmp-'
TXN_DIRNAME = '.aurora_txn'
TXN_JOURNAL = 'journal.json'
GROUP_DIRNAME = '.aurora_txn_groups'
# Project-level folder that whole subject folders are moved into before they are deleted.
TRASH_DIRNAME = '.aurora_trash'

# Older leftovers are swept on project open; younger ones may belong to a write in flight
# on another subject (caches are written outside transactions).
STALE_TEMP_SECONDS = 10 * 60

# Compound extensions that nibabel and friends do not recognise: written as `.nii.gz`
# then renamed to the real name.
_NIFTI_ALIAS_SUFFIXES = ('.nii.mask.gz', '.nii.removal_mask.gz', '.nii.backup.gz')
_KNOWN_EXTENSIONS = (
    '.nii.gz', '.nii', '.npz', '.npy', '.json', '.jsonl', '.ply', '.obj', '.stl', '.glb',
    '.png', '.jpg', '.jpeg', '.mat', '.txt', '.csv', '.md', '.html', '.zip', '.pt', '.pkl',
)


# ---------------------------------------------------------------------------
# Rule 1: atomic files
# ---------------------------------------------------------------------------

def _temp_sibling(path):
    """Hidden temp path in the target's folder that keeps a writer-recognisable extension."""
    directory, base = os.path.split(os.path.abspath(path))
    ext = ''
    lower = base.lower()
    if lower.endswith(_NIFTI_ALIAS_SUFFIXES):
        ext = '.nii.gz'
    else:
        for candidate in _KNOWN_EXTENSIONS:
            if lower.endswith(candidate):
                ext = candidate
                break
        else:
            ext = os.path.splitext(base)[1]  # unknown type: keep its last extension
    return os.path.join(directory, f".{base}{TEMP_MARKER}{uuid.uuid4().hex[:8]}{ext}")


def is_csp_temp_name(name):
    return name.startswith('.') and TEMP_MARKER in name


def _fsync_path(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def fsync_dir(directory):
    """Persist renames in `directory` (no-op where directories cannot be opened, e.g. Windows)."""
    _fsync_path(directory)


def _silent_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _replace_with_retry(src, dst, retries=6, delay=0.05):
    """os.replace, retried: on Windows a reader holding `dst` open makes it fail briefly."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


@contextlib.contextmanager
def atomic_output(path):
    """
    Yield a temp path; when the block finishes it becomes `path` in one os.replace.

        with atomic_output(out_path) as tmp:
            writer(tmp)

    If the block raises (or the process dies) the target is untouched and the temp file
    is removed (or swept on the next project open).
    """
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = _temp_sibling(path)
    try:
        yield tmp
        if not os.path.exists(tmp):
            raise FileNotFoundError(f"atomic_output: writer did not create {tmp}")
        _fsync_path(tmp)
        _replace_with_retry(tmp, path)
        fsync_dir(directory)
    except BaseException:
        _silent_remove(tmp)
        raise


def atomic_nib_save(img, path):
    """nibabel.save, atomically (also for `.nii.mask.gz` style names)."""
    import nibabel as nib
    with atomic_output(path) as tmp:
        nib.save(img, tmp)
    return path


def atomic_write_json(path, payload, indent=4, **dump_kwargs):
    """json.dump to `path`, atomically."""
    with atomic_output(path) as tmp:
        with open(tmp, 'w', encoding=dump_kwargs.pop('encoding', None)) as handle:
            json.dump(payload, handle, indent=indent, **dump_kwargs)
            handle.flush()
    return path


def atomic_write_text(path, text, encoding='utf-8'):
    with atomic_output(path) as tmp:
        with open(tmp, 'w', encoding=encoding) as handle:
            handle.write(text)
    return path


def atomic_write_bytes(path, data):
    with atomic_output(path) as tmp:
        with open(tmp, 'wb') as handle:
            handle.write(data)
    return path


def atomic_np_savez(path, compressed=False, **arrays):
    """
    np.savez / np.savez_compressed, atomically. Same file name as numpy would use:
    `.npz` is appended when `path` does not already end with it.
    """
    import numpy as np
    path = str(path)
    if not path.endswith('.npz'):
        path += '.npz'
    with atomic_output(path) as tmp:
        with open(tmp, 'wb') as handle:   # a handle, so numpy adds no suffix to the temp
            (np.savez_compressed if compressed else np.savez)(handle, **arrays)
    return path


def atomic_np_save(path, array):
    """np.save, atomically (`.npy` appended like numpy when missing)."""
    import numpy as np
    path = str(path)
    if not path.endswith('.npy'):
        path += '.npy'
    with atomic_output(path) as tmp:
        with open(tmp, 'wb') as handle:
            np.save(handle, array)
    return path


def atomic_mesh_export(mesh, path, **export_kwargs):
    """trimesh export (format from the extension unless file_type is given), atomically."""
    with atomic_output(path) as tmp:
        if 'file_type' not in export_kwargs:
            export_kwargs['file_type'] = os.path.splitext(path)[1].lstrip('.').lower() or None
        mesh.export(tmp, **export_kwargs)
    return path


def atomic_copy(src, dst):
    """shutil.copy2 to `dst`, atomically."""
    with atomic_output(dst) as tmp:
        shutil.copy2(src, tmp)
    return dst


def atomic_ants_write(image, path):
    import ants
    with atomic_output(path) as tmp:
        ants.image_write(image, tmp)
    return path


# ---------------------------------------------------------------------------
# Rule 2: read without renaming
# ---------------------------------------------------------------------------

def load_nifti_readonly(path):
    """
    Load a NIfTI under any name (`.nii.gz`, `.nii`, `.nii.mask.gz`, `.nii.removal_mask.gz`,
    `.nii.backup.gz`) without renaming or touching the file.

    Same API as nib.load for the result (affine, header, get_fdata(), dataobj).
    Alias names are read into memory (masks are uint8, so this stays small).
    """
    import nibabel as nib
    lower = str(path).lower()
    if lower.endswith('.nii.gz') or lower.endswith('.nii'):
        return nib.load(path)
    with open(path, 'rb') as handle:
        raw = handle.read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return nib.Nifti1Image.from_bytes(raw)


# ---------------------------------------------------------------------------
# Rule 3: subject transactions
# ---------------------------------------------------------------------------

_REGISTRY_LOCK = threading.Lock()
# normalised scan_dir -> {'txn': SubjectTransaction, 'thread': ident, 'depth': int}
_ACTIVE = {}
# normalised scan_dir -> threading.Lock (serialises transactions on one subject)
_SUBJECT_LOCKS = {}


def _key(path):
    return os.path.normcase(os.path.abspath(path))


def _subject_lock(scan_dir):
    with _REGISTRY_LOCK:
        lock = _SUBJECT_LOCKS.get(_key(scan_dir))
        if lock is None:
            lock = threading.Lock()
            _SUBJECT_LOCKS[_key(scan_dir)] = lock
        return lock


def is_subject_active(scan_dir):
    with _REGISTRY_LOCK:
        return _key(scan_dir) in _ACTIVE


def _visible_entries(scan_dir):
    """Top-level names the transaction protects (hidden entries are tool state, not data)."""
    try:
        names = os.listdir(scan_dir)
    except FileNotFoundError:
        return []
    return sorted(name for name in names if not name.startswith('.'))


def _write_journal(txn_dir, journal):
    atomic_write_json(os.path.join(txn_dir, TXN_JOURNAL), journal, indent=1)


class SubjectTransaction:
    """
    All-or-nothing changes to one subject folder (see the module docstring, rule 3).

        with SubjectTransaction(scan_dir, 'n4') as txn:
            atomic_nib_save(img, out_path)
            txn.remove(old_sidecar)
            patch_subject_json(json_path, {...})

    Re-entering the same subject on the same thread joins the outer transaction (helpers
    can open one without caring whether their caller already did). Another thread
    working on the same subject waits until the first one finishes.
    """

    def __init__(self, scan_dir, operation, group_id=None, group_marker=None):
        self.scan_dir = os.path.abspath(scan_dir)
        self.operation = str(operation)
        self.group_id = group_id
        self.group_marker = group_marker
        self.txn_dir = os.path.join(self.scan_dir, TXN_DIRNAME)
        self._outer = None
        self._lock = None
        self._journal = None
        self._linked = True
        self._finished = False

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        key = _key(self.scan_dir)
        me = threading.get_ident()
        with _REGISTRY_LOCK:
            entry = _ACTIVE.get(key)
            if entry and entry['thread'] == me:
                entry['depth'] += 1
                self._outer = entry['txn']
                return self._outer
        self._lock = _subject_lock(self.scan_dir)
        self._lock.acquire()
        try:
            if os.path.isdir(os.path.join(self.scan_dir, TXN_DIRNAME)):
                recover_subject(self.scan_dir, force=True)
            self._begin()
        except BaseException:
            self._lock.release()
            raise
        with _REGISTRY_LOCK:
            _ACTIVE[key] = {'txn': self, 'thread': me, 'depth': 1}
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._outer is not None:
            with _REGISTRY_LOCK:
                _ACTIVE[_key(self.scan_dir)]['depth'] -= 1
            return False
        try:
            if self.group_id is not None:
                # LinkedTransaction decides commit / rollback for the whole group.
                if exc_type is not None:
                    self.rollback()
                return False
            if exc_type is None:
                self.commit()
            else:
                print(f"[crash-safe] {self.operation} failed on {os.path.basename(self.scan_dir)}; rolling back")
                self.rollback()
        finally:
            if self.group_id is None:
                self._release()
        return False

    def _release(self):
        if self._outer is not None:
            return
        with _REGISTRY_LOCK:
            _ACTIVE.pop(_key(self.scan_dir), None)
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def _begin(self):
        os.makedirs(self.scan_dir, exist_ok=True)
        os.makedirs(os.path.join(self.txn_dir, 'keep'), exist_ok=True)
        entries = _visible_entries(self.scan_dir)
        kept = {}
        self._linked = True
        for index, name in enumerate(entries):
            path = os.path.join(self.scan_dir, name)
            if not os.path.isfile(path):
                continue
            keep_path = os.path.join(self.txn_dir, 'keep', f"{index:05d}")
            if name.lower().endswith('.json'):
                # Always a real copy: JSON is small, and some writers still rewrite it in
                # place (open(path, 'w')), which would change a hard link's content too.
                shutil.copy2(path, keep_path)
                kept[name] = keep_path
                continue
            if self._linked:
                try:
                    os.link(path, keep_path)
                    kept[name] = keep_path
                    continue
                except OSError:
                    self._linked = False  # e.g. exFAT/FAT32: only JSON is kept up front
        self._journal = {
            'version': 1,
            'boot_id': ENGINE_BOOT_ID,
            'pid': os.getpid(),
            'operation': self.operation,
            'started_at': time.time(),
            'group_id': self.group_id,
            'group_marker': self.group_marker,
            'entries': entries,
            'kept': kept,          # name -> journal copy of its original
            'removed': [],         # [name, journal path] moved out as part of the operation
            'renames': [],         # [src name, dst name] in the order they happened
            'hard_links': self._linked,
        }
        _write_journal(self.txn_dir, self._journal)
        fsync_dir(self.scan_dir)

    # -- operations that must be reversible --------------------------------
    def _name(self, path):
        path = os.path.abspath(path)
        if os.path.dirname(path) != self.scan_dir:
            raise ValueError(f"{path} is not directly inside {self.scan_dir}")
        return os.path.basename(path)

    def preserve(self, path):
        """
        Keep a real copy of an existing file this transaction will change IN PLACE
        (anything not replaced through an atomic_* helper, e.g. a library that rewrites
        the file it was given). The up-front hard link would see such a change, a copy
        does not. Files only ever replaced atomically need no call.
        """
        if self._outer is not None:
            return self._outer.preserve(path)
        name = self._name(path)
        if not os.path.isfile(path):
            return
        previous = self._journal['kept'].get(name)
        if previous and not os.path.samefile(previous, path):
            return  # already an independent copy
        keep_path = os.path.join(self.txn_dir, 'keep', f"p{len(self._journal['kept']):05d}")
        shutil.copy2(path, keep_path)
        self._journal['kept'][name] = keep_path
        _write_journal(self.txn_dir, self._journal)
        if previous:
            _silent_remove(previous)

    def remove(self, path):
        """Delete a file as part of the operation (undone by a rollback)."""
        if self._outer is not None:
            return self._outer.remove(path)
        name = self._name(path)
        if not os.path.lexists(path):
            return
        held = os.path.join(self.txn_dir, 'removed', f"{len(self._journal['removed']):05d}")
        os.makedirs(os.path.dirname(held), exist_ok=True)
        self._journal['removed'].append([name, held])
        _write_journal(self.txn_dir, self._journal)
        os.replace(path, held)

    def rename(self, src, dst):
        """Rename inside the subject folder (undone by a rollback)."""
        if self._outer is not None:
            return self._outer.rename(src, dst)
        src_name, dst_name = self._name(src), self._name(dst)
        if os.path.exists(dst):
            self.remove(dst)
        self._journal['renames'].append([src_name, dst_name])
        _write_journal(self.txn_dir, self._journal)
        os.replace(src, dst)

    # -- outcome -------------------------------------------------------------
    def commit(self):
        if self._finished or self._outer is not None:
            return
        if not os.path.isdir(self.scan_dir):
            # The operation deleted the whole subject (remove_subject_folder): nothing to commit.
            self._finished = True
            return
        fsync_dir(self.scan_dir)
        journal_path = os.path.join(self.txn_dir, TXN_JOURNAL)
        os.remove(journal_path)          # <- the commit point
        _drop_txn_dir(self.txn_dir)
        self._finished = True

    def rollback(self):
        if self._finished or self._outer is not None:
            return
        if not os.path.isdir(self.scan_dir):
            self._finished = True
            return
        _rollback_from_journal(self.scan_dir, self._journal)
        self._finished = True


def _drop_txn_dir(txn_dir):
    for attempt in range(5):
        try:
            shutil.rmtree(txn_dir)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.1 * (attempt + 1))
    shutil.rmtree(txn_dir, ignore_errors=True)


def _remove_entry(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        _silent_remove(path)


def _rollback_from_journal(scan_dir, journal):
    """Put `scan_dir` back to the state recorded in `journal`, then drop the journal."""
    txn_dir = os.path.join(scan_dir, TXN_DIRNAME)
    entries = set(journal.get('entries') or [])
    kept = journal.get('kept') or {}

    # 1. Undo renames, newest first (so chains like N -> N+1 unwind correctly).
    for src_name, dst_name in reversed(journal.get('renames') or []):
        src, dst = os.path.join(scan_dir, src_name), os.path.join(scan_dir, dst_name)
        if os.path.exists(dst) and not os.path.exists(src):
            os.replace(dst, src)

    # 2. Anything that was not there before is new output: delete it.
    for name in _visible_entries(scan_dir):
        if name not in entries:
            _remove_entry(os.path.join(scan_dir, name))
    for name in os.listdir(scan_dir):
        if is_csp_temp_name(name):
            _silent_remove(os.path.join(scan_dir, name))

    # 3. Files the operation removed come back.
    for name, held in journal.get('removed') or []:
        if os.path.lexists(held):
            target = os.path.join(scan_dir, name)
            if os.path.exists(target) and name in kept:
                continue  # its original is restored from `kept` below
            os.replace(held, target)

    # 4. Changed files get their original content back.
    for name, keep_path in kept.items():
        if not os.path.exists(keep_path):
            continue
        target = os.path.join(scan_dir, name)
        try:
            if os.path.exists(target) and os.path.samefile(target, keep_path):
                continue
        except OSError:
            pass
        os.replace(keep_path, target)

    fsync_dir(scan_dir)
    _silent_remove(os.path.join(txn_dir, TXN_JOURNAL))
    _drop_txn_dir(txn_dir)
    if not entries:
        # The folder was created for this operation (e.g. a scan being extracted):
        # with its outputs gone it goes too, so no empty half-subject is left behind.
        try:
            os.rmdir(scan_dir)
        except OSError:
            pass
    print(f"[crash-safe] rolled back interrupted '{journal.get('operation')}' on {os.path.basename(scan_dir)}")


def _load_journal(txn_dir):
    try:
        with open(os.path.join(txn_dir, TXN_JOURNAL), 'r') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def recover_subject(scan_dir, force=False):
    """
    Finish a transaction left in `scan_dir` by a dead engine. Returns what was done:
    None, 'committed' (journal gone: only cleanup was missing), 'rolled_forward'
    (linked group had committed) or 'rolled_back'.

    Skipped while this process has a live transaction there, unless `force` (used by
    SubjectTransaction itself, which holds the subject lock).
    """
    txn_dir = os.path.join(scan_dir, TXN_DIRNAME)
    if not os.path.isdir(txn_dir):
        return None
    if not force and is_subject_active(scan_dir):
        return None
    journal = _load_journal(txn_dir)
    if journal is None:
        # No journal: either the commit already happened (journal removed first), or the
        # crash hit before the first journal write finished, in which case nothing in
        # the subject folder has been touched yet. Either way only the folder remains.
        _drop_txn_dir(txn_dir)
        return 'committed'
    if journal.get('boot_id') == ENGINE_BOOT_ID and is_subject_active(scan_dir) and not force:
        return None
    marker = journal.get('group_marker')
    if marker and os.path.exists(marker):
        _silent_remove(os.path.join(txn_dir, TXN_JOURNAL))
        _drop_txn_dir(txn_dir)
        print(f"[crash-safe] completed committed linked '{journal.get('operation')}' on {os.path.basename(scan_dir)}")
        return 'rolled_forward'
    _rollback_from_journal(scan_dir, journal)
    return 'rolled_back'


# ---------------------------------------------------------------------------
# Rule 4: linked subjects
# ---------------------------------------------------------------------------

class LinkedTransaction:
    """
    One commit for several subjects (linked scans replaying an operation).

        with LinkedTransaction(directory, 'crop') as group:
            with group.subject(main_dir):
                ...
            for sibling_dir in siblings:
                with group.subject(sibling_dir):
                    ...

    Members stay journalled until the group ends. On success the group marker is
    written first (the commit point for all members), then each member's journal is
    dropped, then the marker. On failure every member is rolled back.
    """

    def __init__(self, directory, operation):
        self.directory = os.path.abspath(directory)
        self.operation = str(operation)
        self.group_id = uuid.uuid4().hex
        # Project root, not extracted/: some listings of extracted/ treat every folder as a subject.
        self.groups_dir = os.path.join(self.directory, GROUP_DIRNAME)
        groups_dir = self.groups_dir
        self.marker = os.path.join(groups_dir, f"{self.group_id}.committed")
        self._members = []

    def __enter__(self):
        return self

    def subject(self, scan_dir):
        """Context manager for one member; joins an existing member for the same folder."""
        for member in self._members:
            if _key(member.scan_dir) == _key(scan_dir):
                return _JoinMember(member)
        txn = SubjectTransaction(scan_dir, self.operation, group_id=self.group_id, group_marker=self.marker)
        self._members.append(txn)
        return txn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                atomic_write_json(self.marker, {'operation': self.operation, 'at': time.time()})
                for member in self._members:
                    member.commit()
                _silent_remove(self.marker)
                try:
                    os.rmdir(self.groups_dir)  # only when empty: other groups may be live
                except OSError:
                    pass
            else:
                print(f"[crash-safe] linked {self.operation} failed; rolling back {len(self._members)} subject(s)")
                for member in reversed(self._members):
                    member.rollback()
        finally:
            for member in self._members:
                member._release()
        return False


class _JoinMember:
    def __init__(self, member):
        self.member = member

    def __enter__(self):
        return self.member

    def __exit__(self, exc_type, exc, tb):
        return False


# ---------------------------------------------------------------------------
# Project-open recovery
# ---------------------------------------------------------------------------

def _subject_dirs(directory):
    roots = [os.path.join(directory, 'extracted')]
    atlas = os.path.join(directory, 'atlas')
    dirs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not name.startswith('.') and os.path.isdir(path):
                dirs.append(path)
    if os.path.isdir(atlas):
        dirs.append(atlas)
    return dirs


def _sweep_stale_files(folder, now):
    """Old hidden CSP temp files and orphaned linked-replay removal masks."""
    removed = 0
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    for name in names:
        if not (is_csp_temp_name(name) or name.endswith('.nii.removal_mask.gz')):
            continue
        path = os.path.join(folder, name)
        try:
            if now - os.path.getmtime(path) < STALE_TEMP_SECONDS:
                continue
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def recover_project(directory):
    """
    Bring every subject of a project back to a consistent state after a crash.
    Safe to call at any time: live transactions of this engine are left alone.
    Returns a summary dict (also printed).
    """
    summary = {'rolled_back': 0, 'rolled_forward': 0, 'committed': 0, 'swept': 0}
    if not directory or not os.path.isdir(directory):
        return summary
    now = time.time()
    for scan_dir in _subject_dirs(directory):
        if is_subject_active(scan_dir):
            continue
        try:
            outcome = recover_subject(scan_dir)
        except Exception as exc:  # never block opening a project
            print(f"[crash-safe] could not recover {scan_dir}: {exc}")
            outcome = None
        if outcome:
            summary[outcome] += 1
        summary['swept'] += _sweep_stale_files(scan_dir, now)
    trash_dir = os.path.join(directory, TRASH_DIRNAME)
    if os.path.isdir(trash_dir):
        # Subject folders whose deletion was cut short (remove_subject_folder).
        shutil.rmtree(trash_dir, ignore_errors=True)
    groups_dir = os.path.join(directory, GROUP_DIRNAME)
    if os.path.isdir(groups_dir):
        # Every member of a committed group has been rolled forward above.
        with _REGISTRY_LOCK:
            busy = bool(_ACTIVE)
        if not busy:
            shutil.rmtree(groups_dir, ignore_errors=True)
    if any(summary.values()):
        print(f"[crash-safe] project recovery for {directory}: {summary}")
    return summary


# ---------------------------------------------------------------------------
# View helper: one decorator line per Django view
# ---------------------------------------------------------------------------

class _RollbackResponse(Exception):
    """Internal: a view returned an error response, so its changes are undone."""


def subject_dir(directory, name):
    """Folder of a subject: `extracted/<name>`, or the top-level atlas folder for 'atlas'."""
    if not directory or not name:
        return None
    name = str(name)
    if name.lower() == 'atlas':
        return os.path.join(directory, 'atlas')
    return os.path.join(directory, 'extracted', name)


@contextlib.contextmanager
def transaction_over(folders, operation):
    """
    One transaction over any number of subject folders: a SubjectTransaction for one,
    a LinkedTransaction for several (all commit or all roll back together). Folders
    that do not exist are skipped; with none left it simply runs the block unprotected.
    Yields nothing useful: code inside reaches the open transactions by opening
    SubjectTransaction(scan_dir, ...) again, which joins the outer one on this thread.
    """
    seen, existing = set(), []
    for folder in folders or []:
        if folder and os.path.isdir(folder) and _key(folder) not in seen:
            seen.add(_key(folder))
            existing.append(os.path.abspath(folder))
    if not existing:
        yield None
        return
    # One global locking order, so two linked operations can never wait on each other.
    existing.sort(key=_key)
    if len(existing) == 1:
        with SubjectTransaction(existing[0], operation) as txn:
            yield txn
        return
    with LinkedTransaction(_project_of(existing[0]), operation) as group, contextlib.ExitStack() as stack:
        for folder in existing:
            stack.enter_context(group.subject(folder))
        yield group


def subject_transaction(directory, name, operation, linked=False):
    """
    Transaction for one subject of a batch loop:

        for scan_name in scan_names:
            try:
                with subject_transaction(directory, scan_name, 'n4'):
                    ...every write for this subject...
            except Exception as exc:
                ...                     # the subject is already rolled back here

    linked=True also covers every scan linked to it (the sibling replays), all or nothing.
    """
    return transaction_over(subject_folders(directory, [name], linked=linked), operation)


def subject_folders(directory, names, linked=False):
    """Subject folders for `names`, plus each one's link group when linked=True."""
    folders = []
    for name in names or []:
        if not isinstance(name, str) or not name:
            continue
        folders.append(subject_dir(directory, name))
        if linked:
            try:
                from .linkedScans import get_link_group
                group = get_link_group(directory, name)
            except Exception:
                group = None
            for member in (group or {}).get('members', []):
                folders.append(subject_dir(directory, member))
    return folders


def crash_safe_view(operation, subjects):
    """
    Run a single-subject view method inside one transaction (rules 3 and 4):

        @crash_safe_view('crop', lambda request: request_subjects(request, 'filename'))
        def post(self, request): ...

    `subjects(request)` returns the subject folders the view may change (see
    transaction_over). A raised exception, or a returned response with status >= 400,
    rolls every folder back; anything else commits. Batch views do not use this: they
    open subject_transaction per subject inside their loop, so a crash on subject 51
    keeps the 50 finished ones.
    """
    def decorate(view_method):
        @functools.wraps(view_method)
        def wrapper(self, request, *args, **kwargs):
            try:
                folders = subjects(request) or []
            except Exception as exc:
                print(f"[crash-safe] {operation}: could not resolve subjects ({exc}); running unprotected")
                folders = []
            result = {}
            try:
                with transaction_over(folders, operation):
                    result['response'] = view_method(self, request, *args, **kwargs)
                    if getattr(result['response'], 'status_code', 200) >= 400:
                        raise _RollbackResponse()
            except _RollbackResponse:
                pass
            return result['response']
        return wrapper
    return decorate


def _project_of(folder):
    """Project root of a subject folder (`<project>/extracted/<name>` or `<project>/atlas`)."""
    parent = os.path.dirname(os.path.abspath(folder))
    if os.path.basename(parent) == 'extracted':
        return os.path.dirname(parent)
    return parent


def request_subjects(request, *keys, linked=False):
    """
    Subject folders named by `request.data[key]` for each key (a name or a list of
    names). linked=True adds each subject's link group (views that replay to siblings).
    """
    data = getattr(request, 'data', None) or {}
    directory = data.get('directory')
    if not directory:
        return []
    names = []
    for key in keys:
        value = data.get(key)
        if isinstance(value, (list, tuple)):
            names.extend(value)
        else:
            names.append(value)
    return subject_folders(directory, names, linked=linked)


def _active_txn_for(path):
    """The transaction this thread has open on the folder that holds `path`, if any."""
    key = _key(os.path.dirname(os.path.abspath(path)))
    with _REGISTRY_LOCK:
        entry = _ACTIVE.get(key)
    if entry and entry['thread'] == threading.get_ident():
        return entry['txn']
    return None


def tracked_rename(src, dst):
    """
    os.replace(src, dst) that the open transaction can undo. On disks without hard links
    (exFAT/FAT32) a plain rename of a data file could not be rolled back; this records it.
    Outside a transaction it is a plain os.replace.
    """
    txn = _active_txn_for(src)
    if txn is not None and os.path.dirname(os.path.abspath(src)) == os.path.dirname(os.path.abspath(dst)):
        txn.rename(src, dst)
    else:
        os.replace(src, dst)


def tracked_remove(path):
    """os.remove(path) that the open transaction can undo (see tracked_rename)."""
    txn = _active_txn_for(path)
    if txn is not None:
        txn.remove(path)
    elif os.path.lexists(path):
        os.remove(path)


def remove_subject_folder(scan_dir):
    """
    Delete a whole subject folder crash-safely. The folder is first moved, in one atomic
    rename, into <project>/.aurora_trash/, so the subject is gone from extracted/ the
    moment this starts; the slow recursive delete happens there. A crash during the
    delete leaves only the trash folder, which recover_project empties. Never use
    shutil.rmtree on a subject folder directly: a crash halfway leaves a subject with
    some of its files, which every tool would then try to read.
    """
    scan_dir = os.path.abspath(scan_dir)
    if not os.path.isdir(scan_dir):
        return
    trash_dir = os.path.join(_project_of(scan_dir), TRASH_DIRNAME)
    os.makedirs(trash_dir, exist_ok=True)
    doomed = os.path.join(trash_dir, f"{os.path.basename(scan_dir)}-{uuid.uuid4().hex[:8]}")
    os.replace(scan_dir, doomed)
    fsync_dir(os.path.dirname(scan_dir))
    shutil.rmtree(doomed, ignore_errors=True)
