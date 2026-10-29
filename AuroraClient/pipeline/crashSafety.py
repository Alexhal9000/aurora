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

   On entry the subject folder is journalled in "<scan_dir>/.aurora_txn/journal.json"
   (journal version 2): the names of its files, the size and modification
   time of each data file, and, in a folder of its own for this one transaction
   (".aurora_txn/<id>/"), a real copy of every JSON file. Nothing else is copied or
   linked up front. Instead the journal is an undo log: just before an existing file is
   replaced, deleted or renamed, the original is moved into ".aurora_txn/<id>/held/" by
   a rename (instant for any size, on any filesystem) and the step is logged first.
   atomic_output does this by itself, so every atomic_* helper is covered; other writers
   use tracked_replace(src, dst), tracked_remove(path), tracked_rename(src, dst) or
   txn.remove / txn.rename. A file that must be changed IN PLACE needs txn.preserve(path)
   (a copy) before it is touched. Deleting the journal file is the commit. If the method
   raises, or the engine dies before the commit, the rollback replays the log newest
   first, deletes every file that was not there before and puts back every JSON file,
   so the subject is byte-for-byte as before. A data file changed any other way (a raw
   os.replace / os.remove) cannot be undone: the commit prints a warning naming it, so
   the writer can be fixed.

   Why no hard links (1.9.4 / 1.9.5 linked every file up front): a hard link carries the
   original's owner, permissions and Windows read-only flag, some network storage
   (SMB) refuses them, and on NFS a linked file still open somewhere leaves a ".nfsXXXX"
   file behind, so journal folders could not always be deleted. Journals written by those
   versions are still recovered (_rollback_v1).

4. LINKED SUBJECTS COMMIT TOGETHER. When one operation changes several subjects that
   must stay in step (linked scans replaying an edit on their siblings), wrap them in
   LinkedTransaction(directory, operation) and add each subject with
   group.subject(scan_dir). A single marker file (<project>/.aurora_txn_groups/) decides the outcome for all of them
   after a crash: present = roll every member forward, absent = roll every member back.

Recovery
--------
recover_project(directory) runs when a project is opened (GetProjectSettingsView), before
the first POST naming the project after an engine start (RecoverInterruptedWork, so a
batch view's "already done?" checks never see a dead run's half-written outputs), and
SubjectTransaction recovers a subject itself before starting on it. A journal belongs
to a live run when it carries this process's ENGINE_BOOT_ID and is registered as
active here, or when the other engine process on this computer that wrote it is still
running (_live_foreign_owner); anything else is left over from a dead engine and is
rolled back (or forward, for a committed linked group). Leftover hidden temp files and orphaned
".nii.removal_mask.gz" files are swept as well.

Journal folders that cannot be deleted
--------------------------------------
Deleting ".aurora_txn" can still fail: a sync client (OneDrive, Dropbox) or antivirus
holding a file open on Windows, storage that lets Aurora create files but not delete
them (some network mounts), or a folder made by another account (desktop Aurora vs the
aurora-tools sharing service). Such a leftover never blocks a tool:
  - every transaction writes into a new ".aurora_txn/<id>/" folder, so it never writes
    onto a leftover file;
  - a ".aurora_txn" folder that will not go, or that this account cannot write in, is
    renamed aside to ".aurora_txn-stale-<id>" and swept on project open; if even that
    fails, the engine log prints a command that gives this account the rights it needs
    (_print_cleanup_help);
  - a rollback first renames journal.json to rollback.json: if it cannot, it stops
    before touching the subject, and once done it can never be replayed over newer work.

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
import socket
import stat
import sys
import threading
import time
import uuid

# One id per engine process. A journal stamped with another id was left by a dead engine.
ENGINE_BOOT_ID = uuid.uuid4().hex

TEMP_MARKER = '.csp-tmp-'
TXN_DIRNAME = '.aurora_txn'
TXN_JOURNAL = 'journal.json'
# journal.json is renamed to this while it is being rolled back (see _claim_journal).
TXN_ROLLBACK = 'rollback.json'
# A ".aurora_txn" folder that could not be deleted or written in is renamed to
# ".aurora_txn-stale-<id>" so the next transaction starts clean (swept on project open).
TXN_STALE_PREFIX = TXN_DIRNAME + '-stale-'
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

def _read_umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


# Mode a plain open() gives a new file in this process (0o666 minus the umask). Writers
# that create their temp file with tempfile.mkstemp (always 0600) set it afterwards: with
# the shared service, a 0600 file in a shared folder is unreadable for the other account
# (desktop Aurora vs the aurora-tools service), because its access-list mask becomes empty.
NEW_FILE_MODE = 0o666 & ~_read_umask()


def match_new_file_mode(path):
    """Give a mkstemp temp file the mode open() would have given it (see NEW_FILE_MODE)."""
    try:
        os.chmod(path, NEW_FILE_MODE)
    except OSError:
        pass


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
        txn = _active_txn_for(path)
        if txn is not None:
            # CRASH-SAFE PROTOCOL: the file being replaced is moved into the open
            # transaction (rule 3), so a rollback can put it back.
            txn.stash(path)
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


# Journal keys that hold paths of kept / removed files. On disk they are relative to the
# subject folder, so recovery still works when the project is reached through another
# path (a storage server mounted elsewhere, another computer, a new drive letter).
# In memory they are absolute. Journals from 1.9.4/1.9.5 hold absolute paths, still read.
def _journal_paths(journal, convert):
    out = dict(journal)
    if 'kept' in journal:
        out['kept'] = {name: convert(path) for name, path in (journal.get('kept') or {}).items()}
    if 'removed' in journal:
        out['removed'] = [[name, convert(path)] for name, path in (journal.get('removed') or [])]
    if 'json_kept' in journal:
        out['json_kept'] = {name: convert(path) for name, path in (journal.get('json_kept') or {}).items()}
    if 'log' in journal:
        # [op, name, held file] for stash / remove / copy; [op, src name, dst name] for rename.
        out['log'] = [step if step[0] == 'rename' else [step[0], step[1], convert(step[2])]
                      for step in journal.get('log') or []]
    if journal.get('work_dir'):
        out['work_dir'] = convert(journal['work_dir'])
    return out


def _write_journal(txn_dir, journal):
    scan_dir = os.path.dirname(txn_dir)
    on_disk = _journal_paths(journal, lambda path: os.path.relpath(path, scan_dir))
    atomic_write_json(os.path.join(txn_dir, TXN_JOURNAL), on_disk, indent=1)


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
        # This transaction's own folder for kept and removed files (see _begin).
        self.work_dir = os.path.join(self.txn_dir, uuid.uuid4().hex[:12])
        self._outer = None
        self._lock = None
        self._journal = None
        # Journal changes can come from more than one thread (a worker pool writing outputs
        # of the subject through atomic_output), so they are serialised.
        self._journal_lock = threading.Lock()
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
        _open_work_dir(self.scan_dir, self.work_dir)
        try:
            self._journal_subject()
        except BaseException:
            # Nothing in the subject has been touched yet: only this attempt's folder goes.
            _drop_txn_dir(self.work_dir)
            _drop_txn_dir_if_empty(self.txn_dir)
            raise

    def _journal_subject(self):
        """
        Journal version 2: the names of the subject's files, a real copy of every JSON file
        and the size and modification time of every other file. Data files are not copied
        or linked up front: whatever the tool replaces, deletes or renames is moved into
        this transaction's folder by a rename at that moment (stash / remove / rename), so
        the cost is the same for a 2 GB scan as for a text file, and no hard links are made.
        """
        entries = _visible_entries(self.scan_dir)
        json_kept = {}
        stats = {}
        os.makedirs(os.path.join(self.work_dir, 'held'), exist_ok=True)
        for index, name in enumerate(entries):
            path = os.path.join(self.scan_dir, name)
            if not os.path.isfile(path):
                continue
            if name.lower().endswith('.json'):
                # Always a real copy: JSON is small, and some writers still rewrite it in
                # place (open(path, 'w')). copyfile, not copy2: the copy is ours to delete
                # later, so it must not take a read-only flag or another account's mode.
                keep_path = os.path.join(self.work_dir, 'keep', f"{index:05d}")
                try:
                    shutil.copyfile(path, keep_path)
                except PermissionError as exc:
                    raise PermissionError(
                        f"Aurora cannot read {path} ({exc.strerror}), so it cannot protect "
                        f"{os.path.basename(self.scan_dir)} against a crash. The file was probably "
                        "made by another account (for example the shared Aurora service); "
                        "give this account read access to it."
                    ) from exc
                json_kept[name] = keep_path
                continue
            info = os.stat(path)
            stats[name] = [info.st_size, info.st_mtime_ns]
        self._journal = {
            'version': 2,
            'boot_id': ENGINE_BOOT_ID,
            'pid': os.getpid(),
            'host': socket.gethostname(),
            'operation': self.operation,
            'started_at': time.time(),
            'group_id': self.group_id,
            'group_marker': self.group_marker,
            'entries': entries,
            'json_kept': json_kept,  # name -> copy of the JSON file as it was
            'stats': stats,          # name -> [size, mtime_ns] of every other file
            'log': [],               # undo log, replayed newest first by a rollback
            'work_dir': self.work_dir,
        }
        _write_journal(self.txn_dir, self._journal)
        fsync_dir(self.scan_dir)

    def _held_path(self):
        return os.path.join(self.work_dir, 'held', f"{len(self._journal['log']):05d}")

    def _log(self, step):
        """Record a step BEFORE doing it: the rollback skips steps that never happened."""
        self._journal['log'].append(step)
        _write_journal(self.txn_dir, self._journal)

    def _logged_names(self):
        names = set()
        for step in self._journal['log']:
            names.add(step[1])
            if step[0] == 'rename':
                names.add(step[2])
        return names

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
        the file it was given). Files replaced through atomic_output, tracked_replace,
        tracked_remove or tracked_rename need no call. JSON files are always copied.
        """
        if self._outer is not None:
            return self._outer.preserve(path)
        name = self._name(path)
        if not os.path.isfile(path) or name in self._journal['json_kept'] or name not in self._journal['entries']:
            return
        with self._journal_lock:
            if name in self._logged_names():
                return  # its original is already held (or it is this run's own output)
            held = self._held_path()
            self._log(['copy', name, held])
            shutil.copyfile(path, held)

    def stash(self, path):
        """
        Move the existing file at `path` into the journal because it is about to be
        replaced (atomic_output and tracked_replace call this). Only the first change of
        an original file is kept; this run's own outputs, and files outside the subject
        folder's top level, need nothing. Returns True when the file was moved away.
        """
        if self._outer is not None:
            return self._outer.stash(path)
        try:
            name = self._name(path)
        except ValueError:
            return False  # e.g. debug_alignment/...: sub-folders are not journalled
        if not os.path.lexists(path) or name not in self._journal['entries'] or name in self._journal['json_kept']:
            return False
        with self._journal_lock:
            if name in self._logged_names():
                return False
            held = self._held_path()
            self._log(['stash', name, held])
            os.replace(path, held)
        return True

    def remove(self, path):
        """Delete a file as part of the operation (undone by a rollback)."""
        if self._outer is not None:
            return self._outer.remove(path)
        name = self._name(path)
        if not os.path.lexists(path):
            return
        with self._journal_lock:
            held = self._held_path()
            self._log(['remove', name, held])
            os.replace(path, held)

    def rename(self, src, dst):
        """Rename inside the subject folder (undone by a rollback)."""
        if self._outer is not None:
            return self._outer.rename(src, dst)
        src_name, dst_name = self._name(src), self._name(dst)
        if os.path.lexists(dst):
            self.remove(dst)
        with self._journal_lock:
            self._log(['rename', src_name, dst_name])
            os.replace(src, dst)

    def _unprotected_changes(self):
        """
        Data files that changed, or went away, without going through the helpers above
        (a raw os.replace / os.remove / in-place write). A rollback cannot undo those;
        they are reported so the writer can be fixed.
        """
        logged = self._logged_names()
        changed = []
        for name, (size, mtime_ns) in (self._journal.get('stats') or {}).items():
            if name in logged:
                continue
            try:
                info = os.stat(os.path.join(self.scan_dir, name))
            except FileNotFoundError:
                changed.append(name)
                continue
            except OSError:
                continue
            if info.st_size != size or info.st_mtime_ns != mtime_ns:
                changed.append(name)
        return changed

    # -- outcome -------------------------------------------------------------
    def commit(self):
        if self._finished or self._outer is not None:
            return
        if not os.path.isdir(self.scan_dir):
            # The operation deleted the whole subject (remove_subject_folder): nothing to commit.
            self._finished = True
            return
        unprotected = self._unprotected_changes()
        if unprotected:
            print(f"[crash-safe] warning: '{self.operation}' changed {', '.join(sorted(unprotected))} in "
                  f"{os.path.basename(self.scan_dir)} without the crash-safe helpers; a crash during "
                  "it could not have undone that (see pipeline/crashSafety.py)")
        fsync_dir(self.scan_dir)
        journal_path = os.path.join(self.txn_dir, TXN_JOURNAL)
        try:
            _remove_with_retry(journal_path)          # <- the commit point
        except OSError as exc:
            raise PermissionError(
                f"'{self.operation}' finished on {os.path.basename(self.scan_dir)}, but Aurora could "
                f"not mark it as done ({journal_path}: {exc.strerror}), so it will be undone the "
                "next time this subject is used. " + _ACCESS_HINT
            ) from exc
        _drop_txn_dir(self.txn_dir)
        self._finished = True

    def rollback(self):
        if self._finished or self._outer is not None:
            return
        if not os.path.isdir(self.scan_dir):
            self._finished = True
            return
        with self._journal_lock:
            _rollback_from_journal(self.scan_dir, self._journal)
        self._finished = True


# Added to the permission errors below: what usually causes them and what fixes them.
_ACCESS_HINT = (
    "Usually another program is holding a file in its hidden .aurora_txn folder (a sync "
    "client such as OneDrive or Dropbox, or antivirus), or the folder was made by another "
    "account (for example the shared Aurora service). Pause syncing for the project or "
    "give this account full access to the project folder, then try again."
)


def _remove_with_retry(path, retries=6, delay=0.05):
    """os.remove, retried: on Windows a reader (sync client, antivirus) holding `path` makes it fail briefly."""
    for attempt in range(retries):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _clear_readonly_and_retry(function, path, exc):
    """
    shutil.rmtree onexc: Windows will not delete a read-only file, so clear the flag and
    try again. Only on a file with no other name: a hard link from a 1.9.4 / 1.9.5 journal
    shares the flag with the subject's own file, which must stay read-only.
    """
    try:
        info = os.lstat(path)
        if os.name == 'nt' and not info.st_mode & stat.S_IWRITE and info.st_nlink <= 1:
            os.chmod(path, stat.S_IWRITE)
            function(path)
            return
    except OSError:
        pass
    raise exc


def _set_aside(txn_dir):
    """
    Rename a ".aurora_txn" folder that will not go away to ".aurora_txn-stale-<id>"
    (swept on project open), so the next transaction does not trip over it. Renaming
    needs write access to the subject folder only. Returns True when the name is free.
    """
    if os.path.basename(txn_dir) != TXN_DIRNAME:
        return False
    stale = os.path.join(os.path.dirname(txn_dir), f"{TXN_STALE_PREFIX}{uuid.uuid4().hex[:8]}")
    try:
        os.replace(txn_dir, stale)
    except OSError as exc:
        print(f"[crash-safe] could not remove or set aside {txn_dir}: {exc}")
        return False
    print(f"[crash-safe] {txn_dir} could not be deleted; renamed to {os.path.basename(stale)}")
    return True


def _drop_txn_dir(txn_dir):
    for attempt in range(5):
        try:
            shutil.rmtree(txn_dir, onexc=_clear_readonly_and_retry)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.1 * (attempt + 1))
    shutil.rmtree(txn_dir, ignore_errors=True)
    if os.path.lexists(txn_dir):
        _set_aside(txn_dir)


def _drop_txn_dir_if_empty(txn_dir):
    try:
        os.rmdir(txn_dir)
    except OSError:
        pass


def _no_write_access(folder, exc):
    import getpass
    try:
        account = getpass.getuser()
    except Exception:
        account = 'this account'
    return PermissionError(
        f"Aurora (running as {account}) cannot write in {folder} ({exc.strerror}), so it cannot "
        "start this tool safely. " + _ACCESS_HINT
    )


def _open_work_dir(scan_dir, work_dir):
    """
    Create a transaction's own folder, `.aurora_txn/<id>/keep`. A `.aurora_txn` this
    account cannot write in, left behind by another account with no journal in it, is
    renamed aside first (see the module docstring).
    """
    txn_dir = os.path.dirname(work_dir)
    keep_dir = os.path.join(work_dir, 'keep')
    try:
        os.makedirs(keep_dir)
        return
    except PermissionError as exc:
        first_error = exc
    has_journal = any(os.path.lexists(os.path.join(txn_dir, name)) for name in (TXN_JOURNAL, TXN_ROLLBACK))
    if not os.path.isdir(txn_dir) or has_journal or not _set_aside(txn_dir):
        raise _no_write_access(txn_dir if os.path.isdir(txn_dir) else scan_dir, first_error) from first_error
    try:
        os.makedirs(keep_dir)
    except PermissionError as exc:
        raise _no_write_access(scan_dir, exc) from exc


def _claim_journal(scan_dir, txn_dir):
    """
    Rename journal.json to rollback.json before a rollback changes anything. If that is
    not possible the rollback stops here, with the subject untouched: a rollback whose
    journal stayed behind would run again later and delete everything made since. After
    a crash halfway through, rollback.json is the journal and rolling back again
    finishes the job (every step is safe to repeat).
    """
    journal_path = os.path.join(txn_dir, TXN_JOURNAL)
    if not os.path.lexists(journal_path):
        return  # already claimed (an interrupted rollback being finished)
    try:
        _replace_with_retry(journal_path, os.path.join(txn_dir, TXN_ROLLBACK))
    except OSError as exc:
        raise PermissionError(
            f"Aurora cannot undo the interrupted work in {os.path.basename(scan_dir)}: "
            f"{journal_path} cannot be changed ({exc.strerror}). Nothing was touched. " + _ACCESS_HINT
        ) from exc


def _remove_entry(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        _silent_remove(path)


def _rollback_from_journal(scan_dir, journal):
    """Put `scan_dir` back to the state recorded in `journal`, then drop the journal."""
    if journal.get('version', 1) >= 2:
        return _rollback_v2(scan_dir, journal)
    return _rollback_v1(scan_dir, journal)


def _restore(held, target):
    """Move a held original back over `target`; a step that never happened has no held file."""
    if os.path.lexists(held):
        _replace_with_retry(held, target)


def _rollback_v2(scan_dir, journal):
    """
    Undo a version-2 journal. Every step is safe to repeat, so a crash during the rollback
    is finished by the next one:
      1. replay the undo log newest first (put back what was stashed, removed or
         rewritten, undo renames);
      2. delete everything that was not there before (this run's new outputs);
      3. put back every JSON file as it was.
    """
    txn_dir = os.path.join(scan_dir, TXN_DIRNAME)
    entries = set(journal.get('entries') or [])
    _claim_journal(scan_dir, txn_dir)

    for step in reversed(journal.get('log') or []):
        op = step[0]
        if op == 'rename':
            src, dst = os.path.join(scan_dir, step[1]), os.path.join(scan_dir, step[2])
            if os.path.lexists(dst):
                # Anything at `src` now was written after the rename: new output, replaced.
                _replace_with_retry(dst, src)
        elif op in ('stash', 'remove', 'copy'):
            _restore(step[2], os.path.join(scan_dir, step[1]))

    for name in _visible_entries(scan_dir):
        if name not in entries:
            _remove_entry(os.path.join(scan_dir, name))
    for name in os.listdir(scan_dir):
        if is_csp_temp_name(name):
            _silent_remove(os.path.join(scan_dir, name))

    for name, keep_path in (journal.get('json_kept') or {}).items():
        _restore(keep_path, os.path.join(scan_dir, name))

    lost = []
    restored = {step[1] for step in journal.get('log') or []}  # put back from the log above
    for name, (size, mtime_ns) in (journal.get('stats') or {}).items():
        if name in restored:
            continue
        try:
            info = os.stat(os.path.join(scan_dir, name))
            if info.st_size != size or info.st_mtime_ns != mtime_ns:
                lost.append(name)
        except FileNotFoundError:
            lost.append(name)
        except OSError:
            pass
    if lost:
        print(f"[crash-safe] warning: could not undo changes to {', '.join(sorted(lost))} in "
              f"{os.path.basename(scan_dir)} (changed without the crash-safe helpers)")
    _finish_rollback(scan_dir, txn_dir, entries, journal)


def _finish_rollback(scan_dir, txn_dir, entries, journal):
    fsync_dir(scan_dir)
    try:
        _remove_with_retry(os.path.join(txn_dir, TXN_ROLLBACK))
    except OSError as exc:
        print(f"[crash-safe] could not remove {os.path.join(txn_dir, TXN_ROLLBACK)}: {exc}")
    _drop_txn_dir(txn_dir)
    if not entries:
        # The folder was created for this operation (e.g. a scan being extracted):
        # with its outputs gone it goes too, so no empty half-subject is left behind.
        try:
            os.rmdir(scan_dir)
        except OSError:
            pass
    print(f"[crash-safe] rolled back interrupted '{journal.get('operation')}' on {os.path.basename(scan_dir)}")


def _rollback_v1(scan_dir, journal):
    """Undo a journal written by 1.9.4 / 1.9.5 (hard links or copies of the originals in `kept`)."""
    txn_dir = os.path.join(scan_dir, TXN_DIRNAME)
    entries = set(journal.get('entries') or [])
    kept = journal.get('kept') or {}
    _claim_journal(scan_dir, txn_dir)

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

    _finish_rollback(scan_dir, txn_dir, entries, journal)


def _load_journal(txn_dir):
    """
    The journal of the transaction left in `txn_dir` (journal.json, or rollback.json for
    a rollback that was cut short), else None. A journal that exists but cannot be read
    is an error, not "no journal": treating it as committed would keep a dead run's
    half-written outputs.
    """
    for name in (TXN_JOURNAL, TXN_ROLLBACK):
        path = os.path.join(txn_dir, name)
        try:
            with open(path, 'r') as handle:
                journal = json.load(handle)
            scan_dir = os.path.dirname(os.path.abspath(txn_dir))
            return _journal_paths(journal, lambda p: p if os.path.isabs(p) else os.path.join(scan_dir, p))
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise PermissionError(
                f"Aurora cannot read {path} ({exc.strerror}), the record of interrupted work in "
                f"{os.path.basename(os.path.dirname(txn_dir))}. " + _ACCESS_HINT
            ) from exc
        except (OSError, ValueError):
            return None
    return None


def _live_foreign_owner(journal):
    """
    Pid of the other engine process on this computer that wrote `journal` and is still
    running, else None. A journal is only a leftover when its engine is gone; the boot id
    alone cannot tell, because two engines (each with its own boot id) can work on one
    project. The process must have started before the journal did, so a pid reused by an
    unrelated program after a reboot does not count.
    """
    pid = journal.get('pid')
    if journal.get('boot_id') == ENGINE_BOOT_ID or not isinstance(pid, int) or pid == os.getpid():
        return None
    host = journal.get('host')
    if host and host != socket.gethostname():
        return None  # written on another computer (shared drive): cannot be checked here
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(pid)
        started = float(journal.get('started_at') or 0)
        if process.create_time() > started + 1:
            return None
    except psutil.AccessDenied:
        return pid  # another account's process (macOS): it exists, so assume it is live
    except Exception:  # no such process: the journal is a leftover
        return None
    return pid


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
    owner = _live_foreign_owner(journal)
    if owner:
        # Another engine on this computer (e.g. the shared service next to a desktop or
        # dev Aurora) is still working on this subject: its journal is not a leftover.
        if force:
            raise RuntimeError(
                f"{os.path.basename(scan_dir)} is being changed by another Aurora engine "
                f"(process {owner}); try again when it has finished."
            )
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


def _sweep_stale_txn_dirs(folder, stuck):
    """
    Journal folders set aside by _set_aside (no journal in them: only kept copies and
    links). Ones that still will not go are added to `stuck` as (path, error).
    """
    removed = 0
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(TXN_STALE_PREFIX):
            continue
        path = os.path.join(folder, name)
        try:
            shutil.rmtree(path, onexc=_clear_readonly_and_retry)
            removed += 1
        except OSError as exc:
            stuck.append((path, exc))  # tried again on the next project open
    return removed


# Projects whose cleanup help was printed since this engine started (once is enough).
_CLEANUP_HELP_SHOWN = set()


def _permission_fix_command(directory, account):
    """Terminal command that gives `account` read, write and delete rights on `directory`."""
    if os.name == 'nt':
        return [f'icacls "{directory}" /grant "{account}:(OI)(CI)M" /T /C']
    import shlex
    folder = shlex.quote(directory)
    if sys.platform == 'darwin':
        rights = ("list,add_file,search,add_subdirectory,delete_child,read,write,append,"
                  "execute,delete,readattr,writeattr,readextattr,writeextattr,file_inherit,directory_inherit")
        return [f'sudo chmod -R +a {shlex.quote(f"{account} allow {rights}")} {folder}']
    # Network mounts (NFS, SMB) often refuse ACLs ("Operation not supported"): plain modes then.
    return [f'sudo setfacl -R -m u:{account}:rwX,d:u:{account}:rwx {folder} 2>/dev/null '
            f'|| sudo chmod -R u+rwX {folder}']


def _print_cleanup_help(directory, stuck):
    """
    Tell the person, in the engine log, that journal folders cannot be deleted here, with
    a command to copy into a terminal. Once that command has run, the next project open
    cleans them up by itself, so nothing is ever deleted by hand.
    """
    if not stuck or _key(directory) in _CLEANUP_HELP_SHOWN:
        return
    _CLEANUP_HELP_SHOWN.add(_key(directory))
    import getpass
    try:
        account = getpass.getuser()
    except Exception:
        account = os.environ.get('USERNAME') or os.environ.get('USER') or 'YOUR_ACCOUNT'
    path, exc = stuck[0]
    terminal = 'an administrator PowerShell or Command Prompt' if os.name == 'nt' else 'a terminal'
    lines = [
        f"[crash-safe] Aurora (account {account}) cannot delete {len(stuck)} old crash-safety folder(s) in {directory}.",
        f"[crash-safe]   First one: {path} ({getattr(exc, 'strerror', None) or exc})",
        "[crash-safe]   They do not stop any tool, but this folder does not let Aurora delete files, so they pile up.",
        f"[crash-safe]   To fix it, close Aurora, run this in {terminal} on this computer, then open the project again:",
    ]
    lines += [f"      {command}" for command in _permission_fix_command(directory, account)]
    lines += [
        "[crash-safe]   If it still fails and the project is on a network drive or storage server (NFS, SMB),",
        "[crash-safe]   the server or its mount options decide who may delete files: ask whoever manages it",
        f"[crash-safe]   to give {account} write and delete rights on that folder.",
    ]
    print('\n'.join(lines))


# Projects this engine has recovered since it started (normalised paths). A running
# engine leaves no journal of its own behind (a failed transaction rolls back on the
# spot), so one pass per project per engine start covers everything a dead engine left.
_RECOVERED_PROJECTS = set()
# One recovery at a time: two requests opening the same project must not both roll the
# same subject back.
_RECOVERY_LOCK = threading.RLock()


def recover_project(directory):
    """
    Bring every subject of a project back to a consistent state after a crash.
    Safe to call at any time: live transactions of this engine are left alone.
    Returns a summary dict (also printed).
    """
    if not directory or not os.path.isdir(directory):
        return {'rolled_back': 0, 'rolled_forward': 0, 'committed': 0, 'swept': 0}
    with _RECOVERY_LOCK:
        summary = _recover_project(directory)
        _RECOVERED_PROJECTS.add(_key(directory))
    return summary


def recover_project_once(directory):
    """
    recover_project, unless this engine already did it for `directory` since it started.

    Project open (GetProjectSettingsView) is not enough on its own: after an engine
    restart the browser page can still be open and send a batch request straight away
    (e.g. align to reference again after a run was cut short). Its "already done?"
    checks would then see the half-written outputs of the dead run, which only the
    rollback removes. RecoverInterruptedWork runs this before every such request.
    """
    if not isinstance(directory, str) or not directory:
        return None
    directory = os.path.expanduser(directory)
    if _key(directory) in _RECOVERED_PROJECTS or not os.path.isdir(directory):
        return None
    return recover_project(directory)


class RecoverInterruptedWork:
    """
    DRF permission class (settings.REST_FRAMEWORK, after NetworkPathPermission, so only
    folders the caller may use are touched). It never refuses a request: for a POST
    naming a project `directory` it first finishes or undoes what a dead engine left in
    that project, once per engine start (recover_project_once).

    Duck-typed (has_permission / has_object_permission) so this module keeps no Django
    or DRF import.
    """

    def has_permission(self, request, view):
        if request.method == 'POST':
            data = getattr(request, 'data', None)
            directory = data.get('directory') if hasattr(data, 'get') else None
            try:
                recover_project_once(directory)
            except Exception as exc:  # never block a request
                print(f"[crash-safe] could not recover {directory}: {exc}")
        return True

    def has_object_permission(self, request, view, obj):
        return True


def _recover_project(directory):
    summary = {'rolled_back': 0, 'rolled_forward': 0, 'committed': 0, 'swept': 0}
    now = time.time()
    stuck = []  # (journal folder, error) that this account cannot delete
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
        txn_dir = os.path.join(scan_dir, TXN_DIRNAME)
        if os.path.isdir(txn_dir) and _load_journal_quietly(txn_dir) is None and not is_subject_active(scan_dir):
            stuck.append((txn_dir, PermissionError('could not be deleted or renamed')))
        summary['swept'] += _sweep_stale_files(scan_dir, now)
        summary['swept'] += _sweep_stale_txn_dirs(scan_dir, stuck)
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
    _print_cleanup_help(directory, stuck)
    return summary


def _load_journal_quietly(txn_dir):
    """_load_journal, with an unreadable journal counted as present (it is not garbage)."""
    try:
        return _load_journal(txn_dir)
    except PermissionError:
        return {}


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
    """
    The transaction open on the subject folder that holds `path`, if any. Any thread:
    a worker pool writing a subject's outputs for the transaction's thread is covered
    too (one transaction per subject at a time, see SubjectTransaction).
    """
    key = _key(os.path.dirname(os.path.abspath(path)))
    with _REGISTRY_LOCK:
        entry = _ACTIVE.get(key)
    return entry['txn'] if entry else None


def tracked_replace(src, dst):
    """
    os.replace(src, dst) over an existing subject file, undoable: the open transaction
    keeps the file being replaced (by a rename, not a copy). Outside a transaction it is a
    plain os.replace. atomic_output does this itself.
    """
    txn = _active_txn_for(dst)
    if txn is not None:
        txn.stash(dst)
    _replace_with_retry(src, dst)


def tracked_rename(src, dst):
    """
    os.replace(src, dst) that the open transaction can undo (a renamed original must
    come back under its own name). Outside a transaction it is a plain os.replace.
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
