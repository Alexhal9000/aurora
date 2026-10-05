"""Per-edit reverse maps from stored NIfTI values back to original intensities.

Extraction records ``original = stored * mapping_scale + mapping_shift``.
Each later edit stores its own copy of that pair, keyed by the lossy filename.
Edits that do not change intensities copy the previous pair. An edit that
applies ``stored_new = stored_scale * stored_old + stored_offset`` composes
the pair so the same formula still recovers the original value.

A bare dict (not a list) is a legacy project: one pair for every edit.
"""

import os


def compose_intensity_mapping(previous, stored_scale=1.0, stored_offset=0.0):
    """Return a new factor dict after ``stored_new = stored_scale * stored_old + stored_offset``."""
    entry = dict(previous) if isinstance(previous, dict) else {}
    try:
        scale = float(stored_scale)
    except (TypeError, ValueError):
        scale = 1.0
    try:
        offset = float(stored_offset)
    except (TypeError, ValueError):
        offset = 0.0
    try:
        old_scale = float(entry.get('mapping_scale', 0.0))
    except (TypeError, ValueError):
        old_scale = 0.0
    try:
        old_shift = float(entry.get('mapping_shift', 0.0))
    except (TypeError, ValueError):
        old_shift = 0.0
    if scale == 0.0:
        new_scale = old_scale
        new_shift = old_shift
    else:
        new_scale = old_scale / scale
        new_shift = old_shift - offset * new_scale
    entry['mapping_scale'] = float(new_scale)
    entry['mapping_shift'] = float(new_shift)
    entry['formula'] = 'original = stored * mapping_scale + mapping_shift'
    entry.pop('filename', None)
    return entry


def resolve_intensity_value_mapping(mapping, filename=None):
    """Factor dict for one lossy file.

    A dict is legacy and applies to every edit. A list is matched by filename.
    When the filename is missing, or this volume has no row yet, the raw
    extraction row is used so older edits keep the single-map behavior.
    """
    if isinstance(mapping, dict):
        return mapping
    if not isinstance(mapping, list):
        return None
    entries = [entry for entry in mapping if isinstance(entry, dict)]
    if not entries:
        return None
    if filename:
        base = os.path.basename(str(filename))
        for entry in entries:
            if entry.get('filename') == base:
                return entry
    for entry in entries:
        name = str(entry.get('filename') or '')
        if '_edit_' not in name:
            return entry
    return entries[0]


def _raw_lossy_filename(metadata):
    name = ''
    if isinstance(metadata, dict):
        name = str(metadata.get('name') or '').strip()
    if not name:
        name = 'scan'
    return f"{name}_lossy.nii.gz"


def record_intensity_value_mapping(metadata, lossy_filename, stored_scale=1.0, stored_offset=0.0):
    """Append or replace the factor row for ``lossy_filename``.

    No-op when the subject has no mapping. A legacy dict is promoted to a list:
    the original pair stays on the raw volume, and this edit gets the composed pair.
    Returns True when ``metadata`` was changed.
    """
    if not isinstance(metadata, dict):
        return False
    mapping = metadata.get('intensity_value_mapping')
    if not mapping:
        return False
    filename = os.path.basename(str(lossy_filename or ''))
    if not filename:
        return False

    # Rewriting the raw lossy file replaces the list with that one row.
    # The atlas copies a reference subject's rows and then saves atlas_lossy;
    # those foreign edit names must not stay attached to the atlas.
    raw_name = _raw_lossy_filename(metadata)
    if filename == raw_name:
        base = resolve_intensity_value_mapping(mapping)
        if not isinstance(base, dict):
            return False
        edited = compose_intensity_mapping(base, stored_scale, stored_offset)
        edited['filename'] = filename
        metadata['intensity_value_mapping'] = [edited]
        return True

    if isinstance(mapping, dict):
        raw_entry = dict(mapping)
        raw_entry['filename'] = raw_name
        edited = compose_intensity_mapping(mapping, stored_scale, stored_offset)
        edited['filename'] = filename
        metadata['intensity_value_mapping'] = [raw_entry, edited]
        return True

    if not isinstance(mapping, list):
        return False

    parent = None
    for entry in mapping:
        if isinstance(entry, dict) and entry.get('filename') != filename:
            parent = entry
    if parent is None:
        parent = next((entry for entry in mapping if isinstance(entry, dict)), None)
    if parent is None:
        return False

    edited = compose_intensity_mapping(parent, stored_scale, stored_offset)
    edited['filename'] = filename
    kept = [
        entry for entry in mapping
        if not (isinstance(entry, dict) and entry.get('filename') == filename)
    ]
    kept.append(edited)
    metadata['intensity_value_mapping'] = kept
    return True


def remove_intensity_mappings_for_filenames(metadata, filenames):
    """Drop list rows whose filename is in ``filenames``. Legacy dicts are left as-is."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get('intensity_value_mapping'), list):
        return False
    names = {os.path.basename(str(name)) for name in (filenames or []) if name}
    if not names:
        return False
    mapping = metadata['intensity_value_mapping']
    kept = [
        entry for entry in mapping
        if not (isinstance(entry, dict) and entry.get('filename') in names)
    ]
    if len(kept) == len(mapping):
        return False
    metadata['intensity_value_mapping'] = kept
    return True


def drop_intensity_mappings_containing(metadata, substring):
    """Drop list rows whose filename contains ``substring``. Legacy dicts are left as-is."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get('intensity_value_mapping'), list):
        return False
    needle = str(substring or '')
    if not needle:
        return False
    mapping = metadata['intensity_value_mapping']
    kept = [
        entry for entry in mapping
        if not (isinstance(entry, dict) and needle in str(entry.get('filename') or ''))
    ]
    if len(kept) == len(mapping):
        return False
    metadata['intensity_value_mapping'] = kept
    return True


def rename_intensity_mapping_filenames(metadata, old, new):
    """Rewrite list filenames that contain ``old``. Legacy dicts are left as-is."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get('intensity_value_mapping'), list):
        return False
    old_s = str(old or '')
    new_s = str(new or '')
    if not old_s or old_s == new_s:
        return False
    changed = False
    for entry in metadata['intensity_value_mapping']:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get('filename') or '')
        if old_s in filename:
            entry['filename'] = filename.replace(old_s, new_s)
            changed = True
    return changed
