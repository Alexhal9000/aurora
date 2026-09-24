"""
Helpers for linked_to metadata groups used by batch filtering and edit replay.

Semantics (same as LinkScansView):
  - Main: linked_to equals its own name
  - Child: linked_to equals the main scan's name
  - Unlinked: field absent
"""
import json
import os

import nibabel as nib

from .atlas_paths import resolve_atlas_dir


def metadata_path(directory, scan_name):
    if scan_name == "atlas":
        return os.path.join(resolve_atlas_dir(directory), "atlas.json")
    return os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")


def full_res_nifti_path(directory, scan_name):
    if scan_name == "atlas":
        return os.path.join(resolve_atlas_dir(directory), "atlas.nii.gz")
    return os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")


def load_scan_metadata(directory, scan_name):
    path = metadata_path(directory, scan_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as jf:
            return json.load(jf)
    except (OSError, json.JSONDecodeError):
        return None


def is_linked_child(directory, scan_name):
    """True when linked_to is set and points at another scan (not self)."""
    metadata = load_scan_metadata(directory, scan_name)
    if not metadata:
        return False
    linked_to = metadata.get("linked_to")
    if not linked_to or not isinstance(linked_to, str):
        return False
    return linked_to != scan_name


def is_linked_main(directory, scan_name):
    metadata = load_scan_metadata(directory, scan_name)
    if not metadata:
        return False
    linked_to = metadata.get("linked_to")
    return isinstance(linked_to, str) and linked_to == scan_name


def filter_out_linked_children(directory, scan_names):
    """
    Batch discovery helper: keep unlinked scans and link mains; drop linked children.
    """
    if not directory or not scan_names:
        return list(scan_names or [])
    return [name for name in scan_names if not is_linked_child(directory, name)]


def read_full_res_dims(directory, scan_name):
    nifti_path = full_res_nifti_path(directory, scan_name)
    if not os.path.isfile(nifti_path):
        return None, f"NIfTI not found for {scan_name}"
    try:
        shape = nib.load(nifti_path).header.get_data_shape()
        return [int(v) for v in shape[:3]], None
    except Exception as e:
        return None, f"Failed to read dims for {scan_name}: {e}"


def get_link_group(directory, scan_name):
    """
    Resolve the link group for scan_name.

    Returns dict:
      {
        'main': str,
        'members': [str, ...],  # main first, then children sorted
        'children': [str, ...],
      }
    or None if the scan is unlinked / metadata missing.
    """
    metadata = load_scan_metadata(directory, scan_name)
    if not metadata:
        return None
    linked_to = metadata.get("linked_to")
    if not linked_to or not isinstance(linked_to, str):
        return None

    main = linked_to if linked_to != scan_name else scan_name
    extracted = os.path.join(directory, "extracted")
    members = set()
    if os.path.isdir(extracted):
        for name in os.listdir(extracted):
            if name == "project_settings.json" or not os.path.isdir(os.path.join(extracted, name)):
                continue
            meta = load_scan_metadata(directory, name)
            if not meta:
                continue
            lt = meta.get("linked_to")
            if lt == main:
                members.add(name)
    members.add(main)
    children = sorted(m for m in members if m != main)
    ordered = [main] + children
    if len(ordered) < 2:
        return None
    return {"main": main, "members": ordered, "children": children}


def latest_edit_stem(directory, scan_name):
    """
    Filename stem of a scan's latest edit ('<name>_edit_<n>_<suffix>'), or the scan name
    itself when it has no edits yet. Returns None when the latest edit is elastic, since
    elastic results are never part of linked replay.
    """
    import glob

    scan_dir = os.path.join(directory, "extracted", scan_name)
    edit_number = 0
    while glob.glob(os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_*.nii.gz")):
        edit_number += 1
    edit_number -= 1
    if edit_number < 0:
        return scan_name

    matches = [
        path for path in glob.glob(os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_*.nii.gz"))
        if not path.endswith(('.nii.mask.gz', '.nii.removal_mask.gz'))
        and not path.endswith(('_mask_temp.nii.gz', '_mask.nii.gz', '_inv.nii.gz', '_fwd.nii.gz'))
        and 'elastic' not in os.path.basename(path).lower()
    ]
    if not matches:
        return scan_name
    stem = os.path.basename(sorted(matches, key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))[0]).replace('.nii.gz', '')
    if 'elastic' in stem.lower():
        return None
    return stem


def get_same_shape_siblings(directory, source_name, member_names=None):
    """
    Return members of the link group (excluding source) that share source's full-res dims.
    """
    group = get_link_group(directory, source_name)
    if not group:
        return [], None
    names = member_names if member_names is not None else group["members"]
    source_dims, err = read_full_res_dims(directory, source_name)
    if err:
        return [], err
    siblings = []
    for name in names:
        if name == source_name:
            continue
        dims, dim_err = read_full_res_dims(directory, name)
        if dim_err or dims != source_dims:
            continue
        siblings.append(name)
    return siblings, None
