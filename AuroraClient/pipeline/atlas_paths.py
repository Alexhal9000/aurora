import os
import json
import shutil

# Subject name used after promoting population atlas into extracted/ as the project reference.
REFERENCE_ATLAS_SUBJECT = "ReferenceAtlas"


def resolve_atlas_dir(directory):
    """Filesystem path to the population atlas folder (top-level only)."""
    return os.path.join(directory, "atlas")


def atlas_sub_path(directory):
    """Relative path segment for the population atlas (not promoted reference)."""
    return "atlas"


def legacy_atlas_dir(directory):
    """Top-level atlas folder used when creating a new atlas."""
    return os.path.join(directory, "atlas")


def reference_atlas_dir(directory):
    """Filesystem path to promoted atlas living in extracted/ as a normal subject."""
    return os.path.join(directory, "extracted", REFERENCE_ATLAS_SUBJECT)


def is_reference_atlas_promoted(directory):
    return os.path.isdir(reference_atlas_dir(directory))


def atlas_filename_to_reference_atlas(filename):
    """Map atlas-prefixed filenames to ReferenceAtlas-prefixed names."""
    if filename == "atlas.json":
        return f"{REFERENCE_ATLAS_SUBJECT}.json"
    if filename.startswith("atlas_"):
        return REFERENCE_ATLAS_SUBJECT + filename[5:]
    if filename.startswith("atlas."):
        return REFERENCE_ATLAS_SUBJECT + filename[5:]
    return filename


def replace_atlas_token_in_string(value):
    """Replace atlas subject tokens inside metadata strings."""
    if not isinstance(value, str):
        return value
    if value == "atlas":
        return REFERENCE_ATLAS_SUBJECT
    if value.startswith("atlas_"):
        return REFERENCE_ATLAS_SUBJECT + value[5:]
    if value.startswith("atlas."):
        return REFERENCE_ATLAS_SUBJECT + value[5:]
    return value


def deep_replace_atlas_tokens(obj):
    if isinstance(obj, dict):
        return {k: deep_replace_atlas_tokens(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_replace_atlas_tokens(v) for v in obj]
    if isinstance(obj, str):
        return replace_atlas_token_in_string(obj)
    return obj


def rename_atlas_files_in_folder(folder_path):
    """Rename atlas-prefixed files in folder to ReferenceAtlas-prefixed names."""
    renames = []
    for entry in os.listdir(folder_path):
        old_path = os.path.join(folder_path, entry)
        if not os.path.isfile(old_path):
            continue
        new_name = atlas_filename_to_reference_atlas(entry)
        if new_name == entry:
            continue
        new_path = os.path.join(folder_path, new_name)
        if os.path.exists(new_path):
            raise FileExistsError(f"Cannot rename {entry} to {new_name}: destination exists")
        renames.append((old_path, new_path))
    for old_path, new_path in renames:
        os.rename(old_path, new_path)


def read_selected_reference(directory):
    """Return selected_reference from extracted/project_settings.json, or ''."""
    project_settings_path = os.path.join(directory, "extracted", "project_settings.json")
    if not os.path.exists(project_settings_path):
        return ""
    try:
        with open(project_settings_path, "r") as f:
            settings = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""
    value = settings.get("selected_reference", "")
    return value if isinstance(value, str) else ""


def update_json_metadata_files(folder_path, previous_reference=None):
    """Rewrite JSON metadata after file renames (edits lists, name field, etc.).

    When promoting, ``previous_reference`` is the project reference that was
    active before the atlas became ReferenceAtlas (for Scientific Report).
    """
    for entry in os.listdir(folder_path):
        if not entry.endswith(".json"):
            continue
        json_path = os.path.join(folder_path, entry)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        data = deep_replace_atlas_tokens(data)
        if entry == f"{REFERENCE_ATLAS_SUBJECT}.json":
            data["name"] = REFERENCE_ATLAS_SUBJECT
            data["reference"] = True
            data["promoted_from_atlas"] = True
            # Record once; keep existing value on idempotent re-finalize.
            existing_previous = data.get("previous_reference")
            if (
                not existing_previous
                and previous_reference
                and previous_reference != REFERENCE_ATLAS_SUBJECT
            ):
                data["previous_reference"] = previous_reference
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)


def write_project_reference(directory, reference_name):
    """Persist selected_reference in extracted/project_settings.json."""
    from .coordinateFrames import load_project_settings, save_project_settings

    settings = load_project_settings(directory)
    settings["selected_reference"] = reference_name
    return save_project_settings(directory, settings)


def delete_stale_average_files(folder_path):
    """Remove intermediate atlas build artifacts (filenames containing 'average')."""
    removed = []
    for entry in os.listdir(folder_path):
        if "average" not in entry.lower():
            continue
        path = os.path.join(folder_path, entry)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(entry)
    return removed


def copy_reference_atlas_nifti_to_project_root(directory, subject_dir=None):
    """
    Copy ReferenceAtlas.nii.gz to the project root so the file browser can pair
    root-level raw NIfTI with extracted/ReferenceAtlas/ (raw + extracted available).
    """
    subject_dir = subject_dir or reference_atlas_dir(directory)
    src = os.path.join(subject_dir, f"{REFERENCE_ATLAS_SUBJECT}.nii.gz")
    dst = os.path.join(directory, f"{REFERENCE_ATLAS_SUBJECT}.nii.gz")
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"{REFERENCE_ATLAS_SUBJECT}.nii.gz not found in {subject_dir}"
        )
    shutil.copy2(src, dst)
    return dst


def finalize_promoted_reference_atlas(directory):
    """
    Run post-move steps: rename leftovers, drop stale averages, refresh JSON,
    copy root pairing NIfTI, and persist selected_reference.

    Captures the prior selected_reference into ReferenceAtlas.json as
    ``previous_reference`` before overwriting project settings.
    """
    dst_dir = reference_atlas_dir(directory)
    if not os.path.isdir(dst_dir):
        raise FileNotFoundError(f"Promoted subject folder not found: {dst_dir}")

    previous_reference = read_selected_reference(directory)

    rename_atlas_files_in_folder(dst_dir)
    removed_averages = delete_stale_average_files(dst_dir)
    update_json_metadata_files(dst_dir, previous_reference=previous_reference)
    root_nifti_path = copy_reference_atlas_nifti_to_project_root(directory, dst_dir)
    project_settings_path = write_project_reference(directory, REFERENCE_ATLAS_SUBJECT)

    stored_previous = None
    meta_path = os.path.join(dst_dir, f"{REFERENCE_ATLAS_SUBJECT}.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            value = meta.get("previous_reference")
            if isinstance(value, str) and value and value != REFERENCE_ATLAS_SUBJECT:
                stored_previous = value
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "subject_dir": dst_dir,
        "removed_average_files": removed_averages,
        "root_nifti_path": root_nifti_path,
        "project_settings_path": project_settings_path,
        "previous_reference": stored_previous,
    }
