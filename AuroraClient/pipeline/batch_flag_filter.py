"""
Apply metadata `flag` filter to batch subject lists after standard exclusions.

The `flag` key is read from each subject's JSON (same pattern as `faulty` in the UI).
"""
import json
import os


def load_flagged_subject_names(directory):
    """Return a set of extracted subject folder names where metadata ``flag`` is true."""
    flagged = set()
    extracted = os.path.join(directory, "extracted")
    if not os.path.isdir(extracted):
        return flagged
    for name in os.listdir(extracted):
        if name == "project_settings.json" or not os.path.isdir(os.path.join(extracted, name)):
            continue
        json_path = os.path.join(extracted, name, f"{name}.json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                metadata = json.load(jf)
            if metadata.get("flag", False) is True:
                flagged.add(name)
        except (OSError, json.JSONDecodeError):
            continue
    return flagged


def apply_flag_filter(scan_names, flagged_set, flag_filter):
    """
    Apply after standard batch exclusions (faulty, completed tool, elastic, etc.).

    flag_filter: ``'off'`` | ``'exclude'`` | ``'only'``
    """
    if not flag_filter or flag_filter == "off":
        return list(scan_names)
    if flag_filter == "exclude":
        return [s for s in scan_names if s not in flagged_set]
    if flag_filter == "only":
        return [s for s in scan_names if s in flagged_set]
    return list(scan_names)


def normalize_flag_filter_value(raw, only_current_scan=False):
    """Validate API value; force ``off`` when batch scope is single-scan only."""
    ff = raw if raw in ("off", "exclude", "only") else "off"
    if only_current_scan:
        return "off"
    return ff
