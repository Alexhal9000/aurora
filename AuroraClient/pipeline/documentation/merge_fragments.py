#!/usr/bin/env python3
"""Merge batch JSON fragments into docs_index.json and optionally seed hashes."""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(_HERE)
_AURORA_CLIENT = os.path.dirname(_PIPELINE)
if _AURORA_CLIENT not in sys.path:
    sys.path.insert(0, _AURORA_CLIENT)

INDEX_PATH = os.path.join(_HERE, "docs_index.json")
FRAGMENTS_DIR = os.path.join(_HERE, "batch_fragments")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _index_methods(index):
    ids = set()
    for cat in index.get("categories", []):
        for sub in cat.get("subcategories", []):
            for m in sub.get("methods", []):
                ids.add(m.get("id"))
    return ids


def _find_category(index, cat_id):
    for cat in index.get("categories", []):
        if cat.get("id") == cat_id:
            return cat
    return None


def _find_sub(cat, sub_id):
    for sub in cat.get("subcategories", []):
        if sub.get("id") == sub_id:
            return sub
    return None


def merge_fragment(index, fragment):
    """Merge fragment categories into index. Skip duplicate method ids."""
    existing = _index_methods(index)
    added = []
    skipped = []
    for cat in fragment.get("categories", []):
        target_cat = _find_category(index, cat["id"])
        if target_cat is None:
            index.setdefault("categories", []).append({
                "id": cat["id"],
                "title": cat.get("title", cat["id"]),
                "subcategories": [],
            })
            target_cat = _find_category(index, cat["id"])
        else:
            if cat.get("title"):
                target_cat["title"] = cat["title"]

        for sub in cat.get("subcategories", []):
            target_sub = _find_sub(target_cat, sub["id"])
            if target_sub is None:
                target_cat.setdefault("subcategories", []).append({
                    "id": sub["id"],
                    "title": sub.get("title", sub["id"]),
                    "methods": [],
                })
                target_sub = _find_sub(target_cat, sub["id"])
            else:
                if sub.get("title"):
                    target_sub["title"] = sub["title"]

            for method in sub.get("methods", []):
                mid = method.get("id")
                if not mid:
                    continue
                if mid in existing:
                    skipped.append(mid)
                    continue
                # Ensure required source fields
                src = method.setdefault("source", {})
                src.setdefault("hash", "")
                src.setdefault("lastChecked", None)
                method.setdefault("related", [])
                method.setdefault("video", None)
                method.setdefault("images", [])
                method.setdefault("options", [])
                method.setdefault(
                    "algorithm",
                    {"text": "", "math": [], "references": [], "diagram": ""},
                )
                target_sub["methods"].append(method)
                existing.add(mid)
                added.append(mid)
    return added, skipped


def apply_related(index, related_map):
    """related_map: {entry_id: [other_ids]} — merge unique related links."""
    methods = {}
    for cat in index.get("categories", []):
        for sub in cat.get("subcategories", []):
            for m in sub.get("methods", []):
                methods[m["id"]] = m
    for mid, links in related_map.items():
        if mid not in methods:
            continue
        cur = methods[mid].setdefault("related", [])
        for link in links:
            if link in methods and link != mid and link not in cur:
                cur.append(link)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragments-dir", default=FRAGMENTS_DIR)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--files", nargs="*", help="Specific fragment JSON files")
    args = parser.parse_args()

    index = _load(INDEX_PATH)
    files = args.files
    if not files:
        if not os.path.isdir(args.fragments_dir):
            print(f"No fragments dir: {args.fragments_dir}")
            return 1
        files = sorted(
            os.path.join(args.fragments_dir, n)
            for n in os.listdir(args.fragments_dir)
            if n.endswith(".json")
        )

    all_added = []
    all_skipped = []
    for path in files:
        frag = _load(path)
        added, skipped = merge_fragment(index, frag)
        print(f"{os.path.basename(path)}: +{len(added)} skip{len(skipped)}")
        all_added.extend(added)
        all_skipped.extend(skipped)

    _save(INDEX_PATH, index)
    print(f"Saved {INDEX_PATH}")
    print(f"Total added={len(all_added)} skipped_dupes={len(all_skipped)}")
    print(f"Method count now={len(_index_methods(index))}")

    if args.seed:
        from pipeline.documentationManager import DocumentationManager
        mgr = DocumentationManager()
        updated = mgr.seed_hashes()
        print(f"Seeded {len(updated)} hashes")
        stale = mgr.check_for_changes()
        errors = [s for s in stale if s.get("error")]
        print(f"After seed: stale_or_error={len(stale)} errors={len(errors)}")
        for e in errors:
            print(f"  ERROR {e.get('entry_id')}: {e.get('error')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
