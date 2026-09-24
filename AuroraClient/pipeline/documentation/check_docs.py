#!/usr/bin/env python3
"""
CLI for Aurora living-documentation change tracking.

Usage (from anywhere):
  python check_docs.py                  # report stale entries
  python check_docs.py --seed           # fill missing hashes from live source
  python check_docs.py --mark <id>      # mark one entry current after prose update
  python check_docs.py --extract <id>   # print verbatim live source for one entry
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Allow running as a script without installing the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE = os.path.dirname(_HERE)
_AURORA_CLIENT = os.path.dirname(_PIPELINE)
if _AURORA_CLIENT not in sys.path:
    sys.path.insert(0, _AURORA_CLIENT)

from pipeline.documentationManager import DocumentationManager  # noqa: E402


def _walk_methods(node, out=None):
    if out is None:
        out = []
    if isinstance(node, dict):
        if "algorithm" in node and isinstance(node.get("algorithm"), dict) and node.get("id"):
            out.append(node)
        for v in node.values():
            _walk_methods(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_methods(v, out)
    return out


def _lint_unique_ids(index_path: str) -> list:
    """
    Every category / subcategory / method id must be unique across the whole tree.

    Shared titles are fine; shared ids are not. A flat id→number map (and related-link
    resolution) collapses when a subcategory reuses a method slug — e.g. both
    "mesh-elastic-registration" made TOC section 7.4 display as 7.4.2.
    """
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen: dict[str, str] = {}
    errors: list[str] = []

    def claim(node_id: str, kind: str, title: str, trail: str) -> None:
        if not node_id:
            errors.append(f"{trail}: missing {kind} id")
            return
        loc = f"{kind} {trail!r} title={title!r}"
        if node_id in seen:
            errors.append(f"duplicate id {node_id!r}: {seen[node_id]} vs {loc}")
        else:
            seen[node_id] = loc

    for cat in data.get("categories") or []:
        cid = cat.get("id") or ""
        claim(cid, "category", cat.get("title") or "", cid or "?")
        for sub in cat.get("subcategories") or []:
            sid = sub.get("id") or ""
            trail = f"{cid}/{sid or '?'}"
            claim(sid, "subcategory", sub.get("title") or "", trail)
            for method in sub.get("methods") or []:
                mid = method.get("id") or ""
                claim(mid, "method", method.get("title") or "", f"{trail}/{mid or '?'}")
    return errors


def _lint_math(index_path: str) -> list:
    """Soft quality checks for algorithm.math / references (does not fail CI by itself)."""
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    warnings = []
    for method in _walk_methods(data):
        mid = method.get("id", "?")
        alg = method.get("algorithm") or {}
        math = alg.get("math")
        text = alg.get("text") or ""
        refs = alg.get("references")

        if math not in (None, "", []):
            if isinstance(math, str):
                warnings.append(
                    f"{mid}: algorithm.math is a legacy string — prefer "
                    '[{"equation": "...", "caption": "symbol: meaning; ..."}] or []'
                )
            elif not isinstance(math, list):
                warnings.append(f"{mid}: algorithm.math must be a list or empty string")
            else:
                for i, item in enumerate(math):
                    if isinstance(item, str):
                        warnings.append(
                            f"{mid} math[{i}]: bare string — use {{equation, caption}} objects"
                        )
                        continue
                    if not isinstance(item, dict):
                        warnings.append(f"{mid} math[{i}]: expected object with equation/caption")
                        continue
                    eq = (item.get("equation") or item.get("tex") or "").strip()
                    cap = (item.get("caption") or "").strip()
                    if not eq:
                        warnings.append(f"{mid} math[{i}]: missing equation")
                        continue
                    if not cap:
                        warnings.append(
                            f"{mid} math[{i}]: missing caption (define each symbol/term in small prose)"
                        )
                    # Parallel assignments on one line: "a = …, b = …" (not f(x, y) = … / order=0 kwargs)
                    if re.search(r"=[^,;]{1,80},\s*[A-Za-z\\][A-Za-z0-9_{}\\^]*\s*=", eq):
                        warnings.append(
                            f"{mid} math[{i}]: put each equation on its own array entry "
                            "(avoid comma-separated parallel equations on one line)"
                        )

        if refs in (None, []):
            continue
        if not isinstance(refs, list):
            warnings.append(f"{mid}: algorithm.references must be a list")
            continue
        if "[[references]]" not in text:
            warnings.append(
                f"{mid}: algorithm.references is set but algorithm.text lacks [[references]] "
                "placement marker (grid will fall back to end of Algorithm)"
            )
        for i, item in enumerate(refs):
            if not isinstance(item, dict):
                warnings.append(f"{mid} references[{i}]: expected object")
                continue
            if not (item.get("label") or "").strip():
                warnings.append(f"{mid} references[{i}]: missing label")
            url = (item.get("url") or "").strip()
            entry_id = (item.get("entryId") or "").strip()
            if not url and not entry_id:
                warnings.append(
                    f"{mid} references[{i}]: need url (external docs) or entryId (in-house docs entry)"
                )
            if url and not url.startswith(("http://", "https://")):
                warnings.append(f"{mid} references[{i}]: url should be an absolute http(s) link")
            if not (item.get("library") or "").strip():
                warnings.append(f"{mid} references[{i}]: missing library badge (e.g. scikit-image or Aurora (in-house))")
            if not (item.get("symbol") or "").strip():
                warnings.append(f"{mid} references[{i}]: missing symbol (package.module.function)")
    return warnings


def _print_stale(stale):
    if not stale:
        print("OK — all documented entries match live source hashes.")
        return
    print(f"STALE — {len(stale)} entr{'y' if len(stale) == 1 else 'ies'} need attention:\n")
    for item in stale:
        print("=" * 72)
        print(f"id:       {item.get('entry_id')}")
        print(f"title:    {item.get('title')}")
        print(f"file:     {item.get('file')}")
        print(f"symbol:   {item.get('symbol')}")
        if item.get("error"):
            print(f"ERROR:    {item['error']}")
            continue
        print(f"old_hash: {item.get('old_hash') or '(none)'}")
        print(f"new_hash: {item.get('new_hash')}")
        if item.get("lineno") is not None:
            print(f"lines:    {item.get('lineno')}-{item.get('end_lineno')}")
        print("--- new_source ---")
        print(item.get("new_source") or "")
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aurora documentation change tracker")
    parser.add_argument("--seed", action="store_true", help="Fill/refresh all stored hashes from live source")
    parser.add_argument("--mark", metavar="ENTRY_ID", help="Mark one entry current (re-extract hash)")
    parser.add_argument("--extract", metavar="ENTRY_ID", help="Print live verbatim source for one entry")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON for stale report")
    parser.add_argument(
        "--lint-math",
        action="store_true",
        help="Lint algorithm.math / references (captions, one eq per line, [[references]] marker)",
    )
    parser.add_argument(
        "--lint-ids",
        action="store_true",
        help="Fail if any category/subcategory/method id is duplicated in docs_index.json",
    )
    args = parser.parse_args(argv)

    mgr = DocumentationManager()
    index_path = getattr(mgr, "index_path", None) or os.path.join(_HERE, "docs_index.json")

    if args.lint_math:
        warnings = _lint_math(index_path)
        if not warnings:
            print("OK — algorithm.math lint clean.")
            return 0
        print(f"MATH LINT — {len(warnings)} warning(s):\n")
        for w in warnings:
            print(f"  - {w}")
        return 1

    if args.lint_ids:
        errors = _lint_unique_ids(index_path)
        if not errors:
            print("OK — all documentation ids are unique.")
            return 0
        print(f"ID LINT — {len(errors)} error(s):\n")
        for err in errors:
            print(f"  - {err}")
        return 1

    if args.seed:
        updated = mgr.seed_hashes()
        print(f"Seeded {len(updated)} entr{'y' if len(updated) == 1 else 'ies'}.")
        for u in updated:
            print(f"  {u['entry_id']}: {u['hash'][:12]}...")
        return 0

    if args.mark:
        result = mgr.mark_updated(args.mark)
        print(f"Marked {result['entry_id']} current @ {result['lastChecked']}")
        print(f"hash: {result['hash']}")
        return 0

    if args.extract:
        live = mgr.get_live_source_for_entry(args.extract)
        print(f"# {live['file']} :: {live['symbol']}  (lines {live.get('lineno')}-{live.get('end_lineno')})")
        print(f"# hash={live['hash']}  stale={live['stale']}")
        print(live["source"])
        return 0

    id_errors = _lint_unique_ids(index_path)
    if id_errors:
        print(f"ID LINT — {len(id_errors)} error(s) (unique ids required):\n")
        for err in id_errors:
            print(f"  - {err}")
        if not args.json:
            return 1

    stale = mgr.check_for_changes()
    if args.json:
        payload = {"id_errors": id_errors, "stale": stale}
        print(json.dumps(payload, indent=2))
    else:
        _print_stale(stale)
        # Soft math warnings (non-fatal) so agents see them while updating docs.
        warnings = _lint_math(index_path)
        if warnings:
            print(f"\nMATH LINT — {len(warnings)} warning(s) (non-fatal; fix when editing prose):")
            for w in warnings[:25]:
                print(f"  - {w}")
            if len(warnings) > 25:
                print(f"  … {len(warnings) - 25} more (run --lint-math)")
    if id_errors:
        return 1
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
