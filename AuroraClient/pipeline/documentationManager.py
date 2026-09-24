"""
DocumentationManager — load/serve living docs and extract verbatim source by symbol.

The master index lives in pipeline/documentation/docs_index.json.
Source files are resolved relative to the pipeline package directory.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(PIPELINE_DIR, "documentation")
INDEX_PATH = os.path.join(DOCS_DIR, "docs_index.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentationManager:
    """Service class for the Aurora living documentation system."""

    def __init__(self, index_path: str = INDEX_PATH, pipeline_dir: str = PIPELINE_DIR):
        self.index_path = index_path
        self.pipeline_dir = pipeline_dir
        self._index: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Index I/O
    # ------------------------------------------------------------------

    def load_index(self, force: bool = False) -> Dict[str, Any]:
        if self._index is not None and not force:
            return self._index
        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"Documentation index not found: {self.index_path}")
        with open(self.index_path, "r", encoding="utf-8") as f:
            self._index = json.load(f)
        return self._index

    def save_index(self) -> None:
        if self._index is None:
            raise RuntimeError("No index loaded to save")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)
            f.write("\n")

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def _iter_methods(self) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Yield (category, subcategory, method, method_parent_list_ref) for every method."""
        index = self.load_index()
        results = []
        for category in index.get("categories", []):
            for subcategory in category.get("subcategories", []):
                for method in subcategory.get("methods", []):
                    results.append((category, subcategory, method, subcategory["methods"]))
        return results

    def _find_method(self, entry_id: str) -> Optional[Dict[str, Any]]:
        for _cat, _sub, method, _ in self._iter_methods():
            if method.get("id") == entry_id:
                return method
        return None

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @staticmethod
    def _method_has_gif(method: Dict[str, Any]) -> bool:
        """True when any published figure path is a GIF animation."""
        for image in method.get("images") or []:
            if not isinstance(image, dict):
                continue
            path = str(image.get("path") or "").split("?", 1)[0].split("#", 1)[0]
            if path.lower().endswith(".gif"):
                return True
        return False

    def get_index(self) -> Dict[str, Any]:
        """Lightweight nav/search payload (no algorithm/source bodies)."""
        index = self.load_index()
        categories = []
        for category in index.get("categories", []):
            subs = []
            for subcategory in category.get("subcategories", []):
                methods = []
                for method in subcategory.get("methods", []):
                    methods.append({
                        "id": method.get("id"),
                        "title": method.get("title"),
                        "keywords": method.get("keywords", []),
                        "summary": method.get("summary", ""),
                        "hasGif": self._method_has_gif(method),
                    })
                subs.append({
                    "id": subcategory.get("id"),
                    "title": subcategory.get("title"),
                    "methods": methods,
                })
            categories.append({
                "id": category.get("id"),
                "title": category.get("title"),
                "subcategories": subs,
            })
        return {
            "version": index.get("version", 1),
            "title": index.get("title", "Aurora Documentation"),
            "categories": categories,
        }

    def get_entry(self, entry_id: str) -> Dict[str, Any]:
        method = self._find_method(entry_id)
        if method is None:
            raise KeyError(f"Documentation entry not found: {entry_id}")
        # Return a copy without mutating stored hash state.
        entry = deepcopy(method)
        return entry

    def get_corpus(self) -> Dict[str, Any]:
        """
        Full machine-readable documentation corpus from the same docs_index.json root.

        Includes method prose and tutorial transcript segments. Strips live-source
        metadata (paths/hashes) that crawlers and external LLM tools do not need.
        """
        index = self.load_index()
        categories: List[Dict[str, Any]] = []
        method_count = 0
        for category in index.get("categories", []):
            subs: List[Dict[str, Any]] = []
            for subcategory in category.get("subcategories", []):
                methods: List[Dict[str, Any]] = []
                for method in subcategory.get("methods", []):
                    entry = deepcopy(method)
                    entry.pop("source", None)
                    methods.append(entry)
                    method_count += 1
                subs.append({
                    "id": subcategory.get("id"),
                    "title": subcategory.get("title"),
                    "methods": methods,
                })
            categories.append({
                "id": category.get("id"),
                "title": category.get("title"),
                "subcategories": subs,
            })
        return {
            "version": index.get("version", 1),
            "title": index.get("title", "Aurora Documentation"),
            "format": "aurora-docs-corpus",
            "generatedAt": _utc_now_iso(),
            "entryCount": method_count,
            "categories": categories,
        }

    # ------------------------------------------------------------------
    # Source extraction (AST)
    # ------------------------------------------------------------------

    def resolve_source_path(self, relative_file: str) -> str:
        path = os.path.normpath(os.path.join(self.pipeline_dir, relative_file))
        if not path.startswith(os.path.normpath(self.pipeline_dir)):
            raise ValueError(f"Source path escapes pipeline directory: {relative_file}")
        return path

    def _find_ast_node(self, tree: ast.AST, symbol: str) -> Optional[ast.AST]:
        """Locate a class, function, or dotted Class.method node by name."""
        parts = symbol.split(".")
        current_nodes: List[ast.AST] = [tree]

        for i, part in enumerate(parts):
            found = None
            for node in current_nodes:
                body = getattr(node, "body", None)
                if body is None:
                    continue
                for child in body:
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.name == part:
                            found = child
                            break
                if found is not None:
                    break
            if found is None:
                return None
            # For intermediate class parts, search inside the class next.
            if i < len(parts) - 1:
                current_nodes = [found]
            else:
                return found
        return None

    def extract_source(self, relative_file: str, symbol: str) -> Dict[str, Any]:
        """
        Pull the exact verbatim text of a class/function/method from the live file.

        Returns: {source, hash, file, symbol, lineno, end_lineno}
        """
        abs_path = self.resolve_source_path(relative_file)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Source file not found: {relative_file}")

        with open(abs_path, "r", encoding="utf-8") as f:
            source_text = f.read()

        try:
            tree = ast.parse(source_text, filename=abs_path)
        except SyntaxError as exc:
            raise ValueError(f"Failed to parse {relative_file}: {exc}") from exc

        node = self._find_ast_node(tree, symbol)
        if node is None:
            raise KeyError(f"Symbol '{symbol}' not found in {relative_file}")

        segment = ast.get_source_segment(source_text, node)
        if segment is None:
            # Fallback: reconstruct from line numbers if get_source_segment fails.
            lines = source_text.splitlines(keepends=True)
            start = (node.lineno or 1) - 1
            end = node.end_lineno or node.lineno or start + 1
            segment = "".join(lines[start:end])

        # Preserve trailing newline for stable hashing / copy-paste.
        if segment and not segment.endswith("\n"):
            segment = segment + "\n"

        return {
            "source": segment,
            "hash": _sha256_text(segment),
            "file": relative_file,
            "symbol": symbol,
            "lineno": getattr(node, "lineno", None),
            "end_lineno": getattr(node, "end_lineno", None),
        }

    def get_live_source_for_entry(self, entry_id: str) -> Dict[str, Any]:
        method = self._find_method(entry_id)
        if method is None:
            raise KeyError(f"Documentation entry not found: {entry_id}")
        src_meta = method.get("source") or {}
        relative_file = src_meta.get("file")
        symbol = src_meta.get("symbol")
        if not relative_file or not symbol:
            raise ValueError(f"Entry '{entry_id}' is missing source.file / source.symbol")

        live = self.extract_source(relative_file, symbol)
        stored_hash = src_meta.get("hash") or ""
        live["stale"] = (stored_hash != live["hash"]) if stored_hash else True
        live["stored_hash"] = stored_hash
        live["entry_id"] = entry_id
        return live

    # ------------------------------------------------------------------
    # Change tracking
    # ------------------------------------------------------------------

    def check_for_changes(self) -> List[Dict[str, Any]]:
        """
        Re-extract every entry's source and return stale entries.

        Each item: {entry_id, title, file, symbol, old_hash, new_hash, new_source, error?}
        """
        stale: List[Dict[str, Any]] = []
        for _cat, _sub, method, _ in self._iter_methods():
            entry_id = method.get("id")
            src_meta = method.get("source") or {}
            # Curated transcript / non-code entries are content-owned, not AST-hashed.
            if src_meta.get("kind") in ("transcript", "curated"):
                continue
            relative_file = src_meta.get("file")
            symbol = src_meta.get("symbol")
            old_hash = src_meta.get("hash") or ""
            item: Dict[str, Any] = {
                "entry_id": entry_id,
                "title": method.get("title"),
                "file": relative_file,
                "symbol": symbol,
                "old_hash": old_hash,
            }
            try:
                if not relative_file or not symbol:
                    raise ValueError("Missing source.file or source.symbol")
                live = self.extract_source(relative_file, symbol)
                if not old_hash or old_hash != live["hash"]:
                    item["new_hash"] = live["hash"]
                    item["new_source"] = live["source"]
                    item["lineno"] = live.get("lineno")
                    item["end_lineno"] = live.get("end_lineno")
                    stale.append(item)
            except Exception as exc:  # noqa: BLE001 — report per-entry failures to the agent
                item["error"] = str(exc)
                stale.append(item)
        return stale

    def mark_updated(self, entry_id: str, new_hash: Optional[str] = None) -> Dict[str, Any]:
        """
        Update stored hash / lastChecked for an entry after docs prose is refreshed.

        If new_hash is omitted, re-extract live source and store its hash.
        """
        method = self._find_method(entry_id)
        if method is None:
            raise KeyError(f"Documentation entry not found: {entry_id}")

        src_meta = method.setdefault("source", {})
        if new_hash is None:
            relative_file = src_meta.get("file")
            symbol = src_meta.get("symbol")
            if not relative_file or not symbol:
                raise ValueError(f"Entry '{entry_id}' is missing source.file / source.symbol")
            live = self.extract_source(relative_file, symbol)
            new_hash = live["hash"]

        src_meta["hash"] = new_hash
        src_meta["lastChecked"] = _utc_now_iso()
        self.save_index()
        return {
            "entry_id": entry_id,
            "hash": new_hash,
            "lastChecked": src_meta["lastChecked"],
        }

    def seed_hashes(self) -> List[Dict[str, Any]]:
        """Fill missing hashes for all entries from live source. Returns updated entry ids."""
        updated = []
        errors = []
        for _cat, _sub, method, _ in self._iter_methods():
            src_meta = method.setdefault("source", {})
            if src_meta.get("kind") in ("transcript", "curated"):
                continue
            relative_file = src_meta.get("file")
            symbol = src_meta.get("symbol")
            if not relative_file or not symbol:
                continue
            try:
                live = self.extract_source(relative_file, symbol)
            except Exception as exc:  # noqa: BLE001
                errors.append({"entry_id": method.get("id"), "error": str(exc)})
                continue
            src_meta["hash"] = live["hash"]
            src_meta["lastChecked"] = _utc_now_iso()
            updated.append({"entry_id": method.get("id"), "hash": live["hash"]})
        if updated:
            self.save_index()
        if errors:
            # Attach for callers that inspect return; also print for CLI.
            for err in errors:
                print(f"SEED ERROR {err['entry_id']}: {err['error']}")
        return updated
