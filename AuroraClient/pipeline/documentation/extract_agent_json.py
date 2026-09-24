#!/usr/bin/env python3
"""Extract categories JSON from a subagent transcript into batch_fragments/."""
import json, sys, pathlib

def extract(path):
    best = None
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("role") != "assistant":
            continue
        for part in rec.get("message", {}).get("content", []):
            if part.get("type") != "text":
                continue
            text = part.get("text", "")
            start = text.find("{")
            if start < 0:
                continue
            depth = 0
            in_str = False
            esc = False
            end = None
            for i, ch in enumerate(text[start:], start):
                if in_str:
                    if esc: esc = False
                    elif ch == "\\": esc = True
                    elif ch == '"': in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                continue
            try:
                data = json.loads(text[start:end])
            except Exception:
                continue
            if "categories" not in data:
                continue
            n = sum(len(s.get("methods", [])) for c in data["categories"] for s in c.get("subcategories", []))
            if best is None or n >= best[0]:
                best = (n, data)
    return best

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    best = extract(src)
    if not best:
        print("FAIL", src)
        sys.exit(1)
    pathlib.Path(dst).write_text(json.dumps(best[1], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OK {dst} methods={best[0]}")
