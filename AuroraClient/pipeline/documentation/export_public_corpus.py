#!/usr/bin/env python3
"""
Publish the Aurora docs corpus for public crawlers / LLM tools.

Uses DocumentationManager.get_corpus() (same root as /docs/corpus/) and writes:

  - aurora-docs-corpus.json  (machine-readable)
  - aurora-docs-corpus.html  (crawlable prose; preferred for Perplexity @-source)
  - aurora-docs-corpus.txt   (plain text for LLM ingestion)

Served by the BHTools frontend host, e.g.:

  https://www.hallgrimssonlab.ca/static/frontend/aurora-docs-corpus.html
  https://www.hallgrimssonlab.ca/static/frontend/aurora-docs-corpus.txt

Run from the BHTools repo root (or any cwd):

  python3 Aurora/AuroraClient/pipeline/documentation/export_public_corpus.py
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_HERE)
_AURORA_CLIENT_DIR = os.path.dirname(_PIPELINE_DIR)

if _AURORA_CLIENT_DIR not in sys.path:
    sys.path.insert(0, _AURORA_CLIENT_DIR)

from pipeline.documentationManager import DocumentationManager  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(_AURORA_CLIENT_DIR, "..", ".."))
_DEFAULT_OUT_DIR = os.path.join(
    _REPO_ROOT,
    "BHTools",
    "frontend",
    "static",
    "frontend",
)

PUBLIC_ORIGIN = "https://www.hallgrimssonlab.ca"
CORPUS_HTML_URL = f"{PUBLIC_ORIGIN}/static/frontend/aurora-docs-corpus.html"
CORPUS_JSON_URL = f"{PUBLIC_ORIGIN}/static/frontend/aurora-docs-corpus.json"
CORPUS_TXT_URL = f"{PUBLIC_ORIGIN}/static/frontend/aurora-docs-corpus.txt"
CORPUS_SPA_URL = f"{PUBLIC_ORIGIN}/MainAurora/documentation"
META_DESCRIPTION = (
    "Aurora image processing documentation for AI assistants and search engines."
)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _plain(value: Any) -> str:
    return "" if value is None else str(value)


def _text_chunks(text: Any) -> List[str]:
    """Normalize summary/algorithm fields into plain-text paragraphs."""
    if isinstance(text, dict):
        chunks: List[str] = []
        for key in ("text", "math", "references", "diagram"):
            val = text.get(key)
            if key == "text" and isinstance(val, str) and val.strip():
                chunks.append(val.replace("[[references]]", "").strip())
            elif isinstance(val, str) and val.strip():
                chunks.append(val.strip())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.strip():
                        chunks.append(item.strip())
                    elif isinstance(item, dict):
                        if key == "references":
                            label = (item.get("label") or "").strip()
                            lib = (item.get("library") or "").strip()
                            sym = (item.get("symbol") or "").strip()
                            url = (item.get("url") or "").strip()
                            entry_id = (item.get("entryId") or "").strip()
                            blurb = (item.get("blurb") or "").strip()
                            line = " — ".join(p for p in (label, lib, sym) if p)
                            if blurb:
                                line = f"{line}: {blurb}" if line else blurb
                            if url:
                                line = f"{line} ({url})" if line else url
                            elif entry_id:
                                line = f"{line} [docs:{entry_id}]" if line else f"[docs:{entry_id}]"
                            if line:
                                chunks.append(line)
                            continue
                        eq = (item.get("equation") or item.get("tex") or "").strip()
                        cap = (item.get("caption") or "").strip()
                        if eq and cap:
                            chunks.append(f"{eq}\n{cap}")
                        elif eq:
                            chunks.append(eq)
                        elif cap:
                            chunks.append(cap)
        body = "\n\n".join(chunks)
    else:
        body = _plain(text)
    return [c.strip() for c in body.split("\n\n") if c.strip()]


def _paragraphs_html(text: Any) -> str:
    chunks = _text_chunks(text)
    if not chunks:
        return ""
    return "\n".join(f"<p>{_esc(chunk).replace(chr(10), '<br/>')}</p>" for chunk in chunks)


def _paragraphs_txt(text: Any) -> str:
    return "\n\n".join(_text_chunks(text))


def _render_options_html(options: List[Dict[str, Any]]) -> str:
    if not options:
        return ""
    rows = []
    for opt in options:
        rows.append(
            "<tr>"
            f"<td>{_esc(opt.get('name', ''))}</td>"
            f"<td>{_esc(opt.get('type', ''))}</td>"
            f"<td>{_esc(opt.get('default', ''))}</td>"
            f"<td>{_esc(opt.get('description', ''))}</td>"
            "</tr>"
        )
    return (
        "<h4>Inputs / options</h4>"
        "<table><thead><tr><th>Name</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_outputs_html(outputs: List[Dict[str, Any]]) -> str:
    if not outputs:
        return ""
    rows = []
    for out in outputs:
        rows.append(
            "<tr>"
            f"<td>{_esc(out.get('name', ''))}</td>"
            f"<td>{_esc(out.get('type', ''))}</td>"
            f"<td>{_esc(out.get('description', ''))}</td>"
            "</tr>"
        )
    return (
        "<h4>Outputs</h4>"
        "<table><thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_transcript_html(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return ""
    items = []
    for seg in segments:
        start = seg.get("startSeconds", seg.get("start", ""))
        text = seg.get("text", "")
        items.append(f"<li><span class='ts'>{_esc(start)}s</span> {_esc(text)}</li>")
    return f"<h4>Tutorial transcript</h4><ol class='transcript'>{''.join(items)}</ol>"


def _related_line(method: Dict[str, Any]) -> str:
    related = method.get("related") or []
    if not related:
        return ""
    return ", ".join(
        r if isinstance(r, str) else str(r.get("id") or r.get("title") or r)
        for r in related
    )


def _video_line(video: Any) -> str:
    if not video:
        return ""
    if isinstance(video, dict):
        yt = (video.get("youtubeId") or "").strip()
        chapter = (video.get("chapterStart") or "").strip()
        start = video.get("startSeconds")
        parts = []
        if yt:
            parts.append(f"https://www.youtube.com/watch?v={yt}")
            if start not in (None, ""):
                try:
                    parts[-1] += f"&t={int(float(start))}s"
                except (TypeError, ValueError):
                    pass
        if chapter:
            parts.append(f"chapter {chapter}")
        elif start not in (None, ""):
            parts.append(f"start {start}s")
        return " · ".join(parts) if parts else _plain(video)
    return _plain(video)


def _options_txt(options: List[Dict[str, Any]]) -> str:
    if not options:
        return ""
    lines = ["Inputs / options:"]
    for opt in options:
        name = _plain(opt.get("name", "")).strip()
        typ = _plain(opt.get("type", "")).strip()
        default = _plain(opt.get("default", "")).strip()
        desc = _plain(opt.get("description", "")).strip()
        head = name or "(unnamed)"
        meta = ", ".join(p for p in (f"type={typ}" if typ else "", f"default={default}" if default else "") if p)
        line = f"- {head}"
        if meta:
            line += f" ({meta})"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines)


def _outputs_txt(outputs: List[Dict[str, Any]]) -> str:
    if not outputs:
        return ""
    lines = ["Outputs:"]
    for out in outputs:
        name = _plain(out.get("name", "")).strip()
        typ = _plain(out.get("type", "")).strip()
        desc = _plain(out.get("description", "")).strip()
        head = name or "(unnamed)"
        line = f"- {head}"
        if typ:
            line += f" ({typ})"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines)


def _transcript_txt(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return ""
    lines = ["Tutorial transcript:"]
    for seg in segments:
        start = seg.get("startSeconds", seg.get("start", ""))
        text = _plain(seg.get("text", "")).strip()
        if start == "" or start is None:
            lines.append(f"- {text}")
        else:
            lines.append(f"- [{start}s] {text}")
    return "\n".join(lines)


def _structured_data(corpus: Dict[str, Any]) -> Dict[str, Any]:
    title = corpus.get("title") or "Aurora Documentation"
    generated = corpus.get("generatedAt") or None
    return {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "TechArticle",
                "@id": f"{CORPUS_HTML_URL}#article",
                "headline": title,
                "name": title,
                "description": META_DESCRIPTION,
                "url": CORPUS_HTML_URL,
                "mainEntityOfPage": CORPUS_HTML_URL,
                "dateModified": generated,
                "inLanguage": "en",
                "isAccessibleForFree": True,
                "author": {
                    "@type": "Organization",
                    "name": "Hallgrímsson Lab",
                    "url": PUBLIC_ORIGIN,
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Hallgrímsson Lab",
                    "url": PUBLIC_ORIGIN,
                },
                "about": {"@id": f"{CORPUS_HTML_URL}#software"},
                "encoding": [
                    {
                        "@type": "MediaObject",
                        "encodingFormat": "text/html",
                        "contentUrl": CORPUS_HTML_URL,
                    },
                    {
                        "@type": "MediaObject",
                        "encodingFormat": "text/plain",
                        "contentUrl": CORPUS_TXT_URL,
                    },
                    {
                        "@type": "MediaObject",
                        "encodingFormat": "application/json",
                        "contentUrl": CORPUS_JSON_URL,
                    },
                ],
            },
            {
                "@type": "SoftwareApplication",
                "@id": f"{CORPUS_HTML_URL}#software",
                "name": "Aurora",
                "alternateName": ["MouseMorph", "Aurora Client"],
                "applicationCategory": "MultimediaApplication",
                "operatingSystem": "Windows, macOS, Linux",
                "url": f"{PUBLIC_ORIGIN}/MainAurora",
                "sameAs": [CORPUS_SPA_URL],
                "description": (
                    "Free batch-oriented image-processing pipeline for morphometric "
                    "research: rigid alignment, elastic registration, 3D segmentation, "
                    "and landmarking."
                ),
                "offers": {
                    "@type": "Offer",
                    "price": "0",
                    "priceCurrency": "CAD",
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Hallgrímsson Lab",
                    "url": PUBLIC_ORIGIN,
                },
            },
        ],
    }


def corpus_to_html(corpus: Dict[str, Any]) -> str:
    """Render get_corpus() payload as a single crawlable HTML document."""
    title = corpus.get("title", "Aurora Documentation")
    schema = json.dumps(_structured_data(corpus), ensure_ascii=False, indent=2)
    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>',
        f"<title>{_esc(title)}</title>",
        f'<link rel="canonical" href="{_esc(CORPUS_HTML_URL)}"/>',
        '<meta name="robots" content="index,follow"/>',
        f'<meta name="description" content="{_esc(META_DESCRIPTION)}"/>',
        f'<link rel="alternate" type="text/plain" href="{_esc(CORPUS_TXT_URL)}" '
        'title="Aurora documentation (plain text)"/>',
        f'<link rel="alternate" type="application/json" href="{_esc(CORPUS_JSON_URL)}" '
        'title="Aurora documentation (JSON)"/>',
        f'<script type="application/ld+json">\n{schema}\n</script>',
        "<style>",
        "body{font-family:system-ui,sans-serif;line-height:1.5;max-width:52rem;margin:1.5rem auto;padding:0 1rem;color:#122}",
        "h1,h2,h3{line-height:1.25} table{border-collapse:collapse;width:100%;margin:0.75rem 0}",
        "th,td{border:1px solid #ccd;padding:0.4rem 0.55rem;vertical-align:top;text-align:left}",
        "th{background:#eef3f7} .meta{color:#456;font-size:0.95rem}",
        "article{border-top:1px solid #dde;padding:1rem 0} .ts{color:#678;font-variant-numeric:tabular-nums;margin-right:0.4rem}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{_esc(title)}</h1>",
        "<p class='meta'>Official Hallgrímsson Lab Aurora documentation corpus "
        "(methods and tutorial transcript). Generated for search / LLM assistants.</p>",
        f"<p class='meta'>Entries: {_esc(corpus.get('entryCount', ''))}. "
        f"Generated: {_esc(corpus.get('generatedAt', ''))}.</p>",
        "<p class='meta'>Formats: "
        f"<a href='{_esc(CORPUS_HTML_URL)}'>HTML</a> · "
        f"<a href='{_esc(CORPUS_TXT_URL)}'>plain text</a> · "
        f"<a href='{_esc(CORPUS_JSON_URL)}'>JSON</a> · "
        f"<a href='{_esc(CORPUS_SPA_URL)}'>interactive docs</a>"
        "</p>",
    ]

    for category in corpus.get("categories", []):
        parts.append(f"<h2>{_esc(category.get('title', ''))}</h2>")
        for subcategory in category.get("subcategories", []):
            parts.append(f"<h3>{_esc(subcategory.get('title', ''))}</h3>")
            for method in subcategory.get("methods", []):
                mid = method.get("id", "")
                parts.append(f"<article id='{_esc(mid)}'>")
                parts.append(f"<h3>{_esc(method.get('title', mid))}</h3>")
                if method.get("keywords"):
                    parts.append(
                        f"<p class='meta'><strong>Keywords:</strong> "
                        f"{_esc(', '.join(method.get('keywords') or []))}</p>"
                    )
                if method.get("summary"):
                    parts.append("<h4>Summary</h4>")
                    parts.append(_paragraphs_html(method.get("summary", "")))
                parts.append(_render_options_html(method.get("options") or []))
                parts.append(_render_outputs_html(method.get("outputs") or []))
                if method.get("algorithm"):
                    parts.append("<h4>Algorithm / details</h4>")
                    parts.append(_paragraphs_html(method.get("algorithm")))
                related = _related_line(method)
                if related:
                    parts.append(f"<p class='meta'><strong>Related:</strong> {_esc(related)}</p>")
                video = _video_line(method.get("video"))
                if video:
                    parts.append(f"<p class='meta'><strong>Video:</strong> {_esc(video)}</p>")
                parts.append(_render_transcript_html(method.get("transcriptSegments") or []))
                parts.append("</article>")

    parts.extend(["</body>", "</html>", ""])
    return "\n".join(parts)


def corpus_to_txt(corpus: Dict[str, Any]) -> str:
    """Render get_corpus() payload as plain text for LLM ingestion."""
    title = _plain(corpus.get("title") or "Aurora Documentation")
    lines: List[str] = [
        title,
        "=" * len(title),
        "",
        META_DESCRIPTION,
        "",
        f"Canonical HTML: {CORPUS_HTML_URL}",
        f"Plain text:     {CORPUS_TXT_URL}",
        f"JSON:           {CORPUS_JSON_URL}",
        f"Interactive:    {CORPUS_SPA_URL}",
        f"Entries:        {_plain(corpus.get('entryCount', ''))}",
        f"Generated:      {_plain(corpus.get('generatedAt', ''))}",
        "",
    ]

    for category in corpus.get("categories", []):
        cat_title = _plain(category.get("title", "")).strip() or "Category"
        lines.extend([cat_title, "-" * len(cat_title), ""])
        for subcategory in category.get("subcategories", []):
            sub_title = _plain(subcategory.get("title", "")).strip()
            if sub_title:
                lines.extend([sub_title, ""])
            for method in subcategory.get("methods", []):
                mid = _plain(method.get("id", "")).strip()
                mtitle = _plain(method.get("title") or mid).strip()
                heading = f"### {mtitle}" + (f" [{mid}]" if mid and mid != mtitle else "")
                lines.append(heading)
                lines.append("")
                if method.get("keywords"):
                    lines.append("Keywords: " + ", ".join(method.get("keywords") or []))
                    lines.append("")
                summary = _paragraphs_txt(method.get("summary", ""))
                if summary:
                    lines.extend(["Summary", summary, ""])
                opts = _options_txt(method.get("options") or [])
                if opts:
                    lines.extend([opts, ""])
                outs = _outputs_txt(method.get("outputs") or [])
                if outs:
                    lines.extend([outs, ""])
                algo = _paragraphs_txt(method.get("algorithm"))
                if algo:
                    lines.extend(["Algorithm / details", algo, ""])
                related = _related_line(method)
                if related:
                    lines.extend([f"Related: {related}", ""])
                video = _video_line(method.get("video"))
                if video:
                    lines.extend([f"Video: {video}", ""])
                transcript = _transcript_txt(method.get("transcriptSegments") or [])
                if transcript:
                    lines.extend([transcript, ""])
                lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    # Normalize accidental runs of blank lines from sparse entries.
    while "\n\n\n\n" in text:
        text = text.replace("\n\n\n\n", "\n\n\n")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        default=_DEFAULT_OUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    corpus = DocumentationManager().get_corpus()
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "aurora-docs-corpus.json")
    html_path = os.path.join(out_dir, "aurora-docs-corpus.html")
    txt_path = os.path.join(out_dir, "aurora-docs-corpus.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(corpus_to_html(corpus))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(corpus_to_txt(corpus))

    print(
        f"Wrote {corpus.get('entryCount', '?')} entries →\n"
        f"  {json_path} ({os.path.getsize(json_path) / 1024.0:.1f} KiB)\n"
        f"  {html_path} ({os.path.getsize(html_path) / 1024.0:.1f} KiB)\n"
        f"  {txt_path} ({os.path.getsize(txt_path) / 1024.0:.1f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
