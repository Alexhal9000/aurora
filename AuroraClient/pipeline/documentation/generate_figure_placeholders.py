#!/usr/bin/env python3
"""Legacy placeholder PNG generator (superseded by figure_generation/run_all.py).

Prefer::

    figure_generation/run_all.py --wire

Permanent assets belong in BHTools/frontend/static/docs-figures/.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
INDEX = HERE / "docs_index.json"
AURORA_FIG = HERE / "figure_generation" / "_work" / "staging"
# BHTools repo layout: Aurora/AuroraClient/pipeline/documentation → ../../../../BHTools/...
BHTOOLS_FIG = HERE.parents[3] / "BHTools" / "frontend" / "static" / "docs-figures"
URL_PREFIX = "/static/docs-figures"


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TITLE = load_font(28, True)
FONT_LABEL = load_font(18, True)
FONT_SMALL = load_font(13)
FONT_TINY = load_font(11)

BG = (236, 238, 242)
HEADER_BG = (45, 55, 72)
HEADER_FG = (255, 255, 255)
PANEL_FILL = (255, 255, 255)
PANEL_EDGE = (70, 90, 120)
PANEL_DASH = (120, 140, 170)
LABEL_BG = (70, 90, 120)
HINT = (90, 100, 120)
GRID = (210, 216, 225)


def wrap(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def dashed_rect(draw, box, fill, outline, width=2, dash=10, gap=6):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=fill, outline=outline, width=width)
    for x in range(x0 + 8, x1 - 4, dash + gap):
        draw.line([(x, y0 + 4), (min(x + dash, x1 - 4), y0 + 4)], fill=PANEL_DASH, width=1)
        draw.line([(x, y1 - 4), (min(x + dash, x1 - 4), y1 - 4)], fill=PANEL_DASH, width=1)
    for y in range(y0 + 8, y1 - 4, dash + gap):
        draw.line([(x0 + 4, y), (x0 + 4, min(y + dash, y1 - 4))], fill=PANEL_DASH, width=1)
        draw.line([(x1 - 4, y), (x1 - 4, min(y + dash, y1 - 4))], fill=PANEL_DASH, width=1)


def draw_crosshair(draw, box, color=(200, 205, 215)):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    draw.line([(cx, y0 + 12), (cx, y1 - 12)], fill=color, width=1)
    draw.line([(x0 + 12, cy), (x1 - 12, cy)], fill=color, width=1)
    r = min(x1 - x0, y1 - y0) // 6
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=1)


def draw_mesh_glyph(draw, box):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    s = min(x1 - x0, y1 - y0) // 4
    pts = [(cx - s, cy - s // 2), (cx + s, cy - s // 2), (cx + s, cy + s), (cx - s, cy + s)]
    ox, oy = s // 2, -s // 2
    bpts = [(px + ox, py + oy) for px, py in pts]
    col = (160, 170, 190)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        draw.line([a, b], fill=col, width=2)
    for a, b in zip(bpts, bpts[1:] + bpts[:1]):
        draw.line([a, b], fill=col, width=2)
    for a, b in zip(pts, bpts):
        draw.line([a, b], fill=col, width=2)


def draw_slice_glyph(draw, box):
    x0, y0, x1, y1 = box
    pad = 16
    draw.ellipse([x0 + pad, y0 + pad, x1 - pad, y1 - pad], outline=(150, 160, 180), width=2)
    draw.ellipse(
        [x0 + pad * 2, y0 + pad * 2, x1 - pad * 2, y1 - pad * 2],
        outline=(170, 180, 200),
        width=1,
    )
    for i in range(4):
        yy = y0 + pad + 20 + i * ((y1 - y0 - 2 * pad) // 5)
        draw.line([(x0 + pad + 10, yy), (x1 - pad - 10, yy)], fill=(200, 205, 215), width=1)


def draw_heatmap_glyph(draw, box):
    x0, y0, x1, y1 = box
    draw_mesh_glyph(draw, box)
    lx0, ly0 = x1 - 28, y0 + 20
    lx1, ly1 = x1 - 12, y1 - 20
    h = ly1 - ly0
    for i in range(h):
        t = i / max(h - 1, 1)
        if t < 0.33:
            u = t / 0.33
            c = (int(40 + 80 * u), int(80 + 120 * u), 220)
        elif t < 0.66:
            u = (t - 0.33) / 0.33
            c = (int(120 + 100 * u), int(200 - 40 * u), int(220 - 180 * u))
        else:
            u = (t - 0.66) / 0.34
            c = (220, int(160 - 120 * u), 40)
        draw.line([(lx0, ly0 + i), (lx1, ly0 + i)], fill=c, width=1)


def draw_histogram_glyph(draw, box):
    x0, y0, x1, y1 = box
    n = 24
    w = (x1 - x0 - 40) // n
    base = y1 - 24
    for i in range(n):
        h = int(20 + 80 * abs(math.sin(i / 3.2)) * (0.4 + 0.6 * (i / n)))
        bx0 = x0 + 20 + i * w
        draw.rectangle([bx0, base - h, bx0 + w - 2, base], outline=(140, 150, 170), width=1)


def draw_ui_chrome_glyph(draw, box):
    x0, y0, x1, y1 = box
    draw.rectangle([x0 + 8, y0 + 8, x1 - 8, y0 + 36], outline=(160, 170, 190), width=1)
    for i in range(4):
        draw.rectangle(
            [x0 + 16 + i * 70, y0 + 14, x0 + 70 + i * 70, y0 + 30],
            outline=(180, 190, 205),
            width=1,
        )
    draw.rectangle([x0 + 8, y0 + 44, x0 + 90, y1 - 8], outline=(160, 170, 190), width=1)
    for i in range(6):
        yy = y0 + 56 + i * 22
        draw.rectangle([x0 + 16, yy, x0 + 80, yy + 12], outline=(190, 200, 215), width=1)
    draw.rectangle([x0 + 100, y0 + 44, x1 - 8, y1 - 8], outline=(160, 170, 190), width=1)


GLYPHS = {
    "mesh": draw_mesh_glyph,
    "slice": draw_slice_glyph,
    "heatmap": draw_heatmap_glyph,
    "histogram": draw_histogram_glyph,
    "ui": draw_ui_chrome_glyph,
    "cross": draw_crosshair,
}


def panel(draw, box, title, subtitle, glyph=None):
    dashed_rect(draw, box, PANEL_FILL, PANEL_EDGE)
    (GLYPHS.get(glyph) or draw_crosshair)(draw, box)
    x0, y0, x1, y1 = box
    lines = wrap(draw, title, FONT_LABEL, x1 - x0 - 24)
    chip_h = 10 + 20 * len(lines)
    chip_w = max(120, int(draw.textlength(lines[0], FONT_LABEL)) + 20)
    draw.rectangle([x0 + 8, y0 + 8, x0 + 8 + chip_w, y0 + 8 + chip_h], fill=LABEL_BG)
    ty = y0 + 12
    for ln in lines:
        draw.text((x0 + 16, ty), ln, fill=(255, 255, 255), font=FONT_LABEL)
        ty += 20
    if subtitle:
        sub_lines = wrap(draw, subtitle, FONT_SMALL, x1 - x0 - 28)
        sy = y1 - 14 - 16 * len(sub_lines)
        for ln in sub_lines:
            draw.text((x0 + 14, sy), ln, fill=HINT, font=FONT_SMALL)
            sy += 16


def header_bar(draw, W, title, entry_id, priority, kind):
    draw.rectangle([0, 0, W, 64], fill=HEADER_BG)
    draw.text((20, 12), title, fill=HEADER_FG, font=FONT_TITLE)
    draw.text(
        (20, 42),
        f"{entry_id}  ·  {priority}  ·  {kind}  ·  PLACEHOLDER — paint over regions",
        fill=(180, 190, 205),
        font=FONT_TINY,
    )


def footer(draw, W, H, shot):
    draw.rectangle([0, H - 52, W, H], fill=(226, 230, 236))
    lines = wrap(draw, "Brief: " + shot, FONT_SMALL, W - 40)
    y = H - 46
    for ln in lines[:2]:
        draw.text((20, y), ln, fill=HINT, font=FONT_SMALL)
        y += 16


def new_canvas(W, H):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    for x in range(0, W, 40):
        draw.line([(x, 64), (x, H - 52)], fill=GRID, width=1)
    for y in range(64, H - 52, 40):
        draw.line([(0, y), (W, y)], fill=GRID, width=1)
    return img, draw


def layout_before_after_slices(W, H, left="BEFORE", right="AFTER"):
    m, top, bot = 20, 80, H - 68
    gap, mid = 16, W // 2
    return [
        ((m, top, mid - gap // 2, bot), left, "Paste matching FOV slice", "slice"),
        ((mid + gap // 2, top, W - m, bot), right, "Paste result at same FOV", "slice"),
    ]


def layout_before_after_mesh(W, H, left="BEFORE", right="AFTER"):
    m, top, bot = 20, 80, H - 68
    gap, mid = 16, W // 2
    return [
        ((m, top, mid - gap // 2, bot), left, "3D mesh / point cloud", "mesh"),
        ((mid + gap // 2, top, W - m, bot), right, "3D mesh / point cloud", "mesh"),
    ]


def layout_single(W, H, title, sub, glyph):
    return [((20, 80, W - 20, H - 68), title, sub, glyph)]


def layout_ui_plus_result(W, H, left="UI CONTROLS", right="RESULT", rg="mesh"):
    m, top, bot = 20, 80, H - 68
    gap = 16
    split = int(W * 0.38)
    return [
        ((m, top, split - gap // 2, bot), left, "Screenshot Aurora panel / dialog", "ui"),
        ((split + gap // 2, top, W - m, bot), right, "Paste visual outcome", rg),
    ]


def layout_triple_ortho(W, H):
    m, top, bot = 20, 80, H - 68
    gap = 12
    w = (W - 2 * m - 2 * gap) // 3
    out = []
    for i, lab in enumerate(("AXIAL", "CORONAL", "SAGITTAL")):
        x0 = m + i * (w + gap)
        out.append(((x0, top, x0 + w, bot), lab, "Orthogonal slice + mask overlay", "slice"))
    return out


def layout_slice_plus_metrics(W, H):
    m, top, bot = 20, 80, H - 68
    gap = 16
    split = int(W * 0.62)
    return [
        ((m, top, split - gap // 2, bot), "REGISTERED OVERLAY / SLICES", "Reference vs subject", "slice"),
        (
            (split + gap // 2, top, W - m, top + (bot - top) // 2 - gap // 2),
            "QC METRICS",
            "Dice · surface score · NCC",
            "cross",
        ),
        (
            (split + gap // 2, top + (bot - top) // 2 + gap // 2, W - m, bot),
            "OPTIONAL HEATMAP",
            "Surface-distance or local-NCC",
            "heatmap",
        ),
    ]


def layout_grid_thumbs(W, H):
    m, top, bot = 20, 80, H - 68
    cols, rows, gap = 4, 3, 10
    pw = (W - 2 * m - (cols - 1) * gap) // cols
    ph = (bot - top - (rows - 1) * gap) // rows
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = m + c * (pw + gap)
            y0 = top + r * (ph + gap)
            out.append(((x0, y0, x0 + pw, y0 + ph), f"THUMB {r * cols + c + 1}", "Projection PNG", "slice"))
    return out


def layout_atlas(W, H):
    m, top, bot = 20, 80, H - 68
    gap = 14
    left_w = int((W - 2 * m - gap) * 0.45)
    return [
        ((m, top, m + left_w, bot), "ATLAS", "Mean intensity / mesh", "slice"),
        ((m + left_w + gap, top, W - m, top + (bot - top) // 2 - gap // 2), "SUBJECT A", "Warped", "slice"),
        ((m + left_w + gap, top + (bot - top) // 2 + gap // 2, W - m, bot), "SUBJECT B", "Warped", "slice"),
    ]


def layout_semi_landmarks(W, H):
    m, top, bot = 20, 80, H - 68
    gap = 14
    split = int(W * 0.55)
    return [
        ((m, top, split - gap // 2, bot), "OUTER SHELL + CUTTING PLANE", "Plane through endpoints", "mesh"),
        ((split + gap // 2, top, W - m, bot), "ARC-LENGTH SEMI-LANDMARKS", "Ordered samples on path", "mesh"),
    ]


def layout_alpaca_clouds(W, H):
    m, top, bot = 20, 80, H - 68
    gap = 14
    hmid = top + (bot - top) // 2
    return [
        ((m, top, W // 2 - gap // 2, hmid - gap // 2), "REFERENCE CLOUD", "Poisson / FPFH points", "mesh"),
        ((W // 2 + gap // 2, top, W - m, hmid - gap // 2), "SUBJECT CLOUD", "Before alignment", "mesh"),
        ((m, hmid + gap // 2, W - m, bot), "ALIGNED OVERLAY", "After RANSAC + ICP", "mesh"),
    ]


def layout_histogram_match(W, H):
    m, top, bot = 20, 80, H - 68
    gap = 14
    third = (W - 2 * m - 2 * gap) // 3
    return [
        ((m, top, m + third, bot), "REFERENCE SLICE", "", "slice"),
        ((m + third + gap, top, m + 2 * third + gap, bot), "MATCHED SUBJECT", "Same FOV", "slice"),
        ((m + 2 * third + 2 * gap, top, W - m, bot), "HISTOGRAMS", "Ref vs subject", "histogram"),
    ]


def layout_outer_shell(W, H):
    return layout_before_after_mesh(W, H, "FULL MESH", "OUTER SHELL ONLY")


def layout_poisson(W, H):
    return layout_single(W, H, "OUTER SHELL + POISSON POINTS", "~5000 evenly spaced samples", "mesh")


def layout_heatmap_ref(W, H):
    m, top, bot = 20, 80, H - 68
    gap, mid = 14, W // 2
    return [
        ((m, top, mid - gap // 2, bot), "FULL REFERENCE PLY", "Landmark mesh", "mesh"),
        ((mid + gap // 2, top, W - m, bot), "DECIMATED HEATMAP MESH", "Display budget", "heatmap"),
    ]


LAYOUTS = {
    "before_after_slices": layout_before_after_slices,
    "before_after_mesh": layout_before_after_mesh,
    "outer_shell": layout_outer_shell,
    "poisson": layout_poisson,
    "alpaca_clouds": layout_alpaca_clouds,
    "slice_metrics": layout_slice_plus_metrics,
    "atlas": layout_atlas,
    "semi": layout_semi_landmarks,
    "histogram_match": layout_histogram_match,
    "ui_plus_mesh": lambda W, H: layout_ui_plus_result(W, H, "AURORA UI", "3D / MESH RESULT", "mesh"),
    "ui_plus_result_slice": lambda W, H: layout_ui_plus_result(W, H, "AURORA UI", "VOLUME RESULT", "slice"),
    "triple_ortho": layout_triple_ortho,
    "grid": layout_grid_thumbs,
    "heatmap_ref": layout_heatmap_ref,
    "single_mesh": lambda W, H: layout_single(W, H, "3D MESH VIEWPORT", "Aurora canvas / PLY render", "mesh"),
    "single_ui": lambda W, H: layout_single(W, H, "AURORA UI SCREENSHOT", "Panel or dialog", "ui"),
    "single_slice_ui": lambda W, H: layout_ui_plus_result(W, H, "VIEWER CHROME", "SLICE", "slice"),
}

# id -> (W, H, layout_key, title, priority, kind, shot)
SPECS = {
    "align-to-reference": (1400, 800, "ui_plus_mesh", "Align to reference", "P0", "ui+mesh",
                           "Before/after subjects after Align to reference (ALPACA)."),
    "alpaca-get-outer-mesh": (1280, 720, "outer_shell", "Outer shell extraction", "P0", "mesh-3d",
                             "Full mesh vs outer shell only."),
    "alpaca-create-landmarks-from-mesh": (1200, 900, "poisson", "Poisson-disk sampling", "P0", "mesh-3d",
                                         "Poisson-disk pseudo-landmarks (~5000) on outer shell."),
    "alpaca-align-landmarks-to-mesh": (1280, 900, "alpaca_clouds", "ALPACA rigid alignment", "P0", "science-result",
                                      "Aligned point clouds after FPFH/RANSAC/ICP."),
    "elastic-registration": (1400, 800, "slice_metrics", "Elastic registration (voxel)", "P0", "science-result",
                             "Registered subject vs reference + Dice / surface-distance QC."),
    "mesh-elastic-registration": (1280, 720, "before_after_mesh", "Elastic registration (mesh)", "P0", "mesh-3d",
                                  "Preserved-mesh NR-ICP warp; optional surface-distance coloring."),
    "create-atlas": (1400, 800, "atlas", "Create atlas", "P0", "science-result",
                     "Population atlas beside warped subjects."),
    "transfer-landmarks": (1280, 720, "before_after_mesh", "Transfer landmarks", "P0", "mesh-3d",
                           "Reference landmarks transferred to subject."),
    "get-semi-landmarks": (1280, 720, "semi", "Semi-landmarks", "P0", "mesh-3d",
                           "Cutting plane + equal arc-length semi-landmarks."),
    "n4-bias-correction": (1280, 720, "before_after_slices", "N4 bias correction", "P1", "science-result",
                           "Same slice before/after N4."),
    "match-histogram": (1400, 720, "histogram_match", "Intensity normalization", "P1", "science-result",
                        "Intensity-normalized subject vs reference."),
    "denoise-all-scans": (1280, 720, "before_after_slices", "Denoise", "P1", "science-result",
                          "Noisy vs denoised slice."),
    "gpu-nlm3d": (1280, 720, "before_after_slices", "GPU NLM 3D", "P1", "science-result",
                  "3D GPU NLM before/after."),
    "restore-all-scans": (1280, 720, "before_after_slices", "Restore (CLAHE)", "P1", "science-result",
                          "CLAHE restore before/after."),
    "cleanup-mesh": (1280, 720, "before_after_mesh", "Mesh cleanup", "P1", "mesh-3d",
                     "Before/after island removal."),
    "homogenize-background": (1280, 720, "before_after_slices", "Background homogenization", "P1", "science-result",
                              "Background offset correction."),
    "apply-threshold": (1280, 720, "ui_plus_mesh", "Apply threshold", "P1", "ui+result",
                        "Thresholded mesh/volume vs reference."),
    "interpolate-anisotropic-to-isotropic": (1280, 720, "before_after_slices", "Anisotropic → isotropic", "P1",
                                             "science-result", "Thick-slice vs isotropic fill."),
    "marchingcubes": (1200, 900, "single_mesh", "Marching cubes / Quick Mesh", "P1", "mesh-3d",
                      "Quick Mesh from thresholded volume."),
    "ai-segmentation": (1400, 720, "triple_ortho", "AI segmentation", "P1", "science-result",
                        "AI mask overlay on orthogonal slices."),
    "prepare-heatmap-reference-mesh": (1280, 720, "heatmap_ref", "Heatmap reference mesh", "P1", "mesh-3d",
                                       "Full PLY vs decimated heatmap mesh."),
    "remove-background": (1280, 720, "before_after_slices", "Remove background", "P2", "science-result",
                          "Background voxels collapsed after threshold removal."),
    "apply-crop": (1280, 720, "ui_plus_result_slice", "Apply crop", "P2", "ui+result",
                   "Crop box with padding."),
    "apply-rotation": (1280, 720, "ui_plus_mesh", "Apply rotation", "P2", "ui+result",
                       "Quick Mesh rotation + reoriented volume."),
    "display-nifti": (1200, 800, "single_slice_ui", "Display NIfTI", "P2", "ui-screenshot",
                      "Slice viewer on a lossy NIfTI edit."),
    "extract-scans": (1280, 800, "single_ui", "Extract all scans", "P2", "ui-screenshot",
                      "Raw import → extracted/ structure."),
    "list-directories": (1200, 800, "single_ui", "File browser", "P2", "ui-screenshot",
                         "File browser listing extracted scans."),
    "make-atlas-reference": (1200, 800, "single_ui", "Make atlas reference", "P2", "ui-screenshot",
                             "Atlas promoted as ReferenceAtlas."),
    "physically-accurate-mesh": (1200, 900, "single_mesh", "Physically accurate mesh", "P2", "mesh-3d",
                                 "Exported PLY in physical mm."),
    "random-landmarking": (1200, 900, "single_mesh", "Random landmarking", "P2", "mesh-3d",
                           "Auto-distributed landmarks."),
    "rebalance-landmarks-along-path": (1280, 720, "before_after_mesh", "Rebalance landmarks", "P2", "mesh-3d",
                                       "Curve landmarks before/after rebalance."),
    "snap-to-mesh": (1280, 720, "before_after_mesh", "Snap to mesh", "P2", "mesh-3d",
                     "Landmark snapped onto outer shell."),
    "invert-alignment-landmarks": (1280, 720, "before_after_mesh", "Invert alignment landmarks", "P2", "mesh-3d",
                                   "Landmarks inverted to pre-alignment space."),
}


def filename_for(entry_id: str) -> str:
    return f"docs_{entry_id.replace('-', '_')}.png"


def generate_pngs(out_dirs: list[Path]) -> list[tuple[str, str, int, int]]:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
    generated = []
    for eid, (W, H, layout_key, title, priority, kind, shot) in SPECS.items():
        img, draw = new_canvas(W, H)
        header_bar(draw, W, title, eid, priority, kind)
        for box, lab, sub, glyph in LAYOUTS[layout_key](W, H):
            panel(draw, box, lab, sub, glyph)
        footer(draw, W, H, shot)
        fname = filename_for(eid)
        for d in out_dirs:
            img.save(d / fname, "PNG", optimize=True)
        generated.append((eid, fname, W, H))
        print(f"wrote {fname} ({W}x{H}) → {', '.join(str(d) for d in out_dirs)}")
    for d in out_dirs:
        (d / "README.md").write_text(
            textwrap.dedent(
                f"""\
                # Aurora docs figures

                Placeholder PNGs for the Documentation side panel. Paint/paste real
                screenshots into the outlined regions (keep canvas size).

                - Served by Aurora at `{URL_PREFIX}/<filename>`
                - Wired on **methods only** in docs_index.json → images[].path
                """
            )
        )
    return generated


def wire_docs_index(generated: list[tuple[str, str, int, int]]) -> None:
    data = json.loads(INDEX.read_text())
    path_by_id = {eid: f"{URL_PREFIX}/{fname}" for eid, fname, *_ in generated}
    shot_by_id = {eid: SPECS[eid][6] for eid, *_ in generated}

    # Strip misplaced images on categories/subcategories (id collisions).
    for cat in data.get("categories") or []:
        cat.pop("images", None)
        for sub in cat.get("subcategories") or []:
            sub.pop("images", None)

    wired = 0
    for cat in data.get("categories") or []:
        for sub in cat.get("subcategories") or []:
            for method in sub.get("methods") or []:
                eid = method.get("id")
                if eid not in path_by_id:
                    continue
                # Methods always have source; skip anything that somehow isn't a method.
                if "source" not in method:
                    continue
                imgs = method.get("images")
                cap = ""
                if isinstance(imgs, list) and imgs:
                    cap = (imgs[0].get("caption") or "").strip()
                if (
                    not cap
                    or "placeholder" in cap.lower()
                    or cap.startswith("UI / result")
                ):
                    cap = shot_by_id.get(eid) or method.get("title") or eid
                method["images"] = [{"caption": cap, "path": path_by_id[eid]}]
                wired += 1

    INDEX.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wired {wired} method image paths in {INDEX}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-bhtools-mirror", action="store_true")
    parser.add_argument("--skip-wire", action="store_true")
    args = parser.parse_args()

    outs = [AURORA_FIG]
    if not args.no_bhtools_mirror:
        outs.append(BHTOOLS_FIG)

    generated = generate_pngs(outs)
    if not args.skip_wire:
        wire_docs_index(generated)
    print(f"Done — {len(generated)} figures.")


if __name__ == "__main__":
    main()
