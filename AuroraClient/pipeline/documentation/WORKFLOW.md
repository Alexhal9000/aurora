# Aurora Living Documentation — How to Update

Folder: `Aurora/AuroraClient/pipeline/documentation/`

| File | Role |
|------|------|
| `docs_index.json` | Master doc tree (categories → subcategories → methods) |
| `check_docs.py` | Detect stale entries, seed hashes, mark updated, extract source |
| `export_public_corpus.py` | Publish machine-readable corpus JSON for public crawlers / AI Assistant |
| `UPDATE_PROMPT.txt` | Copy-paste instructions for an LLM agent after code changes |
| `WORKFLOW.md` | This file — human-facing how-to for the three update scenarios |

Source of truth for code is always the live Python file under `pipeline/`. The JSON stores prose + a hash of the documented symbol so drift is detectable.

### Machine-readable corpus (same root)

| Endpoint / artifact | Role |
|------|------|
| `GET /docs/corpus/` | Live full corpus from `DocumentationManager.get_corpus()` (methods + tutorial; strips `source`) |
| `BHTools/frontend/static/frontend/aurora-docs-corpus.json` | Public machine-readable dump |
| `BHTools/frontend/static/frontend/aurora-docs-corpus.html` | Public crawlable HTML dump (preferred for Perplexity `@` source attachment) |
| `BHTools/frontend/static/frontend/aurora-docs-corpus.txt` | Public plain-text dump (preferred for general LLM ingestion) |

After editing `docs_index.json`, republish the public dump:

```bash
python3 Aurora/AuroraClient/pipeline/documentation/export_public_corpus.py
```

Public URLs used by the AI Assistant / crawlers:

- HTML (Perplexity): `https://www.hallgrimssonlab.ca/static/frontend/aurora-docs-corpus.html`
- Plain text (LLMs): `https://www.hallgrimssonlab.ca/static/frontend/aurora-docs-corpus.txt`
- JSON: `https://www.hallgrimssonlab.ca/static/frontend/aurora-docs-corpus.json`
- Discovery: `https://www.hallgrimssonlab.ca/robots.txt` and `https://www.hallgrimssonlab.ca/sitemap.xml`
  (sitemap lists public pages + all corpus URLs)

HTTP headers (`Content-Type` with `charset=utf-8`, `X-Robots-Tag: index, follow`) are set by
`frontend/corpus_views.py`. The live nginx load balancer already proxies all paths (including
`/static/`) to Django, so deploying BHTools is enough — no nginx change required unless a
direct `/static/` alias is added later (`BHTools/deploy/nginx-aurora-docs-corpus.conf`).

Run commands from anywhere; paths below are relative to the BHTools repo root:

```bash
python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py
```

---

## Naming convention (match the Aurora UI)

Documentation entry **titles** must use the same wording as the processing tools in the Aurora UI
(`AURORA_PROCESSING_TOOL_DEFS` in `mainAurora.js`).

Canonical map: [`UI_TITLE_MAP.txt`](UI_TITLE_MAP.txt)

| Docs `id` (stable) | UI / docs `title` |
|--------------------|-------------------|
| `match-histogram` | Intensity Normalization |
| `homogenize-background` | Background Offset Correction |
| `apply-threshold` | Align Thresholds |
| `denoise-all-scans` | Denoise |
| `restore-all-scans` | Restore |
| `batch-weld-mesh` | Weld mesh |
| `align-to-reference` | Align to reference |
| `batch-cleanup-mesh` | Batch cleanup mesh |
| `export-landmarks-bundle` | Export landmarks |
| `batch-apply-mask-to-scans` | Apply masks |

Do **not** invent Title Case from the Python class name (`MatchHistogramView` → not “Match Histogram”).
When adding a new documented tool, look up its `label` in `AURORA_PROCESSING_TOOL_DEFS` first and
add a line to `UI_TITLE_MAP.txt`.

---

## Scenario 1 — You changed existing documented code

Use this when you edited a method/class that already has an entry in `docs_index.json`.

1. Change the Python source as usual.
2. Run the checker:
   ```bash
   python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py
   ```
   (Optional while Aurora is running: `GET /docs/check-changes/`)
3. For each **STALE** entry, the report prints `old_hash`, `new_hash`, and the full new verbatim source.
4. Update that entry’s prose in `docs_index.json` only as needed:
   - `summary`
   - `options[]` (inputs: name / type / default / description)
   - `outputs[]` (outputs: name / type / description — from Response/return values)
   - `algorithm` (`text` / `math` / `diagram`)
   - `related[]`, `video`, `images[]` if still accurate
5. If the symbol was renamed or moved, also update `source.file` and `source.symbol`.
6. Mark the entry current:
   ```bash
   python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --mark <entry_id>
   ```
   Example: `--mark alpaca-align-landmarks-to-mesh`

**LLM shortcut:** paste the checker output + `UPDATE_PROMPT.txt` into an agent session and ask it to perform steps 4–6.

**Note:** A stale hash means the *written* docs may be outdated relative to live code. The documentation UI no longer embeds source; source extraction remains available via `check_docs.py --extract` for authors.

---

## Scenario 2 — You added new code that should be documented

Use this when a new method/class should appear in the docs panel.

1. Implement the code in `pipeline/` (or wherever it lives under that package).
2. Add a new method object under the correct `categories[] → subcategories[] → methods[]` in `docs_index.json`.
3. Fill at least:
   ```json
   {
     "id": "my-module-my-method",
     "title": "Human Title",
     "keywords": ["..."],
     "source": {
       "file": "MyModule.py",
       "symbol": "MyClass.my_method",
       "hash": "",
       "lastChecked": null
     },
     "summary": "...",
     "options": [],
     "outputs": [],
     "algorithm": {
       "text": "...",
       "math": [
         {
           "equation": "I_{corr}(x) = I(x) / B(x)",
           "caption": "I: observed intensity; B: smooth bias field; x: voxel."
         }
       ],
       "diagram": ""
     },
     "related": [],
     "video": null,
     "images": [{ "caption": "Optional figure placeholder", "path": null }]
   }
   ```
   - `source.file` is relative to `pipeline/` (e.g. `"ALPACA.py"`).
   - `source.symbol` supports dotted lookup (`Class.method`).
   - `options[]` documents inputs (request params / method args). `outputs[]` documents Response JSON keys or return values — same name/type/description shape, no `default`. Omit or use `[]` until authored; the UI hides empty tables.
   - `images[].path: null` renders a UI placeholder until you add an asset.
   - Figure assets: generate via `pipeline/documentation/figure_generation/`
     (renders go to its gitignored `_work/`; see its README). Permanent files live in
     `BHTools/frontend/static/docs-figures/` and are referenced as
     `/static/docs-figures/docs_<id>.gif` (and `.png`). Wire **method** entries only.
     Do **not** put permanent figures under `static/frontend/` — webpack cleans that dir.
   - **Algorithm math:** Prefer insight over ornament. Use `math: []` when a sentence is enough.
     Otherwise one `{equation, caption}` object per formula (never comma-separated parallel eqs).
     Captions define every symbol **and quote current values for named constants**
     (e.g. `usable_dim = 508` from `MESH_MAX_VOXEL_CUBE` / `MESH_VOXEL_PADDING`).
     Full rules: `UPDATE_PROMPT.txt` → “Algorithm math rules”.
   - **Scientific wording:** Prefer clear math/prose from
     `Hallgrimsson_Image_Processing_Paper/Final_Paper/draft.md` when it matches live code
     (see `UPDATE_PROMPT.txt` → “Preferred scientific wording source”). Code wins on conflict.
   - **Library wrappers:** For tools that dispatch to SciPy/skimage/etc. (e.g. Denoise), put
     official docs links in `algorithm.references` and place `[[references]]` in `algorithm.text`
     on the “apply selected method” step. Do not re-derive library equations. See
     `UPDATE_PROMPT.txt` → “Library backends / references”.
     Lint: `python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --lint-math`
4. Seed hashes from live source (fills empty/`hash` for all entries that can be resolved):
   ```bash
   python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --seed
   ```
   Or, after writing prose for just the new entry:
   ```bash
   python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --mark <new-entry-id>
   ```
5. Open Aurora → Documentation icon → confirm the new entry appears in the tree and loads.

---

## Scenario 3 — You removed code that was documented

Use this when a documented method/class no longer exists (or should no longer be shown).

1. Delete the method object from `docs_index.json`.
2. Remove its `id` from every other entry’s `related[]` array.
3. Optionally confirm nothing is broken:
   ```bash
   python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py
   ```
   Remaining entries should report `OK`. Entries that still point at deleted symbols will show an `ERROR` in the report — fix or remove those too.

---

## Handy commands

```bash
# Status of all documented symbols vs live code
python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py

# JSON report (for agents / tooling)
python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --json

# Print live verbatim source for one entry
python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --extract alpaca-get-outer-mesh

# Fill/refresh stored hashes from live source
python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --seed

# After updating prose for one entry
python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --mark <entry_id>

# Lint equation schema / captions / one-eq-per-line / references
python3 Aurora/AuroraClient/pipeline/documentation/check_docs.py --lint-math
```

---

## Placeholder stubs

If an entry’s `summary` or `algorithm.text` still says *“HTTP APIView … Validates request fields…”*,
it was only seeded. Author real prose from `--extract <id>` (see `UPDATE_PROMPT.txt` →
“Placeholder / stub detection”). Never leave that boilerplate in place.

---

## Current population status

**Mesh surface utilities** (under **Mesh Tools**) documents shared mesh algorithms that are
reused across Aurora — not tied to one registration workflow. Implementations may live in
`ALPACA.py` or elsewhere:

- `alpaca-get-outer-mesh` — outer-shell visibility (`ALPACA.get_outer_mesh`); UI via Quick mesh
  Outer shell → `apply-mesh-shell`

The **ALPACA rigid alignment** subcategory under **Rigid & Elastic Registration** documents
Aurora’s surface-based Align-to-reference path (rigid stages only — no CPD landmark transfer).
Align to reference covers method choice; this subcategory deepens ALPACA itself on preserved
PLY meshes and voxel subjects after marching cubes:

- `alpaca-create-landmarks-from-mesh` — Poisson-disk sampling
- `alpaca-align-landmarks-to-mesh` — FPFH → stochastic RANSAC pool → ICP

Optional outer-shell prep for ALPACA links to `alpaca-get-outer-mesh` under Mesh Tools rather
than treating the shell as an ALPACA-exclusive stage. Batch orchestration lives in
`align-to-reference`. Hashes are seeded against live `ALPACA.py`.
**You do not need to re-seed those entries for the docs panel to work.**

To use it in the UI: start the Aurora engine, open the Aurora page, click the book icon next to the profile. The panel loads from `/docs/index/` etc. Other modules are not documented yet — add them via Scenario 2 when ready.
