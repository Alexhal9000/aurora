# Aurora AI Fine-Tuning Architecture

This note describes the first-milestone fine-tuning framework. User-facing docs in `docs_index.json` are unchanged; this file is for developers.

## Where it lives

| Concern | Location |
|---------|----------|
| Task contract + adapter registry | `pipeline/aiModels/` |
| Discovery, splits, dataset, train, eval, packaging | `pipeline/aiFineTuning/` |
| HTTP API | `pipeline/aiFineTuning/api.py` (`/ai-finetune/...`) |
| Model list / inference dispatch | `GET /get-ai-models/`, `POST /run-ai-segmentation/` via `pipeline/aiModels/registry.py` |
| Frontend dialog | `Aurora/AuroraClient/ui_src/components/aurora/fineTuning/FineTuningDialog.js` |
| Entry from segmentation | “Fine-tune model” in `segmentationView.js` |

## Training run vs registered model

A **training run** is an immutable experiment under:

`<Documents>/Aurora AI Models/runs/<run-id>/`

It may fail or be cancelled. It is never listed in the segmentation model selector.

A **registered model** is a selected checkpoint plus `metadata.json` under:

`<Documents>/Aurora AI Models/<model-id>/`

Only successful runs can be registered. The stable identity is the UUID directory name, not the display name.

Override the root in tests with `AURORA_AI_MODELS_DIR`.

## Task contract

Two axes only:

- dimensionality: `2D` or `3D`
- paradigm: `prompt_guided` or `direct`

Bundled MedSAM2 is `3D` + `prompt_guided`. Direct / 2D MedSAM2 is not supported; the UI hides those options from the adapter capability payload.

Fine-tuned MedSAM2 models reuse the existing intersecting-plane / mask-prior inference path with a different checkpoint. Direct models must not be forced into that workflow.

## How adapters declare capabilities

`pipeline/aiModels/contracts.py` (`medsam2_train_capabilities`) is the source of UI controls: hyperparameters, presets, augmentations, evaluation strategies. The dialog renders from this payload instead of hard-coding `if model === "MedSAM2"`.

Trainable presets map to SAM2 module names in `sam2_base.py`:

- `decoder_focused` — `sam_mask_decoder`, `sam_prompt_encoder`
- `encoder_decoder` — decoder + prompt + encoder neck + memory; trunk frozen
- `full` — all parameters

## Training implementation

The upstream Hydra `training/` package is **not** in this repository. Training uses `SAM2VideoTrainer` plus an Aurora loop in `pipeline/aiFineTuning/train_worker.py`, launched as a subprocess. Progress is written to `status.json` and optionally broadcast on `ws/progress/` from the Django process (InMemoryChannelLayer is not shared with the worker).

Checkpoint selection uses **2D clip validation loss**, not last epoch and not a raw Dice max. After each validation check Aurora scores 8-slice clips on all three views (same objective as training). A new low in that val loss always wins, even if Dice dropped — an early high Dice at a still-falling loss is treated as not yet converged. Dice is allowed to break a tie only when loss is on the same plateau (within `1e-4` absolute or `0.2%` relative).

The Training step has three **validation speed** modes:

- **Fast** (default): 2D clip loss and 2D clip Dice only. The Run tab overlay is the same fixed middle-slab clip painted back onto the volume on all three axes (8 slices × axial / coronal / sagittal). That is not 3D majority-vote segmentation.
- **Intermediate**: same 2D clip loss, plus full-volume Dice on **one** validation subject. That volume Dice is computed **after** 3-view (axial / coronal / sagittal) majority vote, the same method as production inference. The Run tab overlay updates every validation epoch from that volume.
- **Slow**: same 2D clip loss, plus full-volume Dice **averaged** over every validation subject, each after that same 3-view majority vote. Overlays are written for each of those volumes every validation epoch.

Full **3-view volume Dice** still runs once at the end for the before/after figures and the registered 3D score, regardless of mode.

During each epoch, **loss** is tracked separately from **Dice**:

- **Loss figure** (`training_loss.png`): train loss + validation loss (clip objective averaged over all 3 views, same as backprop). This is the metric that picks the checkpoint.
- **Dice figure** (`training_dice.png`): validation Dice from those same 2D clips (train Dice is no longer scored every epoch — that pass was slower than training itself). Dice never overrides a worse loss.

**Multi-view training:** every subject is trained on **all three** axes each epoch (axial, coronal, sagittal). Drawn-organ prompts and whole-slice ones prompts both reorient the volume before building the 8-frame clip, so mask + box (or full-frame ones) supervision applies in each view.

**GPU batching:** with **auto batch size** enabled (default), each fold probes free VRAM once and picks the largest safe clip batch up to the configured max (default 12). Training and 2D clip validation use that same batch size. Epoch validation does **not** re-score the training set; it reuses cached val clips (built once per fold) and runs one forward per batch for both loss and Dice. Disable auto batch size in Advanced hyperparameters to use a fixed max batch size exactly.

The fine-tune dialog polls ``GET /ai-finetune/gpu/`` every 2 seconds (nvidia-smi when available) and shows GPU name, total / used / free VRAM, utilization, and temperature. Utilization will still rise and fall: MedSAM2 propagates slices sequentially inside each clip, so a step is a short burst rather than a flat 100%. Validation should now look like a few training-sized batches, not a long low-VRAM stall. Steady 95–100% is still not expected without changing the model.

**Run-tab preview:** after training, the end-of-run 3D compare writes a lossy image (`max dim 192`) plus the chosen checkpoint’s predicted mask under `preview/`. The Fine-tuning Run tab (and **See existing models** after register) charts train/val loss, scrubs epochs with a slider, and overlays that mask in cyan on `gpuSliceCanvas`. Registering copies `preview/` into the model folder.

**Timing log:** each run writes ``logs/timing.txt`` (human table), ``logs/timing.json`` (summary + verdict), and ``logs/timing.jsonl`` (every span). The live status message includes ``Xs (Ys train / Zs val)`` after each epoch. Registering a model copies those files into the model folder.

**Multi-view validation & inference:** `pipeline/aiFineTuning/volume_inference.py` covers the six half-axes (left and right from the middle slice, on axial / coronal / sagittal). Drawn-organ models still chain each ray from the previous predicted mask (video tracking). Whole-slice ones models reset every 8-slice clip to a full-frame box + ones mask, matching training. Forwards are limited to 2 clips at a time so validation cannot fill a 16 GB card that training is already using. Masks are mapped back to `(D,H,W)` and voxels with **≥2/3** view agreement are kept. Fine-tuned models use this path in both validation figures and `POST /run-ai-segmentation/`. Foundation MedSAM2 still uses `MedSAM2Segmenter`.

Prompt strategies (Model step):

- **Drawn organ (default).** Mask + box on a random middle-slab slice, then propagate.
- **Whole slice as ones.** MedSAM2 cannot actually train with no prompt. This mode uses a full-frame box and a mask of all ones as the first-frame hint, while the loss still uses your organ drawings. At inference the same whole-slice ones prompt is used, so you do not need an intersecting-plane sketch of that organ. Prefer this when fine-tuning to a single structure.

Registered models are listed from `<Documents>/Aurora AI Models/<model-uuid>/`. The dropdown is a live directory scan plus a frontend cache: deleting the folder in the file manager does not refresh an already-open segmentation menu. Use **See existing models** in the fine-tune dialog to inspect metadata/figures and to delete/export/import; delete removes that UUID folder so the model leaves the menu.

K-fold trains **every** fold from the foundation weights (they do not continue from each other). Each fold keeps its own lowest-val-loss checkpoint (Dice only if loss is tied). The registered model is the fold with the lowest validation loss. Figures for each fold's curves plus a fold-comparison bar chart are written under the run's `figures/` folder.

After training, the validation group of the selected fold is scored with foundation MedSAM2 and the fine-tuned weights using the **same 3-view trainer ensemble** as production inference. Each validation subject gets a high-resolution PNG: 10 evenly spaced slices through the structure’s bounding box in axial, coronal, and sagittal, with foundation in magenta, fine-tuned in cyan, and overlap in blue. Loss curves, Dice curves, Dice bars, and those grids are written under the run’s `figures/` folder.

## Prompt slice

For each `(subject, label)` example, Aurora finds every slice that contains that structure, keeps the **middle half** of that occupied extent (the middle slab), and uses one of those slices as the first frame of the training clip (mask + box prompt). Neighbouring slices follow so the model learns to continue from that hint.

- **Training:** a random occupied slice from the middle slab (`rng.choice`), so the model does not always start from the same view.
- **Validation / test:** the centre occupied slice of the same slab, so scores stay comparable.

The clip is built so frame 0 **is** the prompt slice (not a window merely centred on it). If there are not enough slices after the prompt, the clip walks backwards instead.

## Working-space volumes

Training uses the latest **non-elastic** full-res image/mask pair in `extracted/<scan>/`. Elastic registrations (and warp sidecars such as `_fwd` / `_inv`) are skipped even when they have a mask. If there are no non-elastic edits, the raw `{scan}.nii.gz` is that working volume.

Discover lists those pairs from the directory tree and NIfTI headers only. It does not decompress mask voxels; which labels are actually drawn is resolved when training examples are built.

## Tests

`python manage.py test pipeline` from `AuroraClient/` with `DJANGO_SETTINGS_MODULE=AuroraClient.settings`.

## Out of scope (do not pretend these exist)

- Segment All / automatic prompt generation
- Registration-assisted priors
- Bootstrap evaluation
- Resume-from-checkpoint
- Direct MedSAM2 / empty prompt
- Vendoring Facebook/Bowang `training/`
