# MedSAM2 Standalone - Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install torch torchvision torchaudio  # PyTorch (or use cuda version)
pip install nibabel scikit-image Pillow
pip install hydra-core omegaconf
```

### 2. Use in Your Project

**Structure:**
```
your_project/
├── main.py
└── portable/                    ← Copy entire portable folder here
    ├── medsam2_standalone.py
    ├── models/
    ├── configs/
    ├── sam2/
    └── requirements.txt
```

**Code:**
```python
import sys
from pathlib import Path

# Add portable to path
portable_dir = Path(__file__).parent / "portable"
sys.path.insert(0, str(portable_dir))

from medsam2_standalone import MedSAM2Segmenter
import nibabel as nib
import numpy as np

# Initialize segmenter
segmenter = MedSAM2Segmenter()

# Load image
img = nib.load("scan.nii.gz").get_fdata()

# Segment
mask = segmenter.segment(
    img,
    bbox_min=(100, 50, 75),
    bbox_max=(400, 250, 350),
    ensemble=False  # or True for 3-view consensus
)

# Save
out = nib.Nifti1Image(mask, nib.load("scan.nii.gz").affine)
nib.save(out, "result.nii.gz")
```

## Required Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | >=2.7.0 | Deep learning framework |
| **torchvision** | >=0.16.0 | Vision utilities |
| **numpy** | >=1.21.0 | Numerical computing |
| **scikit-image** | >=0.19.0 | Connected components (post-processing) |
| **Pillow** | >=8.0.0 | Image processing |
| **nibabel** | >=4.0.0 | NIfTI file I/O (medical imaging) |
| **hydra-core** | >=1.1.0 | Configuration management |
| **omegaconf** | >=2.1.0 | Hydra dependency |

## GPU Support

### CUDA 13.0 (Recommended for RTX 5070 Ti and newer)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### CUDA 11.8 (Older GPUs)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU Only
```bash
pip install torch torchvision torchaudio
```

## Troubleshooting

### Error: "Primary config module 'sam2' not found"
**Solution:** Make sure `portable/sam2/` folder exists with all files including `__init__.py`

### Error: "Error locating target 'sam2.modeling.backbones.image_encoder.ImageEncoder'"
**Solution:** 
1. Verify all directories under `portable/sam2/` have `__init__.py`
2. Check that `sys.path` includes the portable directory
3. Ensure you're calling from the same level or using proper path resolution

### Error: "hydra.errors.MissingConfigException"
**Solution:** Make sure `portable/configs/sam2.1_hiera_t512.yaml` exists

### CUDA Error: "no kernel image is available for execution"
**Solution:** Reinstall PyTorch with correct CUDA version (see GPU Support above)

### Out of Memory (OOM)
**Solution:**
- Use CPU mode: `MedSAM2Segmenter(force_cpu=True)`
- Use efficient model: `checkpoint_path="models/efficienttam_ti_512x512.pt"`
- Process smaller images or crop regions

## Folder Structure

```
portable/
├── medsam2_standalone.py          ← Main class (only .py file needed)
├── requirements.txt               ← Dependencies (this file)
├── models/
│   └── MedSAM2_latest.pt         ← Default model (must exist)
├── configs/
│   ├── sam2.1_hiera_t512.yaml    ← Default config (must exist)
│   └── sam2.1/                   ← Additional configs
└── sam2/                          ← SAM2 module (must exist)
    ├── __init__.py               ← IMPORTANT: must exist
    ├── build_sam.py
    ├── _C.so (CUDA module)
    ├── modeling/
    │   ├── __init__.py           ← IMPORTANT: must exist
    │   ├── backbones/
    │   └── ...
    ├── configs/
    │   ├── __init__.py           ← IMPORTANT: must exist
    │   └── ...
    ├── utils/
    │   ├── __init__.py           ← IMPORTANT: must exist
    │   └── ...
    └── csrc/
        └── connected_components.cu
```

## Usage Examples

### Example 1: Single-View (Fast)
```python
from medsam2_standalone import MedSAM2Segmenter
import nibabel as nib

img = nib.load("scan.nii.gz").get_fdata()
seg = MedSAM2Segmenter()
mask = seg.segment(img, (100, 50, 75), (400, 250, 350), ensemble=False)
```

### Example 2: Ensemble (Robust)
```python
mask = seg.segment(img, (100, 50, 75), (400, 250, 350), ensemble=True)
```

### Example 3: Different Model
```python
seg = MedSAM2Segmenter(checkpoint_path="models/MedSAM2_CTLesion.pt")
mask = seg.segment(img, bbox_min, bbox_max)
```

### Example 4: Force CPU Mode
```python
seg = MedSAM2Segmenter(force_cpu=True)
mask = seg.segment(img, bbox_min, bbox_max)
```

### Example 5: Batch Processing
```python
from pathlib import Path

seg = MedSAM2Segmenter()

for img_path in Path("data").glob("*.nii.gz"):
    img = nib.load(img_path).get_fdata()
    mask = seg.segment(img, (100, 50, 75), (400, 250, 350))
    
    out = nib.Nifti1Image(mask, nib.load(img_path).affine)
    nib.save(out, img_path.parent / f"{img_path.stem}_seg.nii.gz")
```

## Default Paths (Auto-Resolved)

When using `MedSAM2Segmenter()` without arguments:
- Model: `models/MedSAM2_latest.pt` (relative to medsam2_standalone.py)
- Config: `configs/sam2.1_hiera_t512.yaml` (relative to script)
- Work dir: Script directory (SCRIPT_DIR)

## Performance Expectations

| Mode | Speed | GPU Memory | Result |
|------|-------|-----------|--------|
| Single-view | ~2 sec | 2-3 GB | 348k voxels |
| Ensemble | ~8 sec | 6-8 GB | 3.1M voxels |

(Times for 548×302×402 image on RTX 5070 Ti)

## Advanced: Custom Paths

```python
seg = MedSAM2Segmenter(
    checkpoint_path="/custom/path/model.pt",
    config_path="/custom/path/config.yaml",
    device="cuda",  # or "cpu"
    work_dir="/custom/work/dir",
    force_cpu=False
)
```

## Support

For issues or questions, check:
1. Ensure all `__init__.py` files exist in sam2 subfolders
2. Verify `portable/configs/` and `portable/models/` exist
3. Check that dependencies are installed: `pip list | grep -E "torch|nibabel|hydra"`
4. Try running on CPU: `MedSAM2Segmenter(force_cpu=True)`

---

**Last Updated:** November 15, 2025
**Status:** Production Ready ✅

