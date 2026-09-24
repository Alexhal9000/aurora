# Third-party code in this folder

This folder contains source code from the following projects, each licensed
under the Apache License, Version 2.0 (full text in `LICENSE` next to this file):

| Folder / file | Origin | Copyright |
|---------------|--------|-----------|
| `sam2/` | [SAM 2](https://github.com/facebookresearch/sam2) | Meta Platforms, Inc. and affiliates |
| `efficient_track_anything/` | [EfficientTAM](https://github.com/yformer/EfficientTAM) | Meta Platforms, Inc. and affiliates |
| `configs/` and MedSAM2 changes to the above | [MedSAM2](https://github.com/bowang-lab/MedSAM2) | MedSAM2 authors (Bo Wang Lab) |

`medsam2_standalone.py` and `SETUP.md` are Aurora's wrapper around the MedSAM2
inference code, written by the Hallgrimsson Lab (University of Calgary). Other
files in this folder may also have been modified by the Hallgrimsson Lab to
integrate them with Aurora. All of this is provided under the same Apache
License, Version 2.0.

The MedSAM2 model weights are not part of this folder. Aurora downloads them
separately after the user accepts their license terms.
