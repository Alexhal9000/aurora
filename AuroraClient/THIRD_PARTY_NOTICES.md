# Third-party notices

Aurora's backend is licensed under the GNU Affero General Public License v3.0
(see `LICENSE` in the public repository). It includes or depends on the
third-party work below, which remains under its own license.

## Code included in this source tree

| Component | Where | License | Notice |
|-----------|-------|---------|--------|
| SAM 2 (Meta) | `pipeline/MedSAM2/sam2/` | Apache-2.0 | `pipeline/MedSAM2/LICENSE`, `pipeline/MedSAM2/NOTICE.md` |
| EfficientTAM (Meta) | `pipeline/MedSAM2/efficient_track_anything/` | Apache-2.0 | same as above |
| MedSAM2 (Bo Wang Lab) | `pipeline/MedSAM2/` | Apache-2.0 | same as above |
| ALPACA (SlicerMorph Project, Arthur Porto) | adapted in `pipeline/ALPACA.py` | BSD-2-Clause | `pipeline/third_party_licenses/SlicerMorph-BSD-2-Clause.txt` |

Apache-2.0 and BSD-2-Clause code may be combined into an AGPL-3.0 work; the
original notices above must be kept.

## Model weights (not in this source tree)

Downloaded by the app from the Hallgrimsson Lab server only after the user
accepts each provider's terms, which are shown verbatim in the app
(`pipeline/aiModels/licenses/`).

| Model | License |
|-------|---------|
| MedSAM2 | Apache-2.0 |
| DINOv3 ViT-L/16 (Meta; not offered in this release) | DINOv3 License (`pipeline/dinoReg/LICENSE.md`), not an open-source license and not covered by the AGPL |

## Python dependencies

Installed from PyPI on the user's machine at install time (`requirements.txt`);
they are not redistributed in this repository. Nearly all are under permissive
licenses (BSD, MIT, Apache-2.0, ISC, PSF). Notable exceptions:

- **FireANTs** (Rohit Jena, Pratik Chaudhari, James C. Gee): custom license
  derived from Apache-2.0. It allows use as a dependency installed from PyPI,
  provided its license, copyright notices and bibliography are kept; copying
  its code without those is prohibited. Aurora only imports it. If you use
  Aurora's GPU rigid registration in research, please also cite:

  > Jena R, Chaudhari P, Gee JC. FireANTs: Adaptive Riemannian Optimization for
  > Multi-Scale Diffeomorphic Registration. *Nature Communications* (2024).

- **usd-core** (Pixar): Tomorrow Open Source Technology License 1.0
  (Apache-2.0 with a trademark clause).
- **certifi**, **fqdn**: MPL-2.0 (file-level copyleft; used unmodified).
