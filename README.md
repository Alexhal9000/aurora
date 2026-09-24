# Aurora backend

This repository is the public Python backend for [Aurora](https://hallgrimssonlab.ca),
a 3D morphological analysis tool from the Hallgrimsson Lab, University of Calgary.

It contains the Django engine and the image-processing and analysis pipeline
used by the desktop app, published so the scientific community can inspect,
verify and build on the methods. The user interface source is not published
here. Installers for Ubuntu, Windows, and macOS are attached to
[GitHub Releases](../../releases).

## Installers

Download the installer for your operating system from the release that
matches the version you want, for example `v1.8.6`. Each release tag also marks
the exact backend source that shipped in those installers.

## AI models

Foundation model weights are not in this repository. After you install
Aurora and sign in, the app can download the models you agree to use.

## Development

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install torch torchvision   # choose the build for your CPU/GPU from pytorch.org
pip install -r requirements.txt
cd AuroraClient
python manage.py migrate
```

## License

Aurora's backend is free software under the
[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

In short: you may use, study, modify, and share it, including for research.
If you distribute a modified version, or let others use a modified version
over a network (for example behind a website login), you must make your
modified source code available to those users under the same license.

Third-party code and models keep their own licenses; see
[THIRD_PARTY_NOTICES.md](AuroraClient/THIRD_PARTY_NOTICES.md).

If you use Aurora in published research, please cite the Aurora article.
