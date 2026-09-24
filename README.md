# Aurora backend

This repository is the public Python backend for [Aurora](https://hallgrimssonlab.ca),
a 3D morphological analysis tool from the Hallgrimsson Lab, University of Calgary.

It contains the Django engine and the image-processing and analysis pipeline
used by the desktop app, published so the scientific community can inspect,
verify and build on the methods. The user interface source is not published
here. Installers for Ubuntu, Windows, and macOS are attached to
[GitHub Releases](../../releases).

## Installers

The version cited in the Aurora article is **1.8.7**. These links stay up;
later wizard releases are extra, not replacements:

- [Release v1.8.7](https://github.com/Alexhal9000/aurora/releases/tag/v1.8.7)
- [Ubuntu `.deb`](https://github.com/Alexhal9000/aurora/releases/download/v1.8.7/aurora-tools_1.8.7_amd64.deb)
- [Windows `.exe`](https://github.com/Alexhal9000/aurora/releases/download/v1.8.7/aurora_1.8.7_setup.exe)
- [macOS `.pkg`](https://github.com/Alexhal9000/aurora/releases/download/v1.8.7/aurora_1.8.7_macos.pkg)

Every later GitHub release is also kept. Each tag marks the backend that
shipped in that version's installers. The lab site requires **1.8.7 or newer**.

## AI models

Foundation model weights are not in this repository. After you install
Aurora and sign in, the app can download the models you agree to use.

## Documentation

The methods are explained step by step, with algorithm notes, parameters and
animated figures, in the Aurora documentation:

- [Interactive documentation](https://www.hallgrimssonlab.ca/MainAurora/documentation):
  the same docs panel as in the desktop app, readable online without installing Aurora.
- [Single-page corpus](https://www.hallgrimssonlab.ca/static/frontend/aurora-docs-corpus.html):
  all entries on one page, easy to search or cite. Also available as
  [plain text](https://www.hallgrimssonlab.ca/static/frontend/aurora-docs-corpus.txt) and
  [JSON](https://www.hallgrimssonlab.ca/static/frontend/aurora-docs-corpus.json).

## Reading the source

This snapshot is for inspection, not for running the desktop app. The user
interface is not here, so `manage.py` alone will not give you Aurora.

The analysis logic lives under `AuroraClient/pipeline/`, and the documentation
entries above point to the methods implemented there. Each GitHub release tag
is the backend that shipped in that version's installers.

`requirements.txt` and `requirements-macos.txt` list the Python packages that
version used. They are a record of the stack, not a setup recipe.

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
