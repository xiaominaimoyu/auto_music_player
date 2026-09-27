# Offline OMR component bundle

The source checkout contains only worker templates. A release build uses
`tools/package_omr_components.ps1` to place the verified Windows Audiveris
distribution under `audiveris/` and the separately licensed Jianpu OMR worker
under `jianpu_omr/`. The generated expanded bundle stays in the ignored
`build/` directory (or is supplied to PyInstaller through
`AUTOMUSIC_OMR_BUNDLE_DIR`) instead of committing 300+ MB of third-party
binaries to the source repository.

The PyInstaller spec includes the configured bundle and rejects a bundle larger
than 500 MiB. `main.ensure_omr_components()` copies missing app-managed
components to the selected user data root without overwriting user-installed
components. Each generated component contains `component.json` as its
management/version marker. The component must expose the paths and command-line
contract documented in `core/omr_component.py`.

Audiveris 5.11.0 is an AGPL-3.0 independent process and supports printed
Western staff notation only. jpeditor OMR 0.7.6 is an MIT printed-Jianpu
candidate; it includes ONNX/PaddleOCR-related model/runtime notices and a
verified Node.js 24.14.1 runtime. Its handwritten mode is experimental and has
no accuracy acceptance claim. Before public distribution, record the exact
archive SHA-256, license texts/source offer, and Windows runtime dependencies
in the release manifest.
