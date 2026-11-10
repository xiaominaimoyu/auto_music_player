# Copilot instructions for Auto Music Player

## Repository shape

This is a Windows-first desktop application with a separate Android release line:

- The desktop app is Python 3.10+ with PyQt6. `main.py` loads and merges `config.yaml`, resolves the runtime data directory, loads `profiles/`, creates the database and input/playback services, then starts the GUI.
- `core/` contains the domain and execution pipeline: score parsing and validation, source-score/MIDI/MusicXML/document import, IR/compiler conversion, SQLite persistence, recording/practice, playback, event logging, and Windows input/focus handling.
- `gui/` is the PyQt6 presentation layer. Tabs and dialogs edit or prepare data through the `core/` services rather than implementing a second score model.
- `profiles/` defines game-specific input models. `default` is the 21-key direct-key layout; `delta_force_harmonica` uses seven pitch keys plus mutually exclusive mouse modifiers.
- `android/` is an independent Kotlin/Jetpack Compose application. It shares the jianpu/score JSON concepts and humanization rules with the desktop app but uses Room, screen-coordinate layouts, and Android Accessibility `dispatchGesture` instead of `SendInput`.
- `data/` is runtime state (SQLite database, source files, exports, logs, and OMR components) and is ignored. Store builds use package `LocalState`; portable/installed behavior is selected by `app.data_mode`.

The central desktop flow is **import or record → normalize into a source score/IR → validate and manually correct → compile for the active profile → preview, practice, or play**. Keep source fidelity (tempo, tracks, controls, provenance, and sidecars) separate from the profile-specific playable projection; do not permanently discard source information merely because a target game cannot play it. The default `Player` and profile-aware `EventPlayer` share the same keyboard/mouse driver and event-log format, while their scheduling paths remain distinct.

## Install, run, test, and build

Run commands from the repository root in PowerShell. Use the project virtual environment when available.

```powershell
python -m pip install -r requirements.txt
python main.py
```

The standard-library suite and pytest suite are both maintained. Run them **serially** on Windows because log tests use fixed temporary locations:

```powershell
python -m unittest discover -s tests
python -m pytest -q
```

Useful focused commands:

```powershell
python -m unittest tests.test_parser -v
python -m unittest tests.test_score_io -v
python -m pytest -q tests/test_rest_integration.py
python -m pytest -q tests/test_event_logger_integration.py
python -m pytest -q tests/test_compare_modifier_policies.py
```

For one pytest test, append the node id, for example:

```powershell
python -m pytest -q tests/test_parser.py::TestParser::test_rest
```

Build the Windows executable with PyInstaller:

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean AutoMusicPlayer.spec
```

`AutoMusicPlayer.spec` bundles the default config, icon, profiles, notices, and version metadata. If bundling optional OMR components, set `AUTOMUSIC_OMR_BUNDLE_DIR`; the spec enforces the 500 MB component limit. After a packaged build, manually smoke-test disclaimer acceptance, startup, profile switching, saving a score, a short play, and F8 stop/release.

Build and test the Android line with JDK 17:

```powershell
Set-Location android
.\gradlew.bat :app:assembleDebug
.\gradlew.bat :app:testDebugUnitTest
```

Run one Android JVM test with its Gradle test filter, for example:

```powershell
.\gradlew.bat :app:testDebugUnitTest --tests com.automusic.player.core.JianpuParserTest
```

The debug APK is written to `android/app/build/outputs/apk/debug/app-debug.apk`. Release signing requires an external keystore; never add a keystore or signing credentials to the repository.

## Conventions that affect changes

- **Preserve the score layers.** `core/source_score.py` represents imported musical/source material; `score_model.py` and `ir.py` represent validated playable data; `compiler.py` and profile rules create target input events. Changes to import or adaptation should preserve metadata and provenance while making losses explicit through degradation records/counts.
- **Validate at boundaries.** Use the existing parser, `require_valid`, score I/O, source-score adaptation, and profile validation helpers rather than accepting partially valid notes in GUI code or database code. Imported PDF/DOCX/MusicXML/OMR results must remain in a correction flow before persistence.
- **Profiles are an input-model boundary.** Do not special-case Delta Force behavior in the default keymap or generic player. Add or adjust profile data and the profile/compiler policy so direct keys, modifiers, chord support, and degradation behavior remain isolated.
- **Timing has two meanings.** The desktop playback paths use absolute-clock scheduling and configurable BPM/hold/gap/latency compensation; humanization is session-seeded and must remain reproducible for pause/resume and event-log analysis. Recognition/NPC-style flows intentionally disable or constrain humanization where strict timing is required.
- **Stop and release are safety-critical.** F8, focus loss, shutdown, and error paths must go through the existing stop/abort and release fallbacks. Do not introduce a return path that can leave keyboard keys or mouse modifiers held.
- **Keep logging compatible.** The legacy `PlayLogger` files and the shared structured event logger serve existing log views and retention/audit tooling. Extend their schemas through the existing logger/formatting helpers rather than writing ad-hoc log files from GUI code.
- **Treat configuration and paths as runtime policy.** `config.yaml` is merged with bundled defaults; relative paths resolve differently for source runs, portable executables, installed executables, and MSIX. Use `app_identity`, `data_migration`, and the existing path helpers instead of deriving paths from the current working directory.
- **Avoid real input in automated tests.** Tests use fake drivers, mocks, offscreen Qt, and Windows API guards. Keep parser/compiler/database tests headless; reserve actual keyboard/mouse and foreground-window checks for explicitly platform-gated smoke tests.
- **Keep desktop and Android protocol changes aligned.** The Android parser and score JSON are intended to consume desktop exports, but Android intentionally lacks desktop-only PDF/image OMR, MIDI/MusicXML, recording, practice, and transport features. Update both sides when changing the shared jianpu syntax or exported JSON contract.
- **Do not commit generated/runtime material.** Keep `data/`, logs, personal scores, virtual environments, build output, APKs, signing files, and local configuration out of commits. `THIRD_PARTY_NOTICES.md` and the OMR component license files must remain part of distribution review when packaging.

## Authoritative project references

- `README.md`: user-facing capabilities, configuration semantics, score text protocol, and top-level layout.
- `docs/项目操作文档.md`: maintainer test, packaging, Android build, and release checks.
- `android/README.md`: Android architecture mapping, JDK/build requirements, and device validation.
- `AutoMusicPlayer.spec`: exact Windows packaging inputs and optional OMR bundling behavior.
