# jpeditor Jianpu OMR worker

This directory is a packaging template, not a complete component in the base
source checkout. A release build copies the pinned `jpeditor-omr` v0.7.6
Windows x64 package and the verified Node.js runtime beside these files.

The worker accepts the AutoMusic Player protocol:

```text
jianpu_omr.cmd --input IMAGE --output RESULT.json --mode printed|handwritten
```

It converts jpeditor's 123 output to `auto-music-player-source` JSON and marks
every result as requiring manual confirmation. jpeditor v0.7.6 was verified
only on printed Jianpu bitmap input. The `handwritten` mode is deliberately
labelled experimental and is not an accuracy claim or a validated handwritten
recognizer.

Upstream: https://github.com/lodebar2026/jpeditor/tree/v0.7.6 (MIT). The
assembled package must retain all upstream and dependency license files.
