# Third-party notices

## delta-melodica

Parts of the MIDI source reader and melody adaptation design are adapted from
[gujingyun/delta-melodica](https://github.com/gujingyun/delta-melodica).

MIT License

Copyright (c) 2026 gujingyun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## mido 1.3.3

MIDI file parsing uses `mido`, distributed under the MIT License.

The MIT License

Copyright (c) Ole Martin Bjørndalen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## pypdf

Text-only PDF extraction optionally uses [pypdf](https://github.com/py-pdf/pypdf),
distributed under the BSD-3-Clause license. It is not an OCR engine; scanned pages
are routed to the offline OMR boundary.

## python-docx

Text-only DOCX extraction optionally uses [python-docx](https://github.com/python-openxml/python-docx),
distributed under the MIT License. Its transitive `lxml` dependency keeps its own
license notices in the packaged dependency metadata.

## cryptography

The signed update manifest verifier optionally uses
[cryptography](https://github.com/pyca/cryptography), distributed under the Apache
License 2.0 and BSD-derived license terms described by that project.

## Audiveris (optional OMR component)

If the optional offline staff-notation component is bundled, it contains
[Audiveris](https://github.com/Audiveris/audiveris), distributed under the GNU
Affero General Public License v3.0. The release package must ship the applicable
source/offer and license text before public distribution. Audiveris supports
printed common Western notation and is not a handwriting recognizer.

The verified 5.11.0 Windows console MSI source is:
`https://github.com/Audiveris/audiveris/releases/download/5.11.0/Audiveris-5.11.0-windowsConsole-x86_64.msi`
and its recorded SHA-256 is
`5f1b4e96a12c53c7da426814b76e599363c4181e291855996e0a6878dda95f71`.
The MSI and the extracted Audiveris launcher were not Authenticode-signed in
the local verification; public distribution must retain the AGPL license and
source offer.

## jpeditor Jianpu OMR (optional component)

The printed-Jianpu worker is built from
[lodebar2026/jpeditor v0.7.6](https://github.com/lodebar2026/jpeditor/tree/v0.7.6),
under the MIT License. The verified Windows x64 upstream worker ZIP used for
the assembly has SHA-256
`1025e8757c8a77362f84c5ecc26750719f180817360fb8b91640c13f68e9c2c8`.
The package retains the upstream dependency license files; its ONNX runtime,
Sharp/libvips, PaddleOCR-derived models and Microsoft VC runtime must be
reviewed against their own notices before a public release.

The worker bundles Node.js v24.14.1 from the official Windows x64 distribution;
the verified `node.exe` SHA-256 is
`58e74bf02fc5bbacc41dcb8bef089961cd5bddd37830b87784e4fc624d145d1f`. Node.js is distributed under its
own included license. The worker accepts printed bitmap input, writes the
project rich-source JSON, and requires manual confirmation. No validated
handwritten OMR model is included.
