// Offline worker protocol for AutoMusic Player.
// Upstream jpeditor files (omr.js, parse.js, models/, node_modules/) are copied
// beside this worker by the pinned component assembly script.

import { readFile, writeFile, stat } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";
import { recognizeImage, recognizedToDoc } from "./omr.js";
import { scoreDocToRich } from "./amp_adapter.mjs";

const MAX_INPUT_BYTES = 100 * 1024 * 1024;
const MIME = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".tif": "image/tiff",
  ".tiff": "image/tiff",
};

function usage() {
  console.error("用法：jianpu_omr.cmd --input IMAGE --output RESULT.json --mode printed|handwritten");
}

function parseArgs(argv) {
  const out = { mode: "printed" };
  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i];
    if (value === "--input") out.input = argv[++i];
    else if (value === "--output") out.output = argv[++i];
    else if (value === "--mode") out.mode = argv[++i];
    else if (value === "--help" || value === "-h") out.help = true;
    else throw new Error(`未知参数：${value}`);
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }
  if (!args.input || !args.output) {
    usage();
    throw new Error("必须提供 --input 和 --output");
  }
  if (!["printed", "handwritten"].includes(args.mode)) {
    throw new Error("--mode 必须是 printed 或 handwritten");
  }
  const input = resolve(args.input);
  const output = resolve(args.output);
  const extension = extname(input).toLowerCase();
  const mime = MIME[extension];
  if (!mime) throw new Error("简谱 OMR worker 只接受 PNG/JPEG/WebP/BMP/TIFF 位图");
  const metadata = await stat(input);
  if (metadata.size > MAX_INPUT_BYTES) throw new Error("OMR 输入图片不能超过 100 MB");
  const bytes = await readFile(input);
  const recognized = await recognizeImage(bytes, { mime, format: "123" });
  const doc = recognizedToDoc(recognized.detail.score);
  const warnings = Array.isArray(doc.diagnostics)
    ? doc.diagnostics.map((item) => String(item.message || item.code || item))
    : [];
  const rich = scoreDocToRich(doc, {
    mode: args.mode,
    rawText: recognized.text,
    warnings,
    recognizerVersion: "0.7.6",
  });
  rich.source_metadata = {
    format: "jianpu-omr-source",
    upstream: "lodebar2026/jpeditor",
    upstream_version: "0.7.6",
    input_extension: extension,
    worker_protocol: 1,
    manual_confirmation_required: true,
  };
  await writeFile(output, `${JSON.stringify(rich, null, 2)}\n`, "utf8");
  console.log(`[AutoMusicPlayer] jpeditor OMR wrote ${rich.notes.length} notes to ${output}`);
}

main().catch((error) => {
  console.error(`[AutoMusicPlayer] jianpu OMR failed: ${error.message || error}`);
  process.exitCode = 1;
});
