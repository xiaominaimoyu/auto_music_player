#!/bin/sh
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TARGET_DIR="$ROOT_DIR/."

echo "[Crow5] 启动目录：$TARGET_DIR"
if [ ! -d "$TARGET_DIR" ]; then
  echo "[Crow5] 启动目录不存在"
  exit 1
fi

if [ -f "$TARGET_DIR/package.json" ]; then
  echo "[Crow5] 已检测到 package.json"
fi

if command -v npm >/dev/null 2>&1; then
  echo "[Crow5] npm 可用"
else
  echo "[Crow5] npm 不可用，请先安装 Node.js 或补充 PATH"
fi

echo "[Crow5] 预计启动命令：未识别"
echo "[Crow5] 已登记端口：未识别"
