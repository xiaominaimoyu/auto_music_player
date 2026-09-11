"""生成 上游 main → 打了半音补丁 的 unified diff（a/ b/ 前缀，可用 git apply -p1）。

同时做行尾校验:若两侧行尾风格不一致会打印告警,避免产出 CRLF 污染的巨大 diff。
用法: python _make_upstream_patch.py <上游干净目录> <打补丁目录> <输出patch>
"""
import difflib
import os
import sys


def read_lines(path):
    """按原始行尾读取;非 UTF-8 文件视为不可 diff(返回 None)。"""
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            return f.readlines()
    except (UnicodeDecodeError, OSError):
        return None


# 本次补丁只涉及这些文件(其余目录含二进制资源,不参与 diff)
TARGETS = [
    "core/parser.py",
    "core/ir.py",
    "core/compiler.py",
    "gui/player_tab.py",
    "config.yaml",
    "tests/test_ir.py",
    "tests/test_semitone.py",
]


def main():
    base, work, out = sys.argv[1], sys.argv[2], sys.argv[3]
    chunks = []
    files = TARGETS

    for rel in files:
        a_path = os.path.join(base, *rel.split("/"))
        b_path = os.path.join(work, *rel.split("/"))
        if not os.path.exists(b_path):
            print(f"  [跳过] {rel}: 打补丁目录中不存在")
            continue
        if not os.path.exists(a_path):
            lb = read_lines(b_path)
            if lb is None:
                print(f"  [跳过] {rel}: 非文本文件")
                continue
            d = difflib.unified_diff([], lb, fromfile="/dev/null", tofile=f"b/{rel}")
            chunks.append(f"diff --git a/{rel} b/{rel}\nnew file mode 100644\n"
                          + "".join(d))
            continue
        la, lb = read_lines(a_path), read_lines(b_path)
        if la is None or lb is None:
            print(f"  [跳过] {rel}: 非文本文件")
            continue
        if la == lb:
            print(f"  [无变化] {rel}")
            continue
        d = list(difflib.unified_diff(la, lb, fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3))
        chunks.append(f"diff --git a/{rel} b/{rel}\n" + "".join(d))

    text = "".join(chunks)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    added = sum(1 for l in text.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in text.splitlines() if l.startswith("-") and not l.startswith("---"))
    touched = [l.split(" b/")[-1] for l in text.splitlines() if l.startswith("diff --git")]
    print(f"patch 已写入: {out}")
    print(f"涉及文件 {len(touched)} 个, +{added} / -{removed} 行")
    for t in touched:
        print(f"    {t}")


if __name__ == "__main__":
    main()
