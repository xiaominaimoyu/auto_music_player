"""重放式 unified diff 应用器(本机无 git/patch 时的替代方案)。

算法:对每个文件,按 hunk 顺序"消费旧行、产出新行",不做就地切片替换,
从根上避免索引偏移。支持新建文件。
用法: python _apply_patch.py <patch> <目标目录>
"""
import os
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)")


def main():
    patch_path, target = sys.argv[1], sys.argv[2]
    with open(patch_path, encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")

    applied, created = [], []
    i = 0
    cur = None
    new_file = False
    while i < len(lines):
        line = lines[i]
        m = FILE_RE.match(line)
        if m:
            cur = m.group(2)
            new_file = False
            i += 1
            if i < len(lines) and lines[i].startswith("new file mode"):
                new_file = True
                i += 1
            continue
        h = HUNK_RE.match(line)
        if h and cur is not None:
            old_start = int(h.group(1))
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "diff --git")):
                b = lines[i]
                if b.startswith(("+", "-", " ")) or b == "":
                    body.append(b)
                i += 1

            fpath = os.path.join(target, *cur.split("/"))
            if new_file:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                content = "".join(b[1:] + "\n" for b in body if b.startswith("+"))
                with open(fpath, "w", encoding="utf-8", newline="") as f:
                    f.write(content)
                created.append(cur)
                new_file = False
                continue

            with open(fpath, encoding="utf-8", newline="") as f:
                tl = f.read().split("\n")

            out = []
            idx = 0                      # 已消费的旧行数
            src = old_start - 1          # 本 hunk 在原文中的起点
            # 1) 抄写 hunk 之前的原文
            out.extend(tl[idx:src])
            idx = src
            # 2) 重放本 hunk
            for b in body:
                if b.startswith(" "):
                    out.append(tl[idx])
                    idx += 1
                elif b.startswith("-"):
                    idx += 1             # 丢弃旧行
                elif b.startswith("+"):
                    out.append(b[1:])    # 产出新行
            # 3) 留存剩余原文,留给下一个 hunk 继续消费
            remainder = tl[idx:]
            tl = out + remainder
            with open(fpath, "w", encoding="utf-8", newline="") as f:
                f.write("\n".join(tl))
            applied.append(cur)
            continue
        i += 1

    print("已修改:", ", ".join(sorted(set(applied))))
    print("已新建:", ", ".join(sorted(set(created))) or "(无)")


if __name__ == "__main__":
    main()
