"""校验 unified diff 能否干净应用到目标目录(本机无 git/patch 时的替代方案)。

对每个 hunk:
  - 按 @@ -a,b +c,d @@ 定位
  - 逐行比对上下文行(' ')与删除行('-')是否与目标文件完全一致
  - 统计可应用/失败
用法: python _verify_patch.py <patch> <目标目录> [--reverse]
"""
import os
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)")


def main():
    patch_path, target = sys.argv[1], sys.argv[2]
    reverse = "--reverse" in sys.argv
    with open(patch_path, encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")

    cur = None
    results = []
    i = 0
    hunks = 0
    bad = 0
    while i < len(lines):
        line = lines[i]
        m = FILE_RE.match(line)
        if m:
            cur = m.group(2)
            results.append({"file": cur, "ok": True, "detail": []})
            i += 1
            continue
        h = HUNK_RE.match(line)
        if h and cur is not None:
            hunks += 1
            old_start = int(h.group(1))
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "diff --git")):
                if lines[i].startswith(("+", "-", " ")) or lines[i] == "":
                    body.append(lines[i])
                i += 1
            # 目标文件内容
            fpath = os.path.join(target, *cur.split("/"))
            if not os.path.exists(fpath):
                results[-1]["ok"] = False
                results[-1]["detail"].append(f"文件不存在: {cur}")
                bad += 1
                continue
            with open(fpath, encoding="utf-8", newline="") as f:
                tl = f.read().split("\n")
            # 收集期望的旧内容(上下文 + 删除行);reverse 时改为 上下文 + 新增行
            want = []
            for b in body:
                if b.startswith(" "):
                    want.append(b[1:])
                elif b.startswith("-") and not reverse:
                    want.append(b[1:])
                elif b.startswith("+") and reverse:
                    want.append(b[1:])
            actual = tl[old_start - 1: old_start - 1 + len(want)]
            if actual != want:
                results[-1]["ok"] = False
                bad += 1
                for k, (w, a) in enumerate(zip(want, actual)):
                    if w != a:
                        results[-1]["detail"].append(
                            f"{cur}:{old_start + k} 期望 {w!r} 实际 {a!r}")
                        break
            continue
        i += 1

    mode = "反向" if reverse else "正向"
    print(f"### {mode}应用校验: {patch_path}")
    print(f"### 目标目录: {target}")
    for r in results:
        tag = "OK  " if r["ok"] else "FAIL"
        print(f"  [{tag}] {r['file']}")
        for d in r["detail"]:
            print(f"         {d}")
    print(f"---- hunk 总数 {hunks}, 不匹配 {bad} ----")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
