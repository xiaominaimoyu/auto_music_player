"""补丁决定性校验:逐 hunk 重建,并逐字节比对打补丁后的文件。

判据(不依赖任何外部 patch 工具):
  1. hunk 头声明的行数 == hunk 体内 '-'/' '(旧侧) 与 '+'/' '(新侧) 的实际行数
  2. 用旧块(上下文+删除行)在基线文件里能唯一定位(按内容搜索)
  3. 把定位到的旧块替换为新块(上下文+新增行)后,整个文件与"打补丁后的目标文件"逐字节相同
三项全过 = 补丁干净且等价于实际改动。

用法: python _verify_patch_definitive.py <patch> <基线目录> <目标目录>
"""
import os
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)")


def read_text(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def added_lines(seg):
    """提取补丁中真正的新增内容行(排除 +++ / --- 文件头)。"""
    out = []
    for l in seg.split("\n"):
        if l.startswith("+") and not l.startswith("+++"):
            out.append(l[1:] + "\n")
    return "".join(out)


def main():
    patch_path, base, work = sys.argv[1], sys.argv[2], sys.argv[3]
    text = read_text(patch_path)
    segments = [s for s in re.split(r"(?=^diff --git )", text, flags=re.M) if s.strip()]

    total_hunks = 0
    fails = []
    for seg in segments:
        m = FILE_RE.match(seg.split("\n", 1)[0])
        if not m:
            continue
        rel = m.group(2)
        base_path = os.path.join(base, *rel.split("/"))
        work_path = os.path.join(work, *rel.split("/"))
        if not os.path.exists(work_path):
            fails.append(f"{rel}: 目标文件不存在")
            continue
        target = read_text(work_path)

        if not os.path.exists(base_path):
            # 新增文件:整文件应与 hunk 的 '+' 行一致(排除 +++ 头)
            added = added_lines(seg)
            ok = added == target
            print(f"  [{'OK  ' if ok else 'FAIL'}] {rel}  (新增文件, {len(target)} 字符)")
            if not ok:
                fails.append(f"{rel}: 新增文件内容与补丁不一致")
            continue

        src = read_text(base_path)
        lines = seg.split("\n")
        i = 0
        file_ok = True
        note = []
        while i < len(lines):
            h = HUNK_RE.match(lines[i])
            if not h:
                i += 1
                continue
            total_hunks += 1
            old_n = int(h.group(2)) if h.group(2) is not None else 1
            new_n = int(h.group(4)) if h.group(4) is not None else 1
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "diff --git")):
                if lines[i].startswith(("+", "-", " ")):
                    body.append(lines[i])
                i += 1
            old_cnt = sum(1 for b in body if b[0] in "- ")
            new_cnt = sum(1 for b in body if b[0] in "+ ")
            if (old_cnt, new_cnt) != (old_n, new_n):
                file_ok = False
                note.append(f"行数不符 @@ -{old_n} +{new_n} vs 实际 -{old_cnt} +{new_cnt}")
                continue
            old_block = [b[1:] for b in body if b[0] in "- "]
            new_block = [b[1:] for b in body if b[0] in "+ "]
            hay = "\n".join(old_block)
            cnt = src.count(hay)
            if cnt == 0:
                file_ok = False
                note.append("旧块在基线中找不到")
                continue
            if cnt > 1:
                note.append(f"旧块匹配 {cnt} 处(将替换首处)")
            src = src.replace(hay, "\n".join(new_block), 1)
        ok = file_ok and src == target
        print(f"  [{'OK  ' if ok else 'FAIL'}] {rel}" + (f"  {note}" if note else ""))
        if not ok:
            fails.append(rel)

    print(f"\n---- hunk 总数 {total_hunks}, 失败文件 {len(fails)} ----")
    if fails:
        print("失败:", ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
