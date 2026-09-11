"""校验已打包 exe 内的字节码版本特征。

递归遍历 code 对象的 co_names/co_varnames/co_consts,按"应存在/应不存在"两组标识判定版本。
用法: python _verify_exe_version.py <exe路径> <版本标签> <应存在名,逗号分隔> <应不存在名,逗号分隔>
"""
import os
import sys

from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader

exe, label = sys.argv[1], sys.argv[2]
must_have = [s for s in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if s]
must_not = [s for s in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if s]

TARGETS = ["gui.player_tab", "gui.main_window", "core.ir", "core.parser",
           "core.compiler", "core.humanize", "main"]


def collect(code, names, depth=0):
    if depth > 12:
        return
    names.update(code.co_names)
    names.update(code.co_varnames)
    for c in code.co_consts:
        if isinstance(c, str):
            names.add(c)
        elif hasattr(c, "co_names"):
            collect(c, names, depth + 1)


def main():
    print(f"### 目标: {exe}")
    print(f"### 期望版本: {label}")
    ca = CArchiveReader(exe)
    pyz = [n for n in ca.toc if "pyz" in n.lower()][0]
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v.pyz")
    with open(tmp, "wb") as f:
        f.write(ca.extract(pyz))
    za = ZlibArchiveReader(tmp)

    allnames = set()
    for mod in TARGETS:
        if mod in za.toc:
            collect(za.extract(mod), allnames)
    try:
        os.remove(tmp)
    except OSError:
        pass

    fails = 0
    for m in must_have:
        ok = m in allnames
        print(f"  [{'PASS' if ok else 'FAIL'}] 应存在: {m}")
        fails += 0 if ok else 1
    for m in must_not:
        ok = m not in allnames
        print(f"  [{'PASS' if ok else 'FAIL'}] 应不存在: {m}")
        fails += 0 if ok else 1
    print(f"---- 结论: {'通过' if not fails else f'{fails} 项不符'} ----")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
