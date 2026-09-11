"""校验已打包 exe 内的字节码确实是"修复版"。

PyInstaller 的 ZlibArchiveReader.extract() 直接返回 code 对象;
递归遍历 code.co_consts 与 co_names,查找修复引入的标识(变量名/函数名/文档串)。

用法: python _verify_exe.py <exe路径> [<期望源码目录>]
"""
import os
import sys

from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader

exe = sys.argv[1]
src_dir = sys.argv[2] if len(sys.argv) > 2 else None

# 修复引入的关键标识:仅"修复版"才有的名字(缺失即说明打的是旧码)
REQUIRED = {
    "gui.player_tab": ["play_stop", "_active_player", "_is_playing",
                       "_countdown_seconds", "_last_degradations"],
    "gui.main_window": ["play_stop"],
    "core.ir": ["_item_semitone"],
    "core.parser": ["_SHARP_MARK"],
    "core.compiler": ["pitch_direct_overrides"],
    "core.event_player": ["panic_release"],
}
# 这些标识在修复前后都存在,只作信息展示
INFO = {
    "core.compiler": ["resolve_modifier", "pitch_keys"],
    "core.event_player": ["_release_all"],
}


def collect(code, names, consts, depth=0):
    """递归收集 code 对象里的名字与字符串常量。"""
    if depth > 12:
        return
    names.update(code.co_names)
    names.update(code.co_varnames)
    for c in code.co_consts:
        if isinstance(c, str):
            consts.add(c)
        elif hasattr(c, "co_names"):
            collect(c, names, consts, depth + 1)


def main():
    print(f"### 检查目标: {exe}")
    ca = CArchiveReader(exe)
    pyz_names = [n for n in ca.toc if "pyz" in n.lower()]
    if not pyz_names:
        print("未找到 PYZ 归档")
        return 2
    data = ca.extract(pyz_names[0])
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyz_dump.pyz")
    with open(tmp, "wb") as f:
        f.write(data)
    za = ZlibArchiveReader(tmp)
    print(f"### PYZ: {pyz_names[0]}  条目数 {len(za.toc)}")

    fails, passes = [], []
    for mod, marks in REQUIRED.items():
        if mod not in za.toc:
            print(f"[SKIP] {mod}: 未打包")
            continue
        code = za.extract(mod)
        names, consts = set(), set()
        collect(code, names, consts)
        blob = names | consts
        present = {m: (m in blob) for m in marks}
        missing = [m for m in marks if not present[m]]
        tag = "PASS" if not missing else "FAIL"
        (passes if not missing else fails).append(mod)
        extra = INFO.get(mod, [])
        detail = ", ".join(f"{m}={'Y' if present[m] else 'N'}" for m in marks)
        if extra:
            detail += "  | 参照: " + ", ".join(
                f"{m}={'Y' if m in blob else 'N'}" for m in extra)
        print(f"[{tag}] {mod}  {detail}")

    if src_dir:
        print("\n### 源码对照")
        for mod in REQUIRED:
            rel = os.path.join(src_dir, *mod.split(".")) + ".py"
            if os.path.exists(rel):
                n = sum(1 for _ in open(rel, encoding="utf-8"))
                print(f"    {mod:20} {n} 行")

    try:
        os.remove(tmp)
    except OSError:
        pass
    print(f"\n---- {len(passes)} 个模块含全部修复标识, {len(fails)} 个缺失 ----")
    if fails:
        print("缺失:", ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
