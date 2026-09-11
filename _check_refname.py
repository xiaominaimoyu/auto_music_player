"""用 git 的权威实现(dulwich)实测标签名是否合法 —— 即 GitHub Release 接受与否。

dulwich.refs.check_ref_format 与 git check-ref-format 同源规则。
注意:GitHub 额外要求 Release 的标签通过"引用名"校验,故这里同时测 refname 规则。
"""
import sys

from dulwich.refs import check_ref_format

CASES = [
    # (标签名, 说明)
    ("v1.1.4", "推荐写法"),
    ("v1.1.3", "已存在的版本号(会被 GitHub 判为重名)"),
    ("v1.1.3-semitone", "版本 + 英文后缀"),
    ("v1.2.0-rc1", "预发布"),
    ("v1.1.4 半音", "含空格"),
    ("v1.1.4（半音）", "中文全角括号"),
    ("v1.1.4(半音)", "半角括号"),
    ("v1.1.4-半音", "中文但无空格无标点"),
    ("v1.1.4:半音", "含冒号"),
    ("v1.1.4~1", "含波浪号"),
    ("v1.1.4^", "含插入符"),
    ("v1.1.4?", "含问号"),
    ("v1.1.4*", "含星号"),
    ("v1.1.4[1]", "含方括号"),
    ("v1.1.4\\x", "含反斜杠"),
    ("v1..1.4", "含连续点"),
    ("v1.1.4.", "以点结尾"),
    (".v1.1.4", "以点开头"),
    ("v1.1.4.lock", "以 .lock 结尾"),
    ("-v1.1.4", "以连字符开头"),
    ("v1.1.4/", "以斜杠结尾"),
    ("/v1.1.4", "以斜杠开头"),
    ("v1.1.4//x", "连续斜杠"),
    ("v1.1.4@{1}", "含 @{"),
    ("v1.1.4@", "以 @ 结尾"),
    ("", "空字符串"),
    ("v1.1.4 ", "尾部空格"),
    ("发布v1.1.4", "中文前缀"),
    ("v1.1.4+meta", "含加号"),
    ("v1.1.4#1", "含井号"),
]


def main():
    import inspect
    sig = inspect.signature(check_ref_format)
    print(f"### dulwich.refs.check_ref_format 签名: {sig}")
    print(f"### 文档: {(check_ref_format.__doc__ or '(无)').strip()[:160]}")
    print()
    print(f"{'标签名':<22} {'合法':<6} 说明")
    print("-" * 72)
    ok_list, bad_list = [], []
    for name, desc in CASES:
        full = f"refs/tags/{name}"          # dulwich 要求完整引用名(必须含 '/')
        try:
            ok = bool(check_ref_format(full.encode("utf-8")))
            err = "返回 False"
        except Exception as e:
            ok = False
            err = type(e).__name__
        label = "✅ 可以" if ok else "❌ 不行"
        extra = "" if ok else f"  ({err})"
        print(f"{name!r:<22} {label:<6} {desc}{extra}")
        (ok_list if ok else bad_list).append(name)
    print("-" * 72)
    print(f"合法 {len(ok_list)} 个 / 非法 {len(bad_list)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
