"""游戏档位(GameProfile)定义、加载与注册表。

分区设计:鸣潮/原神与三角洲行动是**两套不同的输入模型**,必须分开存放,
切换档位即切换模型,互不影响(需求:原游戏保留在与三角洲不同的分区)。

    默认分区        21 个独立键(高/中/低 3×7),不使用鼠标修饰键
    三角洲行动分区   7 个音位键 × 4 种鼠标修饰态

档位以独立 yaml 放在 `profiles/` 目录;若目录缺失或为空,回落到 `config.yaml`
的 `keymap` 段合成一个默认档位,保证老用户升级后配置不失效(风险 R8)。
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Sequence

import yaml

from core.compiler import (
    DEFAULT_MODIFIER_BUTTONS,
    ChordPolicy,
    CompileParams,
    Modifier,
    ModifierPolicy,
)

PROFILES_DIRNAME = "profiles"
DEFAULT_PROFILE_ID = "default"

_NOTE_ID_RE = re.compile(r"^(high|mid|low)_[1-7]$")

_MODIFIER_NAMES = {
    "lower": Modifier.LOWER,
    "semitone": Modifier.SEMITONE,
    "higher": Modifier.HIGHER,
}

_POLICY_NAMES = {
    "octave_first": ModifierPolicy.OCTAVE_FIRST,
    "reject_note": ModifierPolicy.REJECT_NOTE,
    "keep_accidental": ModifierPolicy.KEEP_ACCIDENTAL,
}

_CHORD_NAMES = {
    "first": ChordPolicy.CHORD_FIRST,
    "reject": ChordPolicy.CHORD_REJECT,
    "arpeggiate": ChordPolicy.CHORD_ARPEGGIATE,
}


@dataclass(frozen=True)
class GameProfile:
    """一个游戏档位。

    pitch_keys            7 个音位键,从左到右对应 1-7(新模型)
    modifier_buttons      修饰态 → 鼠标键
    pitch_direct_overrides 物理直达键覆盖表(命中后不再按修饰键,决策 D3)
    legacy_keymap         旧 21 键模型;仅默认档位有,三角洲档位为 None
    """

    id: str
    name: str
    group: str = "默认"
    pitch_keys: tuple = ("Z", "X", "C", "V", "B", "N", "M")
    modifier_buttons: dict = field(default_factory=dict)
    pitch_direct_overrides: dict = field(default_factory=dict)
    modifier_policy: ModifierPolicy = ModifierPolicy.OCTAVE_FIRST
    chord_policy: ChordPolicy = ChordPolicy.CHORD_FIRST
    legacy_keymap: dict | None = None

    def __post_init__(self):
        if not self.id:
            raise ValueError("档位缺少 id")
        if len(self.pitch_keys) != 7:
            raise ValueError(f"档位 {self.id}: pitch_keys 必须 7 个")
        for k in self.pitch_direct_overrides:
            if not _NOTE_ID_RE.match(str(k)):
                raise ValueError(f"档位 {self.id}: 非法直达键 id '{k}'")
        for k in self.modifier_buttons:
            if k not in _MODIFIER_NAMES:
                raise ValueError(
                    f"档位 {self.id}: 未知修饰态 '{k}'(应为 {sorted(_MODIFIER_NAMES)})")

    def build_compile_params(self, *, bpm: int, settle_ms: float,
                             release_settle_ms: float, hold_ratio: float,
                             max_hold_ms, gap_ms: float) -> CompileParams:
        """组装编译器参数。修饰态键名从字符串转为 Modifier 枚举。"""
        buttons = {}
        for name, mod in _MODIFIER_NAMES.items():
            btn = self.modifier_buttons.get(name)
            if btn:
                buttons[mod] = btn
        return CompileParams(
            bpm=bpm,
            pitch_keys=self.pitch_keys,
            modifier_buttons=buttons or dict(DEFAULT_MODIFIER_BUTTONS),
            settle_ms=settle_ms,
            release_settle_ms=release_settle_ms,
            hold_ratio=hold_ratio,
            max_hold_ms=max_hold_ms,
            gap_ms=gap_ms,
            pitch_direct_overrides=dict(self.pitch_direct_overrides),
            modifier_policy=self.modifier_policy,
            chord_policy=self.chord_policy,
        )


# ---------------------------------------------------------------- 加载

def _from_dict(data: dict) -> GameProfile:
    if not isinstance(data, dict):
        raise ValueError("档位文件内容必须是对象")
    pid = str(data.get("id") or "").strip()
    if not pid:
        raise ValueError("档位缺少 id")
    mods = data.get("modifier_buttons") or {}
    policy = str(data.get("modifier_policy") or "octave_first")
    chord = str(data.get("chord_policy") or "first")
    if policy not in _POLICY_NAMES:
        raise ValueError(f"档位 {pid}: 未知 modifier_policy '{policy}'")
    if chord not in _CHORD_NAMES:
        raise ValueError(f"档位 {pid}: 未知 chord_policy '{chord}'")
    return GameProfile(
        id=pid,
        name=str(data.get("name") or pid),
        group=str(data.get("group") or "默认"),
        pitch_keys=tuple(data.get("pitch_keys") or ("Z", "X", "C", "V", "B", "N", "M")),
        modifier_buttons={str(k): str(v) for k, v in mods.items()},
        pitch_direct_overrides={str(k): str(v)
                                for k, v in (data.get("pitch_direct_overrides") or {}).items()},
        modifier_policy=_POLICY_NAMES[policy],
        chord_policy=_CHORD_NAMES[chord],
        legacy_keymap=data.get("legacy_keymap"),
    )


def legacy_profile(keymap: dict | None = None) -> GameProfile:
    """由 config.yaml 的 keymap 段合成默认档位(21 键直达,无修饰键)。"""
    km = keymap or {
        "high": list("QWERTYU"),
        "mid": list("ASDFGHJ"),
        "low": list("ZXCVBNM"),
    }
    overrides = {}
    for octave in ("high", "mid", "low"):
        keys = list(km.get(octave) or [])
        for i, key in enumerate(keys[:7], start=1):
            overrides[f"{octave}_{i}"] = str(key)
    return GameProfile(
        id=DEFAULT_PROFILE_ID,
        name="默认(鸣潮 / 原神)",
        group="默认",
        pitch_keys=("Z", "X", "C", "V", "B", "N", "M"),
        pitch_direct_overrides=overrides,
        legacy_keymap=dict(km),
    )


def load_profiles(profiles_dir: str = PROFILES_DIRNAME,
                  fallback_keymap: dict | None = None) -> list[GameProfile]:
    """加载 `profiles/` 下全部档位。目录缺失或为空时回落到 config.yaml 的 keymap。

    yaml 解析失败或内容非法会抛出 ValueError——启动期即暴露,不静默降级。
    """
    profiles: list[GameProfile] = []
    seen: set[str] = set()

    if os.path.isdir(profiles_dir):
        for fn in sorted(os.listdir(profiles_dir)):
            if not fn.lower().endswith((".yaml", ".yml")):
                continue
            path = os.path.join(profiles_dir, fn)
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            try:
                profile = _from_dict(data)
            except Exception as e:
                raise ValueError(f"档位文件 {fn} 解析失败: {e}") from None
            if profile.id in seen:
                raise ValueError(f"档位 id 重复: {profile.id}({fn})")
            seen.add(profile.id)
            profiles.append(profile)

    if not profiles:
        profiles.append(legacy_profile(fallback_keymap))
    elif DEFAULT_PROFILE_ID not in seen:
        # 有 profiles 目录但缺默认档位:补一个,保证老游戏仍可用
        profiles.insert(0, legacy_profile(fallback_keymap))
    return profiles


def resolve_profile(profiles: Sequence[GameProfile], profile_id: str | None) -> GameProfile:
    """按 id 取档位;id 缺失或不存在时回落到 default,再不行取第一个。"""
    for p in profiles:
        if p.id == (profile_id or DEFAULT_PROFILE_ID):
            return p
    for p in profiles:
        if p.id == DEFAULT_PROFILE_ID:
            return p
    return profiles[0]


def grouped(profiles: Sequence[GameProfile]) -> dict:
    """按分区分组,保持文件加载顺序。供 GUI 下拉分组展示。"""
    out: dict = {}
    for p in profiles:
        out.setdefault(p.group, []).append(p)
    return out


# ---------------------------------------------------------------- exe 释放

def ensure_profiles(base_dir: str = ".") -> str:
    """保证工作目录下的 `profiles/` 可用。

    exe(onefile)运行时:打包资源在 _MEIPASS 临时目录,退出即清理,
    用户改过的档位会丢失。故首次运行时释放到 exe 旁,**已存在则不覆盖**。
    开发环境直接返回原路径。
    """
    target = os.path.join(base_dir, PROFILES_DIRNAME)
    if getattr(sys, "frozen", False):
        bundled = os.path.join(getattr(sys, "_MEIPASS", base_dir), PROFILES_DIRNAME)
        if os.path.isdir(bundled) and not os.path.exists(target):
            shutil.copytree(bundled, target)
    return target
