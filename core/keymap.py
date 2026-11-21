"""音符 -> 键盘按键映射。"""

OCTAVES = ("high", "mid", "low")
NOTES = tuple(range(1, 8))


class KeyMap:
    def __init__(self, mapping: dict, *, allow_partial: bool = False):
        """mapping: {"high": [Q..U], "mid": [A..J], "low": [Z..M]}"""
        self._mapping = self._validated_mapping(mapping, allow_partial=allow_partial)

    @staticmethod
    def _validated_mapping(mapping: dict, *, allow_partial: bool) -> dict:
        if not isinstance(mapping, dict):
            raise ValueError("keymap 必须是对象")
        try:
            validated = {octave: list(keys) for octave, keys in mapping.items()}
        except TypeError as exc:
            raise ValueError("keymap 的每个八度必须是按键列表") from exc
        for octave in OCTAVES:
            count = len(validated.get(octave, []))
            if count == 7 or (allow_partial and count == 0):
                continue
            expected = "0 或 7 个按键" if allow_partial else "7 个按键"
            raise ValueError(f"keymap.{octave} 必须包含 {expected}")
        return validated

    def replace(self, mapping: dict, *, allow_partial: bool = False) -> None:
        """完整校验后更新当前对象，失败时保留旧映射。"""
        validated = self._validated_mapping(mapping, allow_partial=allow_partial)
        self._mapping.clear()
        self._mapping.update(validated)

    def key_for(self, note_id: str) -> str | None:
        """note_id 形如 "high_1" / "mid_3" / "low_5",返回键盘按键。"""
        if "_" not in note_id:
            return None
        octave, num = note_id.split("_", 1)
        keys = self._mapping.get(octave)
        if not keys:
            return None
        try:
            idx = int(num) - 1
        except ValueError:
            return None
        if not 0 <= idx < 7:
            return None
        return keys[idx]