"""MusicXML/MXL 到 SourceSong 的窄而可验证的离线适配器。

它不是完整排版引擎，只实现 OMR sidecar 最常见的 part/measure/note/rest/
chord/tie/tempo 子集。遇到无法保证语义的结构会给出 warning，交由人工校对
门禁处理，而不是静默转换成 MIDI 再丢失信息。
"""

from __future__ import annotations

import os
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from core.source_score import SourceNote, SourceSong


MAX_MUSICXML_BYTES = 500 * 1024 * 1024
_STEP_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class MusicXmlImportError(ValueError):
    pass


@dataclass(frozen=True)
class MusicXmlReadResult:
    song: SourceSong
    warnings: list[str] = field(default_factory=list)
    source_metadata: dict = field(default_factory=dict)


def _local(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _children(element, name: str):
    return [child for child in list(element) if _local(child.tag) == name]


def _first(element, name: str):
    for child in list(element):
        if _local(child.tag) == name:
            return child
    return None


def _int_text(element, default=0):
    if element is None or element.text is None:
        return default
    try:
        return int(str(element.text).strip())
    except (TypeError, ValueError):
        return default


def _float_text(element, default=0.0):
    if element is None or element.text is None:
        return default
    try:
        return float(str(element.text).strip())
    except (TypeError, ValueError):
        return default


def _read_payload(path: str) -> bytes:
    resolved = os.path.abspath(os.fspath(path))
    size = os.path.getsize(resolved)
    if size > MAX_MUSICXML_BYTES:
        raise MusicXmlImportError("MusicXML 文件不能超过 500 MB")
    extension = os.path.splitext(resolved)[1].lower()
    if extension != ".mxl":
        with open(resolved, "rb") as stream:
            return stream.read(MAX_MUSICXML_BYTES + 1)
    try:
        with zipfile.ZipFile(resolved) as archive:
            names = archive.namelist()
            root_name = None
            if "META-INF/container.xml" in names:
                container = ET.fromstring(archive.read("META-INF/container.xml"))
                for element in container.iter():
                    if _local(element.tag) == "rootfile":
                        root_name = element.attrib.get("full-path")
                        if root_name:
                            break
            if not root_name:
                candidates = [
                    name for name in names
                    if name.lower().endswith((".musicxml", ".xml"))
                    and not name.lower().startswith("meta-inf/")
                ]
                root_name = candidates[0] if candidates else None
            if not root_name or root_name not in names:
                raise MusicXmlImportError("MXL 容器中没有可识别的 MusicXML 根文件")
            payload = archive.read(root_name)
            if len(payload) > MAX_MUSICXML_BYTES:
                raise MusicXmlImportError("MXL 解压后的 MusicXML 不能超过 500 MB")
            return payload
    except MusicXmlImportError:
        raise
    except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise MusicXmlImportError(f"MXL 容器读取失败：{exc}") from exc


def _pitch(note_element):
    pitch = _first(note_element, "pitch")
    if pitch is None:
        return None
    step_element = _first(pitch, "step")
    step = step_element.text if step_element is not None else None
    if not step or str(step).strip().upper() not in _STEP_TO_PC:
        return None
    octave = _int_text(_first(pitch, "octave"), 4)
    alter = _float_text(_first(pitch, "alter"), 0.0)
    if abs(alter - round(alter)) > 1e-9:
        return None
    midi = 12 * (octave + 1) + _STEP_TO_PC[str(step).strip().upper()] + int(round(alter))
    return midi if 0 <= midi <= 127 else None


def _tempo_from_direction(direction):
    sound = _first(direction, "sound")
    if sound is not None and sound.attrib.get("tempo"):
        try:
            return float(sound.attrib["tempo"])
        except ValueError:
            pass
    direction_type = _first(direction, "direction-type")
    metronome = _first(direction_type, "metronome") if direction_type is not None else None
    per_minute = _first(metronome, "per-minute") if metronome is not None else None
    value = _float_text(per_minute, 0.0)
    return value if value > 0 else None


def _tie_types(note_element) -> set[str]:
    """同时读取 MusicXML 的 sound tie 与 notation tied 标记。"""

    values = {
        str(item.attrib.get("type", "")).strip().lower()
        for item in _children(note_element, "tie")
    }
    notations = _first(note_element, "notations")
    if notations is not None:
        values.update(
            str(item.attrib.get("type", "")).strip().lower()
            for item in _children(notations, "tied")
        )
    return {value for value in values if value in {"start", "stop", "continue"}}


def read_musicxml_source(path: str) -> MusicXmlReadResult:
    payload = _read_payload(path)
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise MusicXmlImportError(f"MusicXML XML 解析失败：{exc}") from exc
    if _local(root.tag) not in ("score-partwise", "score-timewise"):
        raise MusicXmlImportError("仅支持 score-partwise/score-timewise MusicXML")
    if _local(root.tag) == "score-timewise":
        raise MusicXmlImportError("暂不支持 score-timewise，请由 OMR 组件导出 score-partwise")

    part_names = {}
    part_list = _first(root, "part-list")
    if part_list is not None:
        for part in _children(part_list, "score-part"):
            part_id = part.attrib.get("id", "")
            name = _first(part, "part-name")
            part_names[part_id] = str(name.text).strip() if name is not None and name.text else part_id

    warnings = []
    records = []
    tempo_events = []
    time_signatures = []
    parts = _children(root, "part")
    for track_index, part in enumerate(parts):
        part_id = part.attrib.get("id", str(track_index))
        part_name = part_names.get(part_id, part_id)
        part_beat = 0.0
        divisions = 1.0
        previous_note_start = None
        open_ties = {}
        for measure in _children(part, "measure"):
            measure_start = part_beat
            local_beat = 0.0
            previous_note_start = None
            attributes = _first(measure, "attributes")
            if attributes is not None:
                parsed_divisions = _float_text(_first(attributes, "divisions"), divisions)
                if parsed_divisions > 0:
                    divisions = parsed_divisions
                time = _first(attributes, "time")
                if time is not None:
                    numerator = _int_text(_first(time, "beats"), 4)
                    denominator = _int_text(_first(time, "beat-type"), 4)
                    time_signatures.append(
                        {
                            "part": part_id,
                            "beat": measure_start,
                            "numerator": numerator,
                            "denominator": denominator,
                        }
                    )
            for child in list(measure):
                tag = _local(child.tag)
                if tag == "direction":
                    tempo = _tempo_from_direction(child)
                    if tempo:
                        tempo_events.append({"beat": measure_start + local_beat, "bpm": tempo})
                    continue
                if tag == "backup":
                    local_beat -= _int_text(_first(child, "duration"), 0) / divisions
                    continue
                if tag == "forward":
                    local_beat += _int_text(_first(child, "duration"), 0) / divisions
                    continue
                if tag != "note":
                    continue
                duration = _int_text(_first(child, "duration"), 0) / divisions
                if duration <= 0:
                    duration = 1.0 / max(divisions, 1.0)
                    warnings.append(f"part {part_id} 存在无时值音符，按最小时值处理")
                is_chord = _first(child, "chord") is not None
                note_start = (
                    previous_note_start if is_chord and previous_note_start is not None
                    else measure_start + local_beat
                )
                note_end = note_start + duration
                if _first(child, "grace") is not None:
                    warnings.append(f"part {part_id} 的 grace 音符按普通最小时值处理")
                pitch = _pitch(child)
                if pitch is not None:
                    voice_element = _first(child, "voice")
                    staff_element = _first(child, "staff")
                    voice = (
                        str(voice_element.text or "").strip()
                        if voice_element is not None
                        else ""
                    )
                    staff = (
                        str(staff_element.text or "").strip()
                        if staff_element is not None
                        else ""
                    )
                    ties = _tie_types(child)
                    tie_start = bool(ties & {"start", "continue"})
                    tie_stop = bool(ties & {"stop", "continue"})
                    tie_key = (track_index, voice, staff, pitch)
                    record = {
                        "track": track_index,
                        "part": part_id,
                        "part_name": part_name,
                        "pitch": pitch,
                        "start_beat": note_start,
                        "end_beat": note_end,
                        "voice": voice,
                        "staff": staff,
                        "chord": is_chord,
                        "tie_start": tie_start,
                        "tie_stop": tie_stop,
                    }
                    active = open_ties.get(tie_key)
                    contiguous = (
                        active is not None
                        and abs(float(active["end_beat"]) - float(note_start)) <= 1e-7
                    )
                    if tie_stop and contiguous:
                        active["end_beat"] = max(float(active["end_beat"]), note_end)
                        active["tie_segments"] = int(active.get("tie_segments", 1)) + 1
                        active["tie_start"] = tie_start
                        active["tie_stop"] = True
                        if not tie_start:
                            open_ties.pop(tie_key, None)
                    else:
                        if tie_stop:
                            warnings.append(
                                f"part {part_id} voice {voice or '-'} pitch {pitch} "
                                "存在未匹配的 tie stop，已保留为独立音符"
                            )
                        records.append(record)
                        if tie_start:
                            if active is not None:
                                warnings.append(
                                    f"part {part_id} voice {voice or '-'} pitch {pitch} "
                                    "存在重叠 tie start，未猜测合并"
                                )
                            open_ties[tie_key] = record
                else:
                    if _first(child, "rest") is None:
                        warnings.append(f"part {part_id} 存在无法转换的音高，已保留为待校对警告")
                if not is_chord:
                    local_beat += duration
                    previous_note_start = note_start
            part_beat = max(part_beat, measure_start + max(local_beat, 0.0))
        for (_track, voice, _staff, pitch), record in open_ties.items():
            warnings.append(
                f"part {part_id} voice {voice or '-'} pitch {pitch} "
                f"从 beat {record['start_beat']:g} 开始的 tie 未闭合"
            )

    if not records:
        raise MusicXmlImportError("MusicXML 没有可转换的有声音符")
    tempo_events.sort(key=lambda item: item["beat"])
    if not tempo_events:
        tempo_events = [{"beat": 0.0, "bpm": 100.0}]
    elif tempo_events[0]["beat"] > 0:
        tempo_events.insert(0, {"beat": 0.0, "bpm": 100.0})

    def beat_to_seconds(beat: float) -> float:
        seconds = 0.0
        last_beat = 0.0
        bpm = float(tempo_events[0]["bpm"])
        for event in tempo_events[1:]:
            event_beat = float(event["beat"])
            if beat <= event_beat:
                break
            seconds += max(0.0, event_beat - last_beat) * 60.0 / bpm
            last_beat = event_beat
            bpm = float(event["bpm"])
        seconds += max(0.0, beat - last_beat) * 60.0 / bpm
        return seconds

    source_notes = [
        SourceNote(
            beat_to_seconds(float(record["start_beat"])),
            beat_to_seconds(float(record["end_beat"])),
            int(record["pitch"]),
            int(record["track"]),
        )
        for record in records
    ]
    duration_s = max(note.end_s for note in source_notes)
    tracks = {index: part_names.get(part.attrib.get("id", str(index)), str(index)) for index, part in enumerate(parts)}
    metadata = {
        "schema_version": 1,
        "format": "musicxml-source",
        "parts": tracks,
        "tempo_events": tempo_events,
        "time_signatures": time_signatures,
        "notes": records,
        "manual_confirmation_required": True,
    }
    if any(record.get("chord") for record in records):
        warnings.append("MusicXML 包含和弦；现有游戏投影可能按档位降级")
    return MusicXmlReadResult(
        song=SourceSong(
            title=os.path.splitext(os.path.basename(os.fspath(path)))[0],
            notes=source_notes,
            tracks=tracks,
            duration_s=duration_s,
            bpm_hint=float(tempo_events[0]["bpm"]),
            tempo_change_count=max(1, len(tempo_events)),
        ),
        warnings=warnings,
        source_metadata=metadata,
    )


__all__ = ["MusicXmlImportError", "MusicXmlReadResult", "read_musicxml_source"]
