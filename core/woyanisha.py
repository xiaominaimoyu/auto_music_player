"""沃雅妮莎 14 键档位的运行时音域投影。"""

from dataclasses import dataclass

from core.score_model import require_valid
from core.transport import TransportDegradation

_LOW_TO_MID = {f"low_{number}": f"mid_{number}" for number in range(1, 8)}


@dataclass(frozen=True)
class WoyanishaProjection:
    notes: list
    raised_notes: int
    naturalized_elements: int
    merged_notes: int
    degradations: list


def project_woyanisha_notes(notes: list, *, source_index_offset: int = 0) -> WoyanishaProjection:
    """返回低音升中八度、半音自然音化并按原顺序去重的新列表。"""

    require_valid(notes)
    output = []
    degradations = []
    raised_notes = 0
    naturalized_elements = 0
    merged_notes = 0

    for element_index, source in enumerate(notes):
        source_index = source_index_offset + element_index
        original_ids = list(source["notes"])
        raised_ids = [_LOW_TO_MID.get(note_id, note_id) for note_id in original_ids]
        element_raised = sum(
            int(projected_id != note_id)
            for note_id, projected_id in zip(original_ids, raised_ids)
        )
        if element_raised:
            raised_notes += element_raised
            degradations.append(
                TransportDegradation(
                    source_index,
                    f"{element_raised} 个低音",
                    "同音级中音",
                    "沃雅妮莎没有低音键",
                )
            )

        item = dict(source)
        item.pop("semitone", None)
        item.pop("semitones", None)
        naturalized = bool(int(source.get("semitone", 0) or 0)) or any(
            int(value) for value in (source.get("semitones") or [])
        )
        if naturalized:
            naturalized_elements += 1
            degradations.append(
                TransportDegradation(
                    source_index,
                    "半音",
                    "自然音",
                    "沃雅妮莎仅支持自然音",
                )
            )

        projected_ids = list(dict.fromkeys(raised_ids))
        element_merged = len(original_ids) - len(projected_ids)
        if element_merged:
            merged_notes += element_merged
            degradations.append(
                TransportDegradation(
                    source_index,
                    f"和弦 {len(original_ids)} 个音",
                    f"去重为 {len(projected_ids)} 个音",
                    "音域投影后重复物理音",
                )
            )

        item["notes"] = projected_ids
        output.append(item)

    require_valid(output)
    return WoyanishaProjection(
        notes=output,
        raised_notes=raised_notes,
        naturalized_elements=naturalized_elements,
        merged_notes=merged_notes,
        degradations=degradations,
    )