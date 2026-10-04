package com.automusic.player.core

/** 一条校验结论;elementIndex 为 null 表示整谱级问题,否则指向具体元素。 */
data class ScoreIssue(val elementIndex: Int?, val message: String)

data class ValidationResult(val errors: List<ScoreIssue>, val warnings: List<ScoreIssue>) {
    val ok: Boolean get() = errors.isEmpty()
}

/**
 * 入库前校验,规则与桌面版 core/score_model.py 的 require_valid 一致。
 *
 * 安卓端此前完全没有这一层:解析出什么就直接写库,越界时值或非法键位
 * 只会在演奏时表现为"没响"或"按错键",事后无从追溯。
 */
object ScoreValidation {

    const val MAX_DUR_BEATS = 16.0
    const val MIN_BPM = 30
    const val MAX_BPM = 300

    private val NOTE_ID_RE = Regex("^(high|mid|low)_([1-7])$")

    /**
     * @param semitoneSupported 当前档位是否有半音层。21 键档为 false,
     *   此时升号音符按"会被降级为自然音"给警告(与桌面版降级统计对齐),不阻断保存。
     */
    fun validate(
        notes: List<NoteEvent>,
        bpm: Int,
        semitoneSupported: Boolean = false,
    ): ValidationResult {
        val errors = ArrayList<ScoreIssue>()
        val warnings = ArrayList<ScoreIssue>()

        if (notes.isEmpty()) {
            errors += ScoreIssue(null, "谱面为空:没有任何音符")
        }
        if (bpm < MIN_BPM || bpm > MAX_BPM) {
            errors += ScoreIssue(null, "BPM 必须在 $MIN_BPM-$MAX_BPM 之间,当前 $bpm")
        }

        var sounding = 0
        var rests = 0
        var degraded = 0
        var totalBeats = 0.0
        for ((i, el) in notes.withIndex()) {
            if (el.notes.isEmpty()) {
                rests += 1
            } else {
                sounding += 1
                for (nid in el.notes) {
                    if (!NOTE_ID_RE.matches(nid)) {
                        errors += ScoreIssue(i, "音符无效 '$nid'(应为 high/mid/low_1~7)")
                    }
                }
                if (el.semitone && !semitoneSupported) degraded += 1
            }
            val dur = el.dur
            when {
                !dur.isFinite() -> errors += ScoreIssue(i, "时值必须为有限数字")
                dur <= 0.0 -> errors += ScoreIssue(i, "时值必须大于 0")
                dur > MAX_DUR_BEATS ->
                    errors += ScoreIssue(i, "时值超出上限(最大 ${MAX_DUR_BEATS.toInt()} 拍)")
            }
            totalBeats += dur
        }

        if (degraded > 0) {
            warnings += ScoreIssue(
                null,
                "$degraded 个升半音音符将按自然音演奏(当前 21 键档位无半音层)",
            )
        }
        if (rests > 0) {
            warnings += ScoreIssue(null, "含 $rests 处休止,共 $sounding 个发声音符")
        }
        if (notes.isNotEmpty() && sounding == 0) {
            errors += ScoreIssue(null, "整谱只有休止,没有任何可演奏音符")
        }
        warnings += ScoreIssue(null, "总时长 %.1f 拍".format(totalBeats))
        return ValidationResult(errors, warnings)
    }
}
