package com.automusic.player.core

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.math.PI
import kotlin.math.pow
import kotlin.math.sin

/** 将结构化简谱合成为口琴近似音色的本地 PCM，不经过触摸注入通道。 */
class ScorePreviewPlayer {
    @Volatile
    private var track: AudioTrack? = null

    suspend fun play(
        notes: List<NoteEvent>,
        bpm: Int,
        onProgress: (Int) -> Unit = {},
    ) = withContext(Dispatchers.Default) {
        require(notes.isNotEmpty()) { "乐谱为空，无法试听" }
        require(bpm in 1..300) { "BPM 必须在 1-300 之间" }
        val pcm = synthesize(notes, bpm)
        val eventEndFrames = eventEndFrames(notes, bpm)
        currentCoroutineContext().ensureActive()
        val audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(pcm.size)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        track = audioTrack
        try {
            val written = audioTrack.write(pcm, 0, pcm.size)
            if (written != pcm.size) error("试听音频写入失败")
            audioTrack.play()
            val frames = pcm.size / 2
            var eventIndex = 0
            withContext(Dispatchers.Main.immediate) { onProgress(eventIndex) }
            while (audioTrack.playbackHeadPosition < frames) {
                currentCoroutineContext().ensureActive()
                val head = audioTrack.playbackHeadPosition
                while (eventIndex < eventEndFrames.lastIndex && head >= eventEndFrames[eventIndex]) {
                    eventIndex += 1
                    withContext(Dispatchers.Main.immediate) { onProgress(eventIndex) }
                }
                delay(20)
            }
        } finally {
            runCatching { audioTrack.stop() }
            audioTrack.release()
            if (track === audioTrack) track = null
        }
    }

    fun stop() {
        val active = track
        track = null
        if (active != null) runCatching { active.stop() }
    }

    private suspend fun synthesize(notes: List<NoteEvent>, bpm: Int): ByteArray {
        val secondsPerBeat = 60.0 / bpm
        val totalSeconds = notes.sumOf { it.dur * secondsPerBeat }
        require(totalSeconds <= MAX_SECONDS) { "试听时长不能超过 5 分钟" }
        val totalFrames = notes.sumOf {
            (it.dur * secondsPerBeat * SAMPLE_RATE).toInt().coerceAtLeast(1)
        }
        val output = ByteArray(totalFrames * 2)
        var offset = 0
        var checkedFrames = 0
        for (event in notes) {
            require(event.dur > 0.0 && event.dur.isFinite()) { "音符时值无效" }
            val frames = (event.dur * secondsPerBeat * SAMPLE_RATE).toInt().coerceAtLeast(1)
            val frequencies = event.notes.map(::frequency)
            for (frame in 0 until frames) {
                if (++checkedFrames % 4096 == 0) currentCoroutineContext().ensureActive()
                val sample = if (frequencies.isNotEmpty()) {
                    frequencies.sumOf { harmonicaVoice(it, frame, frames) } / frequencies.size
                } else {
                    0.0
                }
                val value = (sample.coerceIn(-1.0, 1.0) * 9800).toInt().toShort().toInt()
                output[offset++] = (value and 0xff).toByte()
                output[offset++] = ((value shr 8) and 0xff).toByte()
            }
        }
        return output
    }

    private fun eventEndFrames(notes: List<NoteEvent>, bpm: Int): IntArray {
        val secondsPerBeat = 60.0 / bpm
        var total = 0
        return IntArray(notes.size) { index ->
            total += (notes[index].dur * secondsPerBeat * SAMPLE_RATE).toInt().coerceAtLeast(1)
            total
        }
    }

    private fun harmonicaVoice(frequency: Double, frame: Int, frames: Int): Double {
        val time = frame.toDouble() / SAMPLE_RATE
        val angle = 2.0 * PI * frequency * time
        val attack = (frame / (SAMPLE_RATE * 0.018)).coerceIn(0.0, 1.0)
        val tail = ((frames - frame).toDouble() / (SAMPLE_RATE * 0.035)).coerceIn(0.0, 1.0)
        return (sin(angle) + 0.31 * sin(3 * angle) + 0.13 * sin(5 * angle)) / 1.44 * attack * tail
    }

    private fun frequency(noteId: String): Double {
        val parts = noteId.split('_')
        require(parts.size == 2) { "无效音符: $noteId" }
        val number = parts[1].toIntOrNull() ?: error("无效音符: $noteId")
        require(number in 1..7) { "无效音符: $noteId" }
        val octave = when (parts[0]) {
            "low" -> -12
            "mid" -> 0
            "high" -> 12
            else -> error("无效音符: $noteId")
        }
        val midi = 60 + octave + MAJOR_OFFSETS[number - 1]
        return 440.0 * 2.0.pow((midi - 69) / 12.0)
    }

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val MAX_SECONDS = 300.0
        private val MAJOR_OFFSETS = intArrayOf(0, 2, 4, 5, 7, 9, 11)
    }
}
