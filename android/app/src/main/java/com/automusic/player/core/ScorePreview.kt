package com.automusic.player.core

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

/** 一次进行中的试听;cancel() 会立刻打断阻塞写入。 */
class PreviewHandle {

    @Volatile
    var cancelled: Boolean = false
        private set

    @Volatile
    private var track: AudioTrack? = null

    internal fun attach(target: AudioTrack) {
        track = target
    }

    fun cancel() {
        cancelled = true
        // 只置标志会等当前块写完;stop() 才能让阻塞的 write 立即返回
        runCatching { track?.stop() }
    }
}

/**
 * 本地试听:把谱面渲染成声音播放,**不经过无障碍注入**,因此不会向游戏发送任何点击。
 *
 * 桌面版用 Windows General MIDI 的大钢琴音色(core/preview_player.py);安卓没有等价的
 * 系统音源接口,这里用正弦叠加 + 指数衰减包络。音色不同,但校对时的用途一致:
 * 在把谱子交给游戏之前,先用耳朵确认写对没有。
 */
object ScorePreview {

    const val SAMPLE_RATE = 22050
    private const val MAX_SEGMENT_MS = 20_000L

    private val OCTAVE_OFFSET = mapOf("low" to -12, "mid" to 0, "high" to 12)
    private val NUM_SEMITONE = intArrayOf(0, 2, 4, 5, 7, 9, 11)

    /** note_id(+半音) -> MIDI 音高编号;中音 1 = C4 = 60。与桌面 note_id_to_midi 一致。 */
    fun noteToMidi(noteId: String, semitone: Boolean = false): Int? {
        val parts = noteId.split('_')
        if (parts.size != 2) return null
        val offset = OCTAVE_OFFSET[parts[0]] ?: return null
        val num = parts[1].toIntOrNull() ?: return null
        if (num !in 1..7) return null
        return 60 + offset + NUM_SEMITONE[num - 1] + if (semitone) 1 else 0
    }

    fun midiToFreq(midi: Int): Double = 440.0 * 2.0.pow((midi - 69) / 12.0)

    /**
     * 后台播放整段谱面。onFinished(true) 表示自然播完,false 表示被取消或出错。
     */
    fun play(notes: List<NoteEvent>, bpm: Int, onFinished: (Boolean) -> Unit): PreviewHandle {
        val handle = PreviewHandle()
        val beatMs = 60000.0 / bpm.coerceAtLeast(1)
        val worker = Thread {
            var completed = true
            try {
                val track = AudioTrack.Builder()
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
                    .setBufferSizeInBytes(SAMPLE_RATE * 2)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .build()
                handle.attach(track)
                track.play()
                for (el in notes) {
                    if (handle.cancelled) {
                        completed = false
                        break
                    }
                    val durMs = (el.dur * beatMs).toLong().coerceIn(1L, MAX_SEGMENT_MS)
                    val freqs = el.notes.mapNotNull { noteToMidi(it, el.semitone) }.map { midiToFreq(it) }
                    if (freqs.isEmpty()) {
                        writeSilence(track, durMs)
                    } else {
                        // 留出 1/10 作为音与音之间的间隙,避免听感糊成一片
                        writeTone(track, freqs, durMs * 9 / 10)
                        writeSilence(track, durMs / 10)
                    }
                }
                runCatching { track.stop() }
                runCatching { track.release() }
            } catch (e: Exception) {
                completed = false
                android.util.Log.e("ScorePreview", "试听失败:${e.message}", e)
            }
            onFinished(completed)
        }
        worker.isDaemon = true
        worker.name = "AmpPreview"
        worker.start()
        return handle
    }

    private fun writeSilence(track: AudioTrack, durationMs: Long) {
        val total = (SAMPLE_RATE * durationMs / 1000.0).toInt()
        var written = 0
        val chunk = ShortArray(2048)
        while (written < total) {
            val n = min(chunk.size, total - written)
            track.write(chunk, 0, n)
            written += n
        }
    }

    private fun writeTone(track: AudioTrack, freqs: List<Double>, durationMs: Long) {
        val total = (SAMPLE_RATE * durationMs / 1000.0).toInt()
        if (total <= 0) return
        var written = 0
        val chunk = ShortArray(2048)
        val gain = 0.42 / freqs.size
        while (written < total) {
            val n = min(chunk.size, total - written)
            for (i in 0 until n) {
                val pos = written + i
                val t = pos.toDouble() / SAMPLE_RATE
                val attack = min(1.0, t / 0.006)
                val release = min(1.0, (total - pos).toDouble() / (SAMPLE_RATE * 0.012))
                var sum = 0.0
                for (f in freqs) sum += sin(2 * PI * f * t) * (0.3 + 0.7 * exp(-1.6 * t))
                chunk[i] = (sum * attack * release * gain * Short.MAX_VALUE).toInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
            track.write(chunk, 0, n)
            written += n
        }
    }
}
