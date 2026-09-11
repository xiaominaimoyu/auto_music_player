package com.automusic.player.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ArrowBack
import androidx.compose.material.icons.outlined.ChevronRight
import androidx.compose.material.icons.outlined.DeleteOutline
import androidx.compose.material.icons.outlined.LibraryMusic
import androidx.compose.material.icons.outlined.MusicNote
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material.icons.outlined.Save
import androidx.compose.material.icons.outlined.SportsEsports
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.automusic.player.AppContainer
import com.automusic.player.core.JianpuEditor
import com.automusic.player.core.JianpuParser
import com.automusic.player.core.NoteCodec
import com.automusic.player.core.NoteEvent
import com.automusic.player.core.Prompt
import com.automusic.player.core.ScorePreviewPlayer
import com.automusic.player.core.db.ScoreEntity
import com.automusic.player.ui.theme.Bg
import com.automusic.player.ui.theme.Brand
import com.automusic.player.ui.theme.BrandSoft
import com.automusic.player.ui.theme.Ink
import com.automusic.player.ui.theme.Ink2
import com.automusic.player.ui.theme.Ink3
import com.automusic.player.ui.theme.Line
import com.automusic.player.ui.theme.StateError
import com.automusic.player.ui.theme.StateSuccess
import com.automusic.player.ui.theme.StateWarning
import com.automusic.player.ui.theme.Surface1
import com.automusic.player.ui.theme.Surface2
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

/** 乐谱库列表及可编辑详情。 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun LibraryScreen(container: AppContainer, onGoPlay: () -> Unit) {
    val scope = rememberCoroutineScope()
    val scores by container.db.scoreDao().observeAll().collectAsState(initial = emptyList())
    var selectedId by remember { mutableStateOf<Long?>(null) }
    var deleting by remember { mutableStateOf<ScoreEntity?>(null) }
    val selected = scores.firstOrNull { it.id == selectedId }

    if (selected != null) {
        ScoreDetail(
            score = selected,
            onBack = { selectedId = null },
            onSave = { updated -> scope.launch { container.db.scoreDao().update(updated) } },
            onPlay = {
                container.selectedScoreId = selected.id
                onGoPlay()
            },
            onDelete = { deleting = selected },
        )
    } else {
        Column(
            modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("乐谱库", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                    Text("${scores.size} 首已保存乐谱", color = Ink2, style = MaterialTheme.typography.bodySmall)
                }
                Icon(Icons.Outlined.LibraryMusic, contentDescription = null, tint = Brand, modifier = Modifier.size(28.dp))
            }

            if (scores.isEmpty()) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    color = Surface1,
                    shape = RoundedCornerShape(8.dp),
                    border = BorderStroke(1.dp, Line),
                ) {
                    Column(
                        Modifier.padding(horizontal = 24.dp, vertical = 36.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Icon(Icons.Outlined.MusicNote, contentDescription = null, tint = Ink3, modifier = Modifier.size(32.dp))
                        Text("还没有乐谱", color = Ink, style = MaterialTheme.typography.titleMedium)
                        Text("识别并保存乐谱后会显示在这里", color = Ink3, style = MaterialTheme.typography.bodySmall)
                    }
                }
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(scores, key = { it.id }) { score ->
                        val noteCount = remember(score.notesJson) {
                            runCatching { NoteCodec.decode(score.notesJson).size }.getOrDefault(0)
                        }
                        Card(
                            colors = CardDefaults.cardColors(containerColor = Surface1),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.fillMaxWidth().clickable { selectedId = score.id },
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 13.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Surface(color = BrandSoft, shape = RoundedCornerShape(6.dp)) {
                                    Box(Modifier.size(42.dp), contentAlignment = Alignment.Center) {
                                        Icon(Icons.Outlined.MusicNote, contentDescription = null, tint = Brand)
                                    }
                                }
                                Column(Modifier.weight(1f).padding(horizontal = 12.dp)) {
                                    Text(score.name, style = MaterialTheme.typography.titleMedium, maxLines = 1)
                                    Text(
                                        "$noteCount 个音符 · ${score.bpmDefault} BPM",
                                        color = Ink2,
                                        style = MaterialTheme.typography.bodySmall,
                                    )
                                    Text(score.createdAt, color = Ink3, style = MaterialTheme.typography.labelSmall)
                                }
                                IconButton(onClick = { deleting = score }) {
                                    Icon(Icons.Outlined.DeleteOutline, contentDescription = "删除", tint = StateError)
                                }
                                Icon(Icons.Outlined.ChevronRight, contentDescription = "查看详情", tint = Ink3)
                            }
                        }
                    }
                }
            }
        }
    }

    deleting?.let { score ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("删除乐谱") },
            text = { Text("确定删除《${score.name}》？该操作不可撤销。") },
            confirmButton = {
                TextButton(onClick = {
                    scope.launch { container.db.scoreDao().delete(score.id) }
                    if (selectedId == score.id) selectedId = null
                    deleting = null
                }) { Text("删除", color = StateError) }
            },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("取消", color = Ink2) } },
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ScoreDetail(
    score: ScoreEntity,
    onBack: () -> Unit,
    onSave: (ScoreEntity) -> Unit,
    onPlay: () -> Unit,
    onDelete: () -> Unit,
) {
    val scope = rememberCoroutineScope()
    val initial = remember(score.id, score.rawText) { Prompt.parseRecognitionResult(score.rawText) }
    val storedNotes = remember(score.notesJson) {
        runCatching { NoteCodec.decode(score.notesJson) }.getOrDefault(emptyList())
    }
    var name by remember(score.id) { mutableStateOf(score.name) }
    var jianpu by remember(score.id) { mutableStateOf(initial.jianpu) }
    var lyricsText by remember(score.id) {
        mutableStateOf(
            if (initial.lyricsLines.isEmpty()) "" else
                JianpuEditor.alignLyrics(initial.jianpu, initial.lyricsLines)
                    .joinToString(" ") { it.ifBlank { "_" } }
        )
    }
    var bpm by remember(score.id) { mutableStateOf(score.bpmDefault.toFloat()) }
    var message by remember(score.id) { mutableStateOf<String?>(null) }
    var previewing by remember(score.id) { mutableStateOf(false) }
    var previewJob by remember(score.id) { mutableStateOf<Job?>(null) }
    val previewPlayer = remember(score.id) { ScorePreviewPlayer() }

    val parsedNotes = remember(jianpu, storedNotes) {
        if (jianpu.isBlank()) storedNotes else JianpuParser.parse(jianpu)
    }
    val lyrics = remember(jianpu, lyricsText, parsedNotes) {
        if (jianpu.isBlank()) List(parsedNotes.size) { "" }
        else JianpuEditor.alignLyrics(jianpu, listOf(lyricsText))
    }

    fun stopPreview() {
        previewJob?.cancel()
        previewJob = null
        previewPlayer.stop()
        previewing = false
    }

    DisposableEffect(score.id) {
        onDispose { stopPreview() }
    }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) { Icon(Icons.Outlined.ArrowBack, contentDescription = "返回") }
            Column(Modifier.weight(1f)) {
                Text("乐谱详情", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                Text("${parsedNotes.size} 个音符 · ${score.createdAt}", color = Ink2, style = MaterialTheme.typography.bodySmall)
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Outlined.DeleteOutline, contentDescription = "删除乐谱", tint = StateError)
            }
        }

        Surface(color = Surface1, shape = RoundedCornerShape(8.dp), border = BorderStroke(1.dp, Line)) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it.take(80); message = null },
                    label = { Text("曲名") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = jianpu,
                    onValueChange = { jianpu = it; message = null; stopPreview() },
                    label = { Text("简谱") },
                    placeholder = { Text("该乐谱没有可编辑的原始简谱") },
                    minLines = 4,
                    textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
                    modifier = Modifier.fillMaxWidth(),
                )
                OutlinedTextField(
                    value = lyricsText,
                    onValueChange = { lyricsText = it; message = null },
                    label = { Text("歌词（空格分隔，一音一字）") },
                    minLines = 2,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        if (parsedNotes.isNotEmpty()) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text("音符与歌词", color = Ink, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                    Text("${parsedNotes.size} 个", color = Ink3, style = MaterialTheme.typography.labelSmall)
                }
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    parsedNotes.forEachIndexed { index, event ->
                        DetailNoteCell(index, event, lyrics.getOrNull(index).orEmpty())
                    }
                }
            }
        }

        Surface(color = Surface1, shape = RoundedCornerShape(8.dp), border = BorderStroke(1.dp, Line)) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("口琴试听", color = Ink, fontWeight = FontWeight.SemiBold)
                        Text("${bpm.toInt()} BPM", color = Brand, style = MaterialTheme.typography.bodySmall)
                    }
                    Text("1 — 300", color = Ink3, style = MaterialTheme.typography.labelSmall)
                }
                Slider(
                    value = bpm,
                    onValueChange = { bpm = it; message = null; stopPreview() },
                    valueRange = 1f..300f,
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        enabled = parsedNotes.isNotEmpty() && !previewing,
                        onClick = {
                            previewing = true
                            message = null
                            previewJob = scope.launch {
                                try {
                                    previewPlayer.play(parsedNotes, bpm.toInt())
                                } catch (e: Exception) {
                                    if (e !is CancellationException) message = "试听失败：${e.message}"
                                } finally {
                                    previewing = false
                                    previewJob = null
                                }
                            }
                        },
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                    ) {
                        Icon(Icons.Outlined.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text(if (previewing) "试听中" else "试听")
                    }
                    OutlinedButton(enabled = previewing, onClick = { stopPreview() }) {
                        Icon(Icons.Outlined.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(5.dp))
                        Text("停止")
                    }
                }
            }
        }

        HorizontalDivider(color = Line)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = { stopPreview(); onPlay() },
                enabled = parsedNotes.isNotEmpty(),
                modifier = Modifier.weight(1f),
            ) {
                Icon(Icons.Outlined.SportsEsports, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("进入演奏")
            }
            Button(
                onClick = {
                    val finalName = name.trim()
                    if (finalName.isBlank()) {
                        message = "曲名不能为空"
                    } else if (parsedNotes.isEmpty()) {
                        message = "简谱中没有可保存的音符"
                    } else {
                        stopPreview()
                        val raw = if (jianpu.isBlank()) score.rawText
                        else buildStoredResult(finalName, jianpu, lyrics, initial.rhythmNotes)
                        onSave(
                            score.copy(
                                name = finalName,
                                rawText = raw,
                                notesJson = NoteCodec.encode(parsedNotes),
                                bpmDefault = bpm.toInt(),
                            )
                        )
                        message = "修改已保存"
                    }
                },
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
            ) {
                Icon(Icons.Outlined.Save, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("保存修改")
            }
        }
        message?.let {
            Text(
                it,
                color = if (it == "修改已保存") StateSuccess else StateWarning,
                style = MaterialTheme.typography.bodySmall,
            )
        }
        Spacer(Modifier.height(8.dp))
    }
}

@Composable
private fun DetailNoteCell(index: Int, event: NoteEvent, lyric: String) {
    val rest = event.notes.isEmpty()
    Surface(
        color = if (rest) StateWarning.copy(alpha = 0.12f) else Surface2,
        shape = RoundedCornerShape(6.dp),
        border = BorderStroke(1.dp, if (rest) StateWarning.copy(alpha = 0.35f) else Line),
    ) {
        Column(
            modifier = Modifier.size(width = 58.dp, height = 64.dp).padding(vertical = 5.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(if (rest) "休" else event.notes.joinToString("+") { libraryNoteLabel(it) }, color = if (rest) StateWarning else Ink)
            Text(lyric.ifBlank { " " }, color = Brand, style = MaterialTheme.typography.labelSmall, maxLines = 1)
            Text("${index + 1} · ${libraryBeatLabel(event.dur)}", color = Ink3, style = MaterialTheme.typography.labelSmall)
        }
    }
}

private fun buildStoredResult(title: String, jianpu: String, lyrics: List<String>, rhythm: List<String>): String = buildString {
    appendLine("TITLE:")
    appendLine(title)
    appendLine("JIANPU:")
    appendLine(jianpu.trim())
    appendLine("LYRICS:")
    appendLine(lyrics.joinToString(" ") { it.ifBlank { "_" } }.ifBlank { "UNKNOWN" })
    appendLine("RHYTHM:")
    append(rhythm.joinToString("；").ifBlank { "OK" })
}

private fun libraryNoteLabel(id: String): String {
    val parts = id.split('_')
    if (parts.size != 2) return id
    return when (parts[0]) {
        "high" -> "${parts[1]}'"
        "low" -> "${parts[1]},"
        else -> parts[1]
    }
}

private fun libraryBeatLabel(value: Double): String =
    if (value % 1.0 == 0.0) value.toInt().toString() else "%.2f".format(value).trimEnd('0')
