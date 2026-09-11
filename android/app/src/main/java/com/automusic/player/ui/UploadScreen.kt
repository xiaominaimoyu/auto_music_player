package com.automusic.player.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.DeleteOutline
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.HourglassTop
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Pause
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material.icons.outlined.Save
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material.icons.outlined.UploadFile
import androidx.compose.material.icons.outlined.WarningAmber
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.dp
import com.automusic.player.AppContainer
import com.automusic.player.core.JianpuParser
import com.automusic.player.core.JianpuEditor
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
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** 识别页:上传 -> 识别 -> 结构化校对 -> 本地试听 -> 保存。 */
@Composable
fun UploadScreen(container: AppContainer, onOpenSettings: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var preview by remember { mutableStateOf<Bitmap?>(null) }
    var scoreName by remember { mutableStateOf("") }
    var jianpuValue by remember { mutableStateOf(TextFieldValue("")) }
    var lyricsLines by remember { mutableStateOf(emptyList<String>()) }
    var lyricsByEvent by remember { mutableStateOf(emptyList<String>()) }
    var rhythmNotes by remember { mutableStateOf(emptyList<String>()) }
    var busy by remember { mutableStateOf(false) }
    var elapsedSeconds by remember { mutableIntStateOf(0) }
    var error by remember { mutableStateOf<String?>(null) }
    var showNoModelDialog by remember { mutableStateOf(false) }
    var editing by remember { mutableStateOf(false) }
    var selectedNoteIndex by remember { mutableStateOf<Int?>(null) }
    var previewConfirmed by remember { mutableStateOf(false) }
    var previewing by remember { mutableStateOf(false) }
    var activePreviewIndex by remember { mutableIntStateOf(-1) }
    var previewJob by remember { mutableStateOf<Job?>(null) }
    var bpm by remember { mutableIntStateOf(100) }
    var savedMessage by remember { mutableStateOf<String?>(null) }
    val previewPlayer = remember { ScorePreviewPlayer() }

    fun stopPreview() {
        previewJob?.cancel()
        previewJob = null
        previewPlayer.stop()
        previewing = false
        activePreviewIndex = -1
    }

    fun invalidatePreview() {
        stopPreview()
        previewConfirmed = false
        savedMessage = null
        selectedNoteIndex = null
    }

    fun applyRecognition(raw: String) {
        val result = Prompt.parseRecognitionResult(raw)
        scoreName = result.title.ifBlank { Prompt.fallbackScoreName() }
        jianpuValue = TextFieldValue(result.jianpu)
        lyricsLines = result.lyricsLines
        lyricsByEvent = JianpuEditor.alignLyrics(result.jianpu, result.lyricsLines)
        rhythmNotes = result.rhythmNotes
        editing = false
        invalidatePreview()
    }

    DisposableEffect(Unit) {
        onDispose { stopPreview() }
    }

    LaunchedEffect(busy) {
        elapsedSeconds = 0
        while (busy) {
            delay(1_000)
            elapsedSeconds += 1
        }
    }

    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            imageUri = uri
            preview = decodePreview(context, uri)
            scoreName = ""
            jianpuValue = TextFieldValue("")
            lyricsLines = emptyList()
            lyricsByEvent = emptyList()
            rhythmNotes = emptyList()
            editing = false
            error = null
            invalidatePreview()
        }
    }

    val notes = remember(jianpuValue.text) { JianpuParser.parse(jianpuValue.text) }
    val noteListState = rememberLazyListState()
    val restCount = remember(notes) { notes.count { it.notes.isEmpty() } }
    val scoreLines = remember(jianpuValue.text) {
        jianpuValue.text.lines().count { it.isNotBlank() }
    }
    val needsRestReview = notes.isNotEmpty() && restCount == 0 && scoreLines > 1
    val bpmValid = bpm in 1..300

    LaunchedEffect(activePreviewIndex) {
        if (activePreviewIndex >= 0) {
            noteListState.animateScrollToItem((activePreviewIndex - 1).coerceAtLeast(0))
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bg)
            .verticalScroll(rememberScrollState()),
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                .background(Surface1)
                .padding(horizontal = 18.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text("识别乐谱", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text("上传谱面，校对音符与节奏后试听保存", color = Ink2, style = MaterialTheme.typography.bodySmall)
        }

        Column(
            Modifier.padding(horizontal = 16.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            if (preview == null) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(8.dp),
                    color = Surface1,
                    border = BorderStroke(1.dp, Line),
                    onClick = { pickImage.launch("image/*") },
                ) {
                    Column(
                        Modifier.padding(horizontal = 24.dp, vertical = 36.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(10.dp),
                    ) {
                        Icon(Icons.Outlined.UploadFile, null, tint = Brand, modifier = Modifier.size(32.dp))
                        Text("选择乐谱图片", color = Ink, fontWeight = FontWeight.SemiBold)
                        Text("JPG、PNG、WEBP、BMP", color = Ink3, style = MaterialTheme.typography.bodySmall)
                    }
                }
            } else {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(8.dp),
                    color = Surface1,
                    border = BorderStroke(1.dp, Line),
                ) {
                    Column {
                        Box(
                            Modifier
                                .fillMaxWidth()
                                .height(240.dp)
                                .background(Color.White),
                        ) {
                            Image(
                                bitmap = preview!!.asImageBitmap(),
                                contentDescription = "乐谱预览",
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Fit,
                            )
                        }
                        Row(
                            Modifier.fillMaxWidth().padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Icon(Icons.Outlined.Image, null, tint = Ink2, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(
                                imageUri?.lastPathSegment ?: "已选择图片",
                                color = Ink2,
                                modifier = Modifier.weight(1f),
                                maxLines = 1,
                            )
                            TextButton(onClick = { pickImage.launch("image/*") }, enabled = !busy) { Text("更换") }
                        }
                    }
                }
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = {
                        val uri = imageUri ?: return@Button
                        busy = true
                        error = null
                        savedMessage = null
                        scope.launch {
                            try {
                                val recognizer = container.createRecognizer()
                                if (recognizer == null) {
                                    showNoModelDialog = true
                                } else {
                                    applyRecognition(recognizer.recognizeImage(context, uri))
                                }
                            } catch (e: Exception) {
                                error = e.message ?: e.toString()
                            } finally {
                                busy = false
                            }
                        }
                    },
                    enabled = imageUri != null && !busy,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                ) {
                    Icon(Icons.Outlined.HourglassTop, null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(if (busy) "正在识别" else "开始识别")
                }
                OutlinedButton(onClick = { copyPromptToClipboard(context) }, enabled = !busy) {
                    Icon(Icons.Outlined.ContentCopy, null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("提示词")
                }
            }

            if (busy) RecognitionProgress(elapsedSeconds)

            if (error != null) {
                Surface(color = StateError.copy(alpha = 0.1f), shape = RoundedCornerShape(6.dp)) {
                    Text(
                        error!!,
                        color = StateError,
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            if (jianpuValue.text.isBlank() && !busy) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    HorizontalDivider(Modifier.weight(1f), color = Line)
                    Text("或", color = Ink3, modifier = Modifier.padding(horizontal = 12.dp), style = MaterialTheme.typography.labelSmall)
                    HorizontalDivider(Modifier.weight(1f), color = Line)
                }
                OutlinedButton(
                    onClick = {
                        val clipboard = context.getSystemService(ClipboardManager::class.java)
                        val pasted = clipboard?.primaryClip?.getItemAt(0)?.coerceToText(context)?.toString().orEmpty()
                        if (pasted.isBlank()) error = "剪贴板中没有可粘贴的识别结果" else applyRecognition(pasted)
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(Icons.Outlined.ContentCopy, null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("粘贴外部 AI 识别结果")
                }
            }

            if (jianpuValue.text.isNotBlank()) {
                HorizontalDivider(color = Line)

                Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text("校对结果", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                            Text(
                                "${notes.size} 个音符 · $restCount 个休止 · ${formatBeats(notes.sumOf { it.dur })} 拍",
                                color = Ink2,
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        val parsed = notes.isNotEmpty()
                        Surface(
                            color = (if (parsed) StateSuccess else StateError).copy(alpha = 0.12f),
                            shape = RoundedCornerShape(6.dp),
                        ) {
                            Row(
                                Modifier.padding(horizontal = 9.dp, vertical = 5.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Icon(
                                    if (parsed) Icons.Outlined.Check else Icons.Outlined.WarningAmber,
                                    null,
                                    tint = if (parsed) StateSuccess else StateError,
                                    modifier = Modifier.size(15.dp),
                                )
                                Spacer(Modifier.width(4.dp))
                                Text(
                                    if (parsed) "已解析" else "需修改",
                                    color = if (parsed) StateSuccess else StateError,
                                    style = MaterialTheme.typography.labelSmall,
                                )
                            }
                        }
                    }

                    OutlinedTextField(
                        value = scoreName,
                        onValueChange = { scoreName = it.take(80) },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("曲名") },
                        singleLine = true,
                        supportingText = { Text("已从图片自动提取，可直接修改") },
                    )

                    if (rhythmNotes.isNotEmpty() || needsRestReview) {
                        RhythmWarning(
                            notes = rhythmNotes,
                            needsRestReview = needsRestReview,
                            onAddLineRests = {
                                val updated = Prompt.addLineBreakRests(jianpuValue.text)
                                jianpuValue = TextFieldValue(updated)
                                lyricsByEvent = JianpuEditor.alignLyrics(updated, lyricsLines)
                                editing = true
                                rhythmNotes = rhythmNotes + "已按原谱分行补入半拍休止，请试听确认"
                                invalidatePreview()
                            },
                        )
                    }

                    Row(
                        Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("音符与歌词", color = Ink, fontWeight = FontWeight.SemiBold)
                            Text(
                                if (previewing && activePreviewIndex in notes.indices) {
                                    val lyric = lyricsByEvent.getOrNull(activePreviewIndex).orEmpty()
                                    "正在试听 ${activePreviewIndex + 1}/${notes.size} · ${lyric.ifBlank { "间奏" }}"
                                } else {
                                    "${notes.size} 个音符，点击可校对"
                                },
                                color = if (previewing) Brand else Ink3,
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        if (previewing) {
                            Surface(color = BrandSoft, shape = RoundedCornerShape(6.dp)) {
                                Text(
                                    lyricsByEvent.getOrNull(activePreviewIndex).orEmpty().ifBlank { "·" },
                                    color = Brand,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 7.dp),
                                )
                            }
                        }
                    }
                    LazyRow(
                        state = noteListState,
                        modifier = Modifier.fillMaxWidth().height(78.dp),
                        contentPadding = PaddingValues(horizontal = 3.dp, vertical = 6.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        items(notes.size, key = { it }) { index ->
                            val note = notes[index]
                            NoteCell(
                                index = index,
                                event = note,
                                lyric = lyricsByEvent.getOrNull(index).orEmpty(),
                                selected = selectedNoteIndex == index,
                                playing = previewing && activePreviewIndex == index,
                                onClick = {
                                    selectedNoteIndex = if (selectedNoteIndex == index) null else index
                                },
                                onInsertBefore = {
                                    jianpuValue = TextFieldValue(
                                        JianpuEditor.insertRest(jianpuValue.text, index, before = true)
                                    )
                                    lyricsByEvent = lyricsByEvent.toMutableList().apply {
                                        while (size < notes.size) add("")
                                        add(index, "")
                                    }
                                    invalidatePreview()
                                },
                                onInsertAfter = {
                                    jianpuValue = TextFieldValue(
                                        JianpuEditor.insertRest(jianpuValue.text, index, before = false)
                                    )
                                    lyricsByEvent = lyricsByEvent.toMutableList().apply {
                                        while (size < notes.size) add("")
                                        add(index + 1, "")
                                    }
                                    invalidatePreview()
                                },
                                onDelete = {
                                    jianpuValue = TextFieldValue(
                                        JianpuEditor.deleteEvent(jianpuValue.text, index)
                                    )
                                    lyricsByEvent = lyricsByEvent.toMutableList().apply {
                                        if (index in indices) removeAt(index)
                                    }
                                    invalidatePreview()
                                },
                            )
                        }
                    }
                    Text("点击任一音符，可在前后插入半拍休止，或删除该音符", color = Ink3, style = MaterialTheme.typography.bodySmall)

                    OutlinedButton(onClick = { editing = !editing }, modifier = Modifier.fillMaxWidth()) {
                        Icon(Icons.Outlined.Edit, null, modifier = Modifier.size(17.dp))
                        Spacer(Modifier.width(7.dp))
                        Text(if (editing) "收起简谱编辑" else "编辑简谱")
                    }

                    if (editing) {
                        OutlinedTextField(
                            value = jianpuValue,
                            onValueChange = {
                                jianpuValue = it
                                lyricsByEvent = List(JianpuParser.parse(it.text).size) { "" }
                                invalidatePreview()
                            },
                            modifier = Modifier.fillMaxWidth(),
                            minLines = 5,
                            label = { Text("规范化简谱") },
                            textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = {
                                jianpuValue = insertAtCursor(jianpuValue, "0_")
                                invalidatePreview()
                            }) {
                                Icon(Icons.Outlined.Pause, null, modifier = Modifier.size(17.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("插入半拍休止")
                            }
                            if (scoreLines > 1 && restCount == 0) {
                                IconButton(
                                    onClick = {
                                        val updated = Prompt.addLineBreakRests(jianpuValue.text)
                                        jianpuValue = TextFieldValue(updated)
                                        lyricsByEvent = JianpuEditor.alignLyrics(updated, lyricsLines)
                                        invalidatePreview()
                                    },
                                ) {
                                    Icon(Icons.Outlined.Add, contentDescription = "按分行补休止")
                                }
                            }
                        }
                    }

                    HorizontalDivider(color = Line)

                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text("口琴试听", color = Ink, fontWeight = FontWeight.SemiBold)
                                Text("拖动调整速度", color = Ink3, style = MaterialTheme.typography.bodySmall)
                            }
                            Surface(color = Surface2, shape = RoundedCornerShape(6.dp)) {
                                Text(
                                    "$bpm BPM",
                                    color = if (bpmValid) Brand else StateError,
                                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                    style = MaterialTheme.typography.labelMedium,
                                )
                            }
                        }
                        Slider(
                            value = bpm.toFloat(),
                            onValueChange = {
                                bpm = it.toInt().coerceIn(0, 300)
                                invalidatePreview()
                            },
                            valueRange = 0f..300f,
                            modifier = Modifier.fillMaxWidth(),
                        )
                        Row(Modifier.fillMaxWidth()) {
                            Text("0", color = Ink3, style = MaterialTheme.typography.labelSmall)
                            Spacer(Modifier.weight(1f))
                            Text(
                                if (bpmValid) "仅使用口琴近似音色" else "0 BPM 无法试听",
                                color = if (bpmValid) Ink3 else StateError,
                                style = MaterialTheme.typography.labelSmall,
                            )
                            Spacer(Modifier.weight(1f))
                            Text("300", color = Ink3, style = MaterialTheme.typography.labelSmall)
                        }
                    }

                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        Button(
                            enabled = notes.isNotEmpty() && bpmValid && !previewing,
                            onClick = {
                                previewConfirmed = true
                                previewing = true
                                error = null
                                previewJob = scope.launch {
                                    try {
                                        previewPlayer.play(notes, bpm) { index ->
                                            activePreviewIndex = index
                                        }
                                    } catch (e: Exception) {
                                        if (e !is CancellationException) {
                                            previewConfirmed = false
                                            error = "试听失败:${e.message ?: e}"
                                        }
                                    } finally {
                                        previewing = false
                                        activePreviewIndex = -1
                                        previewJob = null
                                    }
                                }
                            },
                            modifier = Modifier.weight(1f),
                        ) {
                            Icon(Icons.Outlined.PlayArrow, null, modifier = Modifier.size(19.dp))
                            Spacer(Modifier.width(6.dp))
                            Text(if (previewing) "试听中" else if (previewConfirmed) "重新试听" else "试听")
                        }
                        if (previewing) {
                            OutlinedButton(onClick = { stopPreview() }) {
                                Icon(Icons.Outlined.Stop, null, modifier = Modifier.size(19.dp))
                                Spacer(Modifier.width(5.dp))
                                Text("停止")
                            }
                        }
                    }

                    Button(
                        enabled = notes.isNotEmpty() && bpmValid && previewConfirmed && !previewing && scoreName.isNotBlank(),
                        onClick = {
                            scope.launch {
                                val finalName = scoreName.trim().ifBlank { Prompt.fallbackScoreName() }
                                container.db.scoreDao().insert(
                                    ScoreEntity(
                                        name = finalName,
                                        sourceFile = imageUri?.toString() ?: "",
                                        sourceType = if (imageUri != null) "image" else "text",
                                        rawText = buildResultText(finalName, jianpuValue.text, lyricsByEvent, rhythmNotes),
                                        notesJson = com.automusic.player.core.NoteCodec.encode(notes),
                                        bpmDefault = bpm,
                                        createdAt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date()),
                                    )
                                )
                                savedMessage = "《$finalName》已保存到乐谱库"
                                previewConfirmed = false
                                stopPreview()
                            }
                        },
                        modifier = Modifier.fillMaxWidth(),
                        colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                    ) {
                        Icon(Icons.Outlined.Save, null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(7.dp))
                        Text("保存到乐谱库")
                    }

                    Text(
                        savedMessage ?: if (previewConfirmed) "试听已开始，可确认保存；修改内容后需重新试听" else "请先试听当前校对结果",
                        color = if (savedMessage != null) StateSuccess else Ink3,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            Spacer(Modifier.height(8.dp))
        }
    }

    if (showNoModelDialog) {
        AlertDialog(
            onDismissRequest = { showNoModelDialog = false },
            title = { Text("尚未配置识别模型") },
            text = { Text("开始识别需要一个已激活的多模态模型。也可以复制提示词，到外部 AI 识别后粘贴结果。") },
            confirmButton = {
                TextButton(onClick = {
                    showNoModelDialog = false
                    onOpenSettings()
                }) { Text("去设置", color = Brand) }
            },
            dismissButton = {
                TextButton(onClick = {
                    showNoModelDialog = false
                    copyPromptToClipboard(context)
                }) { Text("复制提示词", color = Ink2) }
            },
        )
    }
}

@Composable
private fun RecognitionProgress(elapsedSeconds: Int) {
    val phase = when {
        elapsedSeconds < 3 -> "正在读取图片与曲名"
        elapsedSeconds < 8 -> "正在定位音符与乐句"
        elapsedSeconds < 15 -> "正在对齐音符与歌词"
        else -> "正在整理校对结果"
    }
    Surface(color = BrandSoft, shape = RoundedCornerShape(8.dp)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.HourglassTop, null, tint = Brand, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Column(Modifier.weight(1f)) {
                    Text("正在识别中", color = Ink, fontWeight = FontWeight.SemiBold)
                    Text(phase, color = Ink2, style = MaterialTheme.typography.bodySmall)
                }
                Text("${elapsedSeconds}s", color = Brand, style = MaterialTheme.typography.labelMedium)
            }
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth(), color = Brand, trackColor = Surface2)
        }
    }
}

@Composable
private fun RhythmWarning(notes: List<String>, needsRestReview: Boolean, onAddLineRests: () -> Unit) {
    Surface(color = StateWarning.copy(alpha = 0.1f), shape = RoundedCornerShape(8.dp)) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.Top) {
                Icon(Icons.Outlined.WarningAmber, null, tint = StateWarning, modifier = Modifier.size(19.dp))
                Spacer(Modifier.width(8.dp))
                Text(
                    (notes + if (needsRestReview) listOf("未识别到休止符；原图若有乐句停顿，请试听校对") else emptyList()).joinToString("\n"),
                    color = Ink,
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.weight(1f),
                )
            }
            if (needsRestReview) {
                TextButton(onClick = onAddLineRests) { Text("按原谱分行补半拍停顿", color = StateWarning) }
            }
        }
    }
}

@Composable
private fun NoteCell(
    index: Int,
    event: NoteEvent,
    lyric: String,
    selected: Boolean,
    playing: Boolean,
    onClick: () -> Unit,
    onInsertBefore: () -> Unit,
    onInsertAfter: () -> Unit,
    onDelete: () -> Unit,
) {
    val isRest = event.notes.isEmpty()
    val scale by animateFloatAsState(if (playing) 1.08f else 1f, label = "preview-note-scale")
    Box(Modifier.graphicsLayer { scaleX = scale; scaleY = scale }) {
        Surface(
            onClick = onClick,
            color = when {
                playing -> Brand
                selected -> BrandSoft
                isRest -> StateWarning.copy(alpha = 0.14f)
                else -> Surface2
            },
            shape = RoundedCornerShape(6.dp),
            border = BorderStroke(
                if (selected) 2.dp else 1.dp,
                when {
                    playing -> Brand
                    selected -> Brand
                    isRest -> StateWarning.copy(alpha = 0.45f)
                    else -> Line
                },
            ),
        ) {
            Column(
                modifier = Modifier.size(width = 58.dp, height = 66.dp).padding(vertical = 5.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
            ) {
                Text(
                    if (isRest) "休" else event.notes.joinToString("+") { noteLabel(it) },
                    color = when {
                        playing -> Bg
                        isRest -> StateWarning
                        else -> Ink
                    },
                    style = MaterialTheme.typography.labelMedium,
                    maxLines = 1,
                )
                Text(
                    lyric.ifBlank { " " },
                    color = if (playing) Bg else Brand,
                    style = MaterialTheme.typography.labelSmall,
                    maxLines = 1,
                )
                Text(
                    "${index + 1} · ${formatBeats(event.dur)}",
                    color = if (playing) Bg.copy(alpha = 0.7f) else Ink3,
                    style = MaterialTheme.typography.labelSmall,
                    maxLines = 1,
                )
            }
        }
        androidx.compose.material3.DropdownMenu(
            expanded = selected,
            onDismissRequest = onClick,
            offset = DpOffset(0.dp, (-190).dp),
            modifier = Modifier.width(252.dp),
        ) {
            Text(
                "第 ${index + 1} 个 · ${if (isRest) "休止" else event.notes.joinToString("+") { noteLabel(it) }} · ${formatBeats(event.dur)} 拍",
                color = Ink2,
                style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp),
            )
            HorizontalDivider(color = Line)
            DropdownMenuItem(
                text = { Text("前面插入半拍休止") },
                leadingIcon = { Icon(Icons.Outlined.Add, contentDescription = null, tint = Brand) },
                onClick = onInsertBefore,
            )
            DropdownMenuItem(
                text = { Text("后面插入半拍休止") },
                leadingIcon = { Icon(Icons.Outlined.Add, contentDescription = null, tint = Brand) },
                onClick = onInsertAfter,
            )
            HorizontalDivider(color = Line)
            DropdownMenuItem(
                text = { Text("删除这个音符", color = StateError) },
                leadingIcon = { Icon(Icons.Outlined.DeleteOutline, contentDescription = null, tint = StateError) },
                onClick = onDelete,
            )
        }
    }
}

private fun noteLabel(id: String): String {
    val parts = id.split('_')
    if (parts.size != 2) return id
    return when (parts[0]) {
        "high" -> "${parts[1]}'"
        "low" -> "${parts[1]},"
        else -> parts[1]
    }
}

private fun formatBeats(value: Double): String =
    if (value % 1.0 == 0.0) value.toInt().toString() else "%.2f".format(value).trimEnd('0')

private fun insertAtCursor(value: TextFieldValue, token: String): TextFieldValue {
    val start = value.selection.min.coerceIn(0, value.text.length)
    val end = value.selection.max.coerceIn(start, value.text.length)
    val prefix = if (start > 0 && !value.text[start - 1].isWhitespace()) " " else ""
    val suffix = if (end < value.text.length && !value.text[end].isWhitespace()) " " else ""
    val insertion = "$prefix$token$suffix"
    val text = value.text.replaceRange(start, end, insertion)
    val cursor = start + insertion.length
    return TextFieldValue(text, androidx.compose.ui.text.TextRange(cursor))
}

private fun buildResultText(
    title: String,
    jianpu: String,
    lyrics: List<String>,
    rhythmNotes: List<String>,
): String = buildString {
    appendLine("TITLE:")
    appendLine(title)
    appendLine("JIANPU:")
    appendLine(jianpu.trim())
    appendLine("LYRICS:")
    appendLine(lyrics.joinToString(" ") { it.ifBlank { "_" } }.ifBlank { "UNKNOWN" })
    appendLine("RHYTHM:")
    append(rhythmNotes.joinToString("；").ifBlank { "OK" })
}

/** 复制识别提示词到剪贴板，供外部 AI 识别。 */
private fun copyPromptToClipboard(context: Context) {
    val clipboard = context.getSystemService(ClipboardManager::class.java)
    clipboard?.setPrimaryClip(ClipData.newPlainText("简谱识别提示词", Prompt.JIANPU_PROMPT))
    Toast.makeText(context, "提示词已复制，请附上乐谱图片发送给 AI", Toast.LENGTH_LONG).show()
}

/** 预览解码:粗降采样到最长边约 1024，避免大图 OOM。 */
private fun decodePreview(context: Context, uri: Uri): Bitmap? = try {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 1024) sample *= 2
    val opts = BitmapFactory.Options().apply { inSampleSize = sample }
    context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
} catch (e: Exception) {
    null
}
