package com.automusic.player.ui

import android.content.Context
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.automusic.player.AppContainer
import com.automusic.player.core.JianpuParser
import com.automusic.player.core.NoteCodec
import com.automusic.player.core.NoteEvent
import com.automusic.player.core.PreviewHandle
import com.automusic.player.core.Prompt
import com.automusic.player.core.ScoreEditing
import com.automusic.player.core.ScoreJson
import com.automusic.player.core.ScorePreview
import com.automusic.player.core.ScoreValidation
import com.automusic.player.core.db.ScoreEntity
import com.automusic.player.ui.theme.Bg
import com.automusic.player.ui.theme.Brand
import com.automusic.player.ui.theme.Ink2
import com.automusic.player.ui.theme.Ink3
import com.automusic.player.ui.theme.StateError
import com.automusic.player.ui.theme.StateInfo
import com.automusic.player.ui.theme.StateSuccess
import com.automusic.player.ui.theme.StateWarning
import com.automusic.player.ui.theme.Surface1
import com.automusic.player.ui.theme.Surface2
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.launch

/**
 * 识别/录入页,功能对齐桌面版 gui/upload_tab.py:
 * 提示词 -> 粘贴或导入 -> 解析到校对表格 -> 逐音编辑(音符/时值/半音)、插删、撤销
 * -> 本地试听(不注入点击) -> 校验 -> 保存入库。
 *
 * 刻意未对齐:物理键盘实时录制(手机无等价输入源)与图片/PDF 离线 OMR
 * (Audiveris 是桌面 Java 程序)。
 */
@Composable
fun UploadScreen(container: AppContainer) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var resultText by remember { mutableStateOf("") }
    var edit by remember { mutableStateOf(ScoreEditing.load(emptyList())) }
    var selected by remember { mutableStateOf<Int?>(null) }
    var cellText by remember { mutableStateOf("") }
    var durText by remember { mutableStateOf("") }
    var parseMessage by remember { mutableStateOf<String?>(null) }
    var parseFailed by remember { mutableStateOf(false) }
    var editError by remember { mutableStateOf<String?>(null) }
    var name by remember { mutableStateOf("") }
    var bpm by remember { mutableStateOf("100") }
    var showSaveDialog by remember { mutableStateOf(false) }
    var previewing by remember { mutableStateOf(false) }
    var previewStatus by remember { mutableStateOf<String?>(null) }
    var previewHandle by remember { mutableStateOf<PreviewHandle?>(null) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        runCatching {
            context.contentResolver.openInputStream(uri)?.use { input ->
                input.readBytes().toString(Charsets.UTF_8)
            } ?: error("无法读取该文件")
        }.onSuccess {
            resultText = it
            parseFailed = false
            parseMessage = "已读取文件,点「解析到校对表格」载入"
        }.onFailure {
            parseFailed = true
            parseMessage = "导入失败:${it.message}"
        }
    }

    // 离开本页时立刻停掉试听,避免声音继续播
    DisposableEffect(Unit) {
        onDispose { previewHandle?.cancel() }
    }

    // 选中行 -> 回填编辑框
    LaunchedEffect(selected, edit.notes.size) {
        val i = selected
        if (i != null && i in edit.notes.indices) {
            cellText = edit.notes[i].notes.joinToString(",")
            durText = trimNumber(edit.notes[i].dur)
            editError = null
        }
    }

    val bpmInt = bpm.trim().toIntOrNull() ?: 0
    val validation = remember(edit.notes, bpmInt) {
        if (edit.notes.isEmpty()) null else ScoreValidation.validate(edit.notes, bpmInt)
    }

    fun parse() {
        val trimmed = resultText.trim()
        if (trimmed.isEmpty()) {
            parseFailed = true
            parseMessage = "先粘贴简谱文本或导入 JSON 文件"
            return
        }
        val notes: List<NoteEvent>?
        if (ScoreJson.looksLikeJson(trimmed)) {
            val parsed = runCatching { ScoreJson.parse(trimmed) }
            if (parsed.isFailure) {
                parseFailed = true
                parseMessage = "解析失败:${parsed.exceptionOrNull()?.message}"
                return
            }
            notes = parsed.getOrThrow().notes
            if (name.isEmpty()) name = parsed.getOrThrow().name
            parsed.getOrThrow().suggestedBpmOrNull()?.let { bpm = it.toString() }
        } else {
            notes = JianpuParser.parse(trimmed)
            if (notes.isEmpty()) {
                parseFailed = true
                parseMessage = "没有解析出任何音符:请检查写法(例 1 2 3 4 5 6 7 1' 0 5- [1 3 5]- 1#)"
                return
            }
        }
        edit = ScoreEditing.load(notes)
        selected = null
        parseFailed = false
        parseMessage = "解析成功:${notes.size} 个元素,发声音符 ${notes.count { it.notes.isNotEmpty() }}," +
            "和弦 ${notes.count { it.notes.size > 1 }} 个"
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("乐谱识别", style = MaterialTheme.typography.titleLarge)

        // ---- 步骤 1 ----
        Card(colors = CardDefaults.cardColors(containerColor = Surface1)) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("第 1 步 · 取得简谱", color = Brand, style = MaterialTheme.typography.titleSmall)
                Text(
                    "无需配置任何 API Key:点「复制提示词」,粘贴到任意外部 AI 工具"
                        + "(如 ChatGPT / 豆包 / Kimi),附上乐谱图片发送,即可得到规范化简谱。"
                        + "也可以直接导入电脑端「导出 JSON」得到的 .json 文件。",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { copyPromptToClipboard(context) },
                        colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                    ) { Text("复制提示词") }
                    OutlinedButton(onClick = { resultText = Prompt.SAMPLE_JIANPU }) { Text("内置样例") }
                    OutlinedButton(onClick = { picker.launch("*/*") }) { Text("导入 JSON") }
                }
            }
        }

        // ---- 步骤 2 ----
        Card(colors = CardDefaults.cardColors(containerColor = Surface1)) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("第 2 步 · 解析到校对表格", color = Brand, style = MaterialTheme.typography.titleSmall)
                OutlinedTextField(
                    value = resultText,
                    onValueChange = { resultText = it },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 4,
                    textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
                    placeholder = { Text("粘贴简谱文本,或桌面导出的 JSON", color = Ink3) },
                )
                Button(
                    onClick = { parse() },
                    colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                ) { Text("解析到校对表格") }
                parseMessage?.let {
                    Text(it, color = if (parseFailed) StateError else StateSuccess)
                }
            }
        }

        // ---- 步骤 3:校对表格 ----
        if (edit.notes.isNotEmpty()) {
            Card(colors = CardDefaults.cardColors(containerColor = Surface1)) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "第 3 步 · 校对表格(点行选中后可修改)",
                        color = Brand,
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Row(Modifier.fillMaxWidth()) {
                        Text("#", color = Ink3, fontSize = 12.sp, modifier = Modifier.width(40.dp))
                        Text("音符", color = Ink3, fontSize = 12.sp, modifier = Modifier.weight(1f))
                        Text("时值(拍)", color = Ink3, fontSize = 12.sp, modifier = Modifier.width(70.dp))
                        Text("半音", color = Ink3, fontSize = 12.sp, modifier = Modifier.width(48.dp))
                    }
                    Column(
                        Modifier
                            .fillMaxWidth()
                            .heightIn(max = 260.dp)
                            .verticalScroll(rememberScrollState())
                    ) {
                        edit.notes.forEachIndexed { i, el ->
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .background(if (selected == i) Surface2 else Bg.copy(alpha = 0f))
                                    .clickable { selected = i },
                            ) {
                                Text("${i + 1}", color = Ink3, fontSize = 13.sp, modifier = Modifier.width(40.dp))
                                Text(
                                    if (el.notes.isEmpty()) "休止" else el.notes.joinToString("+"),
                                    color = if (el.notes.isEmpty()) Ink3 else StateInfo,
                                    fontSize = 13.sp,
                                    fontFamily = FontFamily.Monospace,
                                    modifier = Modifier.weight(1f),
                                )
                                Text(
                                    trimNumber(el.dur),
                                    color = Ink2,
                                    fontSize = 13.sp,
                                    modifier = Modifier.width(70.dp),
                                )
                                Text(
                                    if (el.semitone) "♯" else "",
                                    color = StateWarning,
                                    fontSize = 13.sp,
                                    modifier = Modifier.width(48.dp),
                                )
                            }
                        }
                    }

                    selected?.let { i ->
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(
                                value = cellText,
                                onValueChange = { cellText = it },
                                label = { Text("音符(逗号分隔=和弦,留空=休止)") },
                                singleLine = true,
                                modifier = Modifier.weight(1.4f),
                            )
                            OutlinedTextField(
                                value = durText,
                                onValueChange = { durText = it },
                                label = { Text("拍") },
                                singleLine = true,
                                modifier = Modifier.width(90.dp),
                            )
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = {
                                val (s1, e1) = ScoreEditing.setCells(edit, i, cellText)
                                val (s2, e2) = if (e1 == null) ScoreEditing.setDur(s1, i, durText) else s1 to null
                                editError = e1 ?: e2
                                if (editError == null) edit = s2
                            }) { Text("应用修改") }
                            OutlinedButton(onClick = { edit = ScoreEditing.toggleSemitone(edit, i) }) {
                                Text(if (edit.notes.getOrNull(i)?.semitone == true) "取消升号" else "升半音")
                            }
                        }
                        editError?.let { Text(it, color = StateError, style = MaterialTheme.typography.bodySmall) }
                    }

                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        OutlinedButton(
                            onClick = {
                                val at = selected ?: (edit.notes.size - 1)
                                edit = ScoreEditing.insertRest(edit, at, before = true)
                            },
                        ) { Text("前插半拍休止") }
                        OutlinedButton(
                            onClick = {
                                val at = selected ?: (edit.notes.size - 1)
                                edit = ScoreEditing.insertRest(edit, at, before = false)
                            },
                        ) { Text("后插半拍休止") }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = {
                            selected?.let { edit = ScoreEditing.deleteAt(edit, it) }
                            selected = null
                        }) { Text("删除选中行") }
                        OutlinedButton(onClick = { edit = ScoreEditing.appendRow(edit) }) { Text("追加行") }
                        OutlinedButton(
                            enabled = edit.canUndo,
                            onClick = { edit = edit.undo(); selected = null },
                        ) { Text("撤销") }
                        OutlinedButton(onClick = { edit = ScoreEditing.clear(edit); selected = null }) {
                            Text("清空")
                        }
                    }
                }
            }

            // ---- 校验与试听 ----
            Card(colors = CardDefaults.cardColors(containerColor = Surface1)) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    validation?.errors?.forEach {
                        Text(
                            "错误 ${it.elementIndex?.let { n -> "#${n + 1}" } ?: "（整谱）"}:${it.message}",
                            color = StateError,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    validation?.warnings?.forEach {
                        Text("提示:${it.message}", color = StateWarning, style = MaterialTheme.typography.bodySmall)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(
                            onClick = {
                                if (previewing) {
                                    previewHandle?.cancel()
                                    return@OutlinedButton
                                }
                                previewStatus = "试听中…不会向游戏发送任何点击"
                                previewing = true
                                previewHandle = ScorePreview.play(edit.notes, bpmInt) { finished ->
                                    previewing = false
                                    previewStatus = if (finished) "试听完成" else "试听已停止"
                                }
                            },
                            enabled = validation?.ok == true,
                        ) { Text(if (previewing) "停止试听" else "试听当前乐谱") }
                        OutlinedTextField(
                            value = bpm,
                            onValueChange = { bpm = it.filter { c -> c.isDigit() } },
                            label = { Text("BPM") },
                            singleLine = true,
                            modifier = Modifier.width(110.dp),
                        )
                    }
                    previewStatus?.let { Text(it, color = Ink2, style = MaterialTheme.typography.bodySmall) }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = name,
                    onValueChange = { name = it },
                    label = { Text("乐谱名称") },
                    singleLine = true,
                    modifier = Modifier.weight(1f),
                )
                Button(
                    enabled = validation?.ok == true,
                    onClick = { showSaveDialog = true },
                    colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                ) { Text("保存入库") }
            }
            if (validation != null && !validation.ok) {
                Text("存在上面的错误,修正后才能保存", color = StateError, style = MaterialTheme.typography.bodySmall)
            }
        }

        Spacer(Modifier.height(8.dp))
    }

    if (showSaveDialog) {
        val notes = edit.notes
        val sourceType = if (ScoreJson.looksLikeJson(resultText)) "json" else "manual"
        var saving by remember { mutableStateOf(false) }
        AlertDialog(
            onDismissRequest = { if (!saving) showSaveDialog = false },
            title = { Text("确认保存") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("名称:${name.ifBlank { "未命名" }}", color = Ink2)
                    Text("BPM:$bpmInt · ${notes.size} 个元素", color = Ink2)
                    validation?.warnings?.forEach { Text("提示:${it.message}", color = StateWarning) }
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    if (saving) return@TextButton
                    saving = true
                    scope.launch {
                        container.db.scoreDao().insert(
                            ScoreEntity(
                                name = name.ifBlank {
                                    "乐谱_" + SimpleDateFormat("MMdd_HHmm", Locale.getDefault()).format(Date())
                                },
                                sourceFile = "",
                                sourceType = sourceType,
                                rawText = resultText,
                                notesJson = NoteCodec.encode(notes),
                                bpmDefault = bpmInt,
                                createdAt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date()),
                            )
                        )
                        saving = false
                        showSaveDialog = false
                        previewHandle?.cancel()
                        resultText = ""
                        name = ""
                        edit = ScoreEditing.load(emptyList())
                        selected = null
                        parseMessage = "已保存到乐谱库"
                        parseFailed = false
                    }
                }) { Text("保存", color = Brand) }
            },
            dismissButton = {
                TextButton(onClick = { if (!saving) showSaveDialog = false }) { Text("取消", color = Ink2) }
            },
        )
    }
}

/** JSON 里带了 bpm 才回填,避免把默认值当成用户意图。 */
private fun ScoreJson.Parsed.suggestedBpmOrNull(): Int? = bpm.takeIf { it in 30..300 }

private fun trimNumber(value: Double): String =
    if (value == value.toLong().toDouble()) value.toLong().toString() else value.toString()

/** 复制识别提示词到剪贴板,供任意外部 AI 工具识别乐谱图片,免配置 API Key。 */
private fun copyPromptToClipboard(context: Context) {
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
    clipboard.setPrimaryClip(android.content.ClipData.newPlainText("简谱识别提示词", Prompt.JIANPU_PROMPT))
    android.widget.Toast.makeText(
        context, "提示词已复制:粘贴到外部 AI 工具并附上乐谱图片", android.widget.Toast.LENGTH_LONG
    ).show()
}
