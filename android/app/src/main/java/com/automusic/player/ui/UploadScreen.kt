package com.automusic.player.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.automusic.player.AppContainer
import com.automusic.player.core.JianpuParser
import com.automusic.player.core.Prompt
import com.automusic.player.core.db.ScoreEntity
import com.automusic.player.ui.theme.Bg
import com.automusic.player.ui.theme.Brand
import com.automusic.player.ui.theme.Ink2
import com.automusic.player.ui.theme.Ink3
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.launch

/** 识别页:复制提示词 -> 外部 AI 识别 -> 粘贴简谱 -> 校对预览 -> 保存入库(免配置)。 */
@Composable
fun UploadScreen(container: AppContainer) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var resultText by remember { mutableStateOf("") }
    var showSaveDialog by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("乐谱识别", style = MaterialTheme.typography.titleLarge)

        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("第 1 步 · 复制识别提示词", color = Brand, style = MaterialTheme.typography.titleSmall)
                Text(
                    "无需配置任何 API Key:点「复制提示词」,粘贴到任意外部 AI 工具"
                        + "(如 ChatGPT / 豆包 / Kimi),附上乐谱图片发送,即可得到规范化简谱。",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { copyPromptToClipboard(context) },
                        colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                    ) { Text("复制提示词") }
                    OutlinedButton(onClick = { resultText = Prompt.SAMPLE_JIANPU }) {
                        Text("填入内置样例")
                    }
                }
            }
        }

        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("第 2 步 · 粘贴简谱并校对", color = Brand, style = MaterialTheme.typography.titleSmall)
                Text(
                    "把外部 AI 返回的简谱粘贴到下方(也支持手动输入与修改),解析通过后即可保存到乐谱库。",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )
                OutlinedTextField(
                    value = resultText,
                    onValueChange = { resultText = it },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 4,
                    textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
                    placeholder = {
                        Text(
                            "外部 AI 工具识别出的简谱直接粘贴到这里 · 支持手动输入与修改",
                            color = Ink3,
                        )
                    },
                )
                val notes = remember(resultText) { JianpuParser.parse(resultText) }
                if (notes.isNotEmpty()) {
                    val chordCount = notes.count { it.notes.size > 1 }
                    val beats = notes.sumOf { it.dur }
                    Text(
                        "解析成功:共 ${notes.size} 个音符,其中和弦 $chordCount 个,总时长 %.1f 拍".format(beats),
                        color = Ink2,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Button(
                    enabled = notes.isNotEmpty(),
                    onClick = { showSaveDialog = true },
                    colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                ) { Text("保存到乐谱库") }
            }
        }

        Spacer(Modifier.height(8.dp))
    }

    if (showSaveDialog) {
        var name by remember { mutableStateOf("乐谱_${SimpleDateFormat("MMdd_HHmm", Locale.getDefault()).format(Date())}") }
        AlertDialog(
            onDismissRequest = { showSaveDialog = false },
            title = { Text("保存乐谱") },
            text = {
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("乐谱名称") })
            },
            confirmButton = {
                TextButton(onClick = {
                    scope.launch {
                        container.db.scoreDao().insert(
                            ScoreEntity(
                                name = name,
                                sourceFile = "",
                                sourceType = "manual",
                                rawText = resultText,
                                notesJson = com.automusic.player.core.NoteCodec.encode(
                                    JianpuParser.parse(resultText)
                                ),
                                bpmDefault = 100,
                                createdAt = SimpleDateFormat(
                                    "yyyy-MM-dd HH:mm:ss", Locale.getDefault()
                                ).format(Date()),
                            )
                        )
                        showSaveDialog = false
                    }
                }) { Text("保存", color = Brand) }
            },
            dismissButton = {
                TextButton(onClick = { showSaveDialog = false }) { Text("取消", color = Ink2) }
            },
        )
    }
}

/** 复制识别提示词到剪贴板,供任意外部 AI 工具识别乐谱图片,免配置 API Key。 */
private fun copyPromptToClipboard(context: android.content.Context) {
    val clipboard = context.getSystemService(android.content.ClipboardManager::class.java)
    clipboard?.setPrimaryClip(
        android.content.ClipData.newPlainText("简谱识别提示词", Prompt.JIANPU_PROMPT)
    )
    android.widget.Toast.makeText(
        context, "提示词已复制:粘贴到外部 AI 工具并附上乐谱图片", android.widget.Toast.LENGTH_LONG
    ).show()
}
