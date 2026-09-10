package com.automusic.player.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.automusic.player.AppContainer
import com.automusic.player.core.JianpuParser
import com.automusic.player.core.db.ScoreEntity
import com.automusic.player.core.recognizer.Prompt
import com.automusic.player.ui.theme.Bg
import com.automusic.player.ui.theme.Brand
import com.automusic.player.ui.theme.Ink2
import com.automusic.player.ui.theme.Ink3
import com.automusic.player.ui.theme.StateError
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.launch

/** 识别页:上传乐谱图片/粘贴文本 -> 大模型识别 -> 解析预览 -> 保存入库。 */
@Composable
fun UploadScreen(container: AppContainer) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var preview by remember { mutableStateOf<Bitmap?>(null) }
    var resultText by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var showSaveDialog by remember { mutableStateOf(false) }

    val pickImage = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            imageUri = uri
            error = null
            preview = decodePreview(context, uri)
        }
    }

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
                Text("第 1 步 · 上传乐谱", color = Brand, style = MaterialTheme.typography.titleSmall)
                Text(
                    "支持 jpg / png / webp / bmp 乐谱图片;文档类谱面可直接把文本粘贴到下方结果框。",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { pickImage.launch("image/*") }) { Text("选择图片") }
                    OutlinedButton(onClick = {
                        imageUri = null
                        preview = null
                        resultText = Prompt.SAMPLE_JIANPU
                    }) { Text("填入内置样例") }
                }
                preview?.let { bmp ->
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                            .background(com.automusic.player.ui.theme.Surface2, RoundedCornerShape(8.dp))
                    ) {
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "乐谱预览",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Fit,
                        )
                    }
                }
            }
        }

        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("第 2 步 · 大模型识别", color = Brand, style = MaterialTheme.typography.titleSmall)
                Text(
                    "在线识别需先在「设置」页配置模型;无模型时可点「复制提示词」,把提示词和乐谱图片发给任意外部 AI 工具(如 ChatGPT / 豆包 / Kimi),再把返回的简谱粘贴到下方结果框直接校对入库。",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Button(
                        enabled = !busy && imageUri != null,
                        onClick = {
                            val uri = imageUri ?: return@Button
                            busy = true
                            error = null
                            scope.launch {
                                try {
                                    val recognizer = container.createRecognizer()
                                    resultText = if (recognizer != null) {
                                        recognizer.recognizeImage(context, uri)
                                    } else {
                                        Prompt.SAMPLE_JIANPU
                                    }
                                } catch (e: Exception) {
                                    error = e.message ?: e.toString()
                                } finally {
                                    busy = false
                                }
                            }
                        },
                        colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                    ) { Text(if (busy) "识别中..." else "开始识别") }
                    OutlinedButton(onClick = { copyPromptToClipboard(context) }) { Text("复制提示词") }
                    if (busy) CircularProgressIndicator(Modifier.height(24.dp))
                }
                if (error != null) {
                    Text(error!!, color = StateError, style = MaterialTheme.typography.bodySmall)
                }
            }
        }

        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("第 3 步 · 校对结果", color = Brand, style = MaterialTheme.typography.titleSmall)
                OutlinedTextField(
                    value = resultText,
                    onValueChange = { resultText = it },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 4,
                    textStyle = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
                    placeholder = { Text("识别结果(规范化简谱);外部 AI 工具识别的简谱可直接粘贴到这里", color = Ink3) },
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
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        enabled = notes.isNotEmpty(),
                        onClick = { showSaveDialog = true },
                        colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                    ) { Text("保存到乐谱库") }
                }
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
                                sourceFile = imageUri?.toString() ?: "",
                                sourceType = if (imageUri != null) "image" else "text",
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

/** 预览解码:粗降采样到最长边约 1024,避免大图 OOM。 */
private fun decodePreview(context: android.content.Context, uri: Uri): Bitmap? = try {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, bounds)
    }
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 1024) sample *= 2
    val opts = BitmapFactory.Options().apply { inSampleSize = sample }
    context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, opts)
    }
} catch (e: Exception) {
    null
}
