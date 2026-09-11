package com.automusic.player.ui

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.AutoFixHigh
import androidx.compose.material.icons.outlined.DeleteOutline
import androidx.compose.material.icons.outlined.FileDownload
import androidx.compose.material.icons.outlined.FileUpload
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Save
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material.icons.outlined.TouchApp
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedIconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.automusic.player.AppContainer
import com.automusic.player.calib.KeyLayout
import com.automusic.player.calib.LayoutStore
import com.automusic.player.core.KeyPointMap
import com.automusic.player.core.ScreenMetrics
import com.automusic.player.input.TouchInjector
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** 游戏琴键坐标标定与布局管理。 */
@Composable
fun CalibScreen(container: AppContainer) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val layoutState by container.layouts.state.collectAsState()
    val initialLayout = layoutState.active

    var editingLayout by remember { mutableStateOf(initialLayout) }
    var editingName by remember { mutableStateOf(initialLayout?.name.orEmpty()) }
    var points by remember(editingLayout?.id) {
        mutableStateOf<Map<String, Pair<Float, Float>>>(editingLayout?.points ?: emptyMap())
    }
    var cursor by remember { mutableIntStateOf(0) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var notice by remember { mutableStateOf<String?>(null) }
    var noticeError by remember { mutableStateOf(false) }
    var countdown by remember { mutableStateOf<Int?>(null) }
    var showNewDialog by remember { mutableStateOf(false) }
    var showDeleteDialog by remember { mutableStateOf(false) }

    val notes = KeyPointMap.ALL_NOTES
    val currentNote = notes.getOrNull(cursor) ?: notes.first()
    val completeCount = points.keys.count { it in notes }

    fun selectLayout(layout: KeyLayout) {
        editingLayout = layout
        editingName = layout.name
        cursor = 0
        notice = null
    }

    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            bitmap = decodeFull(context, uri)
            noticeError = bitmap == null
            notice = if (bitmap == null) "无法读取所选截图" else null
        }
    }

    val importLayout = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        try {
            val json = context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                ?: throw IllegalArgumentException("无法读取所选文件")
            val outcome = container.layouts.importJson(json)
            selectLayout(outcome.layout)
            noticeError = false
            notice = buildString {
                append("已导入《${outcome.layout.name}》，包含 ${outcome.layout.points.size}/21 个键位")
                outcome.warnings.forEach { append("\n$it") }
            }
        } catch (e: Exception) {
            noticeError = true
            notice = "导入失败:${e.message}"
        }
    }

    fun exportCurrentLayout() {
        val target = editingLayout ?: layoutState.active
        if (target == null) {
            noticeError = true
            notice = "没有可导出的布局"
            return
        }
        runCatching {
            val json = LayoutStore.exportLayoutJson(target.copy(name = editingName.ifBlank { target.name }, points = points))
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "application/json"
                putExtra(Intent.EXTRA_TITLE, "amp_layout_${target.id}.json")
                putExtra(Intent.EXTRA_TEXT, json)
            }
            context.startActivity(Intent.createChooser(send, "分享布局"))
        }.onSuccess {
            noticeError = false
            notice = "布局分享内容已生成"
        }.onFailure {
            noticeError = true
            notice = "导出失败:${it.message}"
        }
    }

    Box(Modifier.fillMaxSize().background(Bg)) {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            Column(
                Modifier.fillMaxWidth().background(Surface1).padding(horizontal = 18.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text("琴键标定", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                Text(
                    editingLayout?.let { "${it.name} · $completeCount/21 键" } ?: "选择或新建琴键布局",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            Column(
                Modifier.padding(horizontal = 16.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = Surface1),
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(11.dp)) {
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text("布局文件", color = Ink, fontWeight = FontWeight.SemiBold)
                                Text("${layoutState.layouts.size} 个本地布局", color = Ink3, style = MaterialTheme.typography.bodySmall)
                            }
                            IconButton(onClick = { showNewDialog = true }) {
                                Icon(Icons.Outlined.Add, contentDescription = "新建布局", tint = Brand)
                            }
                            IconButton(onClick = { importLayout.launch("*/*") }) {
                                Icon(Icons.Outlined.FileDownload, contentDescription = "导入布局", tint = Ink2)
                            }
                            IconButton(enabled = editingLayout != null, onClick = { exportCurrentLayout() }) {
                                Icon(Icons.Outlined.Share, contentDescription = "导出布局", tint = Ink2)
                            }
                        }

                        if (layoutState.layouts.isEmpty()) {
                            Text("暂无布局", color = Ink3, style = MaterialTheme.typography.bodySmall)
                        } else {
                            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                items(layoutState.layouts, key = { it.id }) { layout ->
                                    FilterChip(
                                        selected = editingLayout?.id == layout.id,
                                        onClick = { selectLayout(layout) },
                                        label = { Text(layout.name) },
                                    )
                                }
                            }
                        }

                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            OutlinedTextField(
                                value = editingName,
                                onValueChange = { editingName = it.take(40) },
                                enabled = editingLayout != null,
                                label = { Text("布局名称") },
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                            Spacer(Modifier.size(8.dp))
                            OutlinedIconButton(
                                enabled = editingLayout != null,
                                onClick = { showDeleteDialog = true },
                            ) {
                                Icon(Icons.Outlined.DeleteOutline, contentDescription = "删除布局", tint = StateError)
                            }
                        }
                    }
                }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = Surface1),
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text("标定画布", color = Ink, fontWeight = FontWeight.SemiBold)
                                Text(
                                    if (bitmap == null) "使用游戏演奏界面的横屏截图" else "点击截图记录当前键位中心",
                                    color = Ink3,
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                            OutlinedButton(onClick = { pickImage.launch("image/*") }) {
                                Icon(Icons.Outlined.Image, null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.size(6.dp))
                                Text(if (bitmap == null) "选择截图" else "更换")
                            }
                        }

                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            color = BrandSoft,
                            shape = RoundedCornerShape(8.dp),
                            border = BorderStroke(1.dp, Brand.copy(alpha = 0.35f)),
                        ) {
                            Row(
                                Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Surface(color = Brand, shape = CircleShape) {
                                    Text(
                                        "${cursor + 1}",
                                        color = Bg,
                                        fontWeight = FontWeight.Bold,
                                        textAlign = TextAlign.Center,
                                        modifier = Modifier.size(30.dp).padding(top = 5.dp),
                                    )
                                }
                                Column(Modifier.weight(1f).padding(horizontal = 10.dp)) {
                                    Text("当前目标 · ${calibrationNoteLabel(currentNote)}", color = Brand, fontWeight = FontWeight.SemiBold)
                                    Text(
                                        if (points.containsKey(currentNote)) "已有坐标，点击截图可重新标定" else "等待记录坐标",
                                        color = Ink2,
                                        style = MaterialTheme.typography.bodySmall,
                                    )
                                }
                                Text("$completeCount/21", color = Brand, style = MaterialTheme.typography.labelLarge)
                            }
                        }

                        if (bitmap != null) {
                            val source = bitmap!!
                            Box(
                                Modifier.fillMaxWidth()
                                    .aspectRatio(source.width.toFloat() / source.height)
                                    .background(Surface2)
                                    .pointerInput(currentNote, points) {
                                        detectTapGestures { offset ->
                                            val nx = (offset.x / size.width).coerceIn(0f, 1f)
                                            val ny = (offset.y / size.height).coerceIn(0f, 1f)
                                            points = points + (currentNote to (nx to ny))
                                            if (cursor < notes.lastIndex) cursor += 1
                                        }
                                    },
                            ) {
                                Image(
                                    bitmap = source.asImageBitmap(),
                                    contentDescription = "标定截图",
                                    modifier = Modifier.fillMaxSize(),
                                    contentScale = ContentScale.FillBounds,
                                )
                                Canvas(Modifier.fillMaxSize()) {
                                    val radius = size.minDimension * 0.024f
                                    notes.forEach { note ->
                                        val point = points[note] ?: return@forEach
                                        val selected = note == currentNote
                                        drawCircle(
                                            color = if (selected) StateSuccess else Brand,
                                            radius = if (selected) radius * 1.25f else radius,
                                            center = Offset(point.first * size.width, point.second * size.height),
                                        )
                                        drawCircle(
                                            color = Color.White,
                                            radius = if (selected) radius * 1.25f else radius,
                                            center = Offset(point.first * size.width, point.second * size.height),
                                            style = Stroke(width = radius * 0.22f),
                                        )
                                    }
                                }
                            }
                        } else {
                            Surface(
                                modifier = Modifier.fillMaxWidth().aspectRatio(2412f / 1084f),
                                color = Surface2,
                                shape = RoundedCornerShape(8.dp),
                                border = BorderStroke(1.dp, Line),
                                onClick = { pickImage.launch("image/*") },
                            ) {
                                Column(
                                    Modifier.fillMaxSize(),
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    verticalArrangement = Arrangement.Center,
                                ) {
                                    Icon(Icons.Outlined.FileUpload, null, tint = Ink3, modifier = Modifier.size(28.dp))
                                    Spacer(Modifier.height(7.dp))
                                    Text("导入游戏截图", color = Ink2)
                                }
                            }
                        }

                        LinearProgressIndicator(
                            progress = { completeCount / 21f },
                            modifier = Modifier.fillMaxWidth(),
                            color = if (completeCount == 21) StateSuccess else Brand,
                            trackColor = Line,
                        )

                        CalibrationKeyGrid(
                            notes = notes,
                            selectedIndex = cursor,
                            completed = points.keys,
                            onSelect = { cursor = it },
                        )

                        HorizontalDivider(color = Line)

                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(
                                modifier = Modifier.weight(1f).height(46.dp),
                                onClick = {
                                    val high1 = points["high_1"]
                                    val high7 = points["high_7"]
                                    val low1 = points["low_1"]
                                    val low7 = points["low_7"]
                                    if (high1 == null || high7 == null || low1 == null || low7 == null) {
                                        noticeError = true
                                        notice = "请先标定高音 1、高音 7、低音 1、低音 7"
                                    } else {
                                        val grid = KeyPointMap.deriveGrid(high1, high7, low1, low7)
                                        if (grid == null) {
                                            noticeError = true
                                            notice = "四个角点的位置关系无效"
                                        } else {
                                            points = grid
                                            noticeError = false
                                            notice = "已推算 21 个键位，请测试后保存"
                                        }
                                    }
                                },
                            ) {
                                Icon(Icons.Outlined.AutoFixHigh, null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.size(6.dp))
                                Text("四角推算")
                            }
                            OutlinedButton(
                                enabled = points.containsKey(currentNote) && countdown == null,
                                modifier = Modifier.weight(1f).height(46.dp),
                                onClick = {
                                    scope.launch {
                                        if (!TouchInjector.accessibilityReady) {
                                            noticeError = true
                                            notice = "请先在演奏页开启触摸服务"
                                            return@launch
                                        }
                                        try {
                                            val point = points[currentNote] ?: return@launch
                                            for (i in 3 downTo 1) {
                                                countdown = i
                                                delay(1_000)
                                            }
                                            countdown = null
                                            val (w, h) = ScreenMetrics.realSize(context)
                                            TouchInjector.tap(point.first * w, point.second * h)
                                            noticeError = false
                                            notice = "${calibrationNoteLabel(currentNote)} 测试点击已发送"
                                        } catch (e: Exception) {
                                            countdown = null
                                            noticeError = true
                                            notice = "测试失败:${e.message}"
                                        }
                                    }
                                },
                            ) {
                                Icon(Icons.Outlined.TouchApp, null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.size(6.dp))
                                Text("测试当前键")
                            }
                        }
                    }
                }

                notice?.let { message ->
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = (if (noticeError) StateError else StateSuccess).copy(alpha = 0.1f),
                        shape = RoundedCornerShape(8.dp),
                    ) {
                        Text(
                            message,
                            color = if (noticeError) StateError else StateSuccess,
                            modifier = Modifier.padding(12.dp),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }

                Button(
                    enabled = editingLayout != null && points.isNotEmpty() && editingName.isNotBlank(),
                    onClick = {
                        val target = editingLayout ?: return@Button
                        val saved = KeyLayout(target.id, editingName.trim(), points)
                        container.layouts.save(saved)
                        editingLayout = saved
                        noticeError = false
                        notice = "《${saved.name}》已保存并设为当前布局"
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                    modifier = Modifier.fillMaxWidth().height(50.dp),
                ) {
                    Icon(Icons.Outlined.Save, null, modifier = Modifier.size(19.dp))
                    Spacer(Modifier.size(7.dp))
                    Text("保存并使用此布局", fontWeight = FontWeight.SemiBold)
                }

                Spacer(Modifier.height(8.dp))
            }
        }

        countdown?.let { number ->
            Box(Modifier.fillMaxSize().background(Bg.copy(alpha = 0.9f)), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(number.toString(), fontSize = 112.sp, color = Brand, style = MaterialTheme.typography.displayLarge)
                    Text("切换到游戏后将测试 ${calibrationNoteLabel(currentNote)}", color = Ink2)
                }
            }
        }
    }

    if (showNewDialog) {
        var newId by remember { mutableStateOf("") }
        var newName by remember { mutableStateOf("") }
        AlertDialog(
            onDismissRequest = { showNewDialog = false },
            title = { Text("新建布局") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                    OutlinedTextField(
                        value = newName,
                        onValueChange = { newName = it.take(40) },
                        label = { Text("布局名称") },
                        singleLine = true,
                    )
                    OutlinedTextField(
                        value = newId,
                        onValueChange = { newId = it.filter { char -> char.isLetterOrDigit() || char == '_' }.take(32) },
                        label = { Text("英文 ID") },
                        supportingText = { Text("例如 my_phone") },
                        singleLine = true,
                    )
                }
            },
            confirmButton = {
                TextButton(
                    enabled = newId.isNotBlank(),
                    onClick = {
                        val created = KeyLayout(
                            newId.trim(),
                            newName.trim().ifBlank { newId.trim() },
                            KeyPointMap.defaultLayout(KeyPointMap.GAME_WUTHERING),
                        )
                        container.layouts.save(created, makeActive = false)
                        selectLayout(created)
                        showNewDialog = false
                    },
                ) { Text("创建", color = Brand) }
            },
            dismissButton = {
                TextButton(onClick = { showNewDialog = false }) { Text("取消", color = Ink2) }
            },
        )
    }

    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text("删除布局") },
            text = { Text("确认删除《${editingLayout?.name.orEmpty()}》？此操作无法撤销。") },
            confirmButton = {
                TextButton(onClick = {
                    editingLayout?.let { container.layouts.delete(it.id) }
                    val next = container.layouts.state.value.active
                    editingLayout = next
                    editingName = next?.name.orEmpty()
                    cursor = 0
                    noticeError = false
                    notice = "布局已删除"
                    showDeleteDialog = false
                }) { Text("删除", color = StateError) }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) { Text("取消", color = Ink2) }
            },
        )
    }
}

@Composable
private fun CalibrationKeyGrid(
    notes: List<String>,
    selectedIndex: Int,
    completed: Set<String>,
    onSelect: (Int) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        notes.chunked(7).forEachIndexed { rowIndex, rowNotes ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                rowNotes.forEachIndexed { columnIndex, note ->
                    val index = rowIndex * 7 + columnIndex
                    val selected = index == selectedIndex
                    val done = note in completed
                    Surface(
                        modifier = Modifier.weight(1f).height(48.dp),
                        onClick = { onSelect(index) },
                        color = when {
                            selected -> Brand
                            done -> StateSuccess.copy(alpha = 0.12f)
                            else -> Surface2
                        },
                        border = BorderStroke(
                            if (selected) 0.dp else 1.dp,
                            if (done) StateSuccess.copy(alpha = 0.4f) else Line,
                        ),
                        shape = RoundedCornerShape(6.dp),
                    ) {
                        Column(
                            Modifier.fillMaxSize(),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center,
                        ) {
                            Text(
                                note.substringAfter('_'),
                                color = if (selected) Bg else Ink,
                                fontWeight = FontWeight.SemiBold,
                            )
                            Text(
                                if (done) "已标" else when (rowIndex) { 0 -> "高"; 1 -> "中"; else -> "低" },
                                color = if (selected) Bg.copy(alpha = 0.7f) else if (done) StateSuccess else Ink3,
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                    }
                }
            }
        }
    }
}

private fun calibrationNoteLabel(note: String): String {
    val parts = note.split('_')
    val octave = when (parts.firstOrNull()) {
        "high" -> "高音"
        "low" -> "低音"
        else -> "中音"
    }
    return "$octave ${parts.getOrNull(1).orEmpty()}"
}

/** 标定截图全量解码:截图较大时粗降采样到最长边 2048。 */
private fun decodeFull(context: android.content.Context, uri: Uri): Bitmap? = try {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, bounds)
    }
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 2048) sample *= 2
    val opts = BitmapFactory.Options().apply { inSampleSize = sample }
    context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, opts)
    }
} catch (e: Exception) {
    null
}
