package com.automusic.player.ui

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.automusic.player.AppContainer
import com.automusic.player.calib.KeyLayout
import com.automusic.player.calib.LayoutStore
import com.automusic.player.core.KeyPointMap
import com.automusic.player.core.ScreenMetrics
import com.automusic.player.core.delta.DeltaKeyPoint
import com.automusic.player.input.TouchInjector
import com.automusic.player.ui.theme.Brand
import com.automusic.player.ui.theme.Bg
import com.automusic.player.ui.theme.Ink2
import com.automusic.player.ui.theme.Ink3
import com.automusic.player.ui.theme.StateError
import com.automusic.player.ui.theme.StateSuccess
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** 标定页:导入游戏演奏界面截图 -> 依次点击 21 个琴键位 -> 归一化保存。 */
@Composable
fun CalibScreen(container: AppContainer) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val layoutState by container.layouts.state.collectAsState()

    var editingLayout by remember { mutableStateOf<KeyLayout?>(null) }
    var editingName by remember { mutableStateOf("") }
    var points by remember(editingLayout) {
        mutableStateOf<Map<String, Pair<Float, Float>>>(editingLayout?.points ?: emptyMap())
    }
    var cursor by remember { mutableIntStateOf(0) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var canvasSize by remember { mutableStateOf(IntSize.Zero) }
    var notice by remember { mutableStateOf<String?>(null) }
    var countdown by remember { mutableStateOf<Int?>(null) }
    var showNewDialog by remember { mutableStateOf(false) }

    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            bitmap = decodeFull(context, uri)
            notice = null
        }
    }

    val importLayout = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        try {
            val json = context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
                ?: throw IllegalArgumentException("无法读取所选文件")
            val outcome = container.layouts.importJson(json)
            editingLayout = outcome.layout
            editingName = outcome.layout.name
            notice = buildString {
                val keyCount = if (outcome.layout.gameType == DeltaKeyPoint.GAME_DELTA)
                    DeltaKeyPoint.KEY_POINTS.size else KeyPointMap.ALL_NOTES.size
                append(
                    "已导入布局《${outcome.layout.name}》" +
                        "(${outcome.layout.points.size}/$keyCount 键)并设为激活"
                )
                if (outcome.renamed) append("\n原 ID 与现有布局冲突,已自动改名避免覆盖")
                outcome.warnings.forEach { append("\n$it") }
            }
        } catch (e: Exception) {
            notice = "导入失败:${e.message}"
        }
    }

    val gameType = editingLayout?.gameType ?: layoutState.active?.gameType ?: KeyPointMap.GAME_WUTHERING
    val isDelta = gameType == DeltaKeyPoint.GAME_DELTA
    val notes = if (isDelta) DeltaKeyPoint.KEY_POINTS else KeyPointMap.ALL_NOTES
    val currentNote = notes.getOrNull(cursor) ?: notes.last()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("琴键标定", style = MaterialTheme.typography.titleLarge)

        // ---- 布局管理 ----
        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("布局", color = Brand, style = MaterialTheme.typography.titleSmall)
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    itemsIndexed(layoutState.layouts) { _, l ->
                        FilterChip(
                            selected = editingLayout?.id == l.id ||
                                (editingLayout == null && layoutState.active?.id == l.id),
                            onClick = {
                                editingLayout = l
                                editingName = l.name
                            },
                            label = { Text(l.name) },
                        )
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { showNewDialog = true }) { Text("新建布局") }
                    OutlinedButton(onClick = {
                        val target = editingLayout ?: layoutState.active
                        if (target == null) {
                            notice = "没有可导出的布局"
                            return@OutlinedButton
                        }
                        try {
                            val json = LayoutStore.exportLayoutJson(target)
                            val send = Intent(Intent.ACTION_SEND).apply {
                                type = "application/json"
                                putExtra(Intent.EXTRA_TITLE, "amp_layout_${target.id}.json")
                                putExtra(Intent.EXTRA_TEXT, json)
                            }
                            context.startActivity(Intent.createChooser(send, "分享布局"))
                            notice = "已生成布局分享内容(以文本形式发送,可在另一台设备导入)"
                        } catch (e: Exception) {
                            notice = "导出失败:${e.message}"
                        }
                    }) { Text("导出布局") }
                    OutlinedButton(onClick = { importLayout.launch("*/*") }) { Text("导入布局") }
                }
                if (editingLayout != null) {
                    OutlinedTextField(
                        value = editingName,
                        onValueChange = { editingName = it },
                        label = { Text("布局名称") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                    )
                }
            }
        }

        // ---- 截图与标定 ----
        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { pickImage.launch("image/*") },
                        colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                    ) { Text("导入游戏截图") }
                }
                Text(
                    "在截图上点击「当前槽位」对应的琴键中心;默认布局已按真机截图预填,可逐点微调。",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )

                bitmap?.let { bmp ->
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .aspectRatio(bmp.width.toFloat() / bmp.height)
                            .onSizeChanged { canvasSize = it }
                            .pointerInput(points) {
                                detectTapGestures { offset ->
                                    val nx = (offset.x / size.width).coerceIn(0f, 1f)
                                    val ny = (offset.y / size.height).coerceIn(0f, 1f)
                                    points = points + (currentNote to (nx to ny))
                                    if (cursor < notes.size - 1) cursor += 1
                                }
                            },
                    ) {
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "标定截图",
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.FillBounds,
                        )
                        Canvas(Modifier.fillMaxSize()) {
                            val r = size.minDimension * 0.02f
                            for (n in notes) {
                                val p = points[n] ?: continue
                                drawCircle(
                                    color = Brand,
                                    radius = r,
                                    center = Offset(p.first * size.width, p.second * size.height),
                                )
                                drawCircle(
                                    color = Color.White,
                                    radius = r,
                                    center = Offset(p.first * size.width, p.second * size.height),
                                    style = Stroke(width = r * 0.25f),
                                )
                            }
                        }
                    }
                } ?: Box(
                    Modifier
                        .fillMaxWidth()
                        .aspectRatio(2412f / 1084f)
                        .background(com.automusic.player.ui.theme.Surface2),
                    contentAlignment = Alignment.Center,
                ) {
                    Text("尚未导入截图", color = Ink3)
                }

                // 槽位选择
                LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    itemsIndexed(notes) { i, n ->
                        FilterChip(
                            selected = i == cursor,
                            onClick = { cursor = i },
                            label = {
                                Text(
                                    (if (points.containsKey(n)) "●" else "○") + n.replace("_", ""),
                                    style = MaterialTheme.typography.labelSmall,
                                )
                            },
                        )
                    }
                }
                Text(
                    "当前槽位:${currentNote}  (● 已标 / ○ 未标)",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        // ---- 操作 ----
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (!isDelta) {
                OutlinedButton(onClick = {
                    val h1 = points["high_1"]
                    val h7 = points["high_7"]
                    val l1 = points["low_1"]
                    val l7 = points["low_7"]
                    if (h1 == null || h7 == null || l1 == null || l7 == null) {
                        notice = "四角推算需先标定 高音1、高音7、低音1、低音7 四个角点"
                        return@OutlinedButton
                    }
                    KeyPointMap.deriveGrid(h1, h7, l1, l7)?.let { grid ->
                        points = grid
                        notice = "已按四角双线性插值推算全部 21 点,请检查并微调"
                    }
                }) { Text("四角推算") }
            }

            OutlinedButton(onClick = {
                scope.launch {
                    if (!TouchInjector.accessibilityReady) {
                        notice = "请先到「演奏」页开启触摸注入无障碍服务"
                        return@launch
                    }
                    try {
                        val p = points[currentNote]
                        if (p == null) {
                            notice = "当前槽位未标定"
                            return@launch
                        }
                        val (w, h) = ScreenMetrics.realSize(context)
                        // 3 秒倒计时:给用户时间切到游戏验证
                        for (i in 3 downTo 1) {
                            countdown = i
                            delay(1000)
                        }
                        countdown = null
                        TouchInjector.tap(p.first * w, p.second * h)
                        notice = "已向 ${currentNote} 发送测试点击"
                    } catch (e: Exception) {
                        countdown = null
                        notice = "测试失败:${e.message}"
                    }
                }
            }) { Text("测试点击") }
        }

        if (notice != null) {
            Text(notice!!, color = StateSuccess, style = MaterialTheme.typography.bodySmall)
        }

        Button(
            enabled = editingLayout != null && points.isNotEmpty(),
            onClick = {
                val target = editingLayout ?: return@Button
                container.layouts.save(
                    KeyLayout(target.id, editingName.ifBlank { target.name }, points, target.gameType),
                )
                editingLayout = container.layouts.state.value.layouts.first { it.id == target.id }
                notice = "布局已保存并设为激活"
            },
            colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
            modifier = Modifier.fillMaxWidth(),
        ) { Text("保存布局") }

        Spacer(Modifier.height(8.dp))
    }

    countdown?.let { n ->
        Box(
            Modifier
                .fillMaxSize()
                .background(com.automusic.player.ui.theme.Bg.copy(alpha = 0.85f)),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    n.toString(),
                    fontSize = 120.sp,
                    color = Brand,
                    style = MaterialTheme.typography.displayLarge,
                )
                Text("请在倒计时结束前切换到游戏", color = Ink2)
            }
        }
    }

    if (showNewDialog) {
        var newId by remember { mutableStateOf("") }
        var newName by remember { mutableStateOf("") }
        var newGame by remember { mutableStateOf(KeyPointMap.GAME_WUTHERING) }
        AlertDialog(
            onDismissRequest = { showNewDialog = false },
            title = { Text("新建布局") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(value = newId, onValueChange = { newId = it },
                        label = { Text("英文 ID(如 my_phone)") }, singleLine = true)
                    OutlinedTextField(value = newName, onValueChange = { newName = it },
                        label = { Text("显示名称") }, singleLine = true)
                    Text("游戏类型", color = Ink2, style = MaterialTheme.typography.labelSmall)
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        KeyPointMap.GAME_NAMES.forEach { (g, label) ->
                            FilterChip(
                                selected = newGame == g,
                                onClick = { newGame = g },
                                label = { Text(label) },
                            )
                        }
                        FilterChip(
                            selected = newGame == DeltaKeyPoint.GAME_DELTA,
                            onClick = { newGame = DeltaKeyPoint.GAME_DELTA },
                            label = { Text("三角洲") },
                        )
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    if (newId.isNotBlank()) {
                        val defaultPoints = if (newGame == DeltaKeyPoint.GAME_DELTA)
                            DeltaKeyPoint.defaultLayout() else KeyPointMap.defaultLayout(newGame)
                        val created = KeyLayout(newId.trim(), newName.ifBlank { newId.trim() }, defaultPoints, newGame)
                        container.layouts.save(created, makeActive = false)
                        editingLayout = created
                        editingName = created.name
                    }
                    showNewDialog = false
                }) { Text("创建", color = Brand) }
            },
            dismissButton = {
                TextButton(onClick = { showNewDialog = false }) { Text("取消", color = Ink2) }
            },
        )
    }
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
