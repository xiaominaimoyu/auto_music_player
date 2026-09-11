package com.automusic.player.ui

import android.content.Context
import android.content.Intent
import android.provider.Settings
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.automusic.player.AppContainer
import com.automusic.player.core.HumanizeParams
import com.automusic.player.core.NoteCodec
import com.automusic.player.core.ScreenMetrics
import com.automusic.player.core.PlayerEngine
import com.automusic.player.input.AmpAccessibilityService
import com.automusic.player.input.TouchInjector
import com.automusic.player.PlaybackService
import com.automusic.player.ui.theme.Brand
import com.automusic.player.ui.theme.Ink2
import com.automusic.player.ui.theme.Ink3
import com.automusic.player.ui.theme.StateError
import com.automusic.player.ui.theme.StateInfo
import com.automusic.player.ui.theme.StateSuccess
import com.automusic.player.ui.theme.StateWarning
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** 演奏控制:选谱 -> 选布局 -> 调 BPM -> 倒计时切窗 -> 触摸注入演奏。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlayScreen(container: AppContainer) {
    val context = LocalContext.current
    val player = container.player

    val scores by container.db.scoreDao().observeAll().collectAsState(initial = emptyList())
    val layoutState by container.layouts.state.collectAsState()
    val playState by player.state.collectAsState()

    var a11yReady by remember { mutableStateOf(AmpAccessibilityService.ready) }

    LaunchedEffect(Unit) {
        while (true) {
            a11yReady = AmpAccessibilityService.ready
            delay(1500)
        }
    }

    var selectedScoreId by remember { mutableStateOf(-1L) }
    var bpm by remember { mutableStateOf(100f) }
    var holdRatio by remember { mutableStateOf(0.75f) }
    var gapMs by remember { mutableStateOf(20f) }
    var humanizeOn by remember { mutableStateOf(true) }

    var countdown by remember { mutableStateOf<Int?>(null) }
    var notice by remember { mutableStateOf<String?>(null) }
    var scoreMenuOpen by remember { mutableStateOf(false) }
    var layoutMenuOpen by remember { mutableStateOf(false) }

    // 库页点过来的联动
    LaunchedEffect(Unit) {
        val linked = container.selectedScoreId
        if (linked > 0) selectedScoreId = linked
    }

    // 演奏结束 -> 停前台服务
    LaunchedEffect(playState) {
        if (playState is PlayerEngine.State.Finished) {
            PlaybackService.stop(context)
        }
    }

    val selectedScore = scores.firstOrNull { it.id == selectedScoreId }
    val activeLayout = layoutState.active

    fun startPlay() {
        notice = null
        container.appScope.launch {
            if (!TouchInjector.accessibilityReady) {
                notice = "请先开启「触摸注入」无障碍服务(见上方卡片)"
                return@launch
            }
            val score = container.db.scoreDao().getById(selectedScoreId)
            if (score == null) {
                notice = "请先在下方选择要演奏的乐谱"
                return@launch
            }
            val layout = container.layouts.state.value.active
            if (layout == null) {
                notice = "没有可用布局:请到「标定」页创建"
                return@launch
            }
            // 暂停态下"开始"= 继续演奏;否则从头开始
            val startIndex = (playState as? PlayerEngine.State.Paused)?.done ?: 0
            // 3 秒倒计时,期间切到游戏窗口
            for (i in 3 downTo 1) {
                countdown = i
                delay(1000)
            }
            countdown = null
            val notes = NoteCodec.decode(score.notesJson)
            if (notes.isEmpty()) {
                notice = "乐谱为空:请回识别页重新保存"
                return@launch
            }
            val (w, h) = ScreenMetrics.realSize(context)
            PlaybackService.start(context)
            player.play(
                notes = notes,
                bpm = bpm.toInt(),
                holdRatio = holdRatio.toDouble(),
                gapMs = gapMs.toLong(),
                layout = layout.points,
                screenW = w,
                screenH = h,
                startIndex = startIndex,
                humanize = if (humanizeOn) HumanizeParams() else null,
            )
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("演奏控制", style = MaterialTheme.typography.titleLarge)

        // ---- 触摸注入(无障碍) ----
        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier
                            .size(10.dp)
                            .clip(CircleShape)
                            .background(if (a11yReady) StateSuccess else StateError)
                    )
                    Spacer(Modifier.size(8.dp))
                    Text(
                        "触摸注入 · ${if (a11yReady) "就绪" else "未开启"}",
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Spacer(Modifier.weight(1f))
                    if (!a11yReady) {
                        Button(onClick = {
                            // 直达本服务详情页(字符串 action,兼容多数 ROM),失败回退总列表
                            runCatching {
                                val intent = Intent("android.settings.ACCESSIBILITY_DETAILS").apply {
                                    putExtra(
                                        Intent.EXTRA_COMPONENT_NAME,
                                        android.content.ComponentName(context, AmpAccessibilityService::class.java),
                                    )
                                }
                                context.startActivity(intent)
                            }.onFailure {
                                context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                            }
                        }) { Text("去开启") }
                    }
                }
                Text(
                    "通过系统无障碍手势接口注入触摸事件:免 root、免 adb、无需任何外部工具。" +
                        "多指同手势派发实现和弦;请保持本服务在系统无障碍设置中处于开启状态。",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        // ---- 乐谱与布局选择 ----
        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                ExposedDropdownMenuBox(
                    expanded = scoreMenuOpen,
                    onExpandedChange = { scoreMenuOpen = it },
                ) {
                    OutlinedTextField(
                        value = selectedScore?.name ?: "未选择乐谱",
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("乐谱") },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = scoreMenuOpen) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor(),
                    )
                    ExposedDropdownMenu(
                        expanded = scoreMenuOpen,
                        onDismissRequest = { scoreMenuOpen = false },
                    ) {
                        if (scores.isEmpty()) {
                            DropdownMenuItem(text = { Text("乐谱库为空") }, onClick = {})
                        }
                        scores.forEach { s ->
                            DropdownMenuItem(
                                text = { Text(s.name) },
                                onClick = {
                                    selectedScoreId = s.id
                                    bpm = s.bpmDefault.toFloat()
                                    scoreMenuOpen = false
                                },
                            )
                        }
                    }
                }

                ExposedDropdownMenuBox(
                    expanded = layoutMenuOpen,
                    onExpandedChange = { layoutMenuOpen = it },
                ) {
                    OutlinedTextField(
                        value = activeLayout?.name ?: "未选择布局",
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("琴键布局") },
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = layoutMenuOpen) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor(),
                    )
                    ExposedDropdownMenu(
                        expanded = layoutMenuOpen,
                        onDismissRequest = { layoutMenuOpen = false },
                    ) {
                        layoutState.layouts.forEach { l ->
                            DropdownMenuItem(
                                text = { Text(l.name) },
                                onClick = {
                                    container.layouts.setActive(l.id)
                                    layoutMenuOpen = false
                                },
                            )
                        }
                    }
                }
            }
        }

        // ---- 节奏参数 ----
        Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
            Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("节奏参数", color = Brand, style = MaterialTheme.typography.titleSmall)
                LabeledSlider("BPM:${bpm.toInt()}", bpm, 40f..240f) { bpm = it }
                if (humanizeOn) {
                    Text(
                        "真人化已开启:长短音动态按住,按住比例由节奏自动分配",
                        color = Ink2,
                        style = MaterialTheme.typography.bodySmall,
                    )
                } else {
                    LabeledSlider("按住比例:%.2f".format(holdRatio), holdRatio, 0.5f..0.95f) { holdRatio = it }
                }
                LabeledSlider("间隔:${gapMs.toInt()} ms", gapMs, 0f..80f) { gapMs = it }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("真人化演奏", color = Ink2, style = MaterialTheme.typography.bodySmall)
                        Text(
                            "节奏微偏移 · 乐句呼吸 · 动态按住,更贴近真人",
                            color = Ink3,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    Switch(checked = humanizeOn, onCheckedChange = { humanizeOn = it })
                }
            }
        }

        // ---- 播放/停止/重置 ----
        val playing = playState is PlayerEngine.State.Playing
        val pausedState = playState as? PlayerEngine.State.Paused
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(
                enabled = !playing && countdown == null,
                onClick = { startPlay() },
                colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = com.automusic.player.ui.theme.Bg),
                modifier = Modifier.weight(1f),
            ) { Text(if (pausedState != null && pausedState.done > 0) "继续演奏" else "开始演奏") }
            OutlinedButton(
                enabled = playing,
                onClick = {
                    // 停止 = 暂停:进度保留,可「继续演奏」或「重置」
                    player.stop()
                    countdown = null
                    PlaybackService.stop(context)
                },
                modifier = Modifier.weight(1f),
            ) { Text("停止", color = StateError) }
            OutlinedButton(
                enabled = pausedState != null,
                onClick = {
                    // 重置:仅停止(暂停)后可用,清空演奏进度
                    player.reset()
                    countdown = null
                    PlaybackService.stop(context)
                },
                modifier = Modifier.weight(1f),
            ) { Text("重置") }
        }

        if (playing) {
            val st = playState as PlayerEngine.State.Playing
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                LinearProgressIndicator(
                    progress = { if (st.total == 0) 0f else st.done.toFloat() / st.total },
                    color = Brand,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(4.dp)),
                )
                Text(
                    "${st.done} / ${st.total}",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        if (pausedState != null) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                LinearProgressIndicator(
                    progress = { if (pausedState.total == 0) 0f else pausedState.done.toFloat() / pausedState.total },
                    color = com.automusic.player.ui.theme.StateWarning,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(4.dp)),
                )
                Text(
                    "已暂停:${pausedState.done} / ${pausedState.total} · 「继续演奏」或「重置」",
                    color = com.automusic.player.ui.theme.StateWarning,
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        if (playState is PlayerEngine.State.Finished) {
            val st = playState as PlayerEngine.State.Finished
            Text(
                when {
                    st.error != null -> "演奏出错:${st.error}"
                    st.complete -> "演奏完成"
                    else -> "已停止"
                },
                color = if (st.error != null) StateError else Ink2,
                style = MaterialTheme.typography.bodySmall,
            )
        }
        if (notice != null) {
            Text(notice!!, color = StateWarning, style = MaterialTheme.typography.bodySmall)
        }

        // ---- 测试点击 ----
        OutlinedButton(onClick = {
            container.appScope.launch {
                if (!TouchInjector.accessibilityReady) {
                    notice = "请先开启「触摸注入」无障碍服务"
                    return@launch
                }
                try {
                    val layout = container.layouts.state.value.active ?: return@launch
                    val p = layout.points["mid_4"] ?: return@launch
                    val (w, h) = ScreenMetrics.realSize(context)
                    // 3 秒倒计时:给用户时间切到游戏,手势按坐标路由到前台游戏窗口
                    for (i in 3 downTo 1) {
                        countdown = i
                        delay(1000)
                    }
                    countdown = null
                    TouchInjector.tap(p.first * w, p.second * h)
                    notice = "已向「中音 4」位置发送一次测试点击"
                } catch (e: Exception) {
                    countdown = null
                    notice = "测试失败:${e.message}"
                }
            }
        }) { Text("发送测试点击(中音 4)") }
    }

    // ---- 倒计时覆盖层 ----
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
}

@Composable
private fun LabeledSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit) {
    Column {
        Text(label, color = Ink2, style = MaterialTheme.typography.bodySmall)
        Slider(value = value, onValueChange = onChange, valueRange = range)
    }
}
