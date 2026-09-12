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
import com.automusic.player.core.delta.DeltaCompileParams
import com.automusic.player.core.delta.DeltaKeyPoint
import com.automusic.player.core.delta.DeltaLayout
import com.automusic.player.core.delta.DeltaPlayerEngine
import com.automusic.player.core.delta.FreePlayScenario
import com.automusic.player.core.delta.NpcQuestScenario
import com.automusic.player.core.delta.Scenario
import com.automusic.player.core.delta.SessionManager
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

    val deltaPlayer = remember { DeltaPlayerEngine(container.appScope) }
    val sessionManager = remember { SessionManager(deltaPlayer, context) }

    val scores by container.db.scoreDao().observeAll().collectAsState(initial = emptyList())
    val layoutState by container.layouts.state.collectAsState()
    val playState by player.state.collectAsState()
    val deltaPlayState by deltaPlayer.state.collectAsState()

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
    var deltaScenario by remember { mutableStateOf<Scenario>(FreePlayScenario()) }
    var deltaDegradations by remember { mutableStateOf<List<com.automusic.player.core.delta.Degradation>>(emptyList()) }

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
    LaunchedEffect(deltaPlayState) {
        if (deltaPlayState is DeltaPlayerEngine.State.Finished || deltaPlayState is DeltaPlayerEngine.State.Aborted) {
            PlaybackService.stop(context)
        }
    }

    val selectedScore = scores.firstOrNull { it.id == selectedScoreId }
    val activeLayout = layoutState.active
    val isDelta = activeLayout?.gameType == DeltaKeyPoint.GAME_DELTA

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
            val notes = NoteCodec.decode(score.notesJson)
            if (notes.isEmpty()) {
                notice = "乐谱为空:请回识别页重新保存"
                return@launch
            }
            val (w, h) = ScreenMetrics.realSize(context)

            if (layout.gameType == DeltaKeyPoint.GAME_DELTA) {
                // 三角洲路径:DeltaCompiler + Scenario + DeltaPlayerEngine
                val startIndex = (deltaPlayState as? DeltaPlayerEngine.State.Paused)?.done ?: 0
                if (startIndex == 0) {
                    for (i in 3 downTo 1) { countdown = i; delay(1000) }
                    countdown = null
                }
                val deltaLayout = DeltaKeyPoint.buildLayout(layout.points)
                val params = DeltaCompileParams(
                    bpm = bpm.toInt(),
                    holdRatio = holdRatio.toDouble(),
                    gapMs = gapMs.toLong(),
                )
                val timings = if (humanizeOn && deltaScenario.humanize)
                    com.automusic.player.core.Humanize.planTimings(notes) else null
                val result = sessionManager.startSession(notes, params, deltaLayout, deltaScenario, w, h, timings)
                if (result == null) {
                    notice = "演奏启动失败:请检查无障碍服务"
                    return@launch
                }
                deltaDegradations = result.degradations
            } else {
                // 鸣潮/原神路径:PlayerEngine(原有逻辑零改动)
                val startIndex = (playState as? PlayerEngine.State.Paused)?.done ?: 0
                for (i in 3 downTo 1) { countdown = i; delay(1000) }
                countdown = null
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

        // ---- 三角洲专属:场景选择 + 口琴就位提示 ----
        if (isDelta) {
            Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("三角洲口琴", color = Brand, style = MaterialTheme.typography.titleSmall)
                    Text("场景", color = Ink2, style = MaterialTheme.typography.bodySmall)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        androidx.compose.material3.FilterChip(
                            selected = deltaScenario is FreePlayScenario,
                            onClick = { deltaScenario = FreePlayScenario() },
                            label = { Text("自由演奏") },
                        )
                        androidx.compose.material3.FilterChip(
                            selected = deltaScenario is NpcQuestScenario,
                            onClick = { deltaScenario = NpcQuestScenario() },
                            label = { Text("NPC 任务") },
                        )
                    }
                    Text(
                        "请先手动把口琴拿在手里,再点开始演奏",
                        color = StateWarning,
                        style = MaterialTheme.typography.bodySmall,
                    )
                    if (deltaScenario is NpcQuestScenario) {
                        Text(
                            "NPC 任务模式:中断后不可续播,需重新听 NPC 示范;演奏完毕自动提交",
                            color = Ink3,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
            // 降级清单展示
            if (deltaDegradations.isNotEmpty()) {
                Card(colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1)) {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("降级清单(${deltaDegradations.size} 处)", color = StateWarning, style = MaterialTheme.typography.titleSmall)
                        deltaDegradations.take(10).forEach { d ->
                            Text(
                                "#${d.index}: ${d.requested} → ${d.actual}(${d.reason})",
                                color = Ink2,
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        if (deltaDegradations.size > 10) {
                            Text("...共 ${deltaDegradations.size} 处,仅显示前 10 条", color = Ink3, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }

        // ---- 播放/停止/重置 ----
        val playing = if (isDelta) deltaPlayState is DeltaPlayerEngine.State.Playing
            else playState is PlayerEngine.State.Playing
        val deltaPaused = deltaPlayState as? DeltaPlayerEngine.State.Paused
        val legacyPaused = playState as? PlayerEngine.State.Paused
        val pausedDone = if (isDelta) deltaPaused?.done ?: 0 else legacyPaused?.done ?: 0
        val deltaAborted = deltaPlayState as? DeltaPlayerEngine.State.Aborted
        val canResume = if (isDelta) deltaPaused != null && deltaAborted == null
            else legacyPaused != null
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(
                enabled = !playing && countdown == null && (deltaAborted == null || !isDelta),
                onClick = { startPlay() },
                colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = com.automusic.player.ui.theme.Bg),
                modifier = Modifier.weight(1f),
            ) { Text(if (canResume && pausedDone > 0) "继续演奏" else "开始演奏") }
            OutlinedButton(
                enabled = playing,
                onClick = {
                    if (isDelta) sessionManager.endSession()
                    else { player.stop(); PlaybackService.stop(context) }
                    countdown = null
                },
                modifier = Modifier.weight(1f),
            ) { Text("停止", color = StateError) }
            OutlinedButton(
                enabled = if (isDelta) deltaPaused != null || deltaAborted != null
                    else legacyPaused != null,
                onClick = {
                    if (isDelta) sessionManager.resetSession()
                    else { player.reset(); PlaybackService.stop(context) }
                    countdown = null
                    deltaDegradations = emptyList()
                },
                modifier = Modifier.weight(1f),
            ) { Text("重置") }
        }

        if (playing) {
            val done = if (isDelta) (deltaPlayState as DeltaPlayerEngine.State.Playing).done
                else (playState as PlayerEngine.State.Playing).done
            val total = if (isDelta) (deltaPlayState as DeltaPlayerEngine.State.Playing).total
                else (playState as PlayerEngine.State.Playing).total
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                LinearProgressIndicator(
                    progress = { if (total == 0) 0f else done.toFloat() / total },
                    color = Brand,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(4.dp)),
                )
                Text(
                    "$done / $total",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        if (isDelta && deltaPaused != null) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                LinearProgressIndicator(
                    progress = { if (deltaPaused.total == 0) 0f else deltaPaused.done.toFloat() / deltaPaused.total },
                    color = com.automusic.player.ui.theme.StateWarning,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(4.dp)),
                )
                Text(
                    "已暂停:${deltaPaused.done} / ${deltaPaused.total} · 「继续演奏」或「重置」",
                    color = com.automusic.player.ui.theme.StateWarning,
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        if (!isDelta && legacyPaused != null) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                LinearProgressIndicator(
                    progress = { if (legacyPaused.total == 0) 0f else legacyPaused.done.toFloat() / legacyPaused.total },
                    color = com.automusic.player.ui.theme.StateWarning,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(4.dp)),
                )
                Text(
                    "已暂停:${legacyPaused.done} / ${legacyPaused.total} · 「继续演奏」或「重置」",
                    color = com.automusic.player.ui.theme.StateWarning,
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        if (isDelta && deltaAborted != null) {
            Text(
                deltaAborted.reason,
                color = StateError,
                style = MaterialTheme.typography.bodySmall,
            )
        }
        if (isDelta && deltaPlayState is DeltaPlayerEngine.State.Finished) {
            val st = deltaPlayState as DeltaPlayerEngine.State.Finished
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
        if (!isDelta && playState is PlayerEngine.State.Finished) {
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
        val testKey = if (isDelta) "note_4" else "mid_4"
        val testLabel = if (isDelta) "音格 4" else "中音 4"
        OutlinedButton(onClick = {
            container.appScope.launch {
                if (!TouchInjector.accessibilityReady) {
                    notice = "请先开启「触摸注入」无障碍服务"
                    return@launch
                }
                try {
                    val layout = container.layouts.state.value.active ?: return@launch
                    val p = layout.points[testKey] ?: return@launch
                    val (w, h) = ScreenMetrics.realSize(context)
                    for (i in 3 downTo 1) { countdown = i; delay(1000) }
                    countdown = null
                    TouchInjector.tap(p.first * w, p.second * h)
                    notice = "已向「$testLabel」位置发送一次测试点击"
                } catch (e: Exception) {
                    countdown = null
                    notice = "测试失败:${e.message}"
                }
            }
        }) { Text("发送测试点击($testLabel)") }
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
