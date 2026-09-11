package com.automusic.player.ui

import android.content.Intent
import android.provider.Settings
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.ExpandLess
import androidx.compose.material.icons.outlined.ExpandMore
import androidx.compose.material.icons.outlined.MusicNote
import androidx.compose.material.icons.outlined.Pause
import androidx.compose.material.icons.outlined.PlayArrow
import androidx.compose.material.icons.outlined.Replay
import androidx.compose.material.icons.outlined.Stop
import androidx.compose.material.icons.outlined.TouchApp
import androidx.compose.material.icons.outlined.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.automusic.player.AppContainer
import com.automusic.player.PlaybackService
import com.automusic.player.core.HumanizeParams
import com.automusic.player.core.NoteCodec
import com.automusic.player.core.NoteEvent
import com.automusic.player.core.PlayerEngine
import com.automusic.player.core.ScorePreviewPlayer
import com.automusic.player.core.ScreenMetrics
import com.automusic.player.input.AmpAccessibilityService
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
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** 演奏控制:选谱 -> 选布局 -> 调 BPM -> 倒计时切窗 -> 触摸注入演奏。 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlayScreen(container: AppContainer) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val player = container.player
    val scores by container.db.scoreDao().observeAll().collectAsState(initial = emptyList())
    val layoutState by container.layouts.state.collectAsState()
    val playState by player.state.collectAsState()

    val playing = playState is PlayerEngine.State.Playing
    val pausedState = playState as? PlayerEngine.State.Paused
    var a11yReady by remember { mutableStateOf(AmpAccessibilityService.ready) }
    var selectedScoreId by remember { mutableStateOf(-1L) }
    var bpm by remember { mutableStateOf(100f) }
    var holdRatio by remember { mutableStateOf(0.75f) }
    var gapMs by remember { mutableStateOf(20f) }
    var humanizeOn by remember { mutableStateOf(true) }
    var countdown by remember { mutableStateOf<Int?>(null) }
    var countdownMessage by remember { mutableStateOf("切换到游戏后将开始演奏") }
    var notice by remember { mutableStateOf<String?>(null) }
    var scoreMenuOpen by remember { mutableStateOf(false) }
    var layoutMenuOpen by remember { mutableStateOf(false) }
    var advancedExpanded by remember { mutableStateOf(false) }
    var previewing by remember { mutableStateOf(false) }
    var previewIndex by remember { mutableIntStateOf(-1) }
    var previewJob by remember { mutableStateOf<Job?>(null) }
    val previewPlayer = remember { ScorePreviewPlayer() }

    fun stopPreview() {
        previewJob?.cancel()
        previewJob = null
        previewPlayer.stop()
        previewing = false
        previewIndex = -1
    }

    fun openAccessibilitySettings() {
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
    }

    DisposableEffect(Unit) {
        onDispose { stopPreview() }
    }

    LaunchedEffect(Unit) {
        while (true) {
            a11yReady = AmpAccessibilityService.ready
            delay(1_500)
        }
    }

    LaunchedEffect(scores) {
        if (scores.none { it.id == selectedScoreId }) {
            selectedScoreId = scores.firstOrNull { it.id == container.selectedScoreId }?.id
                ?: scores.firstOrNull()?.id
                ?: -1L
        }
    }

    LaunchedEffect(playState) {
        if (playState is PlayerEngine.State.Finished) PlaybackService.stop(context)
    }

    val selectedScore = scores.firstOrNull { it.id == selectedScoreId }
    val activeLayout = layoutState.active
    val previewNotes = remember(selectedScore?.notesJson) {
        selectedScore?.let { runCatching { NoteCodec.decode(it.notesJson) }.getOrDefault(emptyList()) }.orEmpty()
    }

    LaunchedEffect(selectedScore?.id) {
        selectedScore?.let { bpm = it.bpmDefault.toFloat().coerceIn(30f, 300f) }
        stopPreview()
    }

    val readyToPlay = a11yReady && selectedScore != null && previewNotes.isNotEmpty() && activeLayout != null
    val inputEnabled = !playing && countdown == null

    fun startPlay() {
        notice = null
        stopPreview()
        container.appScope.launch {
            if (!TouchInjector.accessibilityReady) {
                notice = "触摸服务尚未开启"
                return@launch
            }
            val score = container.db.scoreDao().getById(selectedScoreId)
            if (score == null) {
                notice = "请选择要演奏的乐谱"
                return@launch
            }
            val layout = container.layouts.state.value.active
            if (layout == null) {
                notice = "请选择可用的琴键布局"
                return@launch
            }
            val notes = runCatching { NoteCodec.decode(score.notesJson) }.getOrDefault(emptyList())
            if (notes.isEmpty()) {
                notice = "当前乐谱没有可演奏的音符"
                return@launch
            }
            val startIndex = (playState as? PlayerEngine.State.Paused)?.done ?: 0
            countdownMessage = "切换到游戏后将开始演奏"
            for (i in 3 downTo 1) {
                countdown = i
                delay(1_000)
            }
            countdown = null
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

    fun testCenterKey() {
        notice = null
        stopPreview()
        container.appScope.launch {
            if (!TouchInjector.accessibilityReady) {
                notice = "开启触摸服务后才能测试键位"
                return@launch
            }
            val point = container.layouts.state.value.active?.points?.get("mid_4")
            if (point == null) {
                notice = "当前布局没有标定中音 4"
                return@launch
            }
            try {
                countdownMessage = "切换到游戏后将测试中音 4"
                for (i in 3 downTo 1) {
                    countdown = i
                    delay(1_000)
                }
                countdown = null
                val (w, h) = ScreenMetrics.realSize(context)
                TouchInjector.tap(point.first * w, point.second * h)
                notice = "中音 4 测试点击已发送"
            } catch (e: Exception) {
                countdown = null
                notice = "键位测试失败:${e.message}"
            }
        }
    }

    Box(Modifier.fillMaxSize().background(Bg)) {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        ) {
            Column(
                Modifier.fillMaxWidth().background(Surface1).padding(horizontal = 18.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text("演奏", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                Text(
                    selectedScore?.name ?: "尚未选择乐谱",
                    color = Ink2,
                    style = MaterialTheme.typography.bodySmall,
                    maxLines = 1,
                )
            }

            Column(
                Modifier.padding(horizontal = 16.dp, vertical = 16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    color = (if (a11yReady) StateSuccess else StateError).copy(alpha = 0.1f),
                    border = BorderStroke(1.dp, (if (a11yReady) StateSuccess else StateError).copy(alpha = 0.35f)),
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Row(
                        Modifier.padding(horizontal = 14.dp, vertical = 11.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Box(
                            Modifier.size(9.dp).clip(CircleShape)
                                .background(if (a11yReady) StateSuccess else StateError)
                        )
                        Column(Modifier.weight(1f).padding(horizontal = 10.dp)) {
                            Text("触摸服务", color = Ink, fontWeight = FontWeight.SemiBold)
                            Text(
                                if (a11yReady) "已就绪" else "未开启，无法向游戏发送按键",
                                color = if (a11yReady) StateSuccess else StateError,
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        if (!a11yReady) {
                            TextButton(onClick = { openAccessibilitySettings() }) { Text("去开启") }
                        }
                    }
                }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = Surface1),
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        SectionTitle("演奏内容", "曲谱与键位布局")

                        ExposedDropdownMenuBox(
                            expanded = scoreMenuOpen,
                            onExpandedChange = { if (inputEnabled) scoreMenuOpen = it },
                        ) {
                            OutlinedTextField(
                                value = selectedScore?.name ?: "未选择乐谱",
                                onValueChange = {},
                                readOnly = true,
                                enabled = inputEnabled,
                                label = { Text("乐谱") },
                                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = scoreMenuOpen) },
                                modifier = Modifier.fillMaxWidth().menuAnchor(),
                            )
                            ExposedDropdownMenu(
                                expanded = scoreMenuOpen,
                                onDismissRequest = { scoreMenuOpen = false },
                            ) {
                                if (scores.isEmpty()) {
                                    DropdownMenuItem(text = { Text("乐谱库为空") }, enabled = false, onClick = {})
                                }
                                scores.forEach { score ->
                                    DropdownMenuItem(
                                        text = { Text(score.name) },
                                        onClick = {
                                            stopPreview()
                                            if (score.id != selectedScoreId && pausedState != null) player.reset()
                                            selectedScoreId = score.id
                                            scoreMenuOpen = false
                                        },
                                    )
                                }
                            }
                        }

                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            color = Surface2,
                            shape = RoundedCornerShape(8.dp),
                            border = BorderStroke(1.dp, Line),
                        ) {
                            Column(Modifier.padding(11.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Surface(color = BrandSoft, shape = RoundedCornerShape(6.dp)) {
                                        Box(Modifier.size(38.dp), contentAlignment = Alignment.Center) {
                                            Icon(Icons.Outlined.MusicNote, null, tint = Brand)
                                        }
                                    }
                                    Column(Modifier.weight(1f).padding(horizontal = 10.dp)) {
                                        Text(
                                            selectedScore?.name ?: "选择乐谱后可试听",
                                            color = Ink,
                                            fontWeight = FontWeight.SemiBold,
                                            maxLines = 1,
                                        )
                                        Text(
                                            if (previewing && previewIndex in previewNotes.indices) {
                                                "试听 ${previewIndex + 1}/${previewNotes.size} · ${playNoteLabel(previewNotes[previewIndex])}"
                                            } else {
                                                "${previewNotes.size} 个音符 · ${bpm.toInt()} BPM · 口琴音色"
                                            },
                                            color = if (previewing) Brand else Ink2,
                                            style = MaterialTheme.typography.bodySmall,
                                            maxLines = 1,
                                        )
                                    }
                                    OutlinedButton(
                                        enabled = previewNotes.isNotEmpty() && !playing && countdown == null,
                                        onClick = {
                                            if (previewing) {
                                                stopPreview()
                                            } else {
                                                previewing = true
                                                previewIndex = 0
                                                notice = null
                                                previewJob = scope.launch {
                                                    try {
                                                        previewPlayer.play(previewNotes, bpm.toInt()) { previewIndex = it }
                                                    } catch (e: Exception) {
                                                        if (e !is CancellationException) notice = "试听失败:${e.message}"
                                                    } finally {
                                                        previewing = false
                                                        previewIndex = -1
                                                        previewJob = null
                                                    }
                                                }
                                            }
                                        },
                                    ) {
                                        Icon(
                                            if (previewing) Icons.Outlined.Stop else Icons.Outlined.PlayArrow,
                                            null,
                                            modifier = Modifier.size(18.dp),
                                        )
                                        Spacer(Modifier.size(5.dp))
                                        Text(if (previewing) "停止" else "试听")
                                    }
                                }
                                if (previewing) {
                                    LinearProgressIndicator(
                                        progress = {
                                            if (previewNotes.isEmpty()) 0f
                                            else (previewIndex + 1).toFloat() / previewNotes.size
                                        },
                                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)),
                                        color = Brand,
                                        trackColor = Line,
                                    )
                                }
                            }
                        }

                        ExposedDropdownMenuBox(
                            expanded = layoutMenuOpen,
                            onExpandedChange = { if (inputEnabled) layoutMenuOpen = it },
                        ) {
                            OutlinedTextField(
                                value = activeLayout?.name ?: "未选择布局",
                                onValueChange = {},
                                readOnly = true,
                                enabled = inputEnabled,
                                label = { Text("琴键布局") },
                                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = layoutMenuOpen) },
                                modifier = Modifier.fillMaxWidth().menuAnchor(),
                            )
                            ExposedDropdownMenu(
                                expanded = layoutMenuOpen,
                                onDismissRequest = { layoutMenuOpen = false },
                            ) {
                                if (layoutState.layouts.isEmpty()) {
                                    DropdownMenuItem(text = { Text("没有可用布局") }, enabled = false, onClick = {})
                                }
                                layoutState.layouts.forEach { layout ->
                                    DropdownMenuItem(
                                        text = { Text("${layout.name} · ${layout.points.size}/21 键") },
                                        onClick = {
                                            container.layouts.setActive(layout.id)
                                            layoutMenuOpen = false
                                        },
                                    )
                                }
                            }
                        }

                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                activeLayout?.let { "已标定 ${it.points.size}/21 个键位" } ?: "需要先创建琴键布局",
                                color = if (activeLayout != null) Ink2 else StateWarning,
                                style = MaterialTheme.typography.bodySmall,
                                modifier = Modifier.weight(1f),
                            )
                            TextButton(
                                enabled = activeLayout?.points?.containsKey("mid_4") == true && inputEnabled,
                                onClick = { testCenterKey() },
                            ) {
                                Icon(Icons.Outlined.TouchApp, null, modifier = Modifier.size(17.dp))
                                Spacer(Modifier.size(5.dp))
                                Text("测试中音 4")
                            }
                        }

                        HorizontalDivider(color = Line)

                        ParameterSlider(
                            title = "演奏速度",
                            valueText = "${bpm.toInt()} BPM",
                            value = bpm,
                            range = 30f..300f,
                            enabled = inputEnabled,
                            startLabel = "30",
                            endLabel = "300",
                            onChange = { stopPreview(); bpm = it },
                        )

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text("真人化演奏", color = Ink2, style = MaterialTheme.typography.bodyMedium)
                                Text(
                                    "节奏微偏移 · 乐句呼吸 · 动态按住",
                                    color = Ink3,
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                            Switch(
                                checked = humanizeOn,
                                enabled = inputEnabled,
                                onCheckedChange = { humanizeOn = it },
                            )
                        }

                        OutlinedButton(
                            onClick = { advancedExpanded = !advancedExpanded },
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Icon(Icons.Outlined.Tune, null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.size(7.dp))
                            Text("高级触控参数")
                            Spacer(Modifier.weight(1f))
                            Text("${(holdRatio * 100).toInt()}% · ${gapMs.toInt()}ms", color = Ink2)
                            Spacer(Modifier.size(4.dp))
                            Icon(
                                if (advancedExpanded) Icons.Outlined.ExpandLess else Icons.Outlined.ExpandMore,
                                null,
                                modifier = Modifier.size(18.dp),
                            )
                        }
                        AnimatedVisibility(visible = advancedExpanded) {
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                ParameterSlider(
                                    title = "按住时长",
                                    valueText = "${(holdRatio * 100).toInt()}%",
                                    value = holdRatio,
                                    range = 0.5f..0.95f,
                                    enabled = inputEnabled,
                                    startLabel = "短",
                                    endLabel = "长",
                                    onChange = { holdRatio = it },
                                )
                                ParameterSlider(
                                    title = "音符间隔",
                                    valueText = "${gapMs.toInt()} ms",
                                    value = gapMs,
                                    range = 0f..80f,
                                    enabled = inputEnabled,
                                    startLabel = "0",
                                    endLabel = "80 ms",
                                    onChange = { gapMs = it },
                                )
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
                        val finished = playState as? PlayerEngine.State.Finished
                        val stateColor = when {
                            playing -> Brand
                            pausedState != null -> StateWarning
                            finished?.error != null -> StateError
                            readyToPlay -> StateSuccess
                            else -> Ink3
                        }
                        val stateTitle = when {
                            playing -> "正在演奏"
                            pausedState != null -> "演奏已暂停"
                            finished?.error != null -> "演奏失败"
                            finished?.complete == true -> "演奏完成"
                            readyToPlay -> "可以开始演奏"
                            else -> "演奏条件未就绪"
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.size(9.dp).clip(CircleShape).background(stateColor))
                            Text(
                                stateTitle,
                                color = stateColor,
                                fontWeight = FontWeight.SemiBold,
                                modifier = Modifier.padding(start = 9.dp).weight(1f),
                            )
                            if (playing) {
                                val state = playState as PlayerEngine.State.Playing
                                Text("${state.done}/${state.total}", color = Ink2, style = MaterialTheme.typography.labelMedium)
                            }
                        }

                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            ReadyPill("乐谱", selectedScore != null && previewNotes.isNotEmpty(), Modifier.weight(1f))
                            ReadyPill("布局", activeLayout != null, Modifier.weight(1f))
                            ReadyPill("触摸", a11yReady, Modifier.weight(1f))
                        }

                        val progressDone = when (val state = playState) {
                            is PlayerEngine.State.Playing -> state.done
                            is PlayerEngine.State.Paused -> state.done
                            else -> 0
                        }
                        val progressTotal = when (val state = playState) {
                            is PlayerEngine.State.Playing -> state.total
                            is PlayerEngine.State.Paused -> state.total
                            else -> 0
                        }
                        if (progressTotal > 0) {
                            LinearProgressIndicator(
                                progress = { progressDone.toFloat() / progressTotal },
                                color = if (pausedState != null) StateWarning else Brand,
                                trackColor = Line,
                                modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(4.dp)),
                            )
                            Text(
                                if (pausedState != null) "已暂停在第 $progressDone 个音符"
                                else "正在演奏第 ${(progressDone + 1).coerceAtMost(progressTotal)} 个音符",
                                color = Ink2,
                                style = MaterialTheme.typography.bodySmall,
                                textAlign = TextAlign.Center,
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }

                        Button(
                            enabled = readyToPlay && !playing && countdown == null,
                            onClick = { startPlay() },
                            colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                            modifier = Modifier.fillMaxWidth().height(50.dp),
                        ) {
                            Icon(
                                if (pausedState != null) Icons.Outlined.PlayArrow else Icons.Outlined.MusicNote,
                                null,
                                modifier = Modifier.size(20.dp),
                            )
                            Spacer(Modifier.size(7.dp))
                            Text(if (pausedState != null) "继续演奏" else "开始演奏", fontWeight = FontWeight.SemiBold)
                        }

                        if (playing) {
                            OutlinedButton(
                                onClick = {
                                    player.stop()
                                    countdown = null
                                    PlaybackService.stop(context)
                                },
                                modifier = Modifier.fillMaxWidth(),
                            ) {
                                Icon(Icons.Outlined.Pause, null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.size(6.dp))
                                Text("暂停并保留进度")
                            }
                        } else if (pausedState != null) {
                            OutlinedButton(
                                onClick = {
                                    player.reset()
                                    PlaybackService.stop(context)
                                },
                                modifier = Modifier.fillMaxWidth(),
                            ) {
                                Icon(Icons.Outlined.Replay, null, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.size(6.dp))
                                Text("放弃进度，从头开始")
                            }
                        }

                        if (finished?.error != null) {
                            Text(finished.error, color = StateError, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }

                notice?.let { message ->
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = (if (message.contains("失败") || message.contains("尚未") || message.startsWith("请")) {
                            StateWarning
                        } else {
                            StateSuccess
                        }).copy(alpha = 0.1f),
                        shape = RoundedCornerShape(8.dp),
                    ) {
                        Text(message, color = Ink2, modifier = Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall)
                    }
                }

                Spacer(Modifier.height(8.dp))
            }
        }

        countdown?.let { number ->
            Box(
                Modifier.fillMaxSize().background(Bg.copy(alpha = 0.9f)),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        number.toString(),
                        fontSize = 112.sp,
                        color = Brand,
                        style = MaterialTheme.typography.displayLarge,
                    )
                    Text(countdownMessage, color = Ink2)
                }
            }
        }
    }
}

@Composable
private fun SectionTitle(title: String, subtitle: String) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(title, color = Ink, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
        Text(subtitle, color = Ink3, style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun ReadyPill(label: String, ready: Boolean, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        color = (if (ready) StateSuccess else Surface2).copy(alpha = if (ready) 0.1f else 1f),
        border = BorderStroke(1.dp, if (ready) StateSuccess.copy(alpha = 0.35f) else Line),
        shape = RoundedCornerShape(6.dp),
    ) {
        Text(
            "${if (ready) "已就绪" else "未就绪"} · $label",
            color = if (ready) StateSuccess else Ink3,
            style = MaterialTheme.typography.labelSmall,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 5.dp, vertical = 7.dp),
            maxLines = 1,
        )
    }
}

@Composable
private fun ParameterSlider(
    title: String,
    valueText: String,
    value: Float,
    range: ClosedFloatingPointRange<Float>,
    enabled: Boolean,
    startLabel: String,
    endLabel: String,
    onChange: (Float) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(1.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(title, color = Ink2, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            Surface(color = Surface2, shape = RoundedCornerShape(6.dp)) {
                Text(
                    valueText,
                    color = Brand,
                    style = MaterialTheme.typography.labelMedium,
                    modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp),
                )
            }
        }
        Slider(value = value, onValueChange = onChange, valueRange = range, enabled = enabled)
        Row(Modifier.fillMaxWidth()) {
            Text(startLabel, color = Ink3, style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.weight(1f))
            Text(endLabel, color = Ink3, style = MaterialTheme.typography.labelSmall)
        }
    }
}

private fun playNoteLabel(event: NoteEvent): String {
    if (event.notes.isEmpty()) return "休止"
    return event.notes.joinToString("+") { id ->
        val parts = id.split('_')
        val prefix = when (parts.firstOrNull()) {
            "high" -> "高"
            "low" -> "低"
            else -> ""
        }
        prefix + (parts.getOrNull(1) ?: id)
    }
}
