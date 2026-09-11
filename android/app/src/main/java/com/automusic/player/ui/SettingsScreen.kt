package com.automusic.player.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.CheckCircle
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Edit
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
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.automusic.player.AppContainer
import com.automusic.player.core.recognizer.ScoreRecognizer
import com.automusic.player.core.settings.Provider
import com.automusic.player.ui.theme.Bg
import com.automusic.player.ui.theme.Brand
import com.automusic.player.ui.theme.Ink2
import com.automusic.player.ui.theme.Ink3
import com.automusic.player.ui.theme.Line
import com.automusic.player.ui.theme.StateError
import com.automusic.player.ui.theme.StateSuccess
import kotlinx.coroutines.launch

/** 模型设置:OpenAI 兼容多模态供应商管理(名称/BaseURL/Key/模型)。 */
@Composable
fun SettingsScreen(container: AppContainer) {
    val scope = rememberCoroutineScope()
    val providers by container.settings.providersFlow.collectAsState(initial = emptyList())
    val activeName by container.settings.activeNameFlow.collectAsState(initial = "")

    var editing by remember { mutableStateOf<Provider?>(null) }
    var creating by remember { mutableStateOf(false) }
    var deleting by remember { mutableStateOf<Provider?>(null) }
    var notice by remember { mutableStateOf<String?>(null) }
    var noticeOk by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("模型设置", style = MaterialTheme.typography.titleLarge)
                Text("管理乐谱识别使用的多模态模型", color = Ink2, style = MaterialTheme.typography.bodySmall)
            }
            Button(
                onClick = { creating = true },
                colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
            ) {
                Icon(Icons.Outlined.Add, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.size(6.dp))
                Text("添加")
            }
        }

        for (p in providers) {
            val active = p.name == activeName
            Card(
                colors = CardDefaults.cardColors(containerColor = com.automusic.player.ui.theme.Surface1),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(
                    Modifier.fillMaxWidth().padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
                        Column(Modifier.weight(1f)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(p.name, style = MaterialTheme.typography.titleMedium)
                                if (active) {
                                    Spacer(Modifier.size(8.dp))
                                    Surface(color = StateSuccess.copy(alpha = 0.12f), shape = RoundedCornerShape(5.dp)) {
                                        Text(
                                            "当前使用",
                                            color = StateSuccess,
                                            style = MaterialTheme.typography.labelSmall,
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                        )
                                    }
                                }
                            }
                            Text(p.model, color = Brand, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                            Text(p.baseUrl, color = Ink3, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
                        }
                        IconButton(onClick = { editing = p }) {
                            Icon(Icons.Outlined.Edit, contentDescription = "编辑", tint = Ink2)
                        }
                        IconButton(onClick = { deleting = p }) {
                            Icon(Icons.Outlined.Delete, contentDescription = "删除", tint = StateError)
                        }
                    }
                    HorizontalDivider(color = Line)
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        OutlinedButton(
                            modifier = Modifier.weight(1f),
                            onClick = {
                            notice = null
                            scope.launch {
                                try {
                                    val r = ScoreRecognizer(p.baseUrl, p.apiKey, p.model)
                                    notice = "连接成功 · ${r.testConnection()}"
                                    noticeOk = true
                                } catch (e: Exception) {
                                    notice = "连接失败：${e.message}"
                                    noticeOk = false
                                }
                            }
                            },
                        ) { Text("测试连接") }
                        if (active) {
                            Button(
                                onClick = {},
                                enabled = false,
                                modifier = Modifier.weight(1f),
                            ) {
                                Icon(Icons.Outlined.CheckCircle, contentDescription = null, modifier = Modifier.size(17.dp))
                                Spacer(Modifier.size(6.dp))
                                Text("当前模型")
                            }
                        } else {
                            Button(
                                onClick = { container.appScope.launch { container.settings.setActive(p.name) } },
                                colors = ButtonDefaults.buttonColors(containerColor = Brand, contentColor = Bg),
                                modifier = Modifier.weight(1f),
                            ) { Text("使用此模型") }
                        }
                    }
                }
            }
        }

        if (notice != null) {
            Text(notice!!, color = if (noticeOk) StateSuccess else StateError, style = MaterialTheme.typography.bodySmall)
        }
        Text(
            "API Key 保存在本机应用数据中。测试渠道额度有限，可随时编辑或替换。",
            color = Ink3,
            style = MaterialTheme.typography.bodySmall,
        )
    }

    val dialogOpen = creating || editing != null
    if (dialogOpen) {
        val initial = editing ?: Provider("", "", "", "")
        var name by remember(editing) { mutableStateOf(initial.name) }
        var baseUrl by remember(editing) { mutableStateOf(initial.baseUrl) }
        var apiKey by remember(editing) { mutableStateOf(initial.apiKey) }
        var model by remember(editing) { mutableStateOf(initial.model) }
        AlertDialog(
            onDismissRequest = { creating = false; editing = null },
            title = { Text(if (editing == null) "添加供应商" else "编辑供应商") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(value = name, onValueChange = { name = it },
                        label = { Text("名称") }, singleLine = true)
                    OutlinedTextField(value = baseUrl, onValueChange = { baseUrl = it },
                        label = { Text("Base URL(如 https://dashscope.aliyuncs.com/compatible-mode/v1)") },
                        singleLine = true)
                    OutlinedTextField(value = apiKey, onValueChange = { apiKey = it },
                        label = { Text("API Key") },
                        visualTransformation = PasswordVisualTransformation(),
                        singleLine = true)
                    OutlinedTextField(value = model, onValueChange = { model = it },
                        label = { Text("模型名称(如 qwen-vl-plus)") }, singleLine = true)
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    if (name.isNotBlank() && baseUrl.isNotBlank() && model.isNotBlank()) {
                        val p = Provider(name.trim(), baseUrl.trim(), apiKey.trim(), model.trim())
                        val previousName = editing?.name
                        val isNew = editing == null
                        scope.launch { container.settings.save(p, previousName = previousName, makeActive = isNew) }
                        creating = false
                        editing = null
                    }
                }) { Text("保存", color = Brand) }
            },
            dismissButton = {
                TextButton(onClick = { creating = false; editing = null }) { Text("取消", color = Ink2) }
            },
        )
    }

    deleting?.let { p ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("删除供应商") },
            text = { Text("确定删除「${p.name}」?") },
            confirmButton = {
                TextButton(onClick = {
                    scope.launch { container.settings.delete(p.name) }
                    deleting = null
                }) { Text("删除", color = StateError) }
            },
            dismissButton = {
                TextButton(onClick = { deleting = null }) { Text("取消", color = Ink2) }
            },
        )
    }
}
