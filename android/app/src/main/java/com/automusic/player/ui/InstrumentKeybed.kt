package com.automusic.player.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.automusic.player.core.KeyPointMap

private val rowTitles = listOf("高音", "中音", "低音")
private val computerKeyRows = listOf("QWERTYU", "ASDFGHJ", "ZXCVBNM")

@Composable
fun InstrumentKeybed(
    gameType: String,
    points: Map<String, Pair<Float, Float>>,
    currentNote: String,
    showKeyLabels: Boolean,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val rows = KeyPointMap.notesFor(gameType).chunked(7)
    val outerShape = RoundedCornerShape(20.dp)

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(outerShape)
            .background(
                Brush.verticalGradient(
                    listOf(Color(0xFF20384B), Color(0xFF101D2A)),
                ),
            )
            .border(1.dp, Color(0xFF496276), outerShape)
            .padding(horizontal = 10.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        rows.forEachIndexed { rowIndex, rowNotes ->
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Text(
                    text = rowTitles[rowIndex],
                    modifier = Modifier.width(30.dp),
                    color = Color(0xFFC4D5E1),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                    textAlign = TextAlign.Center,
                )
                Row(
                    modifier = Modifier.weight(1f),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    rowNotes.forEachIndexed { keyIndex, noteId ->
                        InstrumentKey(
                            noteId = noteId,
                            computerKey = computerKeyRows[rowIndex][keyIndex].toString(),
                            isCurrent = noteId == currentNote,
                            isCalibrated = points.containsKey(noteId),
                            showKeyLabel = showKeyLabels,
                            onSelect = onSelect,
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
            }
        }
        Text(
            text = "金色边框为当前槽位  ·  ● 已标定 / ○ 未标定",
            color = Color(0xFF9EB4C4),
            fontSize = 10.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun InstrumentKey(
    noteId: String,
    computerKey: String,
    isCurrent: Boolean,
    isCalibrated: Boolean,
    showKeyLabel: Boolean,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val shape = RoundedCornerShape(12.dp)
    val background = when {
        isCurrent -> Brush.verticalGradient(listOf(Color(0xFF725B2B), Color(0xFF3D301C)))
        isCalibrated -> Brush.verticalGradient(listOf(Color(0xFF245B54), Color(0xFF173A38)))
        else -> Brush.verticalGradient(listOf(Color(0xFF2C465A), Color(0xFF182B3A)))
    }
    val borderColor = when {
        isCurrent -> Color(0xFFFFD77A)
        isCalibrated -> Color(0xFF68D7BD)
        else -> Color(0xFF587387)
    }
    val status = when {
        isCurrent && isCalibrated -> "● 当前"
        isCurrent -> "○ 当前"
        isCalibrated -> "● 已标"
        else -> "○ 未标"
    }

    Box(
        modifier = modifier
            .aspectRatio(0.72f)
            .clip(shape)
            .background(background)
            .border(if (isCurrent) 2.dp else 1.dp, borderColor, shape)
            .clickable(onClickLabel = "选择标定槽位") { onSelect(noteId) }
            .padding(2.dp),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                text = noteId.substringAfter('_'),
                color = if (isCurrent) Color(0xFFFFE7AC) else Color(0xFFF2F7FA),
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
            )
            if (showKeyLabel) {
                Text(
                    text = computerKey,
                    color = if (isCurrent) Color(0xFFFFD77A) else Color(0xFFAFC4D2),
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                )
            }
            Text(
                text = status,
                color = when {
                    isCurrent -> Color(0xFFFFD77A)
                    isCalibrated -> Color(0xFF8DE3CD)
                    else -> Color(0xFF9AAEBC)
                },
                fontSize = 8.sp,
                maxLines = 1,
                softWrap = false,
            )
        }
    }
}