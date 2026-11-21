package com.automusic.player.core

data class WoyanishaProjection(
    val notes: List<NoteEvent>,
    val raisedNotes: Int,
    val naturalizedElements: Int,
    val mergedNotes: Int,
)

object WoyanishaScore {
    private val LOW_TO_MID = (1..7).associate { "low_$it" to "mid_$it" }

    fun project(notes: List<NoteEvent>): WoyanishaProjection {
        var raisedNotes = 0
        var naturalizedElements = 0
        var mergedNotes = 0
        val projected = notes.map { element ->
            if (element.semitone) naturalizedElements++
            val mapped = element.notes.map { noteId ->
                val raised = LOW_TO_MID[noteId]
                if (raised != null) {
                    raisedNotes++
                    raised
                } else {
                    noteId
                }
            }
            val distinct = mapped.distinct()
            mergedNotes += mapped.size - distinct.size
            element.copy(notes = distinct, semitone = false)
        }
        return WoyanishaProjection(projected, raisedNotes, naturalizedElements, mergedNotes)
    }
}