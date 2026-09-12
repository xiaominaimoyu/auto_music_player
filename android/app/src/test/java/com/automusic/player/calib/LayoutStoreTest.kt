package com.automusic.player.calib

import com.automusic.player.core.KeyPointMap
import com.automusic.player.core.delta.DeltaKeyPoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LayoutStoreTest {

    @Test
    fun parseLayoutJson_legacyDecodesAsWuthering() {
        val json = """{"type":"amp-key-layout","version":1,"id":"test","name":"测试","points":{"mid_4":[0.5,0.7],"high_1":[0.3,0.6]}}"""
        val outcome = LayoutStore.parseLayoutJson(json, emptyList())
        assertEquals(KeyPointMap.GAME_WUTHERING, outcome.layout.gameType)
    }

    @Test
    fun parseLayoutJson_deltaDecodesAsDelta() {
        val json = """{"type":"amp-key-layout","version":1,"id":"delta_test","name":"三角洲测试","gameType":"delta_mobile","points":{"note_1":[0.15,0.62],"note_4":[0.45,0.62],"mod_lower":[0.30,0.78]}}"""
        val outcome = LayoutStore.parseLayoutJson(json, emptyList())
        assertEquals(DeltaKeyPoint.GAME_DELTA, outcome.layout.gameType)
        assertTrue(outcome.layout.points.containsKey("note_1"))
        assertTrue(outcome.layout.points.containsKey("note_4"))
        assertTrue(outcome.layout.points.containsKey("mod_lower"))
    }

    @Test
    fun parseLayoutJson_deltaRejectsLegacyKeys() {
        val json = """{"type":"amp-key-layout","version":1,"id":"delta_test","name":"三角洲","gameType":"delta_mobile","points":{"mid_4":[0.5,0.7],"note_1":[0.15,0.62]}}"""
        val outcome = LayoutStore.parseLayoutJson(json, emptyList())
        assertTrue(outcome.layout.points.containsKey("note_1"))
        assertTrue(!outcome.layout.points.containsKey("mid_4"))
        assertTrue(outcome.warnings.isNotEmpty())
    }

    @Test
    fun parseLayoutJson_legacyRejectsDeltaKeys() {
        val json = """{"type":"amp-key-layout","version":1,"id":"test","name":"测试","points":{"note_1":[0.15,0.62],"mid_4":[0.5,0.7]}}"""
        val outcome = LayoutStore.parseLayoutJson(json, emptyList())
        assertTrue(outcome.layout.points.containsKey("mid_4"))
        assertTrue(!outcome.layout.points.containsKey("note_1"))
    }

    @Test
    fun deltaKeyPoints_sizeIs11() {
        assertEquals(11, DeltaKeyPoint.KEY_POINTS.size)
    }

    @Test
    fun deltaAndLegacyKeyNames_noIntersection() {
        val deltaKeys = DeltaKeyPoint.KEY_POINTS.toSet()
        val legacyKeys = KeyPointMap.ALL_NOTES.toSet()
        assertTrue(deltaKeys.intersect(legacyKeys).isEmpty())
    }

    @Test
    fun deltaDefaultLayout_allCoordsInRange() {
        val layout = DeltaKeyPoint.defaultLayout()
        assertEquals(11, layout.size)
        for ((_, coord) in layout) {
            assertTrue("x out of range: ${coord.first}", coord.first in 0f..1f)
            assertTrue("y out of range: ${coord.second}", coord.second in 0f..1f)
        }
    }

    @Test
    fun deltaBuildLayout_correctlyMaps() {
        val points = DeltaKeyPoint.defaultLayout()
        val layout = DeltaKeyPoint.buildLayout(points)
        assertEquals(7, layout.noteCoords.size)
        assertEquals(3, layout.modifierCoords.size)
        assertTrue(layout.submitCoord != null)
    }
}