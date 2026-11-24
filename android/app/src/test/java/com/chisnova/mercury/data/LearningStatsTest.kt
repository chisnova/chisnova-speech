package com.chisnova.mercury.data

import org.junit.Assert.*
import org.junit.Test

class LearningStatsTest {
    @Test fun topicProgressDeduplicatesAttemptsAndKeepsTopicsSeparate() {
        val stats = LearningStats(
            listOf("2026-09-14 10:00:00" to 1, "2026-09-14 10:01:00" to 1,
                "2026-09-14 10:02:00" to 3, "2026-09-14 10:03:00" to 999),
            mapOf(1 to "art", 2 to "art", 3 to "city", 4 to "festival")
        )
        val art = stats.topicProgress("art")
        assertEquals(1, art.done)
        assertEquals(2, art.total)
        assertEquals(0.5f, art.ratio, 0f)
        assertFalse(art.isComplete)
        assertTrue(stats.topicProgress("city").isComplete)
        assertEquals(0, stats.topicProgress("festival").done)
    }

    @Test fun emptyAndAssetOnlyTopicsHaveZeroProgress() {
        val stats = LearningStats(emptyList(), emptyMap())
        val empty = stats.topicProgress("missing")
        assertEquals(0f, empty.ratio, 0f)
        assertFalse(empty.isComplete)
        val assetOnly = stats.topicProgress("speech", fallbackTotal = 20)
        assertEquals(20, assetOnly.total)
        assertEquals(0, assetOnly.done)
    }

    @Test fun resetAndRestoreProgressDoNotChangePracticeHistory() {
        val date = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US)
            .format(java.util.Date())
        val oldPractice = listOf(date to 1, date to 2)
        val otherTopic = listOf(date to 3)
        val categories = mapOf(1 to "art", 2 to "art", 3 to "city")
        val history = oldPractice + otherTopic
        val reset = LearningStats(history, categories, progressAttempts = otherTopic)
        assertEquals(0f, reset.topicProgress("art").ratio, 0f)
        assertEquals(1f, reset.topicProgress("city").ratio, 0f)
        assertEquals(history.size, reset.weekTotal())

        val newPractice = listOf(date to 1)
        val restarted = LearningStats(history + newPractice, categories, otherTopic + newPractice)
        assertEquals(0.5f, restarted.topicProgress("art").ratio, 0f)
        assertEquals(history.size + 1, restarted.weekTotal())

        val restored = LearningStats(history + newPractice, categories)
        assertEquals(1f, restored.topicProgress("art").ratio, 0f)
        assertEquals(restarted.weekTotal(), restored.weekTotal())
    }
}
