package com.chisnova.mercury.data

import com.chisnova.mercury.database.DatabaseHelper
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/**
 * 홈과 주제 목록이 보여 주는 학습 현황.
 *
 * 기록(Attempt)만 보고 계산한다. 같은 문장을 여러 번 해도 진도는 한 번으로 센다 —
 * 몇 번 했느냐가 아니라 얼마나 훑었느냐가 진도이기 때문이다.
 * iOS 의 LearningStats 와 같은 규칙을 쓴다.
 */
class LearningStats internal constructor(
    private val attempts: List<Pair<String, Int>>,
    private val uidCategory: Map<Int, String>,
    private val progressAttempts: List<Pair<String, Int>> = attempts
) {
    constructor(dbHelper: DatabaseHelper?) : this(
        dbHelper?.queryAllAttemptDates() ?: emptyList(),
        dbHelper?.queryUidCategories() ?: emptyMap(),
        dbHelper?.queryAllAttemptDates(visibleOnly = true) ?: emptyList()
    )

    /** 레벨별 전체 문장 수. 매번 자산을 세면 홈이 느려져 미리 세어 둔 값을 쓴다. */
    companion object {
        val TOTAL = mapOf(1 to 130, 2 to 1138, 3 to 1003, 4 to 973, 5 to 972)
        private val PARSER = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

        /** 주제 폴더 이름으로 레벨(1~5)을 가린다. */
        fun levelOf(category: String): Int = when {
            BASIC_TOPICS.contains(category) -> 1
            SPEECH_TOPICS.contains(category) -> 5
            else -> ArticleLevels.tierOf(category)
        }

        private val BASIC_TOPICS = setOf(
            "greeting_basic_expression", "phone_conversation", "expressing_opinions",
            "telling_time", "ordering_food", "asking_for_directions", "asking_for_help",
            "shopping", "declining_politely", "making_appointment", "introducing_people",
            "making_suggestion", "emergency_situation", "describing_locations",
            "making_invitations"
        )
        private val SPEECH_TOPICS = setOf(
            "lincoln", "mandela", "reagan",
            "conan", "barackobama", "gergebush",
            "jfkennedy", "martinLutherkingjr"
        )
    }

    /** 레벨 -> 연습해 본 고유 문장(uid) 수 */
    private val doneByLevel: Map<Int, Int> = progressAttempts
        .mapNotNull { (_, uid) -> uidCategory[uid]?.let { levelOf(it) to uid } }
        .groupBy({ it.first }, { it.second })
        .mapValues { (_, uids) -> uids.toSet().size }

    fun doneCount(level: Int): Int = doneByLevel[level] ?: 0

    fun progress(level: Int): Float {
        val total = TOTAL[level] ?: return 0f
        if (total == 0) return 0f
        return minOf(1f, doneCount(level).toFloat() / total)
    }

    data class TopicProgress(val done: Int, val total: Int) {
        val ratio: Float get() = if (total > 0) done.toFloat() / total else 0f
        val isComplete: Boolean get() = total > 0 && done == total

    }

    private val totalByTopic = uidCategory.values.groupingBy { it }.eachCount()
    private val doneByTopic = progressAttempts.map { it.second }.toSet()
        .mapNotNull { uidCategory[it] }.groupingBy { it }.eachCount()

    /** 문장 화면과 같은 주제 폴더/UID를 사용한다. DB에 없는 콘텐츠는 자산의 문장 수를 쓴다. */
    fun topicProgress(category: String, fallbackTotal: Int = 0): TopicProgress = TopicProgress(
        done = doneByTopic[category] ?: 0,
        total = totalByTopic[category] ?: fallbackTotal.coerceAtLeast(0)
    )

    /** 가장 최근에 연습한 주제 폴더와 레벨 */
    fun lastPractice(): Pair<String, Int>? {
        val (_, uid) = attempts.firstOrNull() ?: return null
        val category = uidCategory[uid] ?: return null
        return category to levelOf(category)
    }

    private fun dayOf(text: String): Date? = try { PARSER.parse(text) } catch (e: Exception) { null }

    /** 월요일부터 일요일까지 그날 연습한 문장 수 */
    fun thisWeek(): List<Triple<String, Int, Boolean>> {
        val names = listOf("월", "화", "수", "목", "금", "토", "일")
        val cal = Calendar.getInstance()
        cal.firstDayOfWeek = Calendar.MONDAY
        val today = startOfDay(cal.time)

        cal.time = today
        while (cal.get(Calendar.DAY_OF_WEEK) != Calendar.MONDAY) {
            cal.add(Calendar.DAY_OF_MONTH, -1)
        }
        val monday = cal.time

        val counts = attempts.mapNotNull { dayOf(it.first) }.map { startOfDay(it) }
        return (0..6).map { offset ->
            val c = Calendar.getInstance().apply { time = monday; add(Calendar.DAY_OF_MONTH, offset) }
            val day = startOfDay(c.time)
            Triple(names[offset], counts.count { it == day }, day == today)
        }
    }

    fun weekTotal(): Int = thisWeek().sumOf { it.second }

    /** 오늘까지 며칠 내리 연습했는지. 오늘 아직 안 했으면 어제까지로 센다. */
    fun streakDays(): Int {
        val days = attempts.mapNotNull { dayOf(it.first) }.map { startOfDay(it) }.toSet()
        if (days.isEmpty()) return 0

        val cal = Calendar.getInstance()
        var day = startOfDay(cal.time)
        if (!days.contains(day)) {
            cal.add(Calendar.DAY_OF_MONTH, -1)
            day = startOfDay(cal.time)
            if (!days.contains(day)) return 0
        }
        var count = 0
        while (days.contains(day)) {
            count++
            cal.add(Calendar.DAY_OF_MONTH, -1)
            day = startOfDay(cal.time)
        }
        return count
    }

    private fun startOfDay(date: Date): Date {
        val c = Calendar.getInstance()
        c.time = date
        c.set(Calendar.HOUR_OF_DAY, 0); c.set(Calendar.MINUTE, 0)
        c.set(Calendar.SECOND, 0); c.set(Calendar.MILLISECOND, 0)
        return c.time
    }
}
