package com.chisnova.mercury

import android.content.Context
import android.content.res.ColorStateList
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import com.chisnova.mercury.data.LearningStats
import com.chisnova.mercury.database.DatabaseHelper
import kotlin.math.roundToInt

/** 주제 카드의 진도를 연결하고 학습 화면에서 돌아올 때 한 번에 갱신한다. */
class LessonProgressBinder(private val context: Context, private val tintRes: Int) {
    private data class Binding(
        val category: String,
        val fallbackTotal: Int,
        val container: LinearLayout,
        val label: TextView,
        val bar: ProgressBar
    )

    private val bindings = mutableListOf<Binding>()
    private val totals = mutableMapOf<String, Int>()
    private fun dp(value: Int) = (value * context.resources.displayMetrics.density).roundToInt()

    fun addTo(parent: LinearLayout, assetBase: String) {
        val total = totals.getOrPut(assetBase) {
            try {
                context.assets.open("$assetBase.eng").bufferedReader().useLines { lines ->
                    lines.count { it.isNotBlank() }
                }
            } catch (_: java.io.IOException) { 0 }
        }
        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(7) }
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
        }
        val label = TextView(context).apply {
            textSize = 11f
            setTextColor(context.getColor(R.color.text_secondary))
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        val bar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progressDrawable = context.getDrawable(R.drawable.progress_level)?.mutate()
            progressTintList = ColorStateList.valueOf(context.getColor(tintRes))
            layoutParams = LinearLayout.LayoutParams(-1, dp(5)).apply { topMargin = dp(5) }
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        container.addView(label)
        container.addView(bar)
        parent.addView(container)
        bindings.add(Binding(assetBase.substringAfterLast('/'), total, container, label, bar))
    }

    fun refresh() {
        val stats = DatabaseHelper(context).use { LearningStats(it) }
        bindings.forEach { binding ->
            val progress = stats.topicProgress(binding.category, binding.fallbackTotal)
            val percent = (progress.ratio * 100).roundToInt()
            binding.label.text = "${progress.done}/${progress.total}문장 · $percent%"
            binding.bar.progress = percent
            binding.container.contentDescription =
                "학습 진행률, 전체 ${progress.total}문장 중 ${progress.done}문장 연습, ${percent}퍼센트"
        }
    }
}
