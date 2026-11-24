package com.chisnova.mercury

import android.content.res.ColorStateList
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.core.widget.ImageViewCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.RecyclerView

/**
 * 문장 목록의 불변 표현.
 *
 * @param index       0-based sentence index (sentenceScores / sentenceDetails 키와 일치)
 * @param engText     영어 문장 원문 (클릭 가능한 Span 적용 전 plain text)
 * @param korText     한국어 번역
 * @param score       최근 발음 점수 (null = 미채점)
 * @param hasAudio    내 녹음 파일 존재 여부 (recordedUids 기반)
 * @param isSelected  현재 선택된 문장 여부
 */
data class SentenceItem(
    val index: Int,
    val engText: String,
    val korText: String,
    val score: Float?,
    val hasAudio: Boolean,
    val isSelected: Boolean,
    /** 마지막으로 인식된 문장. 연습이 끝나도 지우지 않아 무엇이라 발음했는지 남는다. */
    val transcript: String = "",
    /**
     * 지금 이 문장을 녹음 중인지. 버튼 라벨이 여기서 나온다.
     *
     * 예전에는 Activity 가 findViewHolderForAdapterPosition 으로 ViewHolder 를 찾아
     * 라벨을 직접 "완료" 로 바꿨다. 화면 밖이면 홀더가 null 이라 바뀌지 않았고,
     * 스크롤로 재바인딩되면 XML 기본값으로 되돌아갔다. iOS 처럼 상태에서 파생시킨다.
     */
    val isRecording: Boolean = false,
)

class SentenceDiffCallback(
    private val old: List<SentenceItem>,
    private val new: List<SentenceItem>,
) : DiffUtil.Callback() {

    override fun getOldListSize() = old.size
    override fun getNewListSize() = new.size

    override fun areItemsTheSame(oldPos: Int, newPos: Int) =
        old[oldPos].index == new[newPos].index

    override fun areContentsTheSame(oldPos: Int, newPos: Int) =
        old[oldPos] == new[newPos]

    /**
     * areContentsTheSame이 false일 때 변경된 필드만 payload로 전달.
     * onBindViewHolder(payloads) 에서 전체 rebind 없이 변경 부분만 갱신할 수 있다.
     */
    override fun getChangePayload(oldPosition: Int, newPosition: Int): Any? {
        val oldItem = old[oldPosition]
        val newItem = new[newPosition]
        val changes = mutableSetOf<String>()
        if (oldItem.score != newItem.score) changes += "score"
        if (oldItem.hasAudio != newItem.hasAudio) changes += "audio"
        if (oldItem.isSelected != newItem.isSelected) changes += "selected"
        if (oldItem.transcript != newItem.transcript) changes += "transcript"
        if (oldItem.isRecording != newItem.isRecording) changes += "recording"
        return if (changes.isNotEmpty()) changes else null
    }
}

/**
 * 문장 목록 RecyclerView Adapter.
 *
 * 모든 상태(선택, 점수, 오디오 존재 여부)는 [SentenceItem]에 담겨 외부에서 주입된다.
 * Adapter 내부에서는 어떤 mutable 상태도 보유하지 않는다.
 *
 * 콜백:
 *   - [onItemClick]         : 카드 전체 클릭 → 선택 + 단어 정보 팝업
 *   - [onPracticeClick]     : 발음연습 버튼 클릭
 *   - [onPlayClick]         : 예시 듣기 버튼 클릭
 *   - [onPlayMyVoiceClick]  : 내 녹음 재생 버튼 클릭
 *   - [onScoreClick]        : 점수 배지 클릭 → 점수 상세 다이얼로그
 */
class SentenceAdapter(
    private val formatScore: (Float) -> String,
    private val onItemClick: (index: Int) -> Unit,
    private val onPracticeClick: (index: Int) -> Unit,
    private val onPlayClick: (index: Int) -> Unit,
    private val onPlayMyVoiceClick: (index: Int) -> Unit,
    private val onScoreClick: (index: Int) -> Unit,
    /**
     * onBindViewHolder 완료 후 호출되는 콜백.
     * Activity에서 colorSentenceByScores Span을 적용하는 데 사용한다.
     * ViewHolder가 bind된 직후 position을 전달하므로 스크롤 시 새로 나타나는
     * 항목도 Span이 누락되지 않는다.
     */
    private val onItemBound: ((position: Int, holder: SentenceViewHolder) -> Unit)? = null,
) : RecyclerView.Adapter<SentenceAdapter.SentenceViewHolder>() {

    private var items: List<SentenceItem> = emptyList()

    /**
     * DiffUtil로 최소 변경만 반영한다.
     * 전체 목록 교체 및 단일 항목 점수/선택 상태 갱신 모두 이 메서드 하나로 처리.
     */
    fun submitList(newItems: List<SentenceItem>) {
        val diff = DiffUtil.calculateDiff(SentenceDiffCallback(items, newItems))
        items = newItems
        diff.dispatchUpdatesTo(this)
    }

    override fun getItemCount() = items.size

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SentenceViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_sentence, parent, false)
        return SentenceViewHolder(view)
    }

    override fun onBindViewHolder(holder: SentenceViewHolder, position: Int) {
        holder.bind(items[position])
        onItemBound?.invoke(position, holder)
    }

    /**
     * payload가 있는 경우 변경된 필드만 부분 갱신하여 전체 rebind 비용을 줄인다.
     * payload가 비어 있으면 기본 full bind로 폴백한다.
     */
    override fun onBindViewHolder(holder: SentenceViewHolder, position: Int, payloads: MutableList<Any>) {
        if (payloads.isEmpty()) {
            onBindViewHolder(holder, position)
            return
        }
        val item = items[position]
        @Suppress("UNCHECKED_CAST")
        val changes = payloads.filterIsInstance<Set<String>>().flatten().toSet()
        if ("score" in changes) {
            holder.itemScore.text = item.score?.let { "${formatScore(it)}점" } ?: "점수"
            holder.setEnabledAlpha(holder.itemScore, item.score != null)
        }
        if ("recording" in changes || "score" in changes) {
            holder.bindPracticeLabel(item)
        }
        if ("audio" in changes) {
            holder.setEnabledAlpha(holder.btnPlayMyVoice, item.hasAudio)
        }
        if ("transcript" in changes) {
            holder.bindTranscript(item.transcript)
        }
        if ("selected" in changes) {
            holder.itemView.setBackgroundResource(
                if (item.isSelected) R.drawable.bg_sentence_card_selected
                else R.drawable.bg_sentence_card
            )
        }
        onItemBound?.invoke(position, holder)
    }

    // ──────────────────────────────────────────────────────────────
    // ViewHolder
    // ──────────────────────────────────────────────────────────────

    inner class SentenceViewHolder(view: View) : RecyclerView.ViewHolder(view) {

        val engText: TextView = view.findViewById(R.id.engText)
        val korText: TextView = view.findViewById(R.id.korText)
        val itemScore: TextView = view.findViewById(R.id.itemScore)
        val btnPractice: LinearLayout = view.findViewById(R.id.itemBtnPractice)
        val btnPracticeLabel: TextView = view.findViewById(R.id.itemBtnPracticeLabel)
        val btnPracticeIcon: ImageView = view.findViewById(R.id.itemBtnPracticeIcon)
        val btnPlay: LinearLayout = view.findViewById(R.id.itemBtnPlay)
        val btnPlayMyVoice: LinearLayout = view.findViewById(R.id.itemBtnPlayMyVoice)
        val itemStatus: TextView = view.findViewById(R.id.itemStatus)

        /**
         * 마지막 인식 결과를 버튼 아래에 남긴다.
         * 연습이 끝나도 지우지 않아 내가 뭐라고 발음했는지 계속 확인할 수 있다.
         */
        fun bindTranscript(transcript: String) {
            if (transcript.isBlank()) {
                itemHintText.visibility = View.GONE
            } else {
                // 접두사를 붙이지 않는다 — 스트리밍 중에는 엔진 콜백이 이 칸에 결과를
                // 직접 써서, 접두사가 있으면 인식이 끝나는 순간 글머리가 튄다.
                itemHintText.text = transcript
                itemHintText.visibility = View.VISIBLE
            }
        }

        val itemProgress: ProgressBar = view.findViewById(R.id.itemProgress)
        // itemHintText는 ASR 엔진 결과를 수신하는 invisible TextView — 참조만 보관
        val itemHintText: TextView = view.findViewById(R.id.itemHintText)

        fun bind(item: SentenceItem) {
            // ── 텍스트 (Span은 Activity에서 colorSentenceByScores로 덮어씀) ──
            // engText를 plain text로 세팅하면 colorSentenceByScores 미적용 항목이
            // RecyclerView recycle 후 이전 span을 유지하는 문제를 방지한다.
            if (engText.text.toString() != item.engText) {
                engText.text = item.engText
            }
            korText.text = item.korText

            // ── 점수 배지 ──
            itemScore.text = item.score?.let { "${formatScore(it)}점" } ?: "점수"
            setEnabledAlpha(itemScore, item.score != null)
            bindTranscript(item.transcript)
            bindPracticeLabel(item)

            // ── 내 녹음 버튼 활성화 ──
            setEnabledAlpha(btnPlayMyVoice, item.hasAudio)

            // ── 선택 하이라이트 ──
            itemView.setBackgroundResource(
                if (item.isSelected) R.drawable.bg_sentence_card_selected
                else R.drawable.bg_sentence_card
            )

            // ── 클릭 리스너 (adapterPosition 대신 item.index 사용 — notifyItem 중 안정적) ──
            itemView.setOnClickListener { onItemClick(item.index) }
            btnPractice.setOnClickListener { onPracticeClick(item.index) }
            btnPlay.setOnClickListener { onPlayClick(item.index) }
            btnPlayMyVoice.setOnClickListener { onPlayMyVoiceClick(item.index) }
            itemScore.setOnClickListener { onScoreClick(item.index) }
        }

        /**
         * 연습 버튼의 겉모습을 상태에서 만든다.
         *
         * iOS SentenceListView 의 actionButton 과 같은 조합을 쓴다 — 녹음 중에는
         * 정지 아이콘에 빨간 배경, 그렇지 않으면 마이크 아이콘에 옅은 파란 배경.
         */
        fun bindPracticeLabel(item: SentenceItem) {
            val ctx = itemView.context
            btnPracticeLabel.text = when {
                item.isRecording -> "완료"
                item.score == null -> "발음연습"
                else -> "연습하기"
            }
            btnPracticeIcon.setImageResource(
                if (item.isRecording) R.drawable.ic_stop_24 else R.drawable.ic_mic_24
            )
            btnPracticeIcon.setBackgroundResource(
                if (item.isRecording) R.drawable.bg_mic_button_large_active
                else R.drawable.bg_mic_button_large
            )
            val iconTint = ContextCompat.getColor(
                ctx, if (item.isRecording) android.R.color.white else R.color.ds_primary
            )
            ImageViewCompat.setImageTintList(btnPracticeIcon, ColorStateList.valueOf(iconTint))
            btnPracticeLabel.setTextColor(
                ContextCompat.getColor(
                    ctx, if (item.isRecording) R.color.ds_danger else R.color.ds_primary
                )
            )
        }

        fun setEnabledAlpha(view: View, enabled: Boolean) {
            view.isEnabled = enabled
            view.isClickable = enabled
            view.alpha = if (enabled) 1.0f else 0.5f
        }
    }
}
