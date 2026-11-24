package com.chisnova.mercury.data

/**
 * 주제별 콘텐츠의 난이도 단계(2·3·4).
 *
 * 문장 길이로는 나눌 수 없었다 — 3,114문장이 거의 다 8~10단어라 편차가 없다.
 * 대신 음절/단어(어휘 밀도)가 1.69~2.48 로 뚜렷하게 갈려 그것을 기준으로 삼았다.
 * 다음절 단어가 많다는 것은 추상적·전문적 어휘라는 뜻이고 발음도 그만큼 어렵다.
 * 69개 주제를 23개씩 나눴고 문장 수도 1138 / 1003 / 973 으로 고르다.
 *
 * iOS 의 ContentCatalog.articleLevels 와 같은 값을 갖는다 — 두 앱이 같은 구성을 보여야 한다.
 * 폴더 이름과 LessonData 의 레슨 id 가 다른 주제가 있어 양쪽 열쇠를 모두 담았다.
 */
object ArticleLevels {

    private val table: Map<String, Int> = mapOf(
        "sleep" to 2,   // 음절 1.69
        "movie_casting" to 2,   // 음절 1.72
        "e-sports" to 2,   // 음절 1.73
        "instrument" to 2,   // 음절 1.73
        "k-pop" to 2,   // 음절 1.76
        "city" to 2,   // 음절 1.76
        "soccer" to 2,   // 음절 1.79
        "vitamin" to 2,   // 음절 1.84
        "work_out" to 2,   // 음절 1.84
        "festival" to 2,   // 음절 1.85
        "supply_demand" to 2,   // 음절 1.87
        "swimming" to 2,   // 음절 1.88
        "music_genere" to 2,   // 음절 1.88
        "music_genre" to 2,
        "stress" to 2,   // 음절 1.88
        "art" to 2,   // 음절 1.90
        "movie_music" to 2,   // 음절 1.90
        "museum" to 2,   // 음절 1.90
        "local_food" to 2,   // 음절 1.91
        "musical" to 2,   // 음절 1.94
        "k-webtoon" to 2,   // 음절 1.95
        "unesco" to 2,   // 음절 1.95
        "law" to 2,   // 음절 1.96
        "performance" to 2,   // 음절 1.97
        "universe" to 3,   // 음절 1.98
        "mucle" to 3,   // 음절 1.98
        "muscle" to 3,
        "ott" to 3,   // 음절 1.99
        "physics" to 3,   // 음절 2.01
        "earth" to 3,   // 음절 2.01
        "architect" to 3,   // 음절 2.01
        "architecture" to 3,
        "governments" to 3,   // 음절 2.02
        "government" to 3,
        "mental" to 3,   // 음절 2.03
        "uam" to 3,   // 음절 2.03
        "bitcoin" to 3,   // 음절 2.04
        "disease" to 3,   // 음절 2.06
        "low_birth" to 3,   // 음절 2.06
        "lower_birth" to 3,
        "exchange_rate" to 3,   // 음절 2.06
        "hormon" to 3,   // 음절 2.06
        "gender" to 3,   // 음절 2.07
        "gender_equality" to 3,
        "election" to 3,   // 음절 2.08
        "neuro_science" to 3,   // 음절 2.09
        "dictator" to 3,   // 음절 2.11
        "trend" to 3,   // 음절 2.12
        "obesity" to 3,   // 음절 2.12
        "social" to 3,   // 음절 2.12
        "aging_society" to 3,   // 음절 2.13
        "stock" to 4,   // 음절 2.13
        "parliamentary" to 4,   // 음절 2.14
        "computer" to 4,   // 음절 2.16
        "behavior" to 4,   // 음절 2.16
        "currency" to 4,   // 음절 2.17
        "battery" to 4,   // 음절 2.18
        "addiction" to 4,   // 음절 2.18
        "macro" to 4,   // 음절 2.18
        "digital_economics" to 4,   // 음절 2.19
        "welfare" to 4,   // 음절 2.22
        "democracy" to 4,   // 음절 2.22
        "crime" to 4,   // 음절 2.23
        "poverty" to 4,   // 음절 2.23
        "material" to 4,   // 음절 2.23
        "micro" to 4,   // 음절 2.24
        "generation_gap" to 4,   // 음절 2.25
        "racism" to 4,   // 음절 2.25
        "house" to 4,   // 음절 2.25
        "diplomacy" to 4,   // 음절 2.27
        "party" to 4,   // 음절 2.29
        "robot" to 4,   // 음절 2.29
        "security" to 4,   // 음절 2.34
        "elitism" to 4,   // 음절 2.48
    )

    /** 이 주제가 몇 단계인지. 표에 없으면 입문(2)으로 본다. */
    fun tierOf(id: String): Int = table[id] ?: 2
}
