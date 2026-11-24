package com.chisnova.mercury.database
import com.chisnova.mercury.database.DatabaseHelper
import com.chisnova.mercury.asr.AsrController.FinalScore
import com.google.gson.Gson
import android.util.Log


class FinalScoreSaver {
    companion object {
        private val instance = FinalScoreSaver()
        fun getInstance() = instance
    }

    private val phoneCategories = mapOf(
        "AA" to "모음",
        "AE" to "모음",
        "AH" to "모음",
        "AO" to "모음",
        "AW" to "모음",
        "AY" to "모음",
        "EH" to "모음",
        "ER" to "모음",
        "EY" to "모음",
        "IH" to "모음",
        "IY" to "모음",
        "OW" to "모음",
        "OY" to "모음",
        "UH" to "모음",
        "UW" to "모음",
        "B" to "파열음",
        "D" to "파열음",
        "G" to "파열음",
        "K" to "파열음",
        "P" to "파열음",
        "T" to "파열음",
        "CH" to "파찰음",
        "JH" to "파찰음",
        "DH" to "마찰음",
        "F" to "마찰음",
        "S" to "마찰음",
        "SH" to "마찰음",
        "TH" to "마찰음",
        "V" to "마찰음",
        "Z" to "마찰음",
        "ZH" to "마찰음",
        "HH" to "기음",
        "L" to "유음",
        "R" to "유음",
        "M" to "비음",
        "N" to "비음",
        "NG" to "비음",
        "W" to "반모음",
        "Y" to "반모음"
    )

    fun saveFinalScoreToDatabase(
        finalScore: String?,
        content: DatabaseHelper.ContentItem?,
        secondPassResult: String?,
        dbHelper: DatabaseHelper?
    ): Float {
        val gson = Gson()
        val jsonResult = finalScore ?: ""
        val parsedScore = gson.fromJson(jsonResult, FinalScore::class.java)

        val date = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())
        val aid = java.util.UUID.randomUUID().toString()
        val uid = content?.uid?.toInt() ?: 0
        val numWord = parsedScore?.final_score?.size?.toInt() ?: 0
        val transcript = secondPassResult ?: "" 
        val recordPath = "tmp.wav"
        val fluency = 0.0f
        val pronounce = parsedScore?.avg_score?.toFloat() ?: 0.0f.toFloat()

        dbHelper?.insertAttempt(date, aid, uid, numWord, transcript, recordPath, fluency, pronounce)

        // Insert word history and phone history for each word in the final score
        parsedScore?.final_score?.let { finalScores ->
            for ((wordIdx, wordScore) in finalScores.withIndex()) {
                // Insert word history
                dbHelper?.insertWordHistory(
                    date = date,
                    aid = aid,
                    wordIdx = wordIdx,
                    word = wordScore.word,
                    numPhone = wordScore.phone_score.size,
                    color = "", // Color can be determined based on score if needed
                    score = wordScore.score.toFloat(),
                    comment = ""
                )

                // Insert phone history for each phone in the word
                wordScore.phone_score.forEachIndexed { phoneIdx, phoneScore ->
                    dbHelper?.insertPhoneHistory(
                        date = date,
                        aid = aid,
                        wordIdx = wordIdx,
                        phoneIdx = phoneIdx,
                        phone = phoneScore.word,
                        category = phoneCategories[phoneScore.word.uppercase().replace(Regex("[0-9]"), "")] ?: "기타",
                        score = phoneScore.score.toFloat()
                    )
                }
            }
        }

        val attempt = dbHelper?.queryAttemptsByUid(uid)
        val word_history = dbHelper?.queryWordHistoryWithAID(aid)
        val phone_history = dbHelper?.queryPhoneHistoryWithAID(aid)
        Log.i("[AsrAPI]", "saveFinalScoreToDatabase attempt: $attempt")
        Log.i("[AsrAPI]", "saveFinalScoreToDatabase word_history: $word_history")
        Log.i("[AsrAPI]", "saveFinalScoreToDatabase phone_history: $phone_history")
        return pronounce
    }
}
