package com.chisnova.mercury.billing

object ContentAccess {
    // Match identity AND asset path: a free lesson ID cannot unlock a different asset.
    private val freeAssets = mapOf(
        "greetings" to "greeting_basic_expression",
        "daily_conversation" to "phone_conversation",
        "emotions" to "expressing_opinions",
        "time_date" to "telling_time",
    )

    fun isFree(lessonId: String?, assetBase: String?): Boolean {
        val folder = freeAssets[lessonId] ?: return false
        return assetBase == "contents/1.basic/$folder/$folder"
    }
}
