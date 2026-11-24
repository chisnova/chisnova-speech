package com.chisnova.mercury.data

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

object AssetContentLoader {
    data class BilingualContent(
        val englishLines: List<String>,
        val koreanLines: List<String>
    )

    /**
     * Load .eng and .kor files for a given base path inside assets.
     * Example basePath: "contents/1.basic/asking_for_help/asking_for_help"
     */
    fun loadBilingualContent(context: Context, basePath: String): BilingualContent {
        val eng = readLines(context, "$basePath.eng")
        val kor = readLines(context, "$basePath.kor")
        return BilingualContent(eng, kor)
    }

    private fun readLines(context: Context, assetPath: String): List<String> {
        return try {
            context.assets.open(assetPath).use { input ->
                BufferedReader(InputStreamReader(input)).readLines()
            }
        } catch (_: Exception) {
            emptyList()
        }
    }
}


