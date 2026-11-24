package com.chisnova.mercury.data

import android.content.Context
import android.net.Uri
import android.util.Log
import org.json.JSONObject

data class LegalSection(val title: String, val body: String)
data class LegalDocument(
    val id: String,
    val title: String,
    val effectiveDate: String = "",
    val url: String = "",
    val sections: List<LegalSection> = emptyList()
) {
    val webUri: Uri? get() = LegalConfiguration.webUri(url)
    val available: Boolean get() = sections.isNotEmpty() || webUri != null
}

data class LegalConfiguration(
    val operatorName: String = "",
    val supportEmail: String = "",
    val supportUrl: String = "",
    val documents: List<LegalDocument> = emptyList()
) {
    fun document(id: String) = documents.firstOrNull { it.id == id }
        ?: LegalDocument(id, if (id == "privacy") "개인정보 처리방침" else "이용약관")

    val emailUri: Uri?
        get() = supportEmail.trim().takeIf { it.contains('@') && it.none(Char::isWhitespace) }?.let {
            Uri.parse("mailto:${Uri.encode(it, "@")}\u003fsubject=${Uri.encode("쉐도잉 코치 문의")}")
        }

    companion object {
        fun load(context: Context): LegalConfiguration = try {
            val json = JSONObject(context.assets.open("policies.json").bufferedReader().use { it.readText() })
            val documents = json.optJSONArray("documents")
            LegalConfiguration(
                json.optString("operatorName"), json.optString("supportEmail"), json.optString("supportUrl"),
                (0 until (documents?.length() ?: 0)).map { index ->
                    val document = documents!!.getJSONObject(index)
                    val sections = document.optJSONArray("sections")
                    LegalDocument(
                        document.getString("id"), document.getString("title"),
                        document.optString("effectiveDate"), document.optString("url"),
                        (0 until (sections?.length() ?: 0)).map { sectionIndex ->
                            val section = sections!!.getJSONObject(sectionIndex)
                            LegalSection(section.optString("title"), section.getString("body"))
                        }.filter { it.body.isNotBlank() }
                    )
                }
            )
        } catch (e: Exception) {
            Log.e("Legal", "Could not load policies", e)
            LegalConfiguration()
        }

        fun webUri(value: String): Uri? = Uri.parse(value.trim()).takeIf {
            it.scheme.equals("https", ignoreCase = true) && !it.host.isNullOrBlank() && it.userInfo == null
        }
    }
}
