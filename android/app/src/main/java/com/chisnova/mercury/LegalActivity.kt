package com.chisnova.mercury

import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.chisnova.mercury.data.LegalConfiguration

/** Settings, documents and support use the same native reader and back stack. */
class LegalActivity : BaseActivity() {
    override fun getCurrentScreen() = BottomNavScreen.HOME
    private lateinit var content: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_legal)
        content = findViewById(R.id.legalContent)
        findViewById<ImageView>(R.id.btnBack).setOnClickListener { finish() }
        setupBottomNavigation()
        findViewById<LinearLayout>(R.id.btnNavHome).setOnClickListener {
            startActivity(Intent(this, HomeActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP))
            finish()
        }
        val configuration = LegalConfiguration.load(this)
        val title = findViewById<TextView>(R.id.legalTitle)
        val caption = findViewById<TextView>(R.id.legalCaption)
        when (val page = intent.getStringExtra(EXTRA_PAGE)) {
            "terms", "privacy" -> {
                val document = configuration.document(page)
                title.text = document.title
                caption.text = document.effectiveDate.takeIf { it.isNotBlank() && document.available }?.let { "시행일 · $it" }.orEmpty()
                if (document.sections.isEmpty()) {
                    section("", if (document.webUri == null)
                        "안내 내용을 준비하고 있습니다. 준비되는 대로 이 화면에서 확인하실 수 있습니다."
                        else "아래 웹페이지에서 내용을 확인하실 수 있습니다.")
                } else document.sections.forEach { section(it.title, it.body) }
                document.webUri?.let { uri -> link("웹에서 보기", "") { open(uri) } }
            }
            "support" -> {
                title.text = "문의하기"
                caption.text = "서비스 이용 및 구매 문의"
                val email = configuration.emailUri
                val web = LegalConfiguration.webUri(configuration.supportUrl)
                if (email == null && web == null) section("", "문의 안내를 준비하고 있습니다.")
                if (email != null) {
                    section("이메일 전송 안내", "문의 메일은 Gmail을 통해 운영자에게 전달됩니다. 앱이 녹음이나 학습 기록을 자동 첨부하지 않습니다. 불필요한 개인정보는 보내지 말아 주세요.")
                    section("문의 이메일", configuration.supportEmail)
                    link("이메일로 문의", "") { open(email) }
                }
                web?.let { uri -> link("문의 페이지 열기", "") { open(uri) } }
            }
            else -> {
                title.text = "설정 및 안내"
                caption.text = "이용 조건과 개인정보 보호 안내를 확인하세요."
                for (id in listOf("terms", "privacy")) {
                    val document = configuration.document(id)
                    link(document.title, if (document.available) "내용 확인" else "준비 중") { navigate(id) }
                }
                link("문의하기", "서비스 이용 및 구매 문의") { navigate("support") }
                section("쉐도잉 코치", listOfNotNull(
                    configuration.operatorName.takeIf { it.isNotBlank() }?.let { "운영자 · $it" },
                    "버전 ${BuildConfig.VERSION_NAME}"
                ).joinToString("\n"))
            }
        }
        caption.visibility = if (caption.text.isBlank()) View.GONE else View.VISIBLE
    }

    private fun navigate(page: String) {
        startActivity(Intent(this, LegalActivity::class.java).putExtra(EXTRA_PAGE, page))
    }

    private fun section(title: String, body: String) {
        val view = layoutInflater.inflate(R.layout.item_legal_section, content, false)
        view.findViewById<TextView>(R.id.legalSectionTitle).apply {
            text = title
            visibility = if (title.isBlank()) View.GONE else View.VISIBLE
            androidx.core.view.ViewCompat.setAccessibilityHeading(this, true)
        }
        view.findViewById<TextView>(R.id.legalSectionBody).text = body
        content.addView(view)
    }

    private fun link(title: String, subtitle: String, onClick: () -> Unit) {
        val view = layoutInflater.inflate(R.layout.item_legal_link, content, false)
        view.findViewById<TextView>(R.id.legalLinkTitle).text = title
        view.findViewById<TextView>(R.id.legalLinkSubtitle).apply {
            text = subtitle
            visibility = if (subtitle.isBlank()) View.GONE else View.VISIBLE
        }
        view.setOnClickListener { onClick() }
        content.addView(view)
    }

    private fun open(uri: Uri) {
        val isMail = uri.scheme == "mailto"
        try {
            startActivity(Intent(if (isMail) Intent.ACTION_SENDTO else Intent.ACTION_VIEW, uri))
        } catch (_: ActivityNotFoundException) {
            Toast.makeText(this, if (isMail) "메일 앱을 확인하거나 표시된 이메일 주소를 복사해 이용해 주세요."
                else "링크를 열 수 있는 브라우저를 확인해 주세요.", Toast.LENGTH_LONG).show()
        }
    }

    companion object {
        const val EXTRA_PAGE = "legal_page"
        fun bindPurchaseLinks(activity: android.app.Activity) {
            for ((viewId, page) in listOf(R.id.purchaseTerms to "terms", R.id.purchasePrivacy to "privacy", R.id.purchaseSupport to "support")) {
                activity.findViewById<View>(viewId).setOnClickListener {
                    activity.startActivity(Intent(activity, LegalActivity::class.java).putExtra(EXTRA_PAGE, page))
                }
            }
        }
    }
}
