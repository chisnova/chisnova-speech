package com.chisnova.mercury.data

import com.chisnova.mercury.billing.ContentAccess
import com.chisnova.mercury.billing.PurchaseSignature
import java.security.KeyPairGenerator
import java.security.Signature
import org.junit.Assert.*
import org.junit.Test

class PurchaseSecurityTest {
    @Test fun onlyFourTopicsAreFreeAndIdentityCannotUnlockOtherAssets() {
        val folders = mapOf("greetings" to "greeting_basic_expression", "daily_conversation" to "phone_conversation",
            "emotions" to "expressing_opinions", "time_date" to "telling_time")
        folders.forEach { (id, folder) ->
            assertTrue(ContentAccess.isFree(id, "contents/1.basic/$folder/$folder"))
            assertFalse(ContentAccess.isFree(id, "contents/2.beginner/$folder/$folder"))
        }
        assertFalse(ContentAccess.isFree("greetings", "contents/5.speech/conan/conan"))
        assertFalse(ContentAccess.isFree("opinions_feelings", "contents/1.basic/expressing_opinions/expressing_opinions"))
        assertFalse(ContentAccess.isFree(null, "contents/1.basic/telling_time/telling_time"))
        assertFalse(ContentAccess.isFree("time_date", null))
    }

    @Test fun everyNonBasicLevelOneCategoryRemainsPaid() {
        val categories = LessonData.getLevel1Categories()
        val expected = setOf("greetings", "daily_conversation", "emotions", "time_date")
        assertEquals(expected, categories.single { it.id == "basic" }.lessons.map { it.id }.toSet())
        val freePaths = listOf("greeting_basic_expression", "phone_conversation", "expressing_opinions", "telling_time")
        categories.filter { it.id != "basic" }.flatMap { it.lessons }.forEach { lesson ->
            freePaths.forEach { folder ->
                assertFalse("Paid topic ${lesson.id} must remain locked", ContentAccess.isFree(lesson.id, "contents/1.basic/$folder/$folder"))
            }
        }
    }

    @Test fun tamperedReceiptsWrongKeysAndMalformedSignaturesFailClosed() {
        val generator = KeyPairGenerator.getInstance("RSA").apply { initialize(2048) }
        val key = generator.generateKeyPair()
        val receipt = "{\"productId\":\"full_unlock\",\"purchaseToken\":\"example\"}"
        val signature = Signature.getInstance("SHA1withRSA").run {
            initSign(key.private)
            update(receipt.toByteArray(Charsets.UTF_8))
            sign()
        }
        assertTrue(PurchaseSignature.verify(key.public.encoded, receipt, signature))
        assertFalse(PurchaseSignature.verify(key.public.encoded, receipt.replace("example", "forged"), signature))
        assertFalse(PurchaseSignature.verify(generator.generateKeyPair().public.encoded, receipt, signature))
        assertFalse(PurchaseSignature.verify(byteArrayOf(), receipt, signature))
        assertFalse(PurchaseSignature.verify(key.public.encoded, receipt, byteArrayOf()))
    }
}
