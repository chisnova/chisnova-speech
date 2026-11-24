package com.chisnova.mercury.billing

import java.security.KeyFactory
import java.security.Signature
import java.security.spec.X509EncodedKeySpec

object PurchaseSignature {
    fun verify(publicKey: ByteArray, data: String, signature: ByteArray): Boolean = try {
        val key = KeyFactory.getInstance("RSA").generatePublic(X509EncodedKeySpec(publicKey))
        Signature.getInstance("SHA1withRSA").run {
            initVerify(key)
            update(data.toByteArray(Charsets.UTF_8))
            verify(signature)
        }
    } catch (_: Exception) { false }
}
