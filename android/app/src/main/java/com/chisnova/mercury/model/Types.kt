package com.chisnova.mercury.model

enum class Screen {
    Home,
    Level1,
    Level2,
    Level3,
    Lesson,
    VoiceRecognition
}

data class Lesson(
    val id: Int,
    val title: String
)

data class VoiceRecognitionSentence(
    val id: Int,
    val english: String,
    val korean: String
)

data class VoiceRecognitionResult(
    val recognizedText: String,
    val score: Int,
    val translation: String,
    val feedback: String
)

