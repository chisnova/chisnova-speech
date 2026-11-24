import SwiftUI

/// 문장 목록 + 발음 연습 — 안드로이드 VoiceRecognitionActivity(+ item_sentence.xml) 대응.
struct SentenceListView: View {
    let lesson: Lesson
    @StateObject private var purchases = PurchaseStore.shared

    var body: some View {
        Group {
            if purchases.canAccess(lesson) {
                UnlockedSentenceListView(lesson: lesson)
            } else {
                PurchaseView()
            }
        }
    }
}

private struct UnlockedSentenceListView: View {
    let lesson: Lesson

    @StateObject private var session = AsrSession()
    @StateObject private var player = SamplePlayer()
    @State private var sentences: [Sentence] = []
    @State private var activeSentenceId: Int?
    /// 단어를 누르면 채워진다 — 단어·음소 점수 팝업(안드로이드 wordInfoPopup 대응)
    @State private var selectedWord: AsrSession.WordScore?
    @State private var resetError: String?

    var body: some View {
        VStack(spacing: 0) {
            statusBar

            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(sentences) { sentence in
                        SentenceCard(
                            sentence: sentence,
                            isActive: activeSentenceId == sentence.id,
                            partialText: activeSentenceId == sentence.id ? session.partialText : "",
                            // 연습이 끝나도 남는 결과. 활성 문장이 아니어도 보여 준다.
                            finalText: session.recognisedTexts[sentence.id] ?? "",
                            score: session.scores[sentence.id],
                            wordScores: session.wordScores[sentence.id] ?? [],
                            onSelectWord: { selectedWord = $0 },
                            isRecording: session.isRecording && activeSentenceId == sentence.id,
                            isPlaying: player.playingSentenceId == sentence.id,
                            hasRecording: session.recordingURL(for: sentence.id) != nil,
                            // 녹음 중에는 그 문장의 버튼만 살려 둔다("완료"로 동작).
                            // 전부 살려 두면 다른 문장을 눌렀을 때 연습이 시작되는 대신
                            // 진행 중인 녹음이 끝나 버린다.
                            canPractice: session.state == .ready
                                || (session.isRecording && activeSentenceId == sentence.id),
                            onPractice: { practice(sentence) },
                            onPlaySample: { player.playSample(lesson: lesson, sentenceId: sentence.id) },
                            onPlayMine: {
                                if let url = session.recordingURL(for: sentence.id) {
                                    player.playRecording(at: url, sentenceId: sentence.id)
                                }
                            }
                        )
                    }
                }
                .padding(16)
            }
        }
        .background(DS.bg)
        .navigationTitle(lesson.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button("불러오기") {
                    do { try session.loadHistory(for: sentences) }
                    catch { resetError = error.localizedDescription }
                }
                    .font(.system(size: 14, weight: .semibold))
                    .disabled(session.isRecording || session.state == .finishing)
                Button("초기화") {
                    do {
                        try session.clearScores()
                        player.stop()
                        activeSentenceId = nil
                        selectedWord = nil
                    } catch {
                        resetError = error.localizedDescription
                    }
                }
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(DS.danger)
                    .disabled(session.isRecording || session.state == .finishing)
            }
        }
        .alert("학습 상태를 변경하지 못했습니다", isPresented: Binding(
            get: { resetError != nil }, set: { if !$0 { resetError = nil } }
        )) {
            Button("확인", role: .cancel) { resetError = nil }
        } message: {
            Text(resetError ?? "")
        }
        .task {
            sentences = ContentStore.sentences(of: lesson)
            session.currentLesson = lesson
            // 초기화 상태를 유지하면서 현재 진도에 포함된 문장·단어 점수만 복원한다.
            session.restoreScores(for: sentences)
            await session.loadModelIfNeeded()
            // 마이크를 미리 열어 둔다. 버튼을 누른 뒤에 세션을 활성화하면
            // 그때까지 1초 가까이 마이크가 죽어 있어 첫 음절이 통째로 날아간다.
            session.prewarmMicrophone()
            // 예시를 틀면 세션이 재생용으로 바뀌어 마이크가 끊긴다 — 끝나면 다시 연다
            player.onPlaybackEnded = { session.restoreMicrophone() }
        }
        .onDisappear {
            // 화면을 벗어나면 마이크를 놓아 준다 — 잡고 있으면 다른 앱의 오디오를 막는다
            session.releaseMicrophone()
        }
        .sheet(item: $selectedWord) { word in
            WordScoreSheet(word: word)
                .presentationDetents([.medium, .large])
        }
    }

    private var statusBar: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(session.isRecording ? DS.danger : DS.textTertiary.opacity(0.5))
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.system(size: 12))
                .foregroundStyle(DS.textSecondary)
            Spacer()
            if !session.lastEndReason.isEmpty {
                Text(session.lastEndReason)
                    .font(.system(size: 11))
                    .foregroundStyle(DS.textTertiary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(DS.surface)
        .overlay(Rectangle().frame(height: 1).foregroundStyle(DS.borderLight), alignment: .bottom)
    }

    private var statusText: String {
        switch session.state {
        case .idle:         return "대기"
        case .loadingModel: return "엔진 준비 중…"
        case .ready:        return "준비됨"
        case .recording:    return "듣는 중"
        case .finishing:    return "채점 중…"
        case .failed(let m): return m
        }
    }

    private func practice(_ sentence: Sentence) {
        Task {
            // 녹음 중이면 같은 버튼이 "완료"로 동작한다
            if session.isRecording {
                session.stopRecording()
            } else {
                activeSentenceId = sentence.id
                await session.practice(sentence)
            }
        }
    }
}

/// 문장 카드 — item_sentence.xml 대응 (발음연습 / 예시 듣기 / 내 녹음)
struct SentenceCard: View {
    let sentence: Sentence
    let isActive: Bool
    let partialText: String
    let finalText: String
    let score: Double?
    let wordScores: [AsrSession.WordScore]
    let onSelectWord: (AsrSession.WordScore) -> Void
    let isRecording: Bool
    let isPlaying: Bool
    let hasRecording: Bool
    let canPractice: Bool
    let onPractice: () -> Void
    let onPlaySample: () -> Void
    let onPlayMine: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 8) {
                // 단어마다 점수 색이 입고, 누르면 음소 점수를 볼 수 있다
                ScoredSentence(sentence: sentence.english,
                               wordScores: wordScores,
                               onSelect: onSelectWord)
                Spacer(minLength: 0)
                if let score {
                    Text(String(format: "%.2f점", score))
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(DS.primary)
                        .padding(.horizontal, 11)
                        .padding(.vertical, 6)
                        .background(DS.primarySurface)
                        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
                        .fixedSize()
                }
            }

            if !sentence.korean.isEmpty {
                Text(sentence.korean)
                    .font(.system(size: 14))
                    .foregroundStyle(DS.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(DS.surfaceSecondary)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            }

            // 말하는 동안에는 1패스(CTC1) 결과가 실시간으로 흐르고,
            // 발화가 끝나면 전체 문맥을 본 2패스(CTC2) 결과로 확정된다.
            HStack(spacing: 14) {
                actionButton(
                    title: isRecording ? "완료" : (score == nil ? "발음연습" : "연습하기"),
                    symbol: isRecording ? "stop.fill" : "person.wave.2.fill",
                    iconTint: isRecording ? .white : DS.primary,
                    iconBackground: isRecording ? DS.danger : DS.primarySurface,
                    labelTint: isRecording ? DS.danger : DS.primary,
                    labelBold: true,
                    enabled: canPractice,
                    action: onPractice
                )
                actionButton(
                    title: "예시 듣기",
                    symbol: isPlaying ? "speaker.wave.2.fill" : "speaker.wave.2.fill",
                    iconTint: DS.primary,
                    iconBackground: DS.primarySurface,
                    labelTint: DS.textSecondary,
                    labelBold: false,
                    enabled: true,
                    action: onPlaySample
                )
                actionButton(
                    title: "내 녹음",
                    symbol: "play.fill",
                    iconTint: .white,
                    iconBackground: hasRecording ? DS.textPrimary : DS.textTertiary,
                    labelTint: DS.accentGreen,
                    labelBold: false,
                    enabled: hasRecording,
                    action: onPlayMine
                )
                Spacer(minLength: 0)
            }

            // 인식 결과는 버튼 아래에 한 줄로 조용히 둔다 — 문장과 점수가 주인공이라
            // 박스로 감싸 키우면 화면이 시끄러워진다.
            if !recognisedText.isEmpty {
                HStack(alignment: .top, spacing: 5) {
                    Image(systemName: isStreaming ? "waveform" : "text.quote")
                        .font(.system(size: 10))
                        .foregroundStyle(DS.textTertiary)
                        .padding(.top, 2)
                    Text(recognisedText)
                        .font(.system(size: 12))
                        .foregroundStyle(isStreaming ? DS.textTertiary : DS.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .animation(.easeOut(duration: 0.15), value: recognisedText)
                }
            }
        }
        .dsCard(padding: 18, radius: 18)
    }

    /// 지금 이 문장을 녹음 중이고 최종 결과가 아직 없으면 흐르는 중간 결과다
    private var isStreaming: Bool { isActive && finalText.isEmpty }

    private var recognisedText: String {
        finalText.isEmpty ? partialText : finalText
    }

    /// 안드로이드처럼 동그란 아이콘과 라벨을 가로로 늘어놓는다.
    private func actionButton(title: String, symbol: String,
                              iconTint: Color, iconBackground: Color,
                              labelTint: Color, labelBold: Bool,
                              enabled: Bool,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 7) {
                ZStack {
                    Circle()
                        .fill(iconBackground)
                        .frame(width: 34, height: 34)
                    Image(systemName: symbol)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(iconTint)
                }
                Text(title)
                    .font(.system(size: 14, weight: labelBold ? .bold : .regular))
                    .foregroundStyle(labelTint)
                    .fixedSize()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .opacity(enabled ? 1 : 0.4)
    }
}

struct ScoreBadge: View {
    let score: Double

    private var tint: Color {
        switch score {
        case 80...:  return DS.accentGreen
        case 60..<80: return DS.warning
        default:      return DS.danger
        }
    }

    var body: some View {
        Text("\(Int(score.rounded()))점")
            .font(.system(size: 13, weight: .bold))
            .foregroundStyle(tint)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(tint.opacity(0.12))
            .clipShape(Capsule())
    }
}
