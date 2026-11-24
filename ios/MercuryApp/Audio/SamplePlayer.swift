import AVFoundation
import Foundation

/// 예시 음원과 내 녹음을 재생한다.
///
/// 예시는 레슨 폴더의 opus/<주제><번호>.opus 다 — 안드로이드와 같은 규칙으로 찾는다.
///
/// 그냥 AVPlayer 로 틀지 않고 PCM 을 직접 읽어 손질한 뒤 재생한다. 이유가 둘이다:
///  - 예시 음원에 저역 잡음이 심하다(실측 20~120Hz 가 음성 대역보다 +17.9dB). 웅웅거려
///    듣기 나쁘고, 그 에너지가 음량의 상당 부분을 차지해 명료도를 깎는다.
///  - 마이크를 미리 열어 두려면 .playAndRecord 를 써야 하는데 이 카테고리는 출력이 작다.
///    그런데 원본 피크가 이미 -0.1dBFS 라 그냥 곱하면 잘려서 찌그러진다.
/// 그래서 저역을 걷어내 헤드룸을 만든 다음, 남는 여유만큼만 올리고 소프트 리미터로 받는다.
@MainActor
final class SamplePlayer: ObservableObject {
    @Published private(set) var playingSentenceId: Int?

    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private var engineReady = false
    /// 지금 노드가 어떤 포맷으로 이어져 있는지 — 포맷이 바뀌면 다시 이어야 한다
    private var connectedFormat: AVAudioFormat?
    /// 재생이 끝났는지 판단할 때 쓰는 세대 번호 — 이전 재생의 완료 콜백을 무시한다
    private var generation = 0
    /// PCM 으로 읽지 못했을 때 쓰는 대비책
    private var fallback: AVPlayer?
    private var endObserver: NSObjectProtocol?

    /// 목표 라우드니스(RMS). 피크로 맞추면 순간 최대값에 끌려 실제 음량이 거의 안 오른다 —
    /// 원본 피크가 이미 -0.1dBFS 라 피크 기준으로는 0.96배밖에 못 올린다.
    private static let targetRMS: Float = 0.12
    /// 최대 배수. 실측에서 1.8배가 라우드니스 +5dB, 왜곡 -19dB 로 한계였다.
    /// 더 올리면 리미터가 자주 걸려 거칠어진다.
    private static let maxBoost: Float = 1.8

    func playSample(lesson: Lesson, sentenceId: Int) {
        guard let url = sampleURL(lesson: lesson, sentenceId: sentenceId) else {
            NSLog("[SamplePlayer] 예시 음원을 찾지 못했다: %@ #%d", lesson.id, sentenceId)
            return
        }
        play(url: url, sentenceId: sentenceId)
    }

    func playRecording(at url: URL, sentenceId: Int) {
        play(url: url, sentenceId: sentenceId)
    }

    func stop() {
        generation += 1
        node.stop()
        fallback?.pause()
        fallback = nil
        if let endObserver {
            NotificationCenter.default.removeObserver(endObserver)
            self.endObserver = nil
        }
        let wasPlaying = playingSentenceId != nil
        playingSentenceId = nil
        if wasPlaying {
            // 재생하느라 .playback 으로 바꿔 놨다 — 마이크를 다시 열어 달라고 알린다
            onPlaybackEnded?()
        }
    }

    /// 재생이 끝나 마이크를 다시 열어야 할 때 불린다.
    /// .playback 으로 바꾸면 예열해 둔 입력이 끊기므로 화면 쪽에서 다시 예열한다.
    var onPlaybackEnded: (() -> Void)?

    /// 재생용 세션으로 바꾼다.
    ///
    /// .playAndRecord 는 같은 음원이라도 눈에 띄게 작게 나온다(iOS 가 입력과 공존시키느라
    /// 출력을 낮춘다). 안드로이드만큼의 음량을 내려면 재생 동안에는 .playback 이어야 한다.
    /// 대신 미리 열어 둔 마이크가 끊기므로, 재생이 끝나면 onPlaybackEnded 로 알려 되살린다.
    private func beginPlaybackSession() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default, options: [])
        try? session.setActive(true)
    }

    /// 손질 없이 그대로 트는 대비책.
    private func playDirect(url: URL, token: Int) {
        beginPlaybackSession()

        let item = AVPlayerItem(url: url)
        let player = AVPlayer(playerItem: item)
        fallback = player
        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime, object: item, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation == token else { return }
                self.stop()
            }
        }
        player.play()
    }

    // MARK: - 재생

    private func play(url: URL, sentenceId: Int) {
        stop()
        playingSentenceId = sentenceId
        let token = generation

        Task { [weak self] in
            guard let buffer = await Self.loadProcessedBuffer(url: url) else {
                // 컨테이너를 못 여는 경우가 있다. 손질은 못 해도 소리는 나야 하므로
                // 예전 방식(AVPlayer 직접 재생)으로 물러선다.
                NSLog("[SamplePlayer] PCM 으로 읽지 못해 그대로 재생한다: %@", url.lastPathComponent)
                await MainActor.run {
                    guard let self, self.generation == token else { return }
                    self.playDirect(url: url, token: token)
                }
                return
            }
            await MainActor.run {
                guard let self, self.generation == token else { return }
                self.schedule(buffer, token: token)
            }
        }
    }

    private func schedule(_ buffer: AVAudioPCMBuffer, token: Int) {
        beginPlaybackSession()

        // format: nil 로 이으면 하드웨어 기본 포맷(보통 48kHz)으로 추론된다.
        // 거기에 16kHz 버퍼를 넣으면 그 자리에서 죽는다 — 반드시 버퍼의 포맷으로 잇는다.
        if !engineReady || connectedFormat != buffer.format {
            if engineReady {
                engine.disconnectNodeOutput(node)
            } else {
                engine.attach(node)
                engineReady = true
            }
            engine.connect(node, to: engine.mainMixerNode, format: buffer.format)
            connectedFormat = buffer.format
        }
        if !engine.isRunning {
            engine.prepare()
            do {
                try engine.start()
            } catch {
                NSLog("[SamplePlayer] 재생 엔진을 시작하지 못했다: %@", error.localizedDescription)
                playingSentenceId = nil
                return
            }
        }

        node.scheduleBuffer(buffer, at: nil, options: .interrupts) { [weak self] in
            Task { @MainActor in
                guard let self, self.generation == token else { return }
                self.playingSentenceId = nil
            }
        }
        node.play()
    }

    // MARK: - 읽기·손질

    /// 파일을 PCM 으로 읽고 저역 제거 + 음량 보정까지 마친 버퍼를 만든다.
    private static func loadProcessedBuffer(url: URL) async -> AVAudioPCMBuffer? {
        guard var samples = await decodeMono16k(url: url), !samples.isEmpty else { return nil }

        highpass(&samples)
        normalize(&samples)

        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: 16000, channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { return nil }
        for i in 0..<samples.count { channel[i] = samples[i] }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        return buffer
    }

    /// opus·wav 를 가리지 않고 16kHz mono float 로 뽑는다.
    /// AVAudioFile 은 컨테이너에 따라 열지 못해서 AVAssetReader 로 디코딩한다.
    /// 다만 실기기의 AVAssetReader 는 일부 ogg-opus 를 온전히 주지 않는다(전수검사에서 발견,
    /// macOS 는 정상): 2초 미만 파일은 빈 결과를, 긴 파일도 드물게 앞 1초 남짓만 주고 완료
    /// 처리한다. 그래서 결과가 파일 길이에 못 미치면 AVAudioFile 로 다시 읽는다.
    /// 콘텐츠 전수검사 테스트가 같은 경로를 태우도록 internal 로 둔다.
    static func decodeMono16k(url: URL) async -> [Float]? {
        let asset = AVURLAsset(url: url)
        var expectedSeconds = 0.0
        if let duration = try? await asset.load(.duration), duration.isNumeric {
            expectedSeconds = CMTimeGetSeconds(duration)
        }
        if let samples = await decodeViaAssetReader(asset: asset), !samples.isEmpty,
           Double(samples.count) >= (expectedSeconds - 0.2) * 16000 {
            return samples
        }
        return decodeViaAudioFile(url: url)
    }

    private static func decodeViaAssetReader(asset: AVURLAsset) async -> [Float]? {
        guard let track = try? await asset.loadTracks(withMediaType: .audio).first,
              let reader = try? AVAssetReader(asset: asset) else { return nil }

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        guard reader.canAdd(output) else { return nil }
        reader.add(output)
        guard reader.startReading() else { return nil }

        var samples: [Float] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
            let length = CMBlockBufferGetDataLength(block)
            var bytes = [UInt8](repeating: 0, count: length)
            CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: length, destination: &bytes)
            bytes.withUnsafeBytes { raw in
                let floats = raw.bindMemory(to: Float.self)
                samples.append(contentsOf: floats)
            }
        }
        return reader.status == .completed || !samples.isEmpty ? samples : nil
    }

    /// AVAudioFile + AVAudioConverter 경로. 짧은 ogg-opus 의 AVAssetReader 실패 대응이다.
    private static func decodeViaAudioFile(url: URL) -> [Float]? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let source = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0,
              let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: frames),
              (try? file.read(into: input)) != nil, input.frameLength > 0 else { return nil }

        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                         channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: source, to: target) else { return nil }

        var fed = false
        var samples: [Float] = []
        let chunkFrames: AVAudioFrameCount = 16000
        while true {
            guard let output = AVAudioPCMBuffer(pcmFormat: target,
                                                frameCapacity: chunkFrames) else { return nil }
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError) { _, outStatus in
                if fed {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                fed = true
                outStatus.pointee = .haveData
                return input
            }
            if conversionError != nil || status == .error { return nil }
            if output.frameLength > 0, let channel = output.floatChannelData?[0] {
                samples.append(contentsOf: UnsafeBufferPointer(start: channel,
                                                               count: Int(output.frameLength)))
            }
            if status == .endOfStream { break }
        }
        return samples.isEmpty ? nil : samples
    }

    /// 4차 200Hz 하이패스. 예시 음원 실측에서 저역/음성 비를 +17.9dB → -8.6dB 로 낮췄고
    /// 음성 대역은 거의 건드리지 않는다.
    private static func highpass(_ samples: inout [Float]) {
        let f0: Double = 200, fs: Double = 16000, q: Double = 0.7071
        let w0 = 2 * Double.pi * f0 / fs
        let alpha = sin(w0) / (2 * q), cosW = cos(w0)
        let a0 = 1 + alpha
        let b0 = Float((1 + cosW) / 2 / a0)
        let b1 = Float(-(1 + cosW) / a0)
        let b2 = Float((1 + cosW) / 2 / a0)
        let a1 = Float(-2 * cosW / a0)
        let a2 = Float((1 - alpha) / a0)

        for _ in 0..<2 {   // 2차 두 번 = 4차
            var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0
            for i in 0..<samples.count {
                let x0 = samples[i]
                let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                x2 = x1; x1 = x0
                y2 = y1; y1 = y0
                samples[i] = y0
            }
        }
    }

    /// 라우드니스를 목표치에 맞춘다. 넘치는 봉우리는 잘라 내지 않고 부드럽게 눌러 준다 —
    /// 그냥 자르면 고조파가 생겨 지직거린다.
    private static func normalize(_ samples: inout [Float]) {
        var sumSquare: Double = 0
        for v in samples { sumSquare += Double(v) * Double(v) }
        let rms = Float((sumSquare / Double(samples.count)).squareRoot())
        guard rms > 0.0001 else { return }

        let boost = min(maxBoost, targetRMS / rms)
        if boost > 1.001 {
            for i in 0..<samples.count { samples[i] *= boost }
        }
        softLimit(&samples)
    }

    /// 무릎 위쪽만 tanh 로 눌러 0.98 을 넘지 않게 한다.
    private static func softLimit(_ samples: inout [Float]) {
        let knee: Float = 0.85
        let ceiling: Float = 0.98
        let range = ceiling - knee
        for i in 0..<samples.count {
            let v = samples[i]
            if v > knee {
                samples[i] = knee + range * tanh((v - knee) / range)
            } else if v < -knee {
                samples[i] = -knee + range * tanh((v + knee) / range)
            }
        }
    }

    /// opus/<주제>00000.opus 형식. 번호는 0부터라 문장 번호에서 1을 뺀다.
    private func sampleURL(lesson: Lesson, sentenceId: Int) -> URL? {
        // 1.article 은 <레벨>/<분류>/<주제> 로 한 겹 더 깊다. lesson.id 만 붙이면
        // "1.article/architect/opus" 가 되어 찾지 못한다 — 상대 경로를 그대로 쓴다.
        guard let root = Bundle.main.resourceURL?
            .appendingPathComponent("Assets/contents/\(lesson.relativePath)/opus")
        else { return nil }

        let names = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        let opus = names.filter { $0.hasSuffix(".opus") }.sorted()
        let index = sentenceId - 1
        guard index >= 0, index < opus.count else { return nil }
        return root.appendingPathComponent(opus[index])
    }
}
