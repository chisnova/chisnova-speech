import AVFoundation
import Foundation

/// 마이크 입력을 16kHz mono float PCM 으로 흘려보낸다.
///
/// 안드로이드의 `AudioController` 에 해당한다. AudioRecord 로 직접 읽는 대신
/// AVAudioEngine 의 탭을 쓰고, 하드웨어 표본율(보통 48kHz)에서 16kHz 로 변환한다.
final class AudioRecorder {

    enum RecorderError: Error {
        case permissionDenied
        case converterUnavailable
        case engineStartFailed(Error)
    }

    /// 변환된 16kHz mono 청크. 녹음 스레드에서 불린다.
    var onAudioChunk: ((UnsafePointer<Float>, Int) -> Void)?

    /// 이번 녹음의 전체 샘플. "내 녹음" 재생에 쓴다(안드로이드 saveAudioBuffer 대응).
    private(set) var capturedSamples: [Float] = []
    private var deliveredChunks = 0

    /// 엔진에 넣기 전 샘플을 모아 두는 곳.
    private var pending: [Float] = []
    /// 엔진에 한 번에 넣는 크기 — 100ms @ 16kHz.
    private let deliveryChunkSize = 1600

    /// 녹음이 끝났을 때(자동 종료·수동 종료 모두) 메인 큐에서 불린다.
    var onFinished: (() -> Void)?

    private let engine = AVAudioEngine()
    private let targetSampleRate: Double = 16000
    private var converter: AVAudioConverter?
    /// 실제로 샘플을 모아 내보내는 중인지. 탭은 예열 이후 계속 붙어 있으므로
    /// 이 값이 꺼져 있는 동안 들어온 오디오는 그냥 버린다.
    private var isRunning = false
    /// 세션·엔진·탭이 이미 준비됐는지
    private var isPrewarmed = false

    /// 녹음 중인지 — UI 가 버튼 상태를 정하는 데 쓴다.
    var isRecording: Bool { isRunning }

    /// 손질을 하지 않으므로 항상 1.0 이다. 발화 종료 판정이 증폭분을 빼는 데 쓴다.
    let currentGain: Float = 1.0

    /// 마이크 권한을 확인하고 없으면 요청한다.
    static func requestPermission() async -> Bool {
        if #available(iOS 17.0, *) {
            return await AVAudioApplication.requestRecordPermission()
        } else {
            return await withCheckedContinuation { continuation in
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        }
    }

    /// 세션·엔진·탭을 미리 띄워 둔다. 화면에 들어올 때 한 번 부르면 된다.
    ///
    /// 버튼을 누른 뒤에 setActive/engine.start 를 하면 마이크가 열릴 때까지 1초 가까이
    /// 걸려 첫 음절이 통째로 날아간다. 미리 열어 두고 녹음은 스위치만 켜는 방식으로 바꾼다.
    func prewarm() throws {
        guard !isPrewarmed else { return }

        // 입력이 -61dB 까지 떨어지는 문제가 있었다. .voiceChat 은 에코 제거가 붙어 있어
        // 조합에 따라 입력을 크게 깎고, .allowBluetooth 는 저품질 HFP 마이크를 물어올 수 있다.
        // 녹음만 하면 되므로 가장 단순한 조합으로 두고, 입력은 내장 마이크로 못 박는다.
        // .record 가 아니라 .playAndRecord 를 쓴다. 마이크를 미리 열어 두는 구조에서
        // 재생 때문에 카테고리가 .playback 으로 바뀌면 예열해 둔 입력이 죽어 버린다.
        // 재생기도 같은 카테고리를 쓰므로 전환 없이 둘이 공존한다.
        // .voiceChat 모드는 에코 제거가 입력을 크게 깎으니 .default 를 유지한다.
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default,
                                options: [.defaultToSpeaker, .allowBluetoothA2DP])
        try session.setActive(true, options: [])

        // 내장 마이크를 명시적으로 고른다 — 블루투스/외장이 잡히면 레벨이 들쭉날쭉해진다
        if let builtIn = session.availableInputs?.first(where: { $0.portType == .builtInMic }) {
            try? session.setPreferredInput(builtIn)
            // 아래쪽(입에 가까운) 마이크를 쓰도록 지시한다
            if let front = builtIn.dataSources?.first(where: { $0.orientation == .bottom })
                ?? builtIn.dataSources?.first {
                try? builtIn.setPreferredDataSource(front)
            }
        }
        if session.isInputGainSettable {
            try? session.setInputGain(1.0)
        }

        Diag.log("입력 포트: \(session.currentRoute.inputs.map { $0.portName }.joined(separator: ",")) " +
                 "gainSettable=\(session.isInputGainSettable) gain=\(session.inputGain)")
        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                               sampleRate: targetSampleRate,
                                               channels: 1,
                                               interleaved: false) else {
            throw RecorderError.converterUnavailable
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw RecorderError.converterUnavailable
        }
        self.converter = converter

        // 하드웨어 버퍼 크기는 기기마다 다르다. 시간 기준으로 다루기 위해
        // 변환 후 샘플 수로만 판단한다(안드로이드에서 bufferRecordSize 에 의존하다 겪은 문제).
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            self?.convertAndDeliver(buffer, using: converter, outputFormat: outputFormat)
        }

        Diag.log("입력 포맷 \(Int(inputFormat.sampleRate))Hz \(inputFormat.channelCount)ch -> 16000Hz 1ch (예열)")

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw RecorderError.engineStartFailed(error)
        }
        isPrewarmed = true
    }

    func start() throws {
        guard !isRunning else { return }
        // 예열이 안 됐으면 여기서라도 한다(첫 음절 유실은 감수)
        try prewarm()

        capturedSamples.removeAll(keepingCapacity: true)
        pending.removeAll(keepingCapacity: true)
        deliveredChunks = 0
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        // 먼저 스위치를 내려 탭이 더 이상 pending 에 쌓지 않게 한다
        isRunning = false

        // 링 버퍼에 남은 꼬리를 마저 보낸다(100ms 미만이라도 마지막 음절이 들어 있다)
        if !pending.isEmpty {
            let tail = pending
            pending.removeAll(keepingCapacity: true)
            deliver(tail)
        }
        // 엔진과 세션은 그대로 둔다 — 다음 녹음을 지연 없이 시작하기 위함이다.
        // 정리는 화면을 떠날 때 teardown() 에서 한다.
        DispatchQueue.main.async { [weak self] in
            self?.onFinished?()
        }
    }

    /// 화면을 떠날 때 마이크를 놓아 준다.
    func teardown() {
        guard isPrewarmed else { return }
        isRunning = false
        isPrewarmed = false
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
    }

    /// 모아 둔 샘플을 고정 크기로 잘라 내보낸다.
    ///
    /// 내보낼 몫을 먼저 떼어낸 다음에 콜백을 부른다. pending 을
    /// withUnsafeBufferPointer 로 잡은 채 콜백을 부르면, 그 안에서 pending 에 다시 닿는 순간
    /// 배타적 접근 위반으로 죽는다.
    private func flushPending() {
        while pending.count >= deliveryChunkSize {
            let chunk = Array(pending[0..<deliveryChunkSize])
            pending.removeFirst(deliveryChunkSize)
            deliver(chunk)
        }
    }

    /// 떼어낸 덩어리를 엔진으로 넘긴다.
    private func deliver(_ chunk: [Float]) {
        chunk.withUnsafeBufferPointer { buffer in
            if let base = buffer.baseAddress {
                onAudioChunk?(base, buffer.count)
            }
        }
    }

    private func convertAndDeliver(_ buffer: AVAudioPCMBuffer,
                                   using converter: AVAudioConverter,
                                   outputFormat: AVAudioFormat) {
        // 예열 상태에서는 탭이 붙어 있어도 흘려보내지 않는다
        guard isRunning else { return }
        let ratio = outputFormat.sampleRate / buffer.format.sampleRate
        // 딱 맞게 잡으면 리샘플러 지연분이 잘려 소리가 조금씩 유실된다
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio * 2) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            return
        }

        var consumed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            // 같은 입력 버퍼를 두 번 주면 변환기가 무한히 되돌아온다
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return buffer
        }
        if let error {
            NSLog("[AudioRecorder] 표본율 변환 실패: %@", error.localizedDescription)
            return
        }
        guard output.frameLength > 0, let channel = output.floatChannelData?[0] else {
            return
        }
        let frames = Int(output.frameLength)

        // 실시간 이득 보정: 최근 피크를 보고 목표 레벨에 맞춰 완만하게 끌어올린다.
        // 급히 바꾸면 파형이 일그러지므로 게인은 천천히만 움직인다.
        // 예전에는 여기서 하이패스와 자동 이득을 걸었다. 그런데 이득이 2초 만에 1.1배에서
        // 3.6배까지 출렁이며 파형을 망가뜨려 인식이 무너졌다(같은 녹음을 데스크톱 엔진에
        // 넣어도 똑같이 엉망이었다). 안드로이드는 아무 손질 없이 원본을 넘겨 잘 인식한다.
        // 그래서 엔진에는 마이크가 준 그대로 보낸다. 저역 잡음 제거와 음량 보정은
        // 듣기용으로만 필요하므로 재생할 때 SamplePlayer 가 처리한다.

        capturedSamples.append(contentsOf: UnsafeBufferPointer(start: channel, count: frames))

        // 처음 몇 덩어리의 레벨을 남긴다. 여기가 -90dB 근처면 마이크가 아니라 캡처 문제다.
        // frames 는 변환기가 준 크기이고, 엔진에는 아래 flushPending 이 항상 같은 크기로 넣는다.
        deliveredChunks += 1
        if deliveredChunks <= 3 || deliveredChunks % 50 == 0 {
            var sum = 0.0
            for i in 0..<frames { sum += Double(channel[i]) * Double(channel[i]) }
            let db = 20 * log10(max((sum / Double(max(frames, 1))).squareRoot(), 1e-7))
            Diag.log(String(format: "chunk #%d frames=%d level=%.1fdB total=%d",
                            deliveredChunks, frames, db, capturedSamples.count))
        }
        // 들어온 만큼 모아 두고, 정해진 크기로만 잘라 내보낸다
        pending.append(contentsOf: UnsafeBufferPointer(start: channel, count: frames))
        flushPending()
    }
}
