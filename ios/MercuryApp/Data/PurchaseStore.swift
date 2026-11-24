import StoreKit
import Combine

enum ContentAccess {
    private static let freeFolders = [
        "greetings": "greeting_basic_expression", "daily_conversation": "phone_conversation",
        "emotions": "expressing_opinions", "time_date": "telling_time"
    ]

    static func isFree(_ lesson: Lesson) -> Bool {
        guard let folder = freeFolders[lesson.id] else { return false }
        return lesson.level == .basic && lesson.category == "basic" && lesson.folder == folder
            && lesson.relativePath == "1.basic/\(folder)"
    }
}

/// StoreKit의 검증된 거래만 사용한다. UserDefaults의 구매 여부 플래그는 신뢰하지 않는다.
@MainActor
final class PurchaseStore: ObservableObject {
    static let shared = PurchaseStore()
    static let productID = "com.chisnova.mercury.full_unlock"

    @Published private(set) var hasFullAccess = false
    @Published private(set) var product: Product?
    @Published private(set) var isBusy = false
    @Published private(set) var message: String?
    private var updatesTask: Task<Void, Never>?

    init() {
        updatesTask = Task { [weak self] in
            for await result in Transaction.updates {
                guard !Task.isCancelled else { return }
                await self?.receive(result)
            }
        }
        Task { [weak self] in await self?.refreshEntitlements() }
    }

    deinit { updatesTask?.cancel() }

    func canAccess(_ lesson: Lesson) -> Bool {
        ContentAccess.isFree(lesson) || hasFullAccess
    }

    func refreshEntitlements() async {
        var unlocked = false
        for await result in Transaction.currentEntitlements {
            if case .verified(let transaction) = result,
               transaction.productID == Self.productID,
               transaction.productType == .nonConsumable,
               transaction.revocationDate == nil {
                unlocked = true
            }
        }
        if hasFullAccess && !unlocked { message = "전체 이용권의 구매 권한이 해제되었습니다." }
        hasFullAccess = unlocked
    }

    func loadProduct() async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        message = nil
        do {
            product = try await Product.products(for: [Self.productID]).first { $0.type == .nonConsumable }
            if product == nil { message = "구매 상품을 불러올 수 없습니다. 잠시 후 다시 시도해 주세요." }
        } catch {
            product = nil
            message = "App Store에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요."
        }
    }

    func purchase() async {
        guard !isBusy, !hasFullAccess, let product else { return }
        isBusy = true
        message = nil
        defer { isBusy = false }
        do {
            switch try await product.purchase() {
            case .success(let result): await receive(result)
            case .pending: message = "결제 승인 대기 중입니다. 승인이 완료되면 전체 콘텐츠가 열립니다."
            case .userCancelled: break
            @unknown default: message = "구매 상태를 확인하지 못했습니다. 구매 복원을 시도해 주세요."
            }
        } catch StoreKitError.userCancelled {
            // Cancellation is a normal exit from the payment sheet.
        } catch {
            message = "구매를 완료하지 못했습니다. 잠시 후 다시 시도해 주세요."
        }
    }

    func restore() async {
        guard !isBusy else { return }
        isBusy = true
        message = nil
        defer { isBusy = false }
        do {
            try await AppStore.sync()
            await refreshEntitlements()
            message = hasFullAccess ? "구매가 복원되었습니다." : "이 Apple 계정에서 복원할 구매 내역이 없습니다."
        } catch {
            message = "구매 내역을 확인하지 못했습니다. 잠시 후 다시 시도해 주세요."
        }
    }

    private func receive(_ result: VerificationResult<Transaction>) async {
        guard case .verified(let transaction) = result else {
            message = "구매 검증에 실패했습니다. 구매 복원을 시도해 주세요."
            return
        }
        guard transaction.productID == Self.productID, transaction.productType == .nonConsumable else { return }
        await refreshEntitlements()
        if transaction.revocationDate != nil { message = "전체 이용권의 구매 권한이 해제되었습니다." }
        else { message = nil }
        await transaction.finish()
    }
}
