import XCTest
import StoreKit
import StoreKitTest
@testable import MercuryApp

@MainActor
final class PurchaseTests: XCTestCase {
    func testOnlyFourBasicTopicsAreFreeIncludingPaidAlias() {
        let basic = ContentStore.lessons(in: .basic)
        XCTAssertEqual(Set(basic.filter(ContentAccess.isFree).map(\.id)),
                       Set(["greetings", "daily_conversation", "emotions", "time_date"]))
        XCTAssertFalse(ContentAccess.isFree(basic.first { $0.id == "opinions_feelings" }!))
        let forged = Lesson(id: "greetings", title: "Forged", category: "basic", level: .basic,
                            folder: "conan", relativePath: "5.speech/conan")
        XCTAssertFalse(ContentAccess.isFree(forged))
        for level in ContentLevel.allCases where level != .basic {
            XCTAssertTrue(ContentStore.lessons(in: level).allSatisfy { !ContentAccess.isFree($0) })
        }
    }

    // One serial scenario: StoreKitTest controls a single shared store environment.
    @available(iOS 17.0, *)
    func testPurchaseCancellationPendingApprovalRestoreAndRefund() async throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "FullUnlock", withExtension: "storekit"))
        let testStore = try SKTestSession(contentsOf: url)
        testStore.resetToDefaultState()
        testStore.clearTransactions()
        testStore.disableDialogs = true
        defer { testStore.clearTransactions(); testStore.resetToDefaultState() }
        let purchases = PurchaseStore()
        await purchases.refreshEntitlements()
        XCTAssertFalse(purchases.hasFullAccess)
        await purchases.loadProduct()
        XCTAssertEqual(purchases.product?.id, PurchaseStore.productID)
        XCTAssertEqual(purchases.product?.type, .nonConsumable)

        try await testStore.setSimulatedError(.generic(.userCancelled), forAPI: .purchase)
        await purchases.purchase()
        XCTAssertFalse(purchases.hasFullAccess)
        XCTAssertFalse(purchases.isBusy)
        try await testStore.setSimulatedError(nil, forAPI: .purchase)

        testStore.askToBuyEnabled = true
        await purchases.purchase()
        XCTAssertFalse(purchases.hasFullAccess, "Pending purchases must not unlock lessons")
        let pending = try XCTUnwrap(testStore.allTransactions().first)
        try testStore.approveAskToBuyTransaction(identifier: pending.identifier)
        try await eventually { purchases.hasFullAccess }
        testStore.askToBuyEnabled = false

        let restored = PurchaseStore()
        await restored.restore()
        XCTAssertTrue(restored.hasFullAccess, "A new store instance must restore verified ownership")
        for level in ContentLevel.allCases {
            XCTAssertTrue(ContentStore.lessons(in: level).allSatisfy(restored.canAccess))
        }
        let transaction = try XCTUnwrap(testStore.allTransactions().first)
        try testStore.refundTransaction(identifier: transaction.identifier)
        try await eventually { !restored.hasFullAccess && !purchases.hasFullAccess }
        XCTAssertFalse(restored.canAccess(ContentStore.lessons(in: .speech).first!))

        // Also exercise the direct Product.purchase success path, not just Transaction.updates.
        testStore.clearTransactions()
        await purchases.purchase()
        XCTAssertTrue(purchases.hasFullAccess)
        XCTAssertFalse(purchases.isBusy)
    }

    private func eventually(_ condition: () -> Bool) async throws {
        for _ in 0..<100 {
            if condition() { return }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTFail("StoreKit entitlement update did not arrive within 10 seconds")
    }
}
