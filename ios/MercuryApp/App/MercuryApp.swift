import SwiftUI

@main
struct MercuryApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var purchases = PurchaseStore.shared

    init() {
        do {
            // Also covers files created by earlier app versions.
            try LocalDataStorage.documentsDirectory()
        } catch {
            NSLog("[Privacy] Could not exclude local data from backup: %@", error.localizedDescription)
        }
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .task { await purchases.refreshEntitlements() }
                .onChange(of: scenePhase) { phase in
                    if phase == .active { Task { await purchases.refreshEntitlements() } }
                }
        }
    }
}
