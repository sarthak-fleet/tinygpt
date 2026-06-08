import SwiftUI
import AppKit

@main
struct TinyGPTApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("TinyGPT", id: "main") {
            ContentView()
                .frame(minWidth: 920, minHeight: 600)
                .preferredColorScheme(.dark)
        }
        .windowStyle(.titleBar)
        .windowResizability(.contentSize)
        .commands {
            // Drop the default "New" item — TinyGPT has no document model
            // to create a new instance of, so File→New would be a dead
            // entry. Everything else (Cmd-Q, window minimize/zoom, etc.)
            // stays at SwiftUI defaults.
            CommandGroup(replacing: .newItem) {}
        }
    }
}

/// Force the app to the foreground on launch. Without this, the unsigned
/// .app bundle stays in the background when launched via `open` from CLI;
/// SwiftPM-built apps don't get the activation that LaunchServices does
/// for properly-codesigned ones.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}
