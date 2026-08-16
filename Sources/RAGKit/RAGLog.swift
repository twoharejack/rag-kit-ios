import Foundation
import os

/// Lightweight, self-contained logging for the package so it carries no
/// dependency on any host app's logging facade. Host apps that want to route
/// these somewhere custom can set `RAGLog.handler`.
public enum RAGLog {
    /// Optional sink for host apps. When set, it receives every log line
    /// instead of the default `os.Logger`.
    public static var handler: ((String) -> Void)?

    private static let logger = Logger(subsystem: "RAGKit", category: "rag")

    static func debug(_ message: @autoclosure () -> String) {
        let text = message()
        if let handler {
            handler(text)
        } else {
            logger.debug("\(text, privacy: .public)")
        }
    }

    static func warning(_ message: @autoclosure () -> String) {
        let text = message()
        if let handler {
            handler(text)
        } else {
            logger.warning("\(text, privacy: .public)")
        }
    }

    static func error(_ message: @autoclosure () -> String) {
        let text = message()
        if let handler {
            handler(text)
        } else {
            logger.error("\(text, privacy: .public)")
        }
    }
}
