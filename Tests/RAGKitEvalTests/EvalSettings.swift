// EvalSettings.swift
// ============================================================================
// What the eval runs on, read from the environment, so `swift test` and an
// Xcode scheme configure it the same way. See Docs/Evals.md.
// ============================================================================

import Foundation

struct EvalSettings {
    /// Comma-separated engines, or "all". Default: `EvalEngine.defaults`.
    static let enginesVariable = "RAGKIT_EVAL_ENGINES"
    /// Comma-separated judges, such as "claude,codex" or "claude:opus".
    static let judgesVariable = "RAGKIT_EVAL_JUDGES"
    /// A set file of the bundled fixture's shape. Default: the fixture.
    static let setVariable = "RAGKIT_EVAL_SET"
    /// Where report.md and results.json go. Default: .build/ragkit-eval.
    static let outputVariable = "RAGKIT_EVAL_OUTPUT"
    static let depthVariable = "RAGKIT_EVAL_DEPTH"
    static let concurrencyVariable = "RAGKIT_EVAL_JUDGE_CONCURRENCY"

    /// Whether the eval was asked for. It downloads MiniLM, and with judges
    /// it spends tokens, so a plain `swift test` leaves it out.
    static var isRequested: Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment[enginesVariable] != nil || environment[judgesVariable] != nil
    }

    let engines: [EvalEngine]
    let judges: [any RelevanceJudge]
    let setURL: URL
    let outputDirectory: URL
    let options: EvalOptions

    var isBundledSet: Bool { setURL.standardizedFileURL == EvalSet.bundledURL.standardizedFileURL }

    /// Each judge's grades live beside the set they grade.
    var cacheDirectory: URL {
        setURL.deletingPathExtension().appendingPathExtension("judgments")
    }

    static func fromEnvironment(_ environment: [String: String] = ProcessInfo.processInfo.environment) throws -> EvalSettings {
        let engines = try parseEngines(environment[enginesVariable])
        let judges = try environment[judgesVariable].map(JudgeSpec.judges(from:)) ?? []

        var options = EvalOptions()
        if let depth = environment[depthVariable] {
            guard let value = Int(depth), value > 0 else {
                throw SettingsError(variable: depthVariable, value: depth, expected: "a positive number")
            }
            options.depth = value
        }
        if let concurrency = environment[concurrencyVariable] {
            guard let value = Int(concurrency), value > 0 else {
                throw SettingsError(variable: concurrencyVariable, value: concurrency, expected: "a positive number")
            }
            options.judgeConcurrency = value
        }

        return EvalSettings(
            engines: engines,
            judges: judges,
            setURL: environment[setVariable].map { URL(fileURLWithPath: $0) } ?? EvalSet.bundledURL,
            outputDirectory: environment[outputVariable].map { URL(fileURLWithPath: $0, isDirectory: true) }
                ?? packageRoot.appendingPathComponent(".build/ragkit-eval", isDirectory: true),
            options: options
        )
    }

    static func parseEngines(_ list: String?) throws -> [EvalEngine] {
        guard let list, !list.trimmingCharacters(in: .whitespaces).isEmpty else { return EvalEngine.defaults }
        if list.trimmingCharacters(in: .whitespaces).lowercased() == "all" { return EvalEngine.allCases }
        return try list.split(separator: ",").map { name in
            let name = name.trimmingCharacters(in: .whitespaces).lowercased()
            guard let engine = EvalEngine(rawValue: name) else {
                let engines = EvalEngine.allCases.map(\.rawValue).joined(separator: ", ")
                throw SettingsError(variable: enginesVariable, value: name, expected: "\(engines), or all")
            }
            return engine
        }
    }

    private static let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // RAGKitEvalTests
        .deletingLastPathComponent() // Tests
        .deletingLastPathComponent()
}

struct SettingsError: Error, CustomStringConvertible {
    let variable: String
    let value: String
    let expected: String
    var description: String { "\(variable) is \"\(value)\"; expected \(expected)" }
}
