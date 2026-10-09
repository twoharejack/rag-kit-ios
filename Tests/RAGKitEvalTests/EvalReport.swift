// EvalReport.swift
// ============================================================================
// An eval's result as a Markdown report to read and a JSON file with every
// ranking and grade behind it.
// ============================================================================

import Foundation

struct EvalReport {
    let result: EvalResult
    let date: Date

    init(_ result: EvalResult, date: Date = Date()) {
        self.result = result
        self.date = date
    }

    /// Writes `report.md` and `results.json` into `directory`.
    @discardableResult
    func write(to directory: URL) throws -> (markdown: URL, json: URL) {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let markdownURL = directory.appendingPathComponent("report.md")
        let jsonURL = directory.appendingPathComponent("results.json")
        try Data(markdown.utf8).write(to: markdownURL, options: .atomic)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(details).write(to: jsonURL, options: .atomic)
        return (markdownURL, jsonURL)
    }

    // MARK: - Markdown

    var markdown: String {
        [header, engines, judged, labeled, byKind, judgesSection, perQuery]
            .compactMap { $0 }
            .joined(separator: "\n\n") + "\n"
    }

    private var set: EvalSet { result.set }
    private var depth: Int { result.options.depth }
    private var cutoff: Int { result.precisionCutoff }

    private var header: String {
        let judges = result.judges.isEmpty
            ? "No LLM judge ran, so every score is against the hand labels."
            : "Judged by \(result.judges.map { "`\($0)`" }.joined(separator: " and ")) on a 0–3 scale; 2 or more counts as relevant."
        return """
            # Retrieval eval: \(set.name)

            \(set.documents.count) documents, \(set.queries.count) queries, top \(depth) per query. \
            \(judges) Run \(date.formatted(date: .abbreviated, time: .shortened)).
            """
    }

    private var engines: String {
        var lines = [
            "## Engines",
            "",
            "| Engine | What it is | Indexed | Indexing | Per query |",
            "|---|---|---|---|---|",
        ]
        for run in result.runs {
            let indexed = set.documents.count - run.unindexed.count
            let perQuery = seconds(run.searchTime) / Double(max(set.queries.count, 1))
            lines.append(
                "| \(run.engine.rawValue) | \(run.engine.summary) | \(indexed)/\(set.documents.count) "
                    + "| \(String(format: "%.1f s", seconds(run.indexingTime))) | \(String(format: "%.0f ms", perQuery * 1000)) |"
            )
        }
        let unindexed = result.runs.filter { !$0.unindexed.isEmpty }
        if !unindexed.isEmpty {
            lines.append("")
            for run in unindexed {
                lines.append("- \(run.engine.rawValue) could not index \(run.unindexed.joined(separator: ", ")).")
            }
        }
        return lines.joined(separator: "\n")
    }

    private var judged: String? {
        guard !result.judges.isEmpty else { return nil }
        let judges = result.judges
        let meanColumn = judges.count > 1
        var header = ["Engine"]
        header += judges.map { "nDCG@\(depth) \(short($0))" }
        if meanColumn { header.append("nDCG@\(depth) mean") }
        header += judges.map { "MRR \(short($0))" }
        header += judges.map { "P@\(cutoff) \(short($0))" }

        var lines = [
            "## Judged by the LLMs",
            "",
            "Every judge grades every document any engine returned for a query, plus the labeled ones, "
                + "so nDCG@\(depth) is measured against the best ranking of everything found. "
                + "MRR and P@\(cutoff) count documents graded 2 or more; P@\(cutoff) cannot reach 1 "
                + "for a query with fewer than \(cutoff) relevant documents.",
            "",
            row(header),
            row(header.map { _ in "---" }),
        ]
        for run in result.runs {
            let scores = judges.map { result.judgedScores(of: run, judge: $0) }
            var cells = [run.engine.rawValue]
            cells += scores.map { format($0.ndcg) }
            if meanColumn { cells.append("**\(format(result.headlineNDCG(of: run)))**") }
            cells += scores.map { format($0.mrr) }
            cells += scores.map { format($0.precision) }
            lines.append(row(cells))
        }
        return lines.joined(separator: "\n")
    }

    private var labeled: String? {
        guard set.hasLabels else { return nil }
        var lines = [
            "## Against the hand labels",
            "",
            row(["Engine", "nDCG@\(depth)", "MRR", "P@\(cutoff)", "Recall@\(depth)"]),
            row(Array(repeating: "---", count: 5)),
        ]
        for run in result.runs {
            guard let scores = result.labeledScores(of: run) else { continue }
            lines.append(row([run.engine.rawValue, format(scores.ndcg), format(scores.mrr), format(scores.precision), format(scores.recall)]))
        }
        return lines.joined(separator: "\n")
    }

    private var byKind: String? {
        let kinds = result.kinds
        guard !kinds.isEmpty else { return nil }
        let source = result.judges.isEmpty ? "against the labels" : "mean of the judges"
        var header = ["Engine"]
        for kind in kinds {
            header.append("\(kind) (\(set.queries.filter { $0.kind == kind }.count))")
        }
        var lines = [
            "## By query kind",
            "",
            "nDCG@\(depth), \(source).",
            "",
            row(header),
            row(header.map { _ in "---" }),
        ]
        for run in result.runs {
            var cells = [run.engine.rawValue]
            for kind in kinds {
                cells.append(format(result.headlineNDCG(of: run, queries: set.queries.filter { $0.kind == kind })))
            }
            lines.append(row(cells))
        }
        return lines.joined(separator: "\n")
    }

    private var judgesSection: String? {
        guard !result.judges.isEmpty else { return nil }
        var lines = [
            "## The judges",
            "",
            row(["Judge", "Pairs", "From cache", "Calls", "Failed", "Labeled relevant, judged relevant", "Unlabeled, judged relevant", "κ vs labels"]),
            row(Array(repeating: "---", count: 8)),
        ]
        for judge in result.judges {
            let activity = result.activity[judge] ?? JudgeActivity()
            let labels = result.labelAgreement(of: judge)
            lines.append(row([
                "`\(judge)`",
                "\(activity.cached + activity.graded)",
                "\(activity.cached)",
                "\(activity.calls)",
                "\(activity.failures.count)",
                percent(labels?.labeledRelevantJudgedRelevant),
                percent(labels?.unlabeledJudgedRelevant),
                format(labels?.agreement?.kappa),
            ]))
        }

        let judges = result.judges
        let pairs = judges.indices.flatMap { i in judges.indices.filter { $0 > i }.map { (judges[i], judges[$0]) } }
        if !pairs.isEmpty {
            lines += [
                "",
                "κ is Cohen's kappa on relevant (2 or more) or not: 1 is full agreement, 0 what chance alone would give.",
                "",
                row(["Judges", "Pairs", "Same grade", "Within one", "κ"]),
                row(Array(repeating: "---", count: 5)),
            ]
            for (first, second) in pairs {
                guard let agreement = result.agreement(between: first, and: second) else { continue }
                lines.append(row([
                    "`\(first)` vs `\(second)`",
                    "\(agreement.pairs)",
                    percent(agreement.exact),
                    percent(agreement.withinOne),
                    format(agreement.kappa),
                ]))
            }
        }

        let failures = judges.flatMap { judge in (result.activity[judge]?.failures ?? []).map { "- `\(judge)` \($0)" } }
        if !failures.isEmpty {
            lines += ["", "Failed calls:", ""] + failures.prefix(20)
        }
        return lines.joined(separator: "\n")
    }

    private var perQuery: String {
        let relevance = result.judges.isEmpty
            ? "labeled 2 or more"
            : "the judges grade 2 or more on average"
        var header = ["Query", "Kind"]
        header += result.runs.map(\.engine.rawValue)
        var lines = [
            "## Per query",
            "",
            "Rank of the first document \(relevance); – when none made the top \(depth).",
            "",
            row(header),
            row(header.map { _ in "---" }),
        ]
        for query in set.queries {
            let grades = result.consensusGrades(for: query)
            var cells = [query.text.replacingOccurrences(of: "|", with: "\\|"), query.kind ?? ""]
            for run in result.runs {
                let ranking = run.rankings[query.id] ?? []
                let rank = ranking.firstIndex { (grades[$0] ?? 0) >= Double(RetrievalMetrics.relevantGrade) }
                cells.append(rank.map { "\($0 + 1)" } ?? "–")
            }
            lines.append(row(cells))
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - JSON

    struct Details: Encodable {
        struct Query: Encodable {
            let id: String
            let text: String
            let kind: String?
            let labels: [String: Int]?
            /// Document IDs per engine, best first.
            let rankings: [String: [String]]
            /// Grades per judge, by document ID.
            let grades: [String: [String: Int]]
        }

        let set: String
        let depth: Int
        let engines: [String]
        let judges: [String]
        let queries: [Query]
    }

    var details: Details {
        Details(
            set: set.name,
            depth: depth,
            engines: result.runs.map(\.engine.rawValue),
            judges: result.judges,
            queries: set.queries.map { query in
                Details.Query(
                    id: query.id,
                    text: query.text,
                    kind: query.kind,
                    labels: query.relevant,
                    rankings: Dictionary(uniqueKeysWithValues: result.runs.map { ($0.engine.rawValue, $0.rankings[query.id] ?? []) }),
                    grades: Dictionary(uniqueKeysWithValues: result.judges.map { ($0, result.grades(of: $0, for: query)) })
                )
            }
        )
    }

    // MARK: - Formatting

    private func row(_ cells: [String]) -> String {
        "| " + cells.joined(separator: " | ") + " |"
    }

    private func format(_ value: Double?) -> String {
        value.map { String(format: "%.2f", $0) } ?? "–"
    }

    private func percent(_ value: Double?) -> String {
        value.map { String(format: "%.0f%%", $0 * 100) } ?? "–"
    }

    /// A judge's column name: its CLI, which is all that tells two judges
    /// apart unless two models of one CLI are judging.
    private func short(_ judge: String) -> String {
        let cli = String(judge.prefix { $0 != "-" })
        let sameCLI = result.judges.filter { $0.hasPrefix(cli + "-") }.count
        return sameCLI > 1 ? judge : cli
    }

    private func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}
