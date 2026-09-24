// RAGNaturalLanguageEmbedder.swift
// ============================================================================
// Apple's on-device embeddings (the NaturalLanguage framework) as a RAGKit
// engine: nothing to bundle or fetch from Hugging Face, and every language
// Apple has a model for, each text embedded by the model for its script.
// ============================================================================

import Foundation
import NaturalLanguage
import VecturaKit

/// Embeds text with the NaturalLanguage framework's own models, for corpora in
/// any mix of the languages it is configured with.
///
/// Apple has two kinds of embedding model, and neither covers every language
/// in one vector space:
///
/// - `NLEmbedding.sentenceEmbedding(for:)` has one model per language, seven
///   languages in all. On the development Mac only English was installed, and
///   an app cannot ask for the others.
/// - `NLContextualEmbedding` has one model per script, shared by its
///   languages: Latin (20 languages), Chinese/Japanese/Korean, Cyrillic,
///   Arabic, Indic, and Thai. Latin and CJK were installed; the others are
///   Apple downloads the app can request (`requestMissingAssets()`).
///
/// So each script a corpus uses gets its own block of the vector. A text is
/// embedded by its script's model into its script's block, and every other
/// block stays zero. That puts a query about Chinese notes and a query about
/// French notes in separate spaces, which is the point: two models' vectors
/// cannot be compared, and a query only ever meets documents from its own
/// model. A text in a script with no block goes to the first block (its
/// ranking suffers, but it is still indexed and keyword search still finds
/// it). Cross-language search does not work: an English query found the
/// matching French or Chinese note first in 0 of 7 trials with either model.
///
/// Which model each block uses, measured on macOS 27 with ten notes and 24
/// queries per language (mean reciprocal rank of the right note):
///
/// | Notes in  | English NLEmbedding | NLContextualEmbedding |
/// |-----------|---------------------|-----------------------|
/// | English   | 0.85                | 0.75                  |
/// | French    | 0.54                | 0.83                  |
/// | Chinese   | 0.37                | 0.74                  |
///
/// A Latin block serving English alone therefore uses the English
/// `NLEmbedding`, and every other block uses its script's
/// `NLContextualEmbedding`. Adding a second Latin language (English and
/// French, say) moves the whole Latin block to the contextual model.
///
/// Pooling follows what each model is good at. The sentence model is built for
/// sentences: given a whole note it returns a vector that says little about
/// any one part (a 1,100-character note scored under 0.06 against queries
/// about its own sentences). So a text is embedded in ten-word runs and the
/// runs averaged, reading at most 512 words. The contextual model returns one
/// vector per token, and those are averaged over the text. Its runs scored
/// worse (0.75 and 0.68 against 0.83 and 0.74). It reads 256 tokens at a
/// time, so a longer text is read in two passes, about the same 512-token
/// window the sentence transformer reads.
///
/// Every block is then centered: a fixed vector, the mean of the block's
/// model over reference sentences, is subtracted. Raw contextual vectors all
/// point the same way: unrelated English notes averaged 0.70 against a query
/// and the right note 0.74. That leaves a search threshold nothing to
/// separate, and it flattens the vector half of a hybrid score. Centered,
/// unrelated notes average 0.1–0.3, and ranking moved by no more than 0.05
/// either way.
///
/// Each block's model and revision, their order, and the pooling scheme all go
/// into `spaceIdentifier`. A change of languages that changes the blocks (a new
/// script, or a second Latin language) therefore re-embeds, and one that does
/// not (reordering French and German) keeps the index.
public actor RAGNaturalLanguageEmbedder: RAGEmbedder {
    /// Whether this device can embed a language right now.
    public enum LanguageSupport: Sendable, Equatable {
        /// A model for the language is on the device.
        case ready
        /// Apple has a model, but it has not been downloaded yet;
        /// `requestMissingAssets()` asks the system for it.
        case needsDownload
        /// Apple has no embedding model for the language.
        case unsupported
    }

    /// Bump when pooling, chunking, or the reference sentences change: any of
    /// them moves every vector.
    private static let schemeVersion = 1
    private static let wordsPerRun = 10
    private static let wordLimit = 512
    private static let contextualPasses = 2
    private static let referenceSentencesPerBlock = 12

    /// The configured languages this device has a model for, in the order
    /// given. The first one's block also takes texts in scripts no block
    /// covers.
    public nonisolated let languages: [NLLanguage]
    public nonisolated let dimension: Int
    public nonisolated let spaceIdentifier: String

    private var blocks: [Block]
    private let recognizer = NLLanguageRecognizer()
    private let tokenizer = NLTokenizer(unit: .word)
    /// Routing key per language, `nil` for a language with no model.
    private var routingKeys: [NLLanguage: String?] = [:]

    /// - Parameter languages: The languages the corpus is written in, most
    ///   important first. Empty means the device's preferred languages
    ///   (`devicePreferredLanguages`).
    /// - Throws: `RAGError.naturalLanguageModelUnavailable` when Apple has no
    ///   embedding model for any of the languages.
    public init(languages requested: [NLLanguage] = []) throws {
        let requested = requested.isEmpty ? Self.devicePreferredLanguages : requested
        var groups: [(key: String, model: NLContextualEmbedding, languages: [NLLanguage])] = []
        for language in requested {
            guard let model = NLContextualEmbedding(language: language) else {
                RAGLog.warning("⚠️ No Apple embedding model for \(language.rawValue); its texts go to the first block")
                continue
            }
            if let index = groups.firstIndex(where: { $0.key == model.modelIdentifier }) {
                if !groups[index].languages.contains(language) {
                    groups[index].languages.append(language)
                }
            } else {
                groups.append((model.modelIdentifier, model, [language]))
            }
        }
        guard !groups.isEmpty else {
            throw RAGError.naturalLanguageModelUnavailable(language: requested.first?.rawValue ?? "und")
        }

        var blocks: [Block] = []
        var offset = 0
        for group in groups {
            let model: Block.Model
            if group.languages == [.english], let sentence = NLEmbedding.sentenceEmbedding(for: .english) {
                model = .sentence(sentence)
            } else {
                model = .contextual(group.model)
            }
            let block = Block(routingKey: group.key, languages: group.languages, model: model, offset: offset)
            offset += block.dimension
            blocks.append(block)
        }

        self.languages = groups.flatMap(\.languages)
        self.blocks = blocks
        self.dimension = offset
        self.spaceIdentifier = "apple-nl-v\(Self.schemeVersion):" + blocks.map(\.spaceComponent).joined(separator: "+")
    }

    /// The device's preferred languages (Settings ▸ Language & Region) as
    /// NaturalLanguage languages, most preferred first.
    public static var devicePreferredLanguages: [NLLanguage] {
        var seen: Set<NLLanguage> = []
        return Locale.preferredLanguages.compactMap { identifier in
            guard let language = naturalLanguage(forLocaleIdentifier: identifier),
                  seen.insert(language).inserted else { return nil }
            return language
        }
    }

    /// Whether this device can embed `language`, without loading anything.
    public static func support(for language: NLLanguage) -> LanguageSupport {
        guard let model = NLContextualEmbedding(language: language) else { return .unsupported }
        if model.hasAvailableAssets { return .ready }
        if language == .english, NLEmbedding.sentenceEmbedding(for: .english) != nil { return .ready }
        return .needsDownload
    }

    /// Configured languages whose model has not been downloaded. Their texts
    /// fail to embed (and are left out of the index) until it is.
    public func languagesNeedingDownload() -> [NLLanguage] {
        blocks.filter(\.needsAssets).flatMap(\.languages)
    }

    /// Asks the system to download every model this embedder needs and does
    /// not have. The download is Apple's, not the app's.
    /// - Returns: Whether every model is available afterwards.
    @discardableResult
    public func requestMissingAssets() async -> Bool {
        var allAvailable = true
        for index in blocks.indices where blocks[index].needsAssets {
            guard case .contextual(let model) = blocks[index].model else { continue }
            do {
                let result = try await model.requestAssets()
                allAvailable = allAvailable && result == .available
            } catch {
                RAGLog.warning("⚠️ Could not download the embedding model for \(blocks[index].languages.map(\.rawValue)): \(error)")
                allAvailable = false
            }
        }
        return allAvailable
    }

    // MARK: - RAGEmbedder

    public func embed(texts: [String]) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for text in texts {
            if !vectors.isEmpty {
                // Lets a waiting query embed between texts.
                await Task.yield()
            }
            vectors.append(try vector(for: text))
        }
        return vectors
    }

    public func embed(text: String) async throws -> [Float] {
        try vector(for: text)
    }

    // MARK: - Embedding

    /// The text's vector: its block holds the centered, pooled embedding from
    /// the block's model, and every other block is zero.
    private func vector(for text: String) throws -> [Float] {
        let (index, language) = route(text)
        try prepareBlock(at: index)
        let block = blocks[index]
        guard let pooled = try pooledVector(of: text, in: block, language: language) else {
            throw VecturaError.invalidInput("Apple's embedding model returned no vector for text of length \(text.count)")
        }
        var vector = [Float](repeating: 0, count: dimension)
        for component in 0..<block.dimension {
            vector[block.offset + component] = Float(pooled[component] - (block.center?[component] ?? 0))
        }
        return vector
    }

    /// The block a text belongs to, by the most likely of its languages that
    /// has a block, with that language when the block's model knows it.
    /// Routing by model rather than by exact language means a short query
    /// misread as a neighbouring language still lands in the right block.
    private func route(_ text: String) -> (block: Int, language: NLLanguage?) {
        recognizer.reset()
        recognizer.processString(text)
        let hypotheses = recognizer.languageHypotheses(withMaximum: 3).sorted { $0.value > $1.value }
        for (language, _) in hypotheses {
            guard let key = routingKey(for: language),
                  let index = blocks.firstIndex(where: { $0.routingKey == key }) else { continue }
            return (index, blocks[index].reads(language) ? language : nil)
        }
        return (0, nil)
    }

    private func routingKey(for language: NLLanguage) -> String? {
        if let cached = routingKeys[language] { return cached }
        let key = NLContextualEmbedding(language: language)?.modelIdentifier
        routingKeys[language] = key
        return key
    }

    /// Loads the block's model and computes its center, once.
    private func prepareBlock(at index: Int) throws {
        guard blocks[index].center == nil else { return }
        if case .contextual(let model) = blocks[index].model {
            guard model.hasAvailableAssets else {
                throw RAGError.naturalLanguageModelUnavailable(language: blocks[index].languages[0].rawValue)
            }
            try model.load()
        }
        let reference = Self.referenceSentences(for: blocks[index])
        let vectors = try reference.compactMap { try pooledVector(of: $0.text, in: blocks[index], language: $0.language) }
        blocks[index].center = vectors.isEmpty
            ? [Double](repeating: 0, count: blocks[index].dimension)
            : Self.mean(of: vectors, dimension: blocks[index].dimension)
    }

    private func pooledVector(of text: String, in block: Block, language: NLLanguage?) throws -> [Double]? {
        switch block.model {
        case .sentence(let embedding):
            let vectors = runs(of: text, language: language).compactMap { run in
                embedding.vector(for: run).flatMap { $0.count == block.dimension ? $0 : nil }
            }
            return vectors.isEmpty ? nil : Self.mean(of: vectors, dimension: block.dimension)
        case .contextual(let model):
            return try tokenMean(of: text, model: model, language: language)
        }
    }

    /// Cuts `text` into runs of `wordsPerRun` words, reading at most
    /// `wordLimit` words. A run is the original text from its first word to
    /// its last, so spacing and punctuation inside it come through as written,
    /// including in scripts that put no spaces between words. Text with no
    /// words in it at all (emoji, symbols) is one run.
    private func runs(of text: String, language: NLLanguage?) -> [String] {
        tokenizer.string = text
        tokenizer.setLanguage(language ?? .english)
        var words: [Range<String.Index>] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            words.append(range)
            return words.count < Self.wordLimit
        }
        guard !words.isEmpty else {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? [] : [trimmed]
        }
        return stride(from: 0, to: words.count, by: Self.wordsPerRun).map { first in
            let last = min(first + Self.wordsPerRun, words.count) - 1
            return String(text[words[first].lowerBound..<words[last].upperBound])
        }
    }

    /// The mean of the model's token vectors over the text. The model reads
    /// `maximumSequenceLength` tokens and silently drops the rest, so a text
    /// it stops short of is read again from where it stopped, up to
    /// `contextualPasses` times.
    private func tokenMean(of text: String, model: NLContextualEmbedding, language: NLLanguage?) throws -> [Double]? {
        var sum = [Double](repeating: 0, count: model.dimension)
        var tokens = 0
        var remaining = Substring(text)
        for _ in 0..<Self.contextualPasses {
            let piece = String(remaining)
            guard piece.contains(where: { $0.isLetter || $0.isNumber }) else { break }
            let result = try model.embeddingResult(for: piece, language: language)
            var reached = piece.startIndex
            result.enumerateTokenVectors(in: piece.startIndex..<piece.endIndex) { vector, range in
                for component in 0..<min(vector.count, sum.count) {
                    sum[component] += vector[component]
                }
                tokens += 1
                reached = range.upperBound
                return true
            }
            guard reached > piece.startIndex, reached < piece.endIndex else { break }
            let consumed = piece.distance(from: piece.startIndex, to: reached)
            remaining = remaining[remaining.index(remaining.startIndex, offsetBy: consumed)...]
        }
        return tokens > 0 ? sum.map { $0 / Double(tokens) } : nil
    }

    private static func mean(of vectors: [[Double]], dimension: Int) -> [Double] {
        var sum = [Double](repeating: 0, count: dimension)
        for vector in vectors {
            for component in 0..<min(vector.count, dimension) {
                sum[component] += vector[component]
            }
        }
        return sum.map { $0 / Double(vectors.count) }
    }

    // MARK: - Languages

    /// A locale identifier ("en-US", "zh-Hant-TW", "pt-BR") as the
    /// NaturalLanguage language it is written in.
    static func naturalLanguage(forLocaleIdentifier identifier: String) -> NLLanguage? {
        let locale = Locale(identifier: identifier)
        guard let code = locale.language.languageCode?.identifier else { return nil }
        switch code {
        case "zh":
            let script = locale.language.script?.identifier
            let region = locale.region?.identifier ?? ""
            let traditional = script == "Hant" || (script == nil && ["TW", "HK", "MO"].contains(region))
            return traditional ? .traditionalChinese : .simplifiedChinese
        case "no", "nn":
            return .norwegian
        default:
            return NLLanguage(rawValue: code)
        }
    }

    /// Everyday sentences, about nothing in particular, that each block's
    /// center is the mean of. A contextual block takes the ones in its model's
    /// languages; an English sentence block takes the English ones.
    private static func referenceSentences(for block: Block) -> [(language: NLLanguage, text: String)] {
        let matching = referenceSentences.filter { block.referenceLanguages.contains($0.language) }
        return Array(matching.prefix(referenceSentencesPerBlock))
    }

    private static let referenceSentences: [(language: NLLanguage, text: String)] = [
        // Latin script, one per language before the English ones, so a Latin
        // contextual block averages over many languages.
        (.french, "Le musée ouvre à neuf heures en semaine."),
        (.german, "Der Zug hatte zwanzig Minuten Verspätung."),
        (.spanish, "El equipo celebró después de ganar la final."),
        (.italian, "La connessione a internet continua a cadere."),
        (.portuguese, "A receita pede duas xícaras de farinha."),
        (.dutch, "We hebben de woonkamer lichtgroen geverfd."),
        (.polish, "Spotkanie przeniesiono na czwartek po południu."),
        (.turkish, "Köşede yeni bir kafe açıldı."),
        (.vietnamese, "Sáng nay trời lạnh và nhiều gió."),
        (.indonesian, "Dia membaca beberapa halaman sebelum tidur."),
        (.swedish, "Priserna steg kraftigt under förra kvartalet."),
        (.english, "The weather was cold and windy this morning."),
        (.english, "She reads a few pages before going to sleep."),
        (.english, "Prices rose sharply over the last quarter."),
        (.english, "He left his keys at the office again."),
        (.english, "The museum opens at nine on weekdays."),
        (.english, "We watched a documentary about whales."),
        (.english, "Please send the invoice by the end of the week."),
        (.english, "The train was twenty minutes late."),
        (.english, "They painted the living room a pale green."),
        (.english, "The team celebrated after winning the final."),
        (.english, "A new café opened around the corner."),
        (.english, "The meeting moved to Thursday afternoon."),
        // Chinese, Japanese, Korean
        (.simplifiedChinese, "今天早上天气又冷又有风。"),
        (.traditionalChinese, "博物館平日九點開門。"),
        (.japanese, "電車が二十分遅れた。"),
        (.korean, "모퉁이에 새 카페가 문을 열었다."),
        (.simplifiedChinese, "她睡觉前会看几页书。"),
        (.traditionalChinese, "我們看了一部關於鯨魚的紀錄片。"),
        (.japanese, "彼らはリビングを薄い緑色に塗った。"),
        (.korean, "회의가 목요일 오후로 옮겨졌다."),
        (.simplifiedChinese, "上个季度价格大幅上涨。"),
        (.traditionalChinese, "會議改到星期四下午。"),
        (.japanese, "チームは決勝で勝って祝った。"),
        (.korean, "그는 기타를 배우고 있다."),
        // Cyrillic
        (.russian, "Сегодня утром было холодно и ветрено."),
        (.ukrainian, "Команда святкувала перемогу у фіналі."),
        (.bulgarian, "Срещата беше преместена за четвъртък следобед."),
        (.russian, "Музей открывается в девять часов по будням."),
        (.ukrainian, "На розі відкрилося нове кафе."),
        (.russian, "Поезд опоздал на двадцать минут."),
        // Arabic
        (.arabic, "كان الطقس باردًا وعاصفًا هذا الصباح."),
        (.arabic, "يفتح المتحف أبوابه في التاسعة خلال أيام الأسبوع."),
        (.arabic, "تأخر القطار عشرين دقيقة."),
        (.arabic, "احتفل الفريق بعد الفوز في المباراة النهائية."),
        (.arabic, "افتتح مقهى جديد عند الزاوية."),
        (.arabic, "تم تأجيل الاجتماع إلى بعد ظهر يوم الخميس."),
        // Indic scripts (and Urdu, which the same model reads)
        (.hindi, "आज सुबह मौसम ठंडा और हवादार था।"),
        (.bengali, "ট্রেনটি বিশ মিনিট দেরিতে এসেছিল।"),
        (.tamil, "அணி இறுதிப் போட்டியில் வென்ற பிறகு கொண்டாடியது."),
        (.telugu, "మూలలో ఒక కొత్త కేఫ్ తెరిచారు."),
        (.marathi, "बैठक गुरुवारी दुपारी हलवण्यात आली."),
        (.urdu, "وہ سونے سے پہلے چند صفحات پڑھتی ہے۔"),
        (.gujarati, "ગયા ત્રિમાસિકમાં ભાવ ઝડપથી વધ્યા."),
        (.hindi, "संग्रहालय सप्ताह के दिनों में नौ बजे खुलता है।"),
        // Thai
        (.thai, "เช้านี้อากาศหนาวและมีลมแรง"),
        (.thai, "พิพิธภัณฑ์เปิดเวลาเก้าโมงในวันธรรมดา"),
        (.thai, "รถไฟมาช้ายี่สิบนาที"),
        (.thai, "ทีมฉลองหลังจากชนะในรอบชิงชนะเลิศ"),
        (.thai, "มีร้านกาแฟใหม่เปิดที่หัวมุม"),
        (.thai, "การประชุมถูกเลื่อนไปเป็นบ่ายวันพฤหัสบดี"),
    ]
}

// MARK: - Block

/// One script's slice of the vector: which model fills it, where it sits,
/// and the center subtracted from it.
private struct Block {
    enum Model {
        case sentence(NLEmbedding)
        case contextual(NLContextualEmbedding)
    }

    /// The contextual model identifier of the block's script, which is what
    /// texts are routed by (a sentence block keeps its script's key).
    let routingKey: String
    let languages: [NLLanguage]
    let model: Model
    let offset: Int
    /// Computed on first use, from `RAGNaturalLanguageEmbedder`'s reference
    /// sentences.
    var center: [Double]?

    init(routingKey: String, languages: [NLLanguage], model: Model, offset: Int) {
        self.routingKey = routingKey
        self.languages = languages
        self.model = model
        self.offset = offset
    }

    var dimension: Int {
        switch model {
        case .sentence(let embedding): return embedding.dimension
        case .contextual(let model): return model.dimension
        }
    }

    var needsAssets: Bool {
        guard case .contextual(let model) = model else { return false }
        return !model.hasAvailableAssets
    }

    /// Languages the model itself knows, which pick its reference sentences
    /// and which are worth passing to it as a hint.
    var referenceLanguages: Set<NLLanguage> {
        switch model {
        case .sentence: return [.english]
        case .contextual(let model): return Set(model.languages)
        }
    }

    func reads(_ language: NLLanguage) -> Bool {
        referenceLanguages.contains(language)
    }

    /// This block's part of the space identifier: model kind, identity, and
    /// revision.
    var spaceComponent: String {
        switch model {
        case .sentence(let embedding):
            return "sentence-en-r\(embedding.revision)"
        case .contextual(let model):
            return "contextual-\(model.modelIdentifier)-r\(model.revision)"
        }
    }
}
