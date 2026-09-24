// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "RAGKit",
    platforms: [
        // VecturaKit and VecturaEmbeddingsKit both require iOS 18 / macOS 15.
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(
            name: "RAGKit",
            targets: ["RAGKit"]
        ),
    ],
    dependencies: [
        // Pinned to the revisions the app shipped with before the extraction.
        // VecturaEmbeddingsKit 1.1.1 does not compile against its own resolved
        // dependencies, so stay on exact 1.1.0.
        .package(url: "https://github.com/rryam/VecturaKit", revision: "e5b7cd835b7197b4278d564d134e7bf8e37a6b8c"),
        .package(url: "https://github.com/rryam/VecturaEmbeddingsKit", exact: "1.1.0"),
        // RAGSentenceEmbedder calls swift-embeddings directly. Held below 0.1.0,
        // which changed the encode/batchEncode API out from under
        // VecturaEmbeddingsKit 1.1.0. Not held exact: 0.0.30 needs
        // swift-transformers 1.3.3 or later, and a host on WhisperKit 0.18
        // (swift-transformers 1.1.x) can only resolve 0.0.26.
        .package(url: "https://github.com/jkrukowski/swift-embeddings.git", "0.0.26"..<"0.1.0"),
        .package(url: "https://github.com/weichsel/ZIPFoundation", from: "0.9.20"),
    ],
    targets: [
        .target(
            name: "RAGKit",
            dependencies: [
                .product(name: "VecturaKit", package: "VecturaKit"),
                .product(name: "VecturaEmbeddingsKit", package: "VecturaEmbeddingsKit"),
                .product(name: "Embeddings", package: "swift-embeddings"),
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
            ],
            // Match the host app's Swift 5 language mode.
            swiftSettings: [.swiftLanguageMode(.v5)]
            // No resources: the host app bundles the embedding model folder and
            // any pre-embedded database seed, and passes their URLs in through
            // RAGVectorDatabaseConfiguration — mirroring how SupertonicTTS
            // receives its model directories.
        ),
    ]
)
