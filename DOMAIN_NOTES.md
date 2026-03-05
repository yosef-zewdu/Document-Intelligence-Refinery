# DOMAIN_NOTES.md: Document Intelligence Refinery

This document outlines the conceptual foundation and technical strategy for the Document Intelligence Refinery, focusing on the transition from raw document extraction to structured knowledge.

## Extraction Strategy Decision Tree

The system uses a tiered approach to balance cost and quality:

```mermaid
graph TD
    A[Input Document] --> B[Triage Agent]
    B --> C{Origin Type?}
    C -- Native Digital --> D{Layout Complexity?}
    C -- Scanned/Mixed --> E[Strategy C: Vision-Augmented]
    D -- Simple/Single Column --> F[Strategy A: Fast Text]
    D -- Complex/Multi-Column --> G[Strategy B: Layout-Aware]
    F -- Low Confidence --> G
    G -- Low Confidence --> E
```

### Strategy Tiers
1.  **Strategy A (Fast Text)**: Uses `pdfplumber` for character stream extraction. Fast and cheap. Ideal for non-complex digital PDFs.
2.  **Strategy B (Layout-Aware)**: Uses `Docling` or `MinerU`. Recovers structure (tables, columns, reading order). Moderate cost/latency.
3.  **Strategy C (Vision-Augmented)**: Uses Multimodal LLMs (Gemini Pro Vision, etc.). High fidelity for scanned or extremely complex layouts. Expensive.

---

## Failure Modes & Mitigation

| Failure Mode | Cause | Mitigation |
| :--- | :--- | :--- |
| **Structure Collapse** | Naive OCR flattening multi-column text. | Use Layout-Aware (Docling/MinerU) or Vision-Augmented extraction. |
| **Context Poverty** | Random chunking breaking table rows/captions. | Implement Semantic Chunking (LDU) rules. |
| **Provenance Blindness** | Loss of spatial coordinates during parsing. | Track bounding boxes (BBox) and page refs in every LDU. |
| **Hallucination** | LLM reasoning on corrupted/noisy OCR text. | Escalation Guard: Trigger Vision Extraction if OCR confidence is low. |

---

## Pipeline Architecture (The Refinery)

```mermaid
sequenceDiagram
    participant Docs as Document Corpus
    participant Triage as Triage Agent
    participant Extractor as Extraction Engine (A/B/C)
    participant Chunker as Semantic Chunking Engine
    participant Indexer as PageIndex Builder
    participant Query as Query Interface Agent

    Docs->>Triage: Ingest PDF
    Triage->>Triage: Classify (Origin, Layout, Domain)
    Triage->>Extractor: DocumentProfile
    Extractor->>Extractor: Select Strategy & Extract
    Extractor->>Chunker: ExtractedDocument (BBoxes + Tables)
    Chunker->>Chunker: Apply LDU Rules
    Chunker->>Indexer: Metadata & LDUs
    Indexer->>Indexer: Build Section Tree & Summaries
    Indexer->>Query: PageIndex + Vector DB
    Query->>Query: Multi-Tool Retrieval (Navigate/Search/SQL)
```

