# Document-Intelligence-Refinery
# Document Intelligence Refinery

A Python-based document processing pipeline that automatically classifies documents, selects the optimal extraction strategy, and executes the extraction with high accuracy.

## Features

- **Intelligent Triage**: Automatically classifies documents as `NATIVE_DIGITAL` or `SCANNED_IMAGE`.
- **Layout Analysis**: Detects `SINGLE_COLUMN` vs `MULTI_COLUMN` layouts.
- **Domain Detection**: Identifies document domains (Financial, Legal, Technical, Medical, General).
- **Cost-Aware Extraction**: Selects the most efficient extraction strategy based on document characteristics.
- **Ledger System**: Tracks every extraction with strategy used, confidence, and processing time.

## 🏗️ Project Structure

```text
Document-Intelligence-Refinery/
├── .github/workflows/       # CI/CD pipelines
│   └── ci.yml               # GitHub Actions CI
├── .refinery/               # Output directory for profiles and indices
├── data/                    # Sample PDF documents for processing
├── rubric/                  # Configuration and rule-sets
│   └── extraction_rules.yaml # Logic for triage and escalation
├── src/                     # Core Source Code
│   ├── agents/              # Pipeline Stage Orchestrators
│   │   ├── triage.py        # Stage 1: Classifier
│   │   ├── extractor.py     # Stage 2: Router/Switch
│   │   ├── chunker.py       # Stage 3: Semantic Parser
│   │   ├── indexer.py       # Stage 3: PageIndex Builder
│   │   ├── vector_store.py  # Stage 3: Vector DB Manager
│   │   ├── fact_extractor.py # Stage 4: Structured Data Sink
│   │   └── query_agent.py   # Stage 4: LangGraph Interface
│   ├── models/              # Pydantic data contracts
│   │   ├── __init__.py
│   │   └── types.py
│   ├── strategies/          # Implementation-specific logic
│   │   ├── base.py          # Extractor interface
│   │   ├── fast_text.py     # Strategy A: pdfplumber
│   │   ├── layout_aware.py  # Strategy B: Docling
│   │   └── vision_augmented.py # Strategy C: Multimodal fallback
│   └── exploration.py       # Phase 0 domain analysis tool
├── tests/                   # Integration and unit tests
├── .env.example             # Template for API keys
├── .gitignore               # Standard Python ignores
├── pyproject.toml           # Project configuration & dependencies
├── uv.lock                  # UV lockfile
└── README.md                # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) installed (faster alternative to pip).
- Python 3.12+
- A Google API Key (for the `QueryAgent`).

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yosef-zewdu/Document-Intelligence-Refinery.git
   cd Document-Intelligence-Refinery
   ```

2. Setup environment variables:
   ```bash
   cp .env.example .env
   # Open .env and add your GOOGLE_API_KEY
   ```

3. Install dependencies using uv:
   ```bash
   uv sync --all-extras --dev
   ```

## 🧪 Usage

### Full Pipeline Run
To process the sample documents and build the indices:
```bash
PYTHONPATH=. uv run tests/test_full_refinery.py
```

### Domain Exploration
To analyze document characteristics like character density:
```bash
PYTHONPATH=. uv run src/exploration.py
```

## 🛠️ Tech Stack

- **Core**: Python, Pydantic, SQLAlchemy/SQLite.
- **Extraction**: Docling, pdfplumber.
- **Agentic Orchestration**: LangGraph, LangChain.
- **Indexing**: FAISS, Sentence-Transformers.
- **Quality Control**: Pytest.





