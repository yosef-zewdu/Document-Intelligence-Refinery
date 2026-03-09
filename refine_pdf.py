
"""
Refine PDF - End-to-End Document Intelligence Pipeline
Processes a PDF through: Triage → Extraction → Chunking → Indexing → Vector Store → Fact Table → Query

Usage:
    python refine_pdf.py <pdf_path> [--query "your question"]

Example:
    python refine_pdf.py data/CBE_ANNUAL_REPORT.pdf
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import agents
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder, build_and_save_index
from src.agents.vector_store import VectorStoreManager
from src.agents.fact_extractor import EnhancedFactTableExtractor
from src.agents.query_agent import QueryAgent
from src.models import ExtractedDocument, DocumentProfile

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_banner():
    banner = f"""
{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║   {Colors.MAGENTA}DOCUMENT INTELLIGENCE REFINERY{Colors.CYAN}                                             ║
║   {Colors.BLUE}End-to-End Pipeline - Stage 1 to Stage 6{Colors.CYAN}                                   ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝{Colors.END}
"""
    print(banner)

def print_stage_header(stage_num: int, title: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}[STAGE {stage_num}] {title}{Colors.END}")
    print(f"{Colors.BLUE}{'━' * 80}{Colors.END}")

def print_success(message: str):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_info(message: str):
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.END}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message: str):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def save_artifact(data: Any, path: Path, description: str):
    """Save data as JSON and log"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if hasattr(data, 'model_dump'):
            json.dump(data.model_dump(), f, indent=2, default=str)
        elif hasattr(data, 'model_dump_json'):
            f.write(data.model_dump_json(indent=2))
        else:
            json.dump(data, f, indent=2, default=str)
    print_info(f"Saved {description} to: {path}")

def main():
    parser = argparse.ArgumentParser(description="Process a PDF through the full Document Intelligence Refinery.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--query", help="Initial query to ask the document")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive query loop")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print_error(f"File not found: {pdf_path}")
        sys.exit(1)

    print_banner()
    
    # ─── 0. Initialization & Setup ──────────────────────────────────────────
    doc_name = pdf_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Store everything in a dedicated run directory for "refinery" preservation
    run_dir = Path(".refinery/runs") / f"{doc_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print_info(f"Processing: {Colors.BOLD}{pdf_path.name}{Colors.END}")
    print_info(f"Run Directory: {Colors.BOLD}{run_dir}{Colors.END}")
    start_time = time.time()

    # ─── 1. Triage ────────────────────────────────────────────────────────
    print_stage_header(1, "Document Triage")
    triage_agent = TriageAgent()
    profile = triage_agent.classify(str(pdf_path))
    
    print_info(f"Origin Type: {Colors.BOLD}{profile.origin_type}{Colors.END}")
    print_info(f"Layout Complexity: {Colors.BOLD}{profile.layout_complexity}{Colors.END}")
    print_info(f"Language: {Colors.BOLD}{profile.language}{Colors.END} ({profile.language_confidence:.2f})")
    print_info(f"Domain: {Colors.BOLD}{profile.domain_hint}{Colors.END}")
    print_info(f"Estimated Cost: {Colors.BOLD}{profile.estimated_cost}{Colors.END}")
    
    save_artifact(profile, run_dir / "profile.json", "Document Profile")
    print_success("Triage complete")

    # ─── 2. Extraction ───────────────────────────────────────────────────
    print_stage_header(2, "Intelligent Extraction")
    router = ExtractionRouter()
    
    extract_start = time.time()
    doc = router.route_and_extract(str(pdf_path), profile)
    extract_time = time.time() - extract_start
    
    print_info(f"Strategy Used: {Colors.BOLD}{doc.metadata.get('strategy', 'Unknown')}{Colors.END}")
    print_info(f"Confidence Score: {Colors.BOLD}{doc.confidence.score:.4f}{Colors.END}")
    print_info(f"Blocks: {len(doc.blocks)}, Tables: {len(doc.tables)}")
    print_info(f"Extraction Time: {extract_time:.2f}s")
    
    save_artifact(doc, run_dir / "extraction.json", "Extracted Document")
    print_success("Extraction complete")

    # ─── 3. Chunking ─────────────────────────────────────────────────────
    print_stage_header(3, "Semantic Chunking (LDUs)")
    chunker = ChunkingEngine(max_tokens=512)
    
    chunk_start = time.time()
    chunks = chunker.chunk(doc)
    chunk_time = time.time() - chunk_start
    
    print_info(f"Total LDUs Created: {Colors.BOLD}{len(chunks)}{Colors.END}")
    print_info(f"Chunking Time: {chunk_time:.2f}s")
    
    save_artifact([c.model_dump() for c in chunks], run_dir / "chunks.json", "Semantic Chunks")
    print_success("Chunking complete")

    # ─── 4. Page Indexing ────────────────────────────────────────────────
    print_stage_header(4, "Hierarchical Page Indexing")
    
    index_start = time.time()
    # build_and_save_index handles building and saving to .refinery/indices/
    # but we ALSO want it in our run directory
    page_index = build_and_save_index(doc_id=profile.doc_id, ldus=chunks)
    index_time = time.time() - index_start
    
    print_info(f"Sections Detected: {Colors.BOLD}{len(page_index.root.child_sections)}{Colors.END}")
    print_info(f"Page Range: {page_index.root.page_start} to {page_index.root.page_end}")
    print_info(f"Indexing Time: {index_time:.2f}s")
    
    save_artifact(page_index, run_dir / "page_index.json", "Page Index")
    print_success("Indexing complete")

    # ─── 5. Knowledge Enrichment (Vector Store & Fact Table) ─────────────
    print_stage_header(5, "Knowledge Enrichment")
    
    print_info("Building Vector Store (FAISS)...")
    vs_manager = VectorStoreManager(base_path=str(run_dir / "vectorstore"))
    vs_manager.ingest_ldus(profile.doc_id, chunks)
    
    print_info("Extracting Facts to SQL Table...")
    fact_db_path = run_dir / "facts.db"
    fact_extractor = EnhancedFactTableExtractor(db_path=str(fact_db_path))
    facts_count = fact_extractor.ingest_from_chunks(profile.doc_id, chunks)
    
    print_info(f"Facts Ingested: {Colors.BOLD}{facts_count}{Colors.END}")
    print_success("Knowledge enrichment complete")

    # ─── 6. Query Agent ──────────────────────────────────────────────────
    print_stage_header(6, "Query Agent Initialization")
    
    try:
        # Check if OpenRouter API key is present
        if not os.environ.get("OPENROUTER_API_KEY"):
            print_warning("OPENROUTER_API_KEY not found in environment.")
            print_warning("Query Agent will fall back to basic retrieval or fail.")
            
        agent = QueryAgent(
            doc_id=profile.doc_id,
            page_index_path=str(run_dir / "page_index.json")
        )
        
        # Override components to use run-specific data
        agent.vector_store = vs_manager
        agent.fact_extractor = fact_extractor
        
        print_success("Query Agent initialized and ready")
        
        # Handle initial query if provided
        if args.query:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}Query:{Colors.END} {args.query}")
            print(f"{Colors.YELLOW}{'-' * 80}{Colors.END}")
            
            print(f"{Colors.YELLOW}Thinking...{Colors.END}")
            result = agent.query(args.query)
            
            # Display answer
            answer = result.get('answer', '')
            if answer:
                print(f"\n{Colors.BOLD}{Colors.GREEN}Answer:{Colors.END}")
                print(f"{answer}\n")
            else:
                print(f"\n{Colors.YELLOW}No answer generated{Colors.END}\n")
                # Debug info
                print(f"{Colors.YELLOW}Result keys: {list(result.keys())}{Colors.END}")
                if result.get('messages'):
                    print(f"{Colors.YELLOW}Last 2 messages:{Colors.END}")
                    for msg in result['messages'][-2:]:
                        print(f"  {str(msg)[:150]}...")
            
            # Display provenance
            provenance = result.get('provenance', [])
            if provenance:
                print(f"{Colors.BOLD}{Colors.BLUE}Sources:{Colors.END}")
                for i, p in enumerate(provenance[:5], 1):
                    page_num = p.get('page_number', '?')
                    doc_name = p.get('document_name', 'Unknown')
                    content_hash = p.get('content_hash', '')
                    
                    print(f"  {i}. Page {page_num} - {doc_name}")
                    if content_hash:
                        print(f"     Hash: {content_hash[:20]}...")
                    
                    if p.get('bbox'):
                        bbox = p['bbox']
                        print(f"     BBox: ({bbox.get('x0', 0):.0f}, {bbox.get('y0', 0):.0f}) → ({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f})")
                print()
            else:
                print(f"{Colors.YELLOW}No sources found{Colors.END}\n")
            
            # Show iterations
            iterations = result.get('iterations', 0)
            if iterations > 0:
                print(f"{Colors.CYAN}Tool calls: {iterations}{Colors.END}\n")

        # Interactive loop
        if not args.no_interactive:
            print(f"\n{Colors.BOLD}{Colors.MAGENTA}━━━ Interactive Query Mode (type 'exit' or 'quit' to stop) ━━━{Colors.END}")
            while True:
                try:
                    user_query = input(f"\n{Colors.BOLD}{Colors.CYAN}Ask a question > {Colors.END}").strip()
                    if user_query.lower() in ['exit', 'quit']:
                        break
                    if not user_query:
                        continue
                        
                    print(f"{Colors.YELLOW}Thinking...{Colors.END}")
                    result = agent.query(user_query)
                    
                    # Debug: Show raw result structure
                    if not result.get('answer') or result.get('answer', '').strip() == '':
                        print(f"\n{Colors.YELLOW}━━━ DEBUG INFO ━━━{Colors.END}")
                        print(f"{Colors.YELLOW}Result keys: {list(result.keys())}{Colors.END}")
                        
                        # Show messages if available
                        if result.get('messages'):
                            print(f"{Colors.YELLOW}Messages in result:{Colors.END}")
                            for i, msg in enumerate(result['messages'][-3:], 1):
                                print(f"  {i}. {msg[:150]}...")
                        
                        print(f"{Colors.YELLOW}━━━━━━━━━━━━━━━{Colors.END}\n")
                    
                    # Display answer
                    answer = result.get('answer', '')
                    if answer:
                        print(f"\n{Colors.BOLD}{Colors.GREEN}Answer:{Colors.END}")
                        print(f"{answer}\n")
                    else:
                        print(f"\n{Colors.YELLOW}No answer generated (check if LLM returned empty response){Colors.END}\n")
                    
                    # Display provenance with details
                    provenance = result.get('provenance', [])
                    if provenance:
                        print(f"{Colors.BOLD}{Colors.BLUE}Sources:{Colors.END}")
                        for i, p in enumerate(provenance[:5], 1):
                            page_num = p.get('page_number', '?')
                            doc_name = p.get('document_name', 'Unknown')
                            content_hash = p.get('content_hash', '')
                            
                            print(f"  {i}. Page {page_num} - {doc_name}")
                            if content_hash:
                                print(f"     Hash: {content_hash[:20]}...")
                            
                            # Show bbox if available
                            if p.get('bbox'):
                                bbox = p['bbox']
                                print(f"     BBox: ({bbox.get('x0', 0):.0f}, {bbox.get('y0', 0):.0f}) → ({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f})")
                        print()
                    else:
                        print(f"{Colors.YELLOW}No sources found{Colors.END}\n")
                    
                    # Show iterations
                    iterations = result.get('iterations', 0)
                    if iterations > 0:
                        print(f"{Colors.CYAN}Tool calls: {iterations}{Colors.END}\n")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print_error(f"Query failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
    except Exception as e:
        print_error(f"Query Agent setup failed: {e}")

    # ─── Final Summary ───────────────────────────────────────────────────
    total_time = time.time() - start_time
    print_stage_header(7, "Process Complete")
    print_info(f"Total Document Processing Time: {Colors.BOLD}{total_time:.2f}s{Colors.END}")
    print_info(f"All 'refineries' preserved in: {Colors.UNDERLINE}{run_dir}{Colors.END}")
    print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Document Intelligence Refinery Process Finished Successfully!{Colors.END}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Process interrupted by user.{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
