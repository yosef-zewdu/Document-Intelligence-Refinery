"""
Optimized PageIndex Builder with batching and caching
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.agents.indexer import (
    PageIndexBuilder as BasePageIndexBuilder,
    SummaryProvider as BaseSummaryProvider,
    _norm, _extract_entities, _data_types_in_chunks, _snippet
)
from src.models import LDU, SectionNode, PageIndex

logger = logging.getLogger(__name__)


class OptimizedSummaryProvider(BaseSummaryProvider):
    """
    Enhanced SummaryProvider with caching and batch processing.
    """
    
    def __init__(self, use_llm: bool = False, model_name: str = None, cache_path: str = None):
        super().__init__(use_llm=use_llm, model_name=model_name)
        self.cache_path = cache_path or ".refinery/summary_cache.json"
        self.cache = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _load_cache(self) -> Dict[str, str]:
        """Load summary cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save summary cache to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _cache_key(self, section_title: str, section_text: str, data_types: List[str]) -> str:
        """Generate cache key for a summary request."""
        import hashlib
        content = f"{section_title}|{_snippet(section_text, 500)}|{','.join(sorted(data_types))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def summarize(self, section_title: str, section_text: str, data_types: List[str]) -> str:
        """Summarize with caching."""
        # Check cache first
        cache_key = self._cache_key(section_title, section_text, data_types)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        # Generate new summary
        self.cache_misses += 1
        summary = super().summarize(section_title, section_text, data_types)
        
        # Cache it
        self.cache[cache_key] = summary
        
        # Save cache periodically (every 10 misses)
        if self.cache_misses % 10 == 0:
            self._save_cache()
        
        return summary
    
    def finalize(self):
        """Save cache and print stats."""
        self._save_cache()
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_rate = (self.cache_hits / total) * 100
            logger.info(f"Cache stats: {self.cache_hits} hits, {self.cache_misses} misses ({hit_rate:.1f}% hit rate)")


class OptimizedPageIndexBuilder(BasePageIndexBuilder):
    """
    Enhanced PageIndexBuilder with parallel processing and progress tracking.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
        max_workers: int = 5,
        cache_summaries: bool = True,
    ):
        super().__init__(config=config, use_llm=use_llm)
        self.max_workers = max_workers
        
        # Replace summary provider with optimized version
        if cache_summaries:
            self.summary_provider = OptimizedSummaryProvider(use_llm=use_llm)
    
    def build(self, doc_id: str, ldus: List[LDU]) -> PageIndex:
        """
        Build PageIndex with parallel section processing.
        """
        # Step 1: Segment into sections
        sections = self._segment_into_sections(ldus)
        
        logger.info(f"Building {len(sections)} sections with {self.max_workers} workers...")
        
        # Step 2: Build section nodes in parallel
        if self.use_llm and self.max_workers > 1:
            child_nodes = self._build_sections_parallel(sections)
        else:
            child_nodes = [self._build_section_node(sec) for sec in sections]
        
        # Step 3: Build root node
        all_pages = set()
        for ldu in ldus:
            all_pages.update(ldu.page_refs)
        min_page = min(all_pages) if all_pages else 1
        max_page = max(all_pages) if all_pages else 1
        
        all_text = " ".join(_norm(ldu.content) for ldu in ldus if ldu.content)
        
        root = SectionNode(
            title="Document Root",
            page_start=min_page,
            page_end=max_page,
            summary=self.summary_provider.summarize(
                "Full Document", _snippet(all_text, 500), _data_types_in_chunks(ldus)
            ),
            data_types_present=_data_types_in_chunks(ldus),
            child_sections=child_nodes,
            key_entities=_extract_entities(all_text, max_per_type=8),
        )
        
        # Finalize (save cache, print stats)
        if hasattr(self.summary_provider, 'finalize'):
            self.summary_provider.finalize()
        
        return PageIndex(doc_id=doc_id, root=root)
    
    def _build_sections_parallel(self, sections: List[Dict[str, Any]]) -> List[SectionNode]:
        """Build section nodes in parallel using ThreadPoolExecutor."""
        nodes = []
        total = len(sections)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._build_section_node, sec): i 
                for i, sec in enumerate(sections)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    node = future.result()
                    nodes.append((idx, node))
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        logger.info(f"Progress: {completed}/{total} sections ({completed/total*100:.1f}%)")
                except Exception as e:
                    logger.error(f"Failed to build section {idx}: {e}")
                    # Create a fallback node
                    sec = sections[idx]
                    nodes.append((idx, SectionNode(
                        title=sec["title"],
                        page_start=1,
                        page_end=1,
                        summary=f"Error generating summary: {str(e)[:100]}",
                        data_types_present=[],
                        child_sections=[],
                        key_entities=[],
                    )))
        
        # Sort by original index to maintain order
        nodes.sort(key=lambda x: x[0])
        return [node for _, node in nodes]


def build_and_save_index_optimized(
    doc_id: str,
    ldus: List[LDU],
    output_dir: str = ".refinery/indices",
    use_llm: bool = False,
    max_workers: int = 5,
    cache_summaries: bool = True,
) -> PageIndex:
    """
    Build a PageIndex with optimizations and persist it to disk.
    
    Args:
        doc_id: Document identifier
        ldus: List of LDU chunks
        output_dir: Output directory for index
        use_llm: Whether to use LLM for summaries
        max_workers: Number of parallel workers for LLM calls
        cache_summaries: Whether to cache summaries
    """
    builder = OptimizedPageIndexBuilder(
        use_llm=use_llm,
        max_workers=max_workers,
        cache_summaries=cache_summaries,
    )
    index = builder.build(doc_id, ldus)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{doc_id}_page_index.json")
    with open(out_path, "w") as f:
        f.write(index.model_dump_json(indent=2))
    
    logger.info(f"PageIndex saved to {out_path}")
    return index
