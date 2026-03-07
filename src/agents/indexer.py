"""
Stage 4 – PageIndex Builder
============================
Builds a hierarchical section tree (PageIndex) from chunked LDUs and
generates LLM summaries for each section node.

Key capabilities:
  1. Build a tree of SectionNodes from heading-tagged LDUs
  2. Generate 2-3 sentence summaries per section using a cheap LLM
  3. Extract key entities per section via simple NER heuristics
  4. Catalog data_types_present (tables, figures, lists) per section
  5. Query the tree: given a topic, return the top-3 most relevant sections
"""

import os
import re
import json
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

from src.models import LDU, SectionNode, PageIndex
from src.utils.config_loader import load_refinery_config
from src.llm_factory import get_llm_config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────

_WS = re.compile(r"\s+")

# Simple NER-like patterns for key entities
_MONEY_RE = re.compile(
    r"(?:(?:ETB|USD|EUR|GBP|Birr|\$|€|£)\s*[\d,.]+(?:\s*(?:million|billion|mn|bn|m|b))?)"
    r"|(?:[\d,.]+\s*(?:million|billion|mn|bn|m|b)\s*(?:ETB|USD|EUR|GBP|Birr)?)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?%")
_DATE_RE = re.compile(
    r"\b(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}/\d{2}/\d{2}|FY\s*\d{2,4}(?:/\d{2,4})?)\b",
    re.IGNORECASE,
)
_ORG_RE = re.compile(
    r"\b(?:CBE|NBE|IMF|World Bank|Ministry of Finance|Federal Democratic Republic|"
    r"National Bank of Ethiopia|Commercial Bank of Ethiopia)\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())

ROMAN_LIST_RE = re.compile(r"^\s*[ivxlcdm]+[\)\.]\s+", re.IGNORECASE)
LISTLIKE_RE   = re.compile(r"^\s*(?:\d+(?:\.\d+)*[\)\.]|[•\-])\s+")
MOTTO_RE      = re.compile(r".+!\s*$")  # short slogan lines often end with !


NUM_LEVEL_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")

def _heading_level(title: str) -> int:
    t = _norm(title)
    m = NUM_LEVEL_RE.match(t)
    if m:
        return len(m.group(1).split("."))
    # ALL CAPS short headings -> level 1
    alpha = sum(c.isalpha() for c in t)
    if alpha >= 6:
        upper = sum(c.isupper() for c in t if c.isalpha())
        if upper / alpha > 0.88 and len(t.split()) <= 10:
            return 1
    return 2


def _is_valid_section_heading(title: str) -> bool:
    t = _norm(title)
    if not t:
        return False
    # reject list items pretending to be headings
    if ROMAN_LIST_RE.match(t) or LISTLIKE_RE.match(t):
        return False
    # reject motto-like short exclamations
    if MOTTO_RE.match(t) and len(t.split()) <= 12:
        return False
    return True

def _extract_entities(text: str, max_per_type: int = 5) -> List[str]:
    """Simple pattern-based entity extraction."""
    entities = set()
    for m in _MONEY_RE.finditer(text):
        entities.add(m.group(0).strip())
    for m in _PERCENT_RE.finditer(text):
        entities.add(m.group(0).strip())
    for m in _DATE_RE.finditer(text):
        entities.add(m.group(0).strip())
    for m in _ORG_RE.finditer(text):
        entities.add(m.group(0).strip())
    # Limit per-type to avoid noise
    return sorted(entities)[:max_per_type * 4]


def _data_types_in_chunks(chunks: List[LDU]) -> List[str]:
    """Return which data types (tables, figures, lists, text, headings) are present."""
    types = set()
    for c in chunks:
        ct = c.chunk_type
        if ct == "table":
            types.add("tables")
        elif ct == "figure":
            types.add("figures")
        elif ct == "list":
            types.add("lists")
        elif ct == "heading":
            types.add("headings")
        elif ct == "text":
            types.add("text")
    return sorted(types)


def _snippet(text: str, max_chars: int = 800) -> str:
    """Truncate text for LLM prompts to save tokens."""
    t = _norm(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "..."


# ──────────────────────────────────────────────────────
# LLM Summary Provider (pluggable)
# ──────────────────────────────────────────────────────

class SummaryProvider:
    """
    Generates 2-3 sentence summaries.
    
    Supports multiple LLM backends via llm_factory:
      - OpenRouter: uses OPENROUTER_API_KEY + OPENROUTER_MODEL from .env
      - Gemini: uses GOOGLE_API_KEY + GEMINI_MODEL from .env
      - HuggingFace: uses HF_TOKEN + HF_MODEL from .env
    
    Falls back to a heuristic extractive summary if no API key is available.
    """

    def __init__(self, use_llm: bool = False, model_name: str = None):
        self.use_llm = use_llm
        
        # Get LLM configuration from factory
        self.llm_config = get_llm_config()
        self.provider = self.llm_config["provider"]
        self.model_name = model_name or self.llm_config["model"]
        self.api_key = self.llm_config["api_key"]
        
        # Check if LLM is actually available
        if self.use_llm and not self.llm_config["available"]:
            logger.warning(
                f"{self.provider.upper()}_API_KEY not set, falling back to extractive summary."
            )
            self.use_llm = False

    def summarize(self, section_title: str, section_text: str, data_types: List[str]) -> str:
        if self.use_llm:
            return self._llm_summary(section_title, section_text, data_types)
        return self._extractive_summary(section_title, section_text, data_types)

    def _extractive_summary(self, title: str, text: str, data_types: List[str]) -> str:
        """
        Fast, free, deterministic summary.
        Takes the first 2-3 sentences and data-type info.
        """
        sentences = re.split(r'(?<=[.!?])\s+', _norm(text))
        # Filter out very short noise
        sentences = [s for s in sentences if len(s) > 20]
        lead = ". ".join(sentences[:3])
        if not lead:
            lead = _snippet(text, 200)

        dt_note = ""
        if data_types:
            dt_note = f" Contains: {', '.join(data_types)}."

        return f"Section '{title}': {lead}.{dt_note}"

    def _llm_summary(self, title: str, text: str, data_types: List[str]) -> str:
        """
        Route to the correct LLM backend based on LLM_PROVIDER env var.
        """
        if self.provider == "openrouter":
            return self._openrouter_summary(title, text, data_types)
        elif self.provider == "ollama":
            return self._ollama_summary(title, text, data_types)
        elif self.provider == "gemini":
            return self._gemini_summary(title, text, data_types)
        elif self.provider == "huggingface":
            return self._hf_summary(title, text, data_types)
        else:
            logger.warning(f"Unknown provider '{self.provider}', falling back to extractive.")
            return self._extractive_summary(title, text, data_types)

    def _build_prompt_messages(self, title: str, text: str, data_types: List[str]) -> List[Dict[str, str]]:
        """Build the chat messages for summary generation."""
        dt_str = ", ".join(data_types) if data_types else "text"
        return [
            {
                "role": "system",
                "content": (
                    "You are a precise document summarizer. "
                    "Given a document section, produce exactly 2-3 sentences "
                    "that capture the key information. Be factual and concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarize this document section in exactly 2-3 sentences.\n\n"
                    f"Section title: {title}\n"
                    f"Data types present: {dt_str}\n\n"
                    f"Content:\n{_snippet(text, 1500)}\n\n"
                    f"Summary:"
                ),
            },
        ]

    # ── OpenRouter backend ──

    def _openrouter_summary(self, title: str, text: str, data_types: List[str]) -> str:
        """Call OpenRouter API (OpenAI-compatible chat completions)."""
        import requests

        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set, falling back to extractive summary.")
            return self._extractive_summary(title, text, data_types)

        messages = self._build_prompt_messages(title, text, data_types)

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/DocumentIntelligenceRefinery",
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 500,  # Increased from 200 to avoid truncation
                    "temperature": 0.3,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return _norm(content)

        except Exception as e:
            logger.error(f"OpenRouter summary failed: {e}. Falling back to extractive.")
            return self._extractive_summary(title, text, data_types)

    # ── Ollama backend ──

    def _ollama_summary(self, title: str, text: str, data_types: List[str]) -> str:
        """Call Ollama local LLM API (OpenAI-compatible)."""
        import requests
        import re
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        messages = self._build_prompt_messages(title, text, data_types)

        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
                timeout=60,  # Local LLM might be slower
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            
            # Clean up <think> tags from reasoning models like deepcoder
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            
            return _norm(content)

        except Exception as e:
            logger.error(f"Ollama summary failed: {e}. Falling back to extractive.")
            logger.error(f"Make sure Ollama is running: ollama serve")
            logger.error(f"And model is pulled: ollama pull {self.model_name}")
            return self._extractive_summary(title, text, data_types)

    # ── Gemini backend ──

    def _gemini_summary(self, title: str, text: str, data_types: List[str]) -> str:
        """Call Google Gemini API via LangChain."""
        try:
            from src.llm_factory import get_llm
            
            if not self.api_key:
                logger.warning("GOOGLE_API_KEY not set, falling back to extractive summary.")
                return self._extractive_summary(title, text, data_types)

            llm = get_llm(provider="gemini", model=self.model_name, api_key=self.api_key)
            messages = self._build_prompt_messages(title, text, data_types)
            
            # Convert to LangChain message format
            from langchain_core.messages import SystemMessage, HumanMessage
            lc_messages = [
                SystemMessage(content=messages[0]["content"]),
                HumanMessage(content=messages[1]["content"]),
            ]
            
            # Gemini uses max_output_tokens parameter
            response = llm.invoke(lc_messages, max_output_tokens=500)
            return _norm(response.content)

        except Exception as e:
            logger.error(f"Gemini summary failed: {e}. Falling back to extractive.")
            return self._extractive_summary(title, text, data_types)

    # ── HuggingFace backend ──

    def _hf_summary(self, title: str, text: str, data_types: List[str]) -> str:
        """Call HuggingFace Inference API for summary."""
        try:
            from huggingface_hub import InferenceClient
            
            if not self.api_key:
                logger.warning("HF_TOKEN not set, falling back to extractive summary.")
                return self._extractive_summary(title, text, data_types)

            client = InferenceClient(api_key=self.api_key)
            dt_str = ", ".join(data_types) if data_types else "text"

            prompt = (
                f"Summarize this document section in exactly 2-3 sentences.\n"
                f"Section title: {title}\n"
                f"Data types present: {dt_str}\n"
                f"Content:\n{_snippet(text, 1500)}\n\n"
                f"Summary:"
            )

            response = client.text_generation(
                model=self.model_name,
                prompt=prompt,
                max_new_tokens=500,  # Increased from 150
                temperature=0.3,
            )
            return _norm(response)

        except Exception as e:
            logger.error(f"HF summary failed: {e}. Falling back to extractive.")
            return self._extractive_summary(title, text, data_types)


# ──────────────────────────────────────────────────────
# PageIndex Builder
# ──────────────────────────────────────────────────────

class PageIndexBuilder:
    """
    Builds a hierarchical PageIndex from a list of LDUs.

    The tree structure is derived from:
      1. heading-type LDUs (act as section openers)
      2. parent_section metadata on child chunks
      3. Page boundaries (fallback grouping)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
    ):
        if not config:
            full_config = load_refinery_config()
            self.config = full_config.get("indexer", {})
        else:
            self.config = config

        self.use_llm = use_llm
        self.summary_provider = SummaryProvider(use_llm=use_llm)

    def _finalize_node(
        self,
        node: SectionNode,
        node_text: Dict[int, List[str]],
        node_entity_text: Dict[int, List[str]],
        node_types: Dict[int, set],
    ) -> None:
        txt = " ".join(node_text.get(id(node), []))
        types = node_types.get(id(node), set())

        # compute data_types_present without making fake LDUs
        dt = set()
        for t in types:
            if t == "table": dt.add("tables")
            elif t == "figure": dt.add("figures")
            elif t == "list": dt.add("lists")
            elif t == "heading": dt.add("headings")
            elif t == "text": dt.add("text")
        node.data_types_present = sorted(dt)

        if txt:
            node.summary = self.summary_provider.summarize(
                node.title, _snippet(txt, 3000), node.data_types_present
            )

        ent_src = " ".join(node_entity_text.get(id(node), []))
        if ent_src:
            node.key_entities = _extract_entities(ent_src, max_per_type=8)

        for ch in node.child_sections:
            self._finalize_node(ch, node_text, node_entity_text, node_types)

    def build(self, doc_id: str, ldus: List[LDU]) -> PageIndex:
        """
        Main entry point: build a PageIndex tree from LDUs.

        Strategy:
          - Scan LDUs for heading chunks → section openers
          - Group non-heading LDUs under the most recent heading
          - Build SectionNodes with summaries, entities, data_types
          - Wrap in a root node
        """

        ldus = sorted(ldus, key=lambda l: (min(l.page_refs) if l.page_refs else 10**9))
        
        # 1) root node
        all_pages = set()
        for ldu in ldus:
            all_pages.update(ldu.page_refs)
        min_page = min(all_pages) if all_pages else 1
        max_page = max(all_pages) if all_pages else 1

        root = SectionNode(
            title="Document Root",
            page_start=min_page,
            page_end=max_page,
            summary=None,
            data_types_present=[],
            child_sections=[],
            key_entities=[]
        )

        # 2) stack of (level, node)
        stack: List[Tuple[int, SectionNode]] = [(0, root)]
        current_node = root

        # 3) buffers for each node
        node_text = defaultdict(list)
        node_entity_text = defaultdict(list)
        node_types = defaultdict(set)

        # 4) scan LDUs and build hierarchy
        for ldu in ldus:
            if ldu.chunk_type == "heading" and _is_valid_section_heading(ldu.content):
                lvl = _heading_level(ldu.content)

                node = SectionNode(
                    title=_norm(ldu.content),
                    page_start=min(ldu.page_refs) if ldu.page_refs else min_page,
                    page_end=max(ldu.page_refs) if ldu.page_refs else min_page,
                    summary=None,
                    data_types_present=[],
                    child_sections=[],
                    key_entities=[]
                )

                # pop until parent level < lvl
                while stack and stack[-1][0] >= lvl:
                    stack.pop()

                parent = stack[-1][1] if stack else root

                # merge repeated consecutive headings (optional but very effective)
                if parent.child_sections and _norm(parent.child_sections[-1].title).lower() == _norm(node.title).lower():
                    parent.child_sections[-1].page_end = max(parent.child_sections[-1].page_end, node.page_end)
                    current_node = parent.child_sections[-1]
                    stack.append((lvl, current_node))
                    continue

                parent.child_sections.append(node)
                stack.append((lvl, node))
                current_node = node
                continue

            # assign non-heading LDU to current section
            if ldu.page_refs:
                current_node.page_start = min(current_node.page_start, min(ldu.page_refs))
                current_node.page_end = max(current_node.page_end, max(ldu.page_refs))

            node_types[id(current_node)].add(ldu.chunk_type)

            if ldu.chunk_type in ("text", "list", "table"):
                node_text[id(current_node)].append(_norm(ldu.content))
            if ldu.chunk_type in ("text", "list"):
                node_entity_text[id(current_node)].append(_norm(ldu.content))
            if ldu.chunk_type == "figure":
                cap = (ldu.metadata.get("caption") or "")
                if cap:
                    node_text[id(current_node)].append(_norm(cap))
                    node_entity_text[id(current_node)].append(_norm(cap))

        # 5) finalize summaries/entities/types recursively
        
        self._finalize_node(root, node_text, node_entity_text, node_types)

        # 6) root summary/entities from whole doc (capped)
        all_text = " ".join(_norm(ldu.content) for ldu in ldus if ldu.content)
        root.summary = self.summary_provider.summarize(
            "Full Document", _snippet(all_text, 1500), _data_types_in_chunks(ldus)
        )
        root.data_types_present = _data_types_in_chunks(ldus)
        root.key_entities = _extract_entities(all_text, max_per_type=8)

        return PageIndex(doc_id=doc_id, root=root)

    # ── Section segmentation ──

    def _segment_into_sections(self, ldus: List[LDU]) -> List[Dict[str, Any]]:
        """
        Segments LDUs into sections using headings as boundaries.
        Each section is a dict with 'title', 'heading_ldu', 'children'.
        If no headings exist, creates page-range based sections.
        """
        heading_indices = [
            i for i, ldu in enumerate(ldus)
            if ldu.chunk_type == "heading" and _is_valid_section_heading(ldu.content)
        ]
        if not heading_indices:
            # Fallback: create sections by page ranges (every 5 pages)
            return self._segment_by_pages(ldus, pages_per_section=5)

        sections = []
        for idx, hi in enumerate(heading_indices):
            heading_ldu = ldus[hi]
            # Children are all LDUs between this heading and the next heading
            next_hi = heading_indices[idx + 1] if idx + 1 < len(heading_indices) else len(ldus)
            children = ldus[hi + 1: next_hi]

            sections.append({
                "title": _norm(heading_ldu.content),
                "heading_ldu": heading_ldu,
                "children": children,
            })

        # Handle any LDUs BEFORE the first heading (preamble)
        if heading_indices[0] > 0:
            preamble = ldus[:heading_indices[0]]
            sections.insert(0, {
                "title": "Preamble",
                "heading_ldu": None,
                "children": preamble,
            })

        return sections

    def _segment_by_pages(self, ldus: List[LDU], pages_per_section: int = 5) -> List[Dict[str, Any]]:
        """Fallback segmentation by page ranges."""
        all_pages = set()
        for ldu in ldus:
            all_pages.update(ldu.page_refs)
        if not all_pages:
            return [{"title": "Full Document", "heading_ldu": None, "children": ldus}]

        sorted_pages = sorted(all_pages)
        sections = []
        for i in range(0, len(sorted_pages), pages_per_section):
            page_range = sorted_pages[i:i + pages_per_section]
            p_start, p_end = page_range[0], page_range[-1]
            children = [
                ldu for ldu in ldus
                if any(p in range(p_start, p_end + 1) for p in ldu.page_refs)
            ]
            sections.append({
                "title": f"Pages {p_start}–{p_end}",
                "heading_ldu": None,
                "children": children,
            })

        return sections

    # ── Node construction ──

    def _build_section_node(self, sec: Dict[str, Any]) -> SectionNode:
        """Build a SectionNode from a section dict."""
        title = sec["title"]
        children = sec["children"]

        # Determine page range
        pages = set()
        heading = sec.get("heading_ldu")
        if heading:
            pages.update(heading.page_refs)
        for c in children:
            pages.update(c.page_refs)

        page_start = min(pages) if pages else 1
        page_end = max(pages) if pages else 1

        # Aggregate content for summary and entity extraction
        all_content_parts = []
        if heading:
            all_content_parts.append(_norm(heading.content))
        for c in children:
            all_content_parts.append(_norm(c.content))
        all_text = " ".join(all_content_parts)

        # Data types
        all_ldus = ([heading] if heading else []) + list(children)
        data_types = _data_types_in_chunks(all_ldus)

        # Entities
        entities = _extract_entities(all_text)

        # Summary
        summary = summary = self.summary_provider.summarize(title, _snippet(all_text, 3000), data_types)

        return SectionNode(
            title=title,
            page_start=page_start,
            page_end=page_end,
            summary=summary,
            data_types_present=data_types,
            child_sections=[],  # flat for now; could nest if sub-headings detected
            key_entities=entities,
        )


# ──────────────────────────────────────────────────────
# PageIndex Query Engine
# ──────────────────────────────────────────────────────

class PageIndexQuery:
    """
    Given a PageIndex tree and a topic string, traverse the tree
    and return the top-3 most relevant SectionNodes.

    Scoring uses a lightweight TF-based relevance measure.
    """

    def __init__(self, page_index: PageIndex):
        self.page_index = page_index
        self._all_sections = self._flatten(page_index.root)

    def _flatten(self, node: SectionNode) -> List[SectionNode]:
        """Flatten the tree into a list of all section nodes (excluding root if desired)."""
        result = []
        for child in node.child_sections:
            result.append(child)
            result.extend(self._flatten(child))
        return result

    def query(self, topic: str, top_k: int = 3) -> List[Tuple[SectionNode, float]]:
        """
        Return the top-k most relevant sections for the given topic.

        Scoring:
          1. Term overlap between topic tokens and section title/summary/entities
          2. Bonus for matching data types (e.g., "table" in topic → sections with tables)
          3. Entity match bonus
        """
        topic_tokens = set(self._tokenize(topic))
        if not topic_tokens:
            return []

        scored: List[Tuple[SectionNode, float]] = []

        for section in self._all_sections:
            score = self._score_section(section, topic_tokens, topic)
            scored.append((section, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _score_section(self, section: SectionNode, topic_tokens: set, raw_topic: str) -> float:
        """Compute relevance score for a section."""
        score = 0.0

        # 1. Title match (highest weight)
        title_tokens = set(self._tokenize(section.title))
        title_overlap = len(topic_tokens & title_tokens)
        score += title_overlap * 3.0

        # 2. Summary match
        summary_tokens = set(self._tokenize(section.summary or ""))
        summary_overlap = len(topic_tokens & summary_tokens)
        score += summary_overlap * 1.0

        # 3. Entity match (exact substring)
        for entity in section.key_entities:
            entity_lower = entity.lower()
            if entity_lower in raw_topic.lower():
                score += 2.0
            # Partial: topic token in entity
            for tok in topic_tokens:
                if tok in entity_lower:
                    score += 0.5

        # 4. Data type match
        topic_lower = raw_topic.lower()
        for dt in section.data_types_present:
            if dt.rstrip("s") in topic_lower:  # "table" in "tables"
                score += 1.5

        # 5. Penalize very short sections (likely noise)
        if not section.summary or len(section.summary) < 30:
            score *= 0.5

        return round(score, 4)

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercased word tokenizer, filtering stopwords."""
        stops = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "with", "by", "from", "as", "it",
            "that", "this", "be", "has", "have", "had", "do", "does", "did",
            "will", "would", "can", "could", "should", "may", "might", "shall",
            "not", "no", "but", "if", "what", "which", "who", "when", "where",
            "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "than", "too", "very", "just", "about",
        }
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in stops and len(w) > 1]


# ──────────────────────────────────────────────────────
# Convenience: build + save
# ──────────────────────────────────────────────────────

def build_and_save_index(
    doc_id: str,
    ldus: List[LDU],
    output_dir: str = ".refinery/indices",
    use_llm: bool = False,
) -> PageIndex:
    """Build a PageIndex and persist it to disk."""
    builder = PageIndexBuilder(use_llm=use_llm)
    index = builder.build(doc_id, ldus)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{doc_id}_page_index.json")
    with open(out_path, "w") as f:
        f.write(index.model_dump_json(indent=2))

    logger.info(f"PageIndex saved to {out_path}")
    return index
