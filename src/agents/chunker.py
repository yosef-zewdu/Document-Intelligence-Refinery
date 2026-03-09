# chunking_engine.py
# Chunk script for Stage 3 that respects the 5 rules and outputs List[LDU]
#
# Rules enforced:
# 1) Table cells are never split from their header row (table splitting repeats headers).
# 2) Figure caption is stored as metadata of its parent figure chunk (caption text not emitted separately).
# 3) Numbered list is kept as a single LDU unless it exceeds max_tokens (then split by list items).
# 4) Section headers are stored as parent metadata on all child chunks within that section.
# 5) Cross-references (e.g., "see Table 3") are resolved and stored as chunk relationships.

import re
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

from collections import defaultdict
from src.models import ExtractedDocument, TextBlock, TableStructure, LDU, BBox

# ---------------------------
# Utilities
# ---------------------------

_WS = re.compile(r"\s+")
IMAGE_RE = re.compile(r"^\[IMAGE\].*?\[Saved Image:\s*(.*?)\]", re.IGNORECASE)

LIST_ITEM_RE = re.compile(
    r"^\s*(" 
    r"(?:\d+(?:\.\d+)*[\.\)])"          # 1) 1.2) 3.4.5)
    r"|(?:[ivxlcdm]+[\)\.])"            # i) ii) iv) (roman)
    r"|(?:[•\-])"                       # bullets
    r")\s+",
    re.IGNORECASE
)
TABLE_REF_RE = re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE)
FIG_REF_RE = re.compile(r"\bFigure\s+(\d+)\b|\bFig\.\s*(\d+)\b", re.IGNORECASE)

TABLE_LABEL_LINE_RE = re.compile(r"^\s*Table\s+(\d+)\b", re.IGNORECASE)
FIG_LABEL_LINE_RE = re.compile(r"^\s*(Figure|Fig\.)\s*(\d+)\b", re.IGNORECASE)

def norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())

def approx_tokens(s: str) -> int:
    # word-count proxy (replace with tokenizer later)
    return len(norm_text(s).split())

def bbox_union(bboxes: List[BBox]) -> Optional[BBox]:
    bxs = [b for b in bboxes if b is not None and not _is_zero_bbox(b)]
    if not bxs:
        return None
    x0 = min(b.x0 for b in bxs)
    y0 = min(b.y0 for b in bxs)
    x1 = max(b.x1 for b in bxs)
    y1 = max(b.y1 for b in bxs)
    return BBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1))

def _is_zero_bbox(b: BBox) -> bool:
    return float(b.x0) == 0.0 and float(b.y0) == 0.0 and float(b.x1) == 0.0 and float(b.y1) == 0.0

def bbox_center(b: BBox) -> Tuple[float, float]:
    return ((b.x0 + b.x1) / 2.0, (b.y0 + b.y1) / 2.0)

def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)

def x_overlap_ratio(a: BBox, b: BBox) -> float:
    ow = overlap_1d(a.x0, a.x1, b.x0, b.x1)
    denom = max(min(a.x1 - a.x0, b.x1 - b.x0), 1e-6)
    return ow / denom

# Reading-order sorting:
# Many PDFs in your samples have "higher y means closer to top" (y0 around 777 near top).
# If in your coordinate system y increases upwards, sort by y0 descending.
# If it increases downwards, flip the sign here.
def sort_key_reading_order(block: TextBlock) -> Tuple[int, float, float]:
    # page_num ASC, then y0 DESC, then x0 ASC
    y0 = getattr(block.bbox, "y0", 0.0) if block.bbox else 0.0
    x0 = getattr(block.bbox, "x0", 0.0) if block.bbox else 0.0
    return (block.page_num, -float(y0), float(x0))

# ---------------------------
# Normalized item wrapper
# ---------------------------

@dataclass
class Item:
    kind: str  # "text" | "table" | "image"
    page_num: int
    bbox: Optional[BBox]
    content: Optional[str] = None
    table: Optional[TableStructure] = None
    source: Any = None  # original object

# ---------------------------
# Heading / list / image detection heuristics
# ---------------------------

def is_image_block_text(content: str) -> bool:
    return bool(IMAGE_RE.match((content or "").strip()))

def parse_image_path(content: str) -> Optional[str]:
    m = IMAGE_RE.match((content or "").strip())
    return m.group(1).strip() if m else None

HEADING_NUM_RE = re.compile(r"^\s*\d+(\.\d+)*\s+\S+")
ROMAN_LIST_RE   = re.compile(r"^\s*[ivxlcdm]+[\)\.]\s+", re.IGNORECASE)

KNOWN_HEADINGS = {
    "PROFILE","VISION","MISSION","MOTTO","VALUES","CONTENTS",
    "PRESIDENT'S MESSAGE","BOARD OF DIRECTORS","EXECUTIVE MANAGEMENT"
}

def is_heading_text(text: str) -> bool:
    t = norm_text(text)
    if not t:
        return False
    if len(t) > 180:
        return False

    # reject list markers pretending to be headings
    if ROMAN_LIST_RE.match(t) or LIST_ITEM_RE.match(t):
        return False

    # accept explicit numbering
    if HEADING_NUM_RE.match(t):
        return True

    # accept known headings exactly
    if t.upper() in KNOWN_HEADINGS:
        return True

    # accept ALL CAPS heavy lines
    alpha = sum(c.isalpha() for c in t)
    if alpha >= 6:
        upper = sum(c.isupper() for c in t if c.isalpha())
        if upper / alpha > 0.88 and len(t.split()) <= 14:
            return True

    return False

# def is_heading_text(text: str) -> bool:
#     """
#     Heuristic heading detector:
#     - relatively short line OR all-caps heavy
#     - not ending with punctuation
#     """
#     t = norm_text(text)
#     if not t:
#         return False
#     if len(t) > 160:
#         return False
#     if t.endswith((".", ":", ";", ",")):
#         return False
#     # All caps (allow digits and punctuation)
#     alpha = sum(c.isalpha() for c in t)
#     if alpha >= 8:
#         upper = sum(c.isupper() for c in t if c.isalpha())
#         if upper / max(alpha, 1) > 0.80:
#             return True
#     # Title-ish: not too many words, and mostly starts uppercase
#     words = t.split()
#     if 2 <= len(words) <= 12:
#         starts_upper = sum(1 for w in words if w[:1].isupper())
#         if starts_upper / len(words) > 0.6:
#             return True
#     return False

def is_list_item(text: str) -> bool:
    return bool(LIST_ITEM_RE.match((text or "").strip()))

def is_list_continuation(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("Â") or t.startswith("•") or t.startswith("-")

def build_running_header_set(doc: ExtractedDocument) -> set:
    """
    Detect repeated running headers (e.g., 'COMMERCIAL BANK OF ETHIOPIA ...')
    that appear on many pages near the top margin. These should NOT become headings.
    """
    counts = defaultdict(set)  # text -> set(pages)

    for b in doc.blocks:
        t = norm_text(b.content)
        if not t or len(t) > 180:
            continue

        # only consider all-caps heavy candidates
        alpha = sum(c.isalpha() for c in t)
        if alpha < 6:
            continue
        upper = sum(c.isupper() for c in t if c.isalpha())
        if upper / alpha < 0.85:
            continue

        # top margin heuristic (tune threshold if needed)
        if b.bbox and getattr(b.bbox, "y0", 0.0) > 740:
            counts[t].add(b.page_num)

    # repeated across many pages => running header
    return {txt for txt, pages in counts.items() if len(pages) >= 8}

# ---------------------------
# Chunk Validator
# ---------------------------

class ChunkValidator:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def validate(self, ldus: List[LDU]) -> None:
        """
        Raises ValueError if any constitution rule is violated.
        """
        for ldu in ldus:
            if not ldu.content and ldu.chunk_type not in ("figure",):
                raise ValueError("Empty content in non-figure LDU.")

            if not ldu.page_refs:
                raise ValueError("LDU missing page_refs.")

            if ldu.chunk_type == "table":
                # Rule 1: header must be present for tables (we store it in metadata AND in content format)
                hdr = ldu.metadata.get("table_headers")
                if not hdr or not isinstance(hdr, list) or len(hdr) == 0:
                    raise ValueError("Table LDU missing headers in metadata (Rule 1).")

            if ldu.chunk_type == "figure":
                # Rule 2: caption must live in metadata (if present). We disallow a separate caption chunk.
                # (We can't prove absence of separate caption chunk here, but we ensure caption is in metadata if found.)
                pass

            if ldu.chunk_type == "list":
                # Rule 3: list should not be split unless token_count > max_tokens
                # If it *is* split, it should say so in metadata.
                split = bool(ldu.metadata.get("list_split", False))
                if split is False and ldu.token_count > self.max_tokens:
                    raise ValueError("List LDU exceeds max_tokens without list_split (Rule 3).")

            # Rule 4: parent_section should propagate; validator can warn when missing
            # (We don't hard-fail since some docs truly lack headings.)
            # Rule 5: if references exist, relations should be recorded (unresolved allowed).
            # We'll soft-check:
            refs = ldu.metadata.get("xref_detected", {})
            if refs and "relations" not in ldu.metadata:
                raise ValueError("Cross-references detected but no relations recorded (Rule 5).")

# ---------------------------
# Chunking Engine
# ---------------------------

class ChunkingEngine:
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.validator = ChunkValidator(max_tokens=max_tokens)

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        # 1) Build per-page item stream (reading-order aware)
        items_by_page = self._build_items_by_page(doc)
        running_headers = build_running_header_set(doc)

        # 2) First pass: produce LDUs with section + list + figure rules
        ldus: List[LDU] = []
        section_context: Optional[str] = None

        # Index placeholders for later xref resolution
        table_label_to_hash: Dict[str, str] = {}
        fig_label_to_hash: Dict[str, str] = {}

        for page_num in sorted(items_by_page.keys()):
            items = items_by_page[page_num]

            # Attach captions to images on this page BEFORE creating text LDUs
            image_to_caption_block = self._attach_captions_on_page(items)

            # Iterate items in reading order
            buffer_text_blocks: List[TextBlock] = []
            buffer_list_blocks: List[TextBlock] = []
            list_mode = False

            def flush_text_buffer():
                nonlocal buffer_text_blocks
                if not buffer_text_blocks:
                    return
                for ldu in self._emit_text_ldus(doc.doc_id, buffer_text_blocks, section_context):
                    ldus.append(ldu)
                buffer_text_blocks = []

            def flush_list_buffer(force_split: bool = False):
                nonlocal buffer_list_blocks
                if not buffer_list_blocks:
                    return
                for ldu in self._emit_list_ldus(doc.doc_id, buffer_list_blocks, section_context, force_split=force_split):
                    ldus.append(ldu)
                buffer_list_blocks = []

            for it in items:
                if it.kind == "text":
                    assert it.content is not None
                    t = norm_text(it.content)
                    
                    # Suppress repeated running headers
                    if t in running_headers:
                        continue
                    # Skip caption blocks that were assigned to an image (Rule 2)
                    caption_blocks = set(
                        self._block_key(b) for b in image_to_caption_block.values() if b is not None
                    )
                    if isinstance(it.source, TextBlock) and self._block_key(it.source) in caption_blocks:
                        continue

                    # Heading updates section context (Rule 4)
                    if is_heading_text(t):
                        # Flush any buffered content before switching section
                        flush_list_buffer()
                        flush_text_buffer()
                        section_context = t
                        # Emit heading as its own LDU (helps PageIndex building later)
                        ldus.append(self._make_heading_ldu(doc.doc_id, it, section_context))
                        continue

                    # List handling (Rule 3) with continuation support
                    if is_list_item(t):
                        # entering list mode
                        if buffer_text_blocks:
                            flush_text_buffer()
                        buffer_list_blocks.append(it.source)
                        list_mode = True
                        continue

                    # if already in list mode, keep absorbing continuation lines (bullets, Â artifacts)
                    if list_mode and is_list_continuation(t):
                        buffer_list_blocks.append(it.source)
                        continue

                    # leaving list mode if current line is neither new list item nor continuation
                    if list_mode and buffer_list_blocks:
                        flush_list_buffer()
                        list_mode = False

                    # Normal paragraph text buffering
                    buffer_text_blocks.append(it.source)
                    continue

                if it.kind == "table":
                    # Flush any buffered text/list before table
                    flush_list_buffer()
                    flush_text_buffer()

                    # Emit table LDUs (Rule 1)
                    for tldu in self._emit_table_ldus(doc.doc_id, it.table, section_context):
                        ldus.append(tldu)

                    # Attempt to assign table labels (for Rule 5)
                    # Look for nearby "Table N" in preceding heading/text on same page
                    lbl = self._infer_table_label_from_context(ldus, it.table.page_num)
                    if lbl and tldu.content_hash:
                        table_label_to_hash[lbl] = tldu.content_hash
                    continue

                if it.kind == "image":
                    # Flush any buffered text/list before image
                    flush_list_buffer()
                    flush_text_buffer()

                    # Emit figure LDU with caption in metadata (Rule 2)
                    cap_block = None
                    if isinstance(it.source, TextBlock):
                        cap_block = image_to_caption_block.get(self._block_key(it.source))
                    fig_ldu = self._emit_figure_ldu(doc.doc_id, it, cap_block, section_context)
                    ldus.append(fig_ldu)

                    # index label if caption starts with Figure N (Rule 5)
                    cap = (fig_ldu.metadata.get("caption") or "").strip()
                    m = FIG_LABEL_LINE_RE.match(cap)
                    if m:
                        num = m.group(2)
                        fig_label_to_hash[f"Figure {num}"] = fig_ldu.content_hash
                    continue

            # End page: flush buffers
            if buffer_list_blocks:
                flush_list_buffer()
            if buffer_text_blocks:
                flush_text_buffer()

        # 3) Second pass: resolve cross-references (Rule 5)
        self._resolve_cross_references(ldus, table_label_to_hash, fig_label_to_hash)

        # 4) Validate all rules (hard constraints)
        self.validator.validate(ldus)

        return ldus

    # -----------------------
    # Build items by page
    # -----------------------

    def _build_items_by_page(self, doc: ExtractedDocument) -> Dict[int, List[Item]]:
        items_by_page: Dict[int, List[Item]] = {}

        # Convert blocks into text/image items
        # Ensure blocks are in reading order by bbox
        blocks_sorted = sorted(doc.blocks, key=sort_key_reading_order)

        for b in blocks_sorted:
            items_by_page.setdefault(b.page_num, [])
            if is_image_block_text(b.content):
                items_by_page[b.page_num].append(Item(
                    kind="image", page_num=b.page_num, bbox=b.bbox, content=b.content, source=b
                ))
            else:
                items_by_page[b.page_num].append(Item(
                    kind="text", page_num=b.page_num, bbox=b.bbox, content=b.content, source=b
                ))

        # Add tables
        for t in doc.tables:
            items_by_page.setdefault(t.page_num, [])
            items_by_page[t.page_num].append(Item(
                kind="table", page_num=t.page_num, bbox=t.bbox, table=t, source=t
            ))

        # Sort per page: vertical then x0 (tables/images included)
        for p, items in items_by_page.items():
            def key(it: Item):
                y0 = it.bbox.y0 if it.bbox else 0.0
                x0 = it.bbox.x0 if it.bbox else 0.0
                return (-float(y0), float(x0))
            items_by_page[p] = sorted(items, key=key)

        return items_by_page

    # -----------------------
    # Captions for images (Rule 2)
    # -----------------------

    def _block_key(self, b: TextBlock) -> tuple:
        bb = b.bbox
        bbox_key = None
        if bb is not None:
            bbox_key = (round(float(bb.x0), 2), round(float(bb.y0), 2),
                        round(float(bb.x1), 2), round(float(bb.y1), 2))
        return (int(b.page_num), bbox_key, norm_text(b.content)[:80])

    def _attach_captions_on_page(self, items: List[Item]) -> Dict[tuple, Optional[TextBlock]]:
        """
        Returns mapping: image_block_key -> caption_text_block (or None)
        Caption blocks will be skipped in text emission.
        """
        images = [it for it in items if it.kind == "image" and it.bbox and isinstance(it.source, TextBlock)]
        texts  = [it for it in items if it.kind == "text" and it.bbox and isinstance(it.source, TextBlock)]

        mapping: Dict[tuple, Optional[TextBlock]] = {}

        for img in images:
            img_block: TextBlock = img.source
            img_key = self._block_key(img_block)

            best: Optional[Tuple[float, TextBlock]] = None
            for tx in texts:
                tx_block: TextBlock = tx.source
                t = norm_text(tx.content or "")
                if not t:
                    continue

                cap_bonus = 0.0
                if FIG_LABEL_LINE_RE.match(t):
                    cap_bonus += 2.0
                if len(t.split()) <= 35:
                    cap_bonus += 0.5

                dy = abs(float(tx.bbox.y0) - float(img.bbox.y0))
                xo = x_overlap_ratio(img.bbox, tx.bbox)

                score = cap_bonus + (2.0 * xo) - (dy / 200.0)
                if best is None or score > best[0]:
                    best = (score, tx_block)

            mapping[img_key] = best[1] if best else None

        return mapping

    # -----------------------
    # Emit LDUs
    # -----------------------

    def _make_heading_ldu(self, doc_id: str, it: Item, section_context: Optional[str]) -> LDU:
        content = norm_text(it.content or "")
        bb = it.bbox if it.bbox and not _is_zero_bbox(it.bbox) else None
        return self._make_ldu(
            doc_id=doc_id,
            content=content,
            chunk_type="heading",
            page_refs=[it.page_num],
            bbox=bb,
            parent_section=section_context,
            metadata={"doc_id": doc_id}
        )

    def _emit_text_ldus(self, doc_id: str, blocks: List[TextBlock], section_context: Optional[str]) -> List[LDU]:
        """
        Merge text blocks into LDUs up to max_tokens.
        Excludes image blocks; images are handled separately.
        """
        # Merge blocks respecting order; split only at block boundaries
        texts = [norm_text(b.content) for b in blocks if b.content and not is_image_block_text(b.content)]
        texts = [t for t in texts if t]
        if not texts:
            return []

        # greedy pack by block
        out: List[LDU] = []
        current: List[str] = []
        current_blocks: List[TextBlock] = []
        tok = 0

        for b, t in zip(blocks, texts):
            t_tok = approx_tokens(t)
            if current and (tok + t_tok) > self.max_tokens:
                content = "\n".join(current)
                out.append(self._make_ldu(
                    doc_id=doc_id,
                    content=content,
                    chunk_type="text",
                    page_refs=sorted({blk.page_num for blk in current_blocks}),
                    bbox=bbox_union([blk.bbox for blk in current_blocks if blk.bbox]),
                    parent_section=section_context,
                    metadata={"doc_id": doc_id}
                ))
                current, current_blocks, tok = [], [], 0

            current.append(t)
            current_blocks.append(b)
            tok += t_tok

        if current:
            content = "\n".join(current)
            out.append(self._make_ldu(
                doc_id=doc_id,
                content=content,
                chunk_type="text",
                page_refs=sorted({blk.page_num for blk in current_blocks}),
                bbox=bbox_union([blk.bbox for blk in current_blocks if blk.bbox]),
                parent_section=section_context,
                metadata={"doc_id": doc_id}
            ))

        return out

    def _emit_list_ldus(self, doc_id: str, blocks: List[TextBlock], section_context: Optional[str], force_split: bool=False) -> List[LDU]:
        """
        Rule 3: keep list as single LDU unless exceeds max_tokens.
        If split, split by list item boundaries (blocks).
        """
        items = [norm_text(b.content) for b in blocks if b.content]
        items = [t for t in items if t]
        if not items:
            return []

        total = sum(approx_tokens(t) for t in items)
        if total <= self.max_tokens and not force_split:
            content = "\n".join(items)
            return [self._make_ldu(
                doc_id=doc_id,
                content=content,
                chunk_type="list",
                page_refs=sorted({b.page_num for b in blocks}),
                bbox=bbox_union([b.bbox for b in blocks if b.bbox]),
                parent_section=section_context,
                metadata={"doc_id": doc_id, "list_split": False}
            )]

        # split by item groups
        out: List[LDU] = []
        buf: List[str] = []
        buf_blocks: List[TextBlock] = []
        tok = 0
        for b, t in zip(blocks, items):
            t_tok = approx_tokens(t)
            if buf and (tok + t_tok) > self.max_tokens:
                out.append(self._make_ldu(
                    doc_id=doc_id,
                    content="\n".join(buf),
                    chunk_type="list",
                    page_refs=sorted({x.page_num for x in buf_blocks}),
                    bbox=bbox_union([x.bbox for x in buf_blocks if x.bbox]),
                    parent_section=section_context,
                    metadata={"doc_id": doc_id, "list_split": True}
                ))
                buf, buf_blocks, tok = [], [], 0
            buf.append(t)
            buf_blocks.append(b)
            tok += t_tok

        if buf:
            out.append(self._make_ldu(
                doc_id=doc_id,
                content="\n".join(buf),
                chunk_type="list",
                page_refs=sorted({x.page_num for x in buf_blocks}),
                bbox=bbox_union([x.bbox for x in buf_blocks if x.bbox]),
                parent_section=section_context,
                metadata={"doc_id": doc_id, "list_split": True}
            ))
        return out

    def _emit_table_ldus(self, doc_id: str, table: TableStructure, section_context: Optional[str]) -> List[LDU]:
        """
        Rule 1: if table must be split, repeat headers in each chunk.
        """
        headers = [str(h) for h in (table.headers or [])]
        rows = table.rows or []

        # Convert rows to strings but keep structure in metadata
        def row_to_str(r): return " | ".join("" if c is None else str(c) for c in r)

        header_str = " | ".join(headers)
        sep = "-" * max(len(header_str), 3)
        base = f"{header_str}\n{sep}"

        # Greedy pack rows
        out: List[LDU] = []
        buf: List[str] = []
        buf_rows: List[List[Any]] = []
        tok = approx_tokens(base)

        for r in rows:
            rs = row_to_str(r)
            rs_tok = approx_tokens(rs)
            if buf and (tok + rs_tok) > self.max_tokens:
                content = base + "\n" + "\n".join(buf)
                out.append(self._make_ldu(
                    doc_id=doc_id,
                    content=content,
                    chunk_type="table",
                    page_refs=[table.page_num],
                    bbox=table.bbox if table.bbox and not _is_zero_bbox(table.bbox) else None,
                    parent_section=section_context,
                    metadata={
                        "doc_id": doc_id,
                        "table_headers": headers,
                        "table_rows": buf_rows,
                        "table_split": True
                    }
                ))
                buf, buf_rows, tok = [], [], approx_tokens(base)

            buf.append(rs)
            buf_rows.append(r)
            tok += rs_tok

        if buf or not rows:
            content = base if not buf else base + "\n" + "\n".join(buf)
            out.append(self._make_ldu(
                doc_id=doc_id,
                content=content,
                chunk_type="table",
                page_refs=[table.page_num],
                bbox=table.bbox if table.bbox and not _is_zero_bbox(table.bbox) else None,
                parent_section=section_context,
                metadata={
                    "doc_id": doc_id,
                    "table_headers": headers,
                    "table_rows": buf_rows if buf else [],
                    "table_split": bool(rows) and (len(out) > 0)
                }
            ))

        return out

    def _emit_figure_ldu(self, doc_id: str, it: Item, caption_block: Optional[TextBlock], section_context: Optional[str]) -> LDU:
        path = parse_image_path(it.content or "")
        cap_text = norm_text(caption_block.content) if caption_block else None
        cap_bbox = caption_block.bbox if caption_block and caption_block.bbox and not _is_zero_bbox(caption_block.bbox) else None

        bb = it.bbox if it.bbox and not _is_zero_bbox(it.bbox) else None

        # Rule 2: caption stored as metadata
        md: Dict[str, Any] = {
            "doc_id": doc_id,
            "image_path": path,
            "caption": cap_text,
            "caption_bbox": cap_bbox.model_dump() if cap_bbox else None,
        }

        # IMPORTANT: Include caption in content for searchability
        # The caption contains critical information (e.g., "Known Gaps in Current Judicial Layer")
        # that must be searchable in the vector store
        content = f"[IMAGE: {cap_text}]" if cap_text else "[IMAGE]"
        
        return self._make_ldu(
            doc_id=doc_id,
            content=content,
            chunk_type="figure",
            page_refs=[it.page_num],
            bbox=bb,
            parent_section=section_context,
            metadata=md
        )

    # -----------------------
    # Cross-reference inference (Rule 5)
    # -----------------------

    def _infer_table_label_from_context(self, ldus: List[LDU], page_num: int) -> Optional[str]:
        """
        Heuristic: find last text/heading chunk on the same page that begins with 'Table N'
        """
        for ldu in reversed(ldus):
            if page_num not in ldu.page_refs:
                continue
            if ldu.chunk_type in ("text", "heading"):
                first_line = (ldu.content.splitlines() or [""])[0]
                m = TABLE_LABEL_LINE_RE.match(first_line.strip())
                if m:
                    return f"Table {m.group(1)}"
        return None

    def _resolve_cross_references(self, ldus: List[LDU], table_index: Dict[str, str], fig_index: Dict[str, str]) -> None:
        """
        Scan each LDU for "Table N"/"Figure N" and attach relations in metadata.
        Unresolved refs are recorded too.
        """
        for ldu in ldus:
            text = ldu.content or ""
            trefs = TABLE_REF_RE.findall(text)
            frefs = FIG_REF_RE.findall(text)

            detected = {"tables": [], "figures": []}
            relations = []
            unresolved = []

            # Tables
            for (num,) in trefs:
                label = f"Table {num}"
                detected["tables"].append(label)
                target = table_index.get(label)
                if target:
                    relations.append({"type": "refers_to_table", "target_hash": target, "label": label})
                else:
                    unresolved.append({"type": "refers_to_table", "label": label})

            # Figures: FIG_REF_RE returns tuples like ("2","") or ("","2")
            for a, b in frefs:
                num = a or b
                if not num:
                    continue
                label = f"Figure {num}"
                detected["figures"].append(label)
                target = fig_index.get(label)
                if target:
                    relations.append({"type": "refers_to_figure", "target_hash": target, "label": label})
                else:
                    unresolved.append({"type": "refers_to_figure", "label": label})

            if detected["tables"] or detected["figures"]:
                ldu.metadata["xref_detected"] = detected
                ldu.metadata["relations"] = relations
                if unresolved:
                    ldu.metadata["unresolved_refs"] = unresolved

    # -----------------------
    # LDU creation
    # -----------------------

    def _make_ldu(
        self,
        doc_id: str,
        content: str,
        chunk_type: str,
        page_refs: List[int],
        bbox: Optional[BBox],
        parent_section: Optional[str],
        metadata: Dict[str, Any],
    ) -> LDU:
        # Normalize content (helps stable hashing)
        content_norm = norm_text(content)
        # token_count = approx tokens
        tok = approx_tokens(content_norm)

        # If your LDU model auto-hashes ONLY content, you should set content_hash here instead
        # (recommended) to avoid collisions.
        # We'll compute a provenance-aware hash and pass it explicitly.
        payload = {
            "doc_id": doc_id,
            "chunk_type": chunk_type,
            "content": content_norm,
            "page_refs": page_refs,
            "bbox": bbox.model_dump() if bbox else None,
            "parent_section": parent_section,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        h = "sha256:" + hashlib.sha256(blob).hexdigest()

        return LDU(
            content=content_norm,
            chunk_type=chunk_type,
            page_refs=page_refs,
            bounding_box=bbox,
            parent_section=parent_section,
            token_count=tok,
            content_hash=h,
            metadata=metadata,
        )


# ---------------------------
# Example usage
# ---------------------------
# engine = ChunkingEngine(max_tokens=512)
# ldus = engine.chunk(extracted_doc)
# for l in ldus[:5]:
#     print(l.chunk_type, l.page_refs, l.parent_section, l.content_hash)