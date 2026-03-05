import time
from typing import Optional, Tuple, Dict, Any, List

from docling.document_converter import DocumentConverter
from .base import BaseExtractor
from src.models import (
    ExtractedDocument,
    TextBlock,
    TableStructure,
    DocumentProfile,
    BBox,
)

class LayoutExtractor(BaseExtractor):
    """
    Layout-aware extraction using Docling >= 2.76.
    Produces text blocks + tables with provenance where available.
    Includes dynamic confidence scoring (0..1).
    """

    def __init__(self, *, min_block_per_page: float = 0.8):
        self.converter = DocumentConverter()
        self.min_block_per_page = min_block_per_page

    # ---------- Provenance helpers ----------
    def _get_item_prov_bbox(self, item) -> Tuple[int, Optional[BBox], bool]:
        """
        Returns: (page_num, bbox_or_none, has_prov)
        Page num is returned as-is from docling (often 1-based).
        """
        if not hasattr(item, "prov") or not item.prov:
            return 1, None, False

        p = item.prov[0]  # first provenance entry
        page_num = getattr(p, "page_no", 1)

        if hasattr(p, "bbox") and p.bbox is not None:
            b = p.bbox
            # Docling bbox commonly has l,t,r,b
            # IMPORTANT: coordinate system should be tracked in metadata downstream
            return page_num, BBox(x0=float(b.l), y0=float(b.t), x1=float(b.r), y1=float(b.b)), True

        return page_num, None, True

    def _is_table_item(self, item) -> bool:
        # Robust without importing internal classes:
        return item.__class__.__name__ == "TableItem" or callable(getattr(item, "export_to_dataframe", None))

    def _is_picture_item(self, item) -> bool:
        return item.__class__.__name__ in ("PictureItem", "ImageItem")

    # ---------- Confidence scoring ----------
    def _compute_confidence(
        self,
        *,
        conversion_ok: bool,
        total_pages: int,
        blocks: List[TextBlock],
        tables: List[TableStructure],
        prov_hits: int,
        prov_misses: int,
        total_text_chars: int,
        table_quality: Dict[str, Any],
    ) -> float:
        """
        Confidence in [0,1].
        Intended meaning: "How safe is it to trust this extraction without escalating?"
        """
        if not conversion_ok:
            return 0.0

        # Base score for successful conversion
        score = 0.60

        # 1) Provenance coverage (bbox+page) — critical for audit and chunking
        prov_total = prov_hits + prov_misses
        prov_ratio = (prov_hits / prov_total) if prov_total else 0.0
        # Reward good provenance; penalize missing provenance
        score += 0.25 * min(1.0, prov_ratio / 0.85)     # full credit at ~85% coverage
        score -= 0.10 * min(1.0, (1.0 - prov_ratio) / 0.50)

        # 2) Extraction yield sanity: blocks per page should not be near-zero for digital docs
        # (For very short docs, avoid over-penalizing.)
        pages = max(total_pages, 1)
        blocks_per_page = len(blocks) / pages
        if pages >= 3:
            if blocks_per_page < self.min_block_per_page:
                # low yield: maybe layout parsing failed
                score -= 0.15 * min(1.0, (self.min_block_per_page - blocks_per_page) / self.min_block_per_page)
            else:
                score += 0.05

        # 3) Total text volume check
        # If doc is long and text is tiny, something likely went wrong (or it's image-only).
        if pages >= 10 and total_text_chars < 2000:
            score -= 0.20
        elif total_text_chars > 8000:
            score += 0.05

        # 4) Table quality boosts (if tables exist)
        if tables:
            # table_quality includes avg_cols, avg_rows, nonempty_header_ratio
            avg_cols = table_quality.get("avg_cols", 0)
            avg_rows = table_quality.get("avg_rows", 0)
            nonempty_header_ratio = table_quality.get("nonempty_header_ratio", 0.0)

            if avg_cols >= 3 and avg_rows >= 3:
                score += 0.05
            if nonempty_header_ratio >= 0.7:
                score += 0.05
            if avg_rows < 2 or avg_cols < 2:
                # suspicious "tables" that aren't tables
                score -= 0.10

        # Clamp
        return max(0.0, min(1.0, score))

    # ---------- Public API ----------
    def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        t0 = time.time()
        conversion_ok = False

        blocks: List[TextBlock] = []
        tables: List[TableStructure] = []

        prov_hits = 0
        prov_misses = 0
        total_text_chars = 0

        table_cols = []
        table_rows = []
        header_nonempty = []

        # Dedup sets to avoid duplicate blocks from nested items
        seen_text_keys = set()
        seen_table_keys = set()

        try:
            result = self.converter.convert(pdf_path)
            doc = result.document
            conversion_ok = True

            for item, level in doc.iterate_items():
                # Skip pictures here (you may later create "figure" LDUs)
                if self._is_picture_item(item):
                    continue

                # 1) Tables first (and skip them in text extraction to avoid duplication)
                if self._is_table_item(item):
                    try:
                        df = item.export_to_dataframe(doc=doc)  # Docling >=2.76.0 friendly
                    except Exception:
                        # If dataframe export fails, treat as not confident; skip table
                        continue

                    headers = [str(c) for c in df.columns]
                    rows = [
                        ["" if cell is None else str(cell) for cell in row]
                        for row in df.values.tolist()
                    ]

                    page_num, bbox, has_prov = self._get_item_prov_bbox(item)
                    if bbox is not None:
                        prov_hits += 1
                    else:
                        prov_misses += 1

                    # Dedup key: page + bbox + header signature
                    key = (
                        page_num,
                        (bbox.x0, bbox.y0, bbox.x1, bbox.y1) if bbox else None,
                        tuple(headers[:5]),
                        len(rows),
                        len(headers),
                    )
                    if key in seen_table_keys:
                        continue
                    seen_table_keys.add(key)

                    tables.append(TableStructure(
                        headers=headers,
                        rows=rows,
                        bbox=bbox,
                        page_num=page_num
                    ))

                    # Table quality stats
                    table_cols.append(len(headers))
                    table_rows.append(len(rows))
                    header_nonempty.append(sum(1 for h in headers if h and h.strip()) / max(len(headers), 1))

                    continue  # important: don't also extract table text as blocks

                # 2) Text blocks (paragraphs/headings/captions depending on docling)
                txt = getattr(item, "text", None)
                if txt:
                    txt_norm = " ".join(txt.split())
                    if not txt_norm:
                        continue

                    page_num, bbox, has_prov = self._get_item_prov_bbox(item)
                    if bbox is not None:
                        prov_hits += 1
                    else:
                        prov_misses += 1

                    # Dedup: same page+bbox+prefix
                    key = (page_num,
                           (bbox.x0, bbox.y0, bbox.x1, bbox.y1) if bbox else None,
                           txt_norm[:80])
                    if key in seen_text_keys:
                        continue
                    seen_text_keys.add(key)

                    total_text_chars += len(txt_norm)

                    # NOTE: your TextBlock model requires bbox.
                    # Best practice: make bbox Optional[BBox].
                    # For now: if bbox is missing, store a tiny bbox and flag it in metadata elsewhere.
                    safe_bbox = bbox or BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

                    blocks.append(TextBlock(
                        content=txt_norm,
                        bbox=safe_bbox,
                        page_num=page_num
                    ))

        except Exception as e:
            # conversion_ok remains False
            doc = None

        elapsed = time.time() - t0

        # table quality aggregate
        table_quality = {
            "avg_cols": (sum(table_cols) / len(table_cols)) if table_cols else 0.0,
            "avg_rows": (sum(table_rows) / len(table_rows)) if table_rows else 0.0,
            "nonempty_header_ratio": (sum(header_nonempty) / len(header_nonempty)) if header_nonempty else 0.0,
        }

        # dynamic confidence
        total_pages = int(profile.metadata.get("total_pages", 0)) or 1
        confidence = self._compute_confidence(
            conversion_ok=conversion_ok,
            total_pages=total_pages,
            blocks=blocks,
            tables=tables,
            prov_hits=prov_hits,
            prov_misses=prov_misses,
            total_text_chars=total_text_chars,
            table_quality=table_quality,
        )

        return ExtractedDocument(
            doc_id=profile.doc_id,
            blocks=blocks,
            tables=tables,
            metadata={
                "strategy": "LayoutExtractor(Docling)",
                "docling_version_min": "2.76.0",
                "conversion_ok": conversion_ok,
                "elapsed_sec": round(elapsed, 3),
                "counts": {
                    "blocks": len(blocks),
                    "tables": len(tables),
                },
                "provenance": {
                    "bbox_hits": prov_hits,
                    "bbox_misses": prov_misses,
                    "bbox_ratio": round((prov_hits / (prov_hits + prov_misses)) if (prov_hits + prov_misses) else 0.0, 4),
                    "coord_system": "docling_pdf_bbox(l,t,r,b)",
                    "page_index_base": "unknown(verify_docling)",  # set to 1-based if confirmed
                },
                "table_quality": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in table_quality.items()},
                "text_chars": total_text_chars,
                "confidence": round(confidence, 4),
            }
        )

    def get_confidence_score(self, extracted: ExtractedDocument) -> float:
        """
        Prefer using the computed score from extraction metadata.
        """
        return float(extracted.metadata.get("confidence", 0.0))