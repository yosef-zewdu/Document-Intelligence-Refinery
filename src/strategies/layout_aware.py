import time
import logging
from typing import Optional, Tuple, Dict, Any, List

from docling.document_converter import DocumentConverter
from .base import BaseExtractor
from src.models import (
    ExtractedDocument,
    TextBlock,
    TableStructure,
    DocumentProfile,
    BBox,
    ConfidenceMetadata
)

logger = logging.getLogger(__name__)

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
        if not hasattr(item, "prov") or not item.prov:
            return 1, None, False

        p = item.prov[0]
        page_num = getattr(p, "page_no", 1)

        if hasattr(p, "bbox") and p.bbox is not None:
            b = p.bbox
            try:
                 return page_num, BBox(x0=float(b.l), y0=float(b.t), x1=float(b.r), y1=float(b.b)), True
            except ValueError:
                 return page_num, None, True

        return page_num, None, True

    def _is_table_item(self, item) -> bool:
        return item.__class__.__name__ == "TableItem" or callable(getattr(item, "export_to_dataframe", None))

    def _is_picture_item(self, item) -> bool:
        return item.__class__.__name__ in ("PictureItem", "ImageItem")

    # ---------- Public API ----------
    def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        t0 = time.time()
        conversion_ok = False
        warnings = []
        signals = {}

        blocks: List[TextBlock] = []
        tables: List[TableStructure] = []

        prov_hits = 0
        prov_misses = 0
        total_text_chars = 0

        table_cols = []
        table_rows = []
        header_nonempty = []

        seen_text_keys = set()
        seen_table_keys = set()

        try:
            result = self.converter.convert(pdf_path)
            doc = result.document
            conversion_ok = True

            for item, level in doc.iterate_items():
                if self._is_picture_item(item):
                    continue

                if self._is_table_item(item):
                    try:
                        df = item.export_to_dataframe(doc=doc)
                        if df.empty:
                            continue
                    except Exception as e:
                        warnings.append(f"Table export failed: {e}")
                        continue

                    headers = [str(c) for c in df.columns]
                    rows = [["" if cell is None else str(cell) for cell in row] for row in df.values.tolist()]

                    page_num, bbox, has_prov = self._get_item_prov_bbox(item)
                    if bbox is not None:
                        prov_hits += 1
                    else:
                        prov_misses += 1

                    key = (page_num, (bbox.x0, bbox.y0, bbox.x1, bbox.y1) if bbox else None, tuple(headers[:5]), len(rows))
                    if key in seen_table_keys:
                        continue
                    seen_table_keys.add(key)

                    try:
                        tables.append(TableStructure(headers=headers, rows=rows, bbox=bbox, page_num=page_num))
                        table_cols.append(len(headers))
                        table_rows.append(len(rows))
                        header_nonempty.append(sum(1 for h in headers if h and h.strip()) / max(len(headers), 1))
                    except ValueError as e:
                        warnings.append(f"Invalid table structure: {e}")

                    continue

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

                    key = (page_num, (bbox.x0, bbox.y0, bbox.x1, bbox.y1) if bbox else None, txt_norm[:80])
                    if key in seen_text_keys:
                        continue
                    seen_text_keys.add(key)

                    total_text_chars += len(txt_norm)
                    safe_bbox = bbox or BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

                    try:
                        blocks.append(TextBlock(content=txt_norm, bbox=safe_bbox, page_num=page_num))
                    except ValueError as e:
                        warnings.append(f"Invalid text block: {e}")

        except Exception as e:
            logger.error(f"Docling conversion failed for {pdf_path}: {e}")
            warnings.append(f"Conversion error: {str(e)}")

        elapsed = time.time() - t0
        
        # Table quality aggregate
        table_quality = {
            "avg_cols": (sum(table_cols) / len(table_cols)) if table_cols else 0.0,
            "avg_rows": (sum(table_rows) / len(table_rows)) if table_rows else 0.0,
            "nonempty_header_ratio": (sum(header_nonempty) / len(header_nonempty)) if header_nonempty else 0.0,
        }
        signals.update(table_quality)
        signals["prov_ratio"] = round((prov_hits / (prov_hits + prov_misses)) if (prov_hits + prov_misses) else 0.0, 4)
        signals["total_text_chars"] = total_text_chars

        # Compute confidence
        score = self._compute_confidence_score(conversion_ok, profile, blocks, tables, signals, warnings)
        
        confidence_meta = ConfidenceMetadata(
            score=score,
            method="docling_structural",
            warnings=warnings,
            signals=signals
        )

        return ExtractedDocument(
            doc_id=profile.doc_id,
            blocks=blocks,
            tables=tables,
            metadata={
                "strategy": "LayoutExtractor(Docling)",
                "elapsed_sec": round(elapsed, 3),
                "conversion_ok": conversion_ok
            },
            confidence=confidence_meta
        )

    def _compute_confidence_score(self, conversion_ok: bool, profile: DocumentProfile, blocks: List[TextBlock], tables: List[TableStructure], signals: Dict[str, Any], warnings: List[str]) -> float:
        if not conversion_ok:
            return 0.0
        
        score = 0.60  # Initial base score
        
        # 1. Provenance boost
        prov_ratio = signals.get("prov_ratio", 0.0)
        score += 0.25 * min(1.0, prov_ratio / 0.85)
        
        # 2. Yield check
        total_pages = int(profile.metadata.get("total_pages", 1))
        blocks_per_page = len(blocks) / total_pages
        if blocks_per_page < self.min_block_per_page:
            score -= 0.15
        else:
            score += 0.05
            
        # 3. Table quality
        if tables:
            if signals.get("avg_cols", 0) >= 3 and signals.get("nonempty_header_ratio", 0) > 0.7:
                score += 0.10
        
        # 4. Warnings penalty
        score -= 0.05 * len(warnings)
        
        return max(0.0, min(1.0, score))

    def get_confidence_score(self, extracted: ExtractedDocument) -> float:
        if extracted.confidence:
            return extracted.confidence.score
        return 0.0