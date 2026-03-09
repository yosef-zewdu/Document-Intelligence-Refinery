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
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        # Ensure layout and table structure models are active
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = True # Critical for getting bboxes on all text
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        # Correct way to pass options in Docling 2.x
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.min_block_per_page = min_block_per_page

    # ---------- Provenance helpers ----------
    def _get_item_prov_bbox(self, item, doc=None) -> Tuple[int, Optional[BBox], bool]:
        # In Docling 2, provenance is in item.prov
        prov = getattr(item, "prov", [])
        if not prov:
            return 1, None, False

        p = prov[0]
        page_num = getattr(p, "page_no", 1)

        # Get bbox from p.bbox (BoundingBox)
        b = getattr(p, "bbox", None)
        if b is not None:
             try:
                  # docling BBox uses l, t, r, b. 
                  # Depending on coord_origin, t can be > b. Pydantic requires y1 > y0.
                  l, t, r, bottom = float(b.l), float(b.t), float(b.r), float(b.b)
                  return page_num, BBox(
                      x0=round(min(l, r), 2), 
                      y0=round(min(t, bottom), 2), 
                      x1=round(max(l, r), 2), 
                      y1=round(max(t, bottom), 2)
                  ), True
             except (ValueError, AttributeError) as e:
                  logger.debug(f"BBox mapping failed: {e}")
                  return page_num, None, True

        return page_num, None, True

    def _is_table_item(self, item) -> bool:
        from docling_core.types.doc.document import TableItem
        return isinstance(item, TableItem)

    def _is_formula_item(self, item) -> bool:
        from docling_core.types.doc.document import FormulaItem
        return isinstance(item, FormulaItem)

    def _is_picture_item(self, item) -> bool:
        from docling_core.types.doc.document import PictureItem
        return isinstance(item, PictureItem)

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
                # Capture Tables
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

                    page_num, bbox, has_prov = self._get_item_prov_bbox(item, doc)
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

                # Capture Text, Formulas, and Alt-Text from Pictures
                txt = getattr(item, "text", None)
                label = ""
                
                if self._is_formula_item(item):
                    label = "[FORMULA]"
                    txt = txt or "(no formula text)"
                elif self._is_picture_item(item):
                    label = "[IMAGE]"
                    # Pictures often have labels or captions instead of primary text
                    caption_text = txt or getattr(item, "caption", None) or ""
                    
                    # Try to extract OCR text from the image itself
                    ocr_text = ""
                    try:
                        # Get the image and extract text using OCR
                        img = item.get_image(doc)
                        if img:
                            # Use Docling's OCR or pytesseract to extract text from image
                            try:
                                import pytesseract
                                ocr_text = pytesseract.image_to_string(img).strip()
                                if ocr_text:
                                    logger.debug(f"Extracted OCR text from image: {len(ocr_text)} chars")
                            except ImportError:
                                logger.debug("pytesseract not available, skipping OCR text extraction")
                            except Exception as e:
                                logger.debug(f"OCR extraction failed: {e}")
                    except Exception as e:
                        logger.debug(f"Failed to get image for OCR: {e}")
                    
                    # Combine caption and OCR text
                    if caption_text and ocr_text:
                        txt = f"{caption_text}\n[OCR Text]: {ocr_text}"
                    elif ocr_text:
                        txt = f"[OCR Text]: {ocr_text}"
                    elif caption_text:
                        txt = caption_text
                    else:
                        txt = "(no caption or text)"

                if txt is not None:
                    txt_norm = (f"{label} " if label else "") + " ".join(txt.split())
                    
                    # Try to extract and save the actual image if it's a picture or formula
                    image_path_msg = ""
                    ocr_text_for_save = ""
                    if label in ("[IMAGE]", "[FORMULA]"):
                        try:
                            # docling 2.x method to get image of an item
                            img = item.get_image(doc)
                            if img:
                                import os
                                img_dir = f".refinery/extracted_images/{profile.doc_id}"
                                os.makedirs(img_dir, exist_ok=True)
                                page_no = getattr(item.prov[0], "page_no", 1) if getattr(item, "prov", None) else 1
                                # Give it a unique name using its id or bounding box hash
                                safe_name = f"page{page_no}_{abs(hash(txt_norm + str(item))) % 100000}.png"
                                img_path = os.path.join(img_dir, safe_name)
                                img.save(img_path)
                                image_path_msg = f" [Saved Image: {img_path}]"
                                
                                # Save OCR text alongside image if available
                                if label == "[IMAGE]" and ocr_text:
                                    ocr_text_for_save = ocr_text
                                    txt_file_path = img_path.replace('.png', '_ocr.txt')
                                    try:
                                        with open(txt_file_path, 'w', encoding='utf-8') as f:
                                            f.write(ocr_text)
                                        logger.debug(f"Saved OCR text to {txt_file_path}")
                                    except Exception as e:
                                        logger.debug(f"Failed to save OCR text: {e}")
                        except Exception as e:
                            logger.debug(f"Failed to extract image for {label}: {e}")
                            image_path_msg = f" [Image extraction failed]"

                    txt_norm += image_path_msg

                    if not txt_norm or txt_norm.strip() in ("[IMAGE]", "[FORMULA]"):
                        continue

                    page_num, bbox, has_prov = self._get_item_prov_bbox(item, doc)
                    if bbox is not None and (bbox.x1 - bbox.x0 > 0): # Only count real hits
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