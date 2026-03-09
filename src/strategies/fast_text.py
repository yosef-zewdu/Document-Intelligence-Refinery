import pdfplumber
import logging
from typing import List, Dict, Any
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

class FastTextExtractor(BaseExtractor):
    def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        blocks = []
        tables = []
        warnings = []
        signals = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    return self._empty_document(profile.doc_id, "PDF has no pages")

                for i, page in enumerate(pdf.pages):
                    try:
                        words = page.extract_words()
                        if not words:
                            warnings.append(f"Page {i+1} appears to have no text.")
                            continue

                        lines = {}
                        for w in words:
                            y = round(float(w["top"]), 1)
                            if y not in lines:
                                lines[y] = []
                            lines[y].append(w)
                        
                        for y in sorted(lines.keys()):
                            line_words = sorted(lines[y], key=lambda x: x["x0"])
                            line_text = " ".join([w["text"] for w in line_words])
                            
                            if not line_text.strip():
                                continue

                            x0 = min([float(w["x0"]) for w in line_words])
                            y0 = min([float(w["top"]) for w in line_words])
                            x1 = max([float(w["x1"]) for w in line_words])
                            y1 = max([float(w["bottom"]) for w in line_words])
                            
                            blocks.append(TextBlock(
                                content=line_text,
                                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                                page_num=i + 1
                            ))
                        
                        page_tables = page.extract_tables()
                        for table in page_tables:
                            if table and any(row and any(cell for cell in row) for row in table):
                                clean_table = [[str(cell) if cell is not None else "" for cell in row] for row in table]
                                headers = clean_table[0]
                                rows = clean_table[1:]
                                
                                # Validate table structure
                                if not any(headers) and not any(rows):
                                    continue
                                    
                                tables.append(TableStructure(
                                    headers=headers,
                                    rows=rows,
                                    page_num=i + 1
                                ))
                    except Exception as e:
                        warnings.append(f"Error extracting page {i+1}: {str(e)}")

        except Exception as e:
            logger.error(f"FastTextExtractor failed to open {pdf_path}: {e}")
            return self._empty_document(profile.doc_id, str(e))

        # Compute confidence with detailed signals
        confidence_meta = self._compute_confidence_metadata(pdf_path, blocks, tables, warnings)
        
        return ExtractedDocument(
            doc_id=profile.doc_id,
            blocks=blocks,
            tables=tables,
            metadata={
                "strategy": "FastTextExtractor",
                "warnings": warnings
            },
            confidence=confidence_meta
        )

    def get_confidence_score(self, extracted: ExtractedDocument) -> float:
        if extracted.confidence:
            return extracted.confidence.score
        return 0.0

    def _compute_confidence_metadata(self, pdf_path: str, blocks: List[TextBlock], tables: List[TableStructure], warnings: List[str]) -> ConfidenceMetadata:
        """
        Enhanced confidence calculation with all required signals from spec.
        Implements multi-signal scoring as documented in DOMAIN_NOTES.md and extraction_rules.yaml
        """
        score = 1.0
        signals = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                signals["total_pages"] = total_pages
                
                if total_pages == 0:
                    return ConfidenceMetadata(
                        score=0.0, 
                        method="multi_signal_heuristic", 
                        warnings=["Zero pages"]
                    )

                # Collect metrics from all pages
                page_char_counts = []
                densities = []
                image_ratios = []
                font_metadata_found = False
                
                for page in pdf.pages:
                    # Signal 1: Character count per page
                    text = page.extract_text() or ""
                    char_count = len(text)
                    page_char_counts.append(char_count)
                    
                    # Signal 2: Character density
                    page_area = float(page.width * page.height)
                    density = char_count / page_area if page_area > 0 else 0
                    densities.append(density)
                    
                    # Signal 3: Image area ratio
                    image_area = sum([float(img.get("width", 0) * img.get("height", 0)) for img in page.images])
                    image_ratio = image_area / page_area if page_area > 0 else 0
                    image_ratios.append(image_ratio)
                    
                    # Signal 4: Font metadata presence
                    if not font_metadata_found:
                        chars = page.chars
                        if chars and any(c.get("fontname") for c in chars):
                            font_metadata_found = True
                
                # Calculate averages
                avg_char_count = sum(page_char_counts) / total_pages
                avg_density = sum(densities) / total_pages
                avg_image_ratio = sum(image_ratios) / total_pages
                
                # Store signals
                signals["avg_char_count_per_page"] = round(avg_char_count, 2)
                signals["avg_char_density"] = round(avg_density, 6)
                signals["avg_image_ratio"] = round(avg_image_ratio, 4)
                signals["font_metadata_present"] = font_metadata_found
                
                # Multi-signal confidence scoring (weights from extraction_rules.yaml)
                # Thresholds: char_count >= 100, density >= 0.0005, image_ratio <= 0.50, font_metadata = true
                char_count_signal = 1.0 if avg_char_count >= 100 else (avg_char_count / 100)
                char_density_signal = 1.0 if avg_density >= 0.0005 else (avg_density / 0.0005)
                image_ratio_signal = 1.0 if avg_image_ratio <= 0.50 else max(0.0, 1.0 - (avg_image_ratio - 0.50))
                font_metadata_signal = 1.0 if font_metadata_found else 0.0
                
                # Weighted combination (30% image, 25% char count, 25% density, 20% font)
                score = (
                    0.30 * max(0.0, min(1.0, image_ratio_signal)) +
                    0.25 * max(0.0, min(1.0, char_count_signal)) +
                    0.25 * max(0.0, min(1.0, char_density_signal)) +
                    0.20 * font_metadata_signal
                )
                
                signals["char_count_signal"] = round(char_count_signal, 4)
                signals["char_density_signal"] = round(char_density_signal, 4)
                signals["image_ratio_signal"] = round(image_ratio_signal, 4)
                signals["font_metadata_signal"] = round(font_metadata_signal, 4)
                
                # Add warnings for failed checks
                if avg_char_count < 100:
                    warnings.append(f"Low character count per page ({avg_char_count:.0f} < 100)")
                if avg_density < 0.0005:
                    warnings.append(f"Low character density ({avg_density:.6f} < 0.0005) - likely scanned")
                if avg_image_ratio > 0.50:
                    warnings.append(f"High image area ratio ({avg_image_ratio:.2%} > 50%) - text extraction incomplete")
                if not font_metadata_found:
                    warnings.append("No font metadata detected - may indicate scanned content")
                    
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            score = 0.0
            warnings.append(f"Confidence calculation error: {str(e)}")

        # Penalize for zero content
        if not blocks and not tables:
            score = 0.0
            warnings.append("No content extracted")

        return ConfidenceMetadata(
            score=max(0.0, min(1.0, score)),
            method="multi_signal_heuristic",
            warnings=warnings,
            signals=signals
        )

    def _empty_document(self, doc_id: str, error: str) -> ExtractedDocument:
        return ExtractedDocument(
            doc_id=doc_id,
            metadata={"error": error, "strategy": "FastTextExtractor"},
            confidence=ConfidenceMetadata(score=0.0, method="failure", warnings=[error])
        )
