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
        """Richer confidence calculation for Strategy A."""
        score = 1.0
        signals = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                signals["total_pages"] = total_pages
                
                if total_pages == 0:
                    return ConfidenceMetadata(score=0.0, method="heuristic", warnings=["Zero pages"])

                page_scores = []
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    char_count = len(text)
                    page_area = float(page.width * page.height)
                    density = char_count / page_area if page_area > 0 else 0
                    
                    image_area = sum([float(img["width"] * img["height"]) for img in page.images])
                    image_ratio = image_area / page_area if page_area > 0 else 0
                    
                    page_score = 1.0
                    if density < 0.0005: 
                        page_score -= 0.5
                    if image_ratio > 0.5: 
                        page_score -= 0.4
                    page_scores.append(max(0.0, page_score))
                
                avg_page_score = sum(page_scores) / total_pages
                score = avg_page_score
                signals["avg_page_score"] = round(avg_page_score, 4)
                signals["density_check"] = "pass" if any(s > 0.5 for s in page_scores) else "fail"
        except:
            score = 0.0
            warnings.append("Could not compute confidence signals")

        # Penalize for warnings
        if warnings:
            score -= 0.1 * len(warnings)
            
        # Penalize for zero blocks
        if not blocks and not tables:
            score = 0.0
            warnings.append("No content extracted")

        return ConfidenceMetadata(
            score=max(0.0, min(1.0, score)),
            method="heuristic",
            warnings=warnings,
            signals=signals
        )

    def _empty_document(self, doc_id: str, error: str) -> ExtractedDocument:
        return ExtractedDocument(
            doc_id=doc_id,
            metadata={"error": error, "strategy": "FastTextExtractor"},
            confidence=ConfidenceMetadata(score=0.0, method="failure", warnings=[error])
        )
