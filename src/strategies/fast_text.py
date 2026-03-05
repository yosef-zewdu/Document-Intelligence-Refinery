import pdfplumber
from .base import BaseExtractor
from src.models import (
    ExtractedDocument, 
    TextBlock, 
    TableStructure, 
    DocumentProfile, 
    BBox
)

class FastTextExtractor(BaseExtractor):
    def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        blocks = []
        tables = []
        
        # Pre-compute confidence to include in metadata
        confidence = self._compute_confidence(pdf_path)

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # ... (rest of extraction logic)
                words = page.extract_words()
                lines = {}
                for w in words:
                    y = round(float(w["top"]), 1)
                    if y not in lines:
                        lines[y] = []
                    lines[y].append(w)
                
                for y in sorted(lines.keys()):
                    line_words = sorted(lines[y], key=lambda x: x["x0"])
                    line_text = " ".join([w["text"] for w in line_words])
                    
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
                        tables.append(TableStructure(
                            headers=headers,
                            rows=rows,
                            page_num=i + 1
                        ))
                        
        return ExtractedDocument(
            doc_id=profile.doc_id,
            blocks=blocks,
            tables=tables,
            metadata={
                "strategy": "FastTextExtractor",
                "confidence": round(confidence, 4)
            }
        )

    def get_confidence_score(self, extracted: ExtractedDocument) -> float:
        return float(extracted.metadata.get("confidence", 0.0))

    def _compute_confidence(self, pdf_path: str) -> float:
        """Internal helper for original logic."""
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0: return 0.0
            scores = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                char_count = len(text)
                page_area = float(page.width * page.height)
                density = char_count / page_area if page_area > 0 else 0
                
                image_area = sum([float(img["width"] * img["height"]) for img in page.images])
                image_ratio = image_area / page_area if page_area > 0 else 0
                
                page_score = 1.0
                if density < 0.0005: page_score -= 0.5
                if image_ratio > 0.5: page_score -= 0.4
                scores.append(max(0.0, page_score))
            return sum(scores) / total_pages
