import os
import pdfplumber
from typing import Dict, Any, List
from src.models import (
    DocumentProfile, 
    OriginType, 
    LayoutComplexity, 
    DomainHint, 
    ExtractionCost
)

class TriageAgent:
    def __init__(self, thresholds: Dict[str, Any] = None):
        self.thresholds = thresholds or {
            "scanned_density_max": 0.0005,
            "digital_density_min": 0.001,
            "multi_column_word_gap": 50,
            "table_heavy_word_ratio": 0.4
        }

    def classify(self, pdf_path: str) -> DocumentProfile:
        doc_id = os.path.basename(pdf_path)
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            densities = []
            image_ratios = []
            word_counts = []
            max_x_offsets = []
            
            for page in pdf.pages:
                text = page.extract_text() or ""
                char_count = len(text)
                page_area = float(page.width * page.height)
                density = char_count / page_area if page_area > 0 else 0
                densities.append(density)
                
                image_area = sum([float(img["width"] * img["height"]) for img in page.images])
                image_ratio = image_area / page_area if page_area > 0 else 0
                image_ratios.append(image_ratio)
                
                words = page.extract_words()
                word_counts.append(len(words))
                
                if words:
                    # Collect unique starting x-positions to detect columns
                    x_starts = sorted(list(set([round(w["x0"], 0) for w in words])))
                    max_x_offsets.append(len(x_starts))

            avg_density = sum(densities) / total_pages if total_pages > 0 else 0
            avg_image_ratio = sum(image_ratios) / total_pages if total_pages > 0 else 0
            
            # 1. Determine Origin Type
            if avg_density < self.thresholds["scanned_density_max"] or avg_image_ratio > 0.9:
                origin_type = OriginType.SCANNED_IMAGE
            else:
                origin_type = OriginType.NATIVE_DIGITAL
                
            # 2. Determine Layout Complexity
            avg_x_offsets = sum(max_x_offsets) / total_pages if total_pages > 0 else 0
            if avg_x_offsets > 10: # Heuristic for multiple columns or complex tables
                layout_complexity = LayoutComplexity.MULTI_COLUMN
            else:
                layout_complexity = LayoutComplexity.SINGLE_COLUMN
                
            # 3. Determine Domain Hint (Simple keyword search in first 2 pages)
            domain_hint = self._detect_domain(pdf.pages[:2])
            
            # 4. Estimate Extraction Cost
            if origin_type == OriginType.SCANNED_IMAGE:
                estimated_cost = ExtractionCost.NEEDS_VISION_MODEL
            elif layout_complexity == LayoutComplexity.MULTI_COLUMN:
                estimated_cost = ExtractionCost.NEEDS_LAYOUT_MODEL
            else:
                estimated_cost = ExtractionCost.FAST_TEXT_SUFFICIENT

            return DocumentProfile(
                doc_id=doc_id,
                filename=os.path.basename(pdf_path),
                origin_type=origin_type,
                layout_complexity=layout_complexity,
                domain_hint=domain_hint,
                estimated_cost=estimated_cost,
                metadata={
                    "avg_char_density": f"{avg_density:.6f}",
                    "avg_image_ratio": f"{avg_image_ratio:.6f}",
                    "total_pages": total_pages
                }
            )

    def _detect_domain(self, pages) -> DomainHint:
        text = ""
        for page in pages:
            text += (page.extract_text() or "").lower()
            
        keywords = {
            DomainHint.FINANCIAL: ["report", "financial", "audit", "balance", "revenue", "fiscal", "tax"],
            DomainHint.LEGAL: ["law", "agreement", "contract", "court", "legal", "article"],
            DomainHint.TECHNICAL: ["system", "architecture", "technical", "manual", "specification"],
            DomainHint.MEDICAL: ["patient", "medical", "health", "clinical", "diagnosis"]
        }
        
        for domain, keys in keywords.items():
            if any(key in text for key in keys):
                return domain
                
        return DomainHint.GENERAL

if __name__ == "__main__":
    triage = TriageAgent()
    docs = [
        "data/CBE ANNUAL REPORT 2023-24.pdf",
        "data/Audit Report - 2023.pdf",
        "data/fta_performance_survey_final_report_2022.pdf",
        "data/tax_expenditure_ethiopia_2021_22.pdf"
    ]
    
    for doc in docs:
        if os.path.exists(doc):
            profile = triage.classify(doc)
            print(f"Profile for {doc}:")
            print(profile.model_dump_json(indent=2))
            
            # Save to .refinery/profiles
            os.makedirs(".refinery/profiles", exist_ok=True)
            output_path = f".refinery/profiles/{os.path.basename(doc)}.json"
            with open(output_path, "w") as f:
                f.write(profile.model_dump_json(indent=2))
