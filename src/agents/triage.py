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
        import re
        text = ""
        for page in pages:
            text += (page.extract_text() or "").lower()
            
        # Weighted scoring system using stems and variations
        domain_scores = {
            DomainHint.FINANCIAL: {
                r"financi\w*": 3, r"audit\w*": 5, r"fiscal": 5, r"tax\w*": 4, 
                r"revenue": 3, r"balance\s+sheet": 5, r"expenditure": 4, 
                r"invoic\w*": 3, r"account\w*": 3, r"econom\w*": 2, 
                r"statement": 2, r"profit": 3, r"loss": 2, r"budget": 3
            },
            DomainHint.LEGAL: {
                r"\blaw\w*": 3, r"agreement": 4, r"contract": 5, r"court": 5, 
                r"legal\w*": 3, r"statut\w*": 5, r"provision": 2, 
                r"jurisdiction": 4, r"liability": 3, r"clause": 4,
                r"regulation": 3, r"compliance": 2, r"policy": 1
            },
            DomainHint.TECHNICAL: {
                r"system": 1, r"architect\w*": 4, r"technical": 3, 
                r"complexit\w*": 2, r"specification": 4, r"software": 5, 
                r"hardwar\w*": 5, r"algorithm": 5, r"deployment": 3, 
                r"protocol": 4, r"engineer\w*": 3, r"data": 1
            },
            DomainHint.MEDICAL: {
                r"patient": 5, r"medical": 3, r"health": 2, 
                r"clinical": 5, r"diagnosis": 5, r"treatment": 3, 
                r"hospital": 4, r"physician": 5, r"pharmaceut\w*": 5
            }
        }
        
        scores = {domain: 0 for domain in domain_scores}
        
        for domain, keywords in domain_scores.items():
            for pattern, weight in keywords.items():
                matches = re.findall(pattern, text)
                scores[domain] += len(matches) * weight
        
        if not any(scores.values()):
            return DomainHint.GENERAL
            
        best_domain = max(scores, key=scores.get)
        
        # Confidence threshold
        if scores[best_domain] < 5:
            return DomainHint.GENERAL
            
        return best_domain

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
