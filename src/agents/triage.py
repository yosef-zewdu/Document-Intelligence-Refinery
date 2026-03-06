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
import logging
from src.utils.config_loader import load_refinery_config

# Define the logger for this file
logger = logging.getLogger(__name__)

class TriageAgent:
    def __init__(self, config: Dict[str, Any] = None):
        if not config:
            full_config = load_refinery_config()
            self.config = full_config.get("triage", {})
        else:
            self.config = config
            
        self.thresholds = self.config.get("thresholds", {
            "scanned_density_max": 0.0005,
            "digital_density_min": 0.001,
            "multi_column_x_offsets": 200,
            "table_heavy_word_ratio": 0.4
        })
        self.domain_keywords = self.config.get("domain_keywords", {})

    def classify(self, pdf_path: str) -> DocumentProfile:
        doc_id = os.path.basename(pdf_path)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    return self._default_profile(doc_id, pdf_path, "Empty PDF")
                
                densities = []
                image_ratios = []
                max_x_offsets = []
                
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    char_count = len(text)
                    page_area = float(page.width * page.height)
                    density = char_count / page_area if page_area > 0 else 0
                    densities.append(density)
                    
                    image_area = sum([float(img.get("width", 0) * img.get("height", 0)) for img in page.images])
                    image_ratio = image_area / page_area if page_area > 0 else 0
                    image_ratios.append(image_ratio)
                    
                    words = page.extract_words()
                    if words:
                        x_starts = sorted(list(set([round(w["x0"], 0) for w in words])))
                        max_x_offsets.append(len(x_starts))
                    else:
                        max_x_offsets.append(0)

                avg_density = sum(densities) / total_pages
                avg_image_ratio = sum(image_ratios) / total_pages
                
                # 1. Determine Origin Type
                if avg_density < self.thresholds.get("scanned_density_max", 0.0005) or avg_image_ratio > 0.9:
                    origin_type = OriginType.SCANNED_IMAGE
                else:
                    origin_type = OriginType.NATIVE_DIGITAL
                    
                # 2. Determine Layout Complexity
                avg_x_offsets = sum(max_x_offsets) / total_pages
                if avg_x_offsets > self.thresholds.get("multi_column_x_offsets", 200):
                    layout_complexity = LayoutComplexity.MULTI_COLUMN
                else:
                    layout_complexity = LayoutComplexity.SINGLE_COLUMN
                    
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
        except Exception as e:
            logger.error(f"Failed to triage {pdf_path}: {e}")
            return self._default_profile(doc_id, pdf_path, str(e))

    def _default_profile(self, doc_id: str, pdf_path: str, error: str) -> DocumentProfile:
        return DocumentProfile(
            doc_id=doc_id,
            filename=os.path.basename(pdf_path),
            origin_type=OriginType.NATIVE_DIGITAL,
            layout_complexity=LayoutComplexity.SINGLE_COLUMN,
            domain_hint=DomainHint.GENERAL,
            estimated_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
            metadata={"error": error, "is_fallback": True}
        )

    def _detect_domain(self, pages) -> DomainHint:
        import re
        text = ""
        for page in pages:
            text += (page.extract_text() or "").lower()
            
        scores = {domain: 0 for domain in self.domain_keywords}
        
        # Track best domain based on weighted keyword hits
        for domain_key, data in self.domain_keywords.items():
            stems = data.get("stems", {})
            for pattern, weight in stems.items():
                matches = re.findall(pattern, text)
                # Map yaml string key to DomainHint enum
                try:
                    enum_val = DomainHint(domain_key)
                    scores[domain_key] += len(matches) * weight
                except ValueError:
                    continue
        
        if not any(scores.values()):
            return DomainHint.GENERAL
            
        best_key = max(scores, key=scores.get)
        best_score = scores[best_key]
        
        # Confidence threshold from config
        required_score = self.domain_keywords.get(best_key, {}).get("confidence_threshold", 5)
        
        if best_score < required_score:
            return DomainHint.GENERAL
            
        return DomainHint(best_key)
