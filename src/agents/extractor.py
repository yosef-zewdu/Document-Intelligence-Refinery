import logging
import json
import os
from datetime import datetime
from typing import List, Optional
from src.models import DocumentProfile, ExtractedDocument, ExtractionCost, OriginType
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.vision_augmented import VisionExtractor

class ExtractionRouter:
    def __init__(self, ledger_path: str = ".refinery/extraction_ledger.jsonl"):
        self.fast_extractor = FastTextExtractor()
        self.layout_extractor = LayoutExtractor()
        self.vision_extractor = VisionExtractor()
        self.ledger_path = ledger_path
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)

    def route_and_extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        strategy_used = "Strategy A"
        extractor = self.fast_extractor
        
        if profile.origin_type == OriginType.SCANNED_IMAGE or \
           profile.estimated_cost == ExtractionCost.NEEDS_VISION_MODEL:
            strategy_used = "Strategy C"
            extractor = self.vision_extractor
        elif profile.estimated_cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
            strategy_used = "Strategy B"
            extractor = self.layout_extractor
        
        # Execute Extraction
        start_time = datetime.now()
        doc = extractor.extract(pdf_path, profile)
        confidence = extractor.get_confidence_score(doc)
        end_time = datetime.now()
        
        # Escalation Guard
        if confidence < 0.5 and strategy_used == "Strategy A":
            logging.info(f"Low confidence ({confidence}) for Strategy A. Escalating to Strategy B.")
            strategy_used = "Strategy B (Escalated)"
            extractor = self.layout_extractor
            doc = extractor.extract(pdf_path, profile)
            confidence = extractor.get_confidence_score(doc)

        # Log to Ledger
        self._log_to_ledger(profile.doc_id, strategy_used, confidence, (end_time - start_time).total_seconds())
        
        return doc

    def _log_to_ledger(self, doc_id: str, strategy: str, confidence: float, duration: float):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "strategy_used": strategy,
            "confidence_score": confidence,
            "processing_time_sec": duration,
            "cost_estimate": 0.0 # Placeholder for cost tracking
        }
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
