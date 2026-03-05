import logging
import json
import os
from datetime import datetime
from typing import List, Optional, Any, Dict
from src.models import DocumentProfile, ExtractedDocument, ExtractionCost, OriginType, ConfidenceMetadata
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout_aware import LayoutExtractor
from src.strategies.vision_augmented import VisionExtractor

from src.utils.config_loader import load_refinery_config

logger = logging.getLogger(__name__)

class ExtractionRouter:
    def __init__(self, ledger_path: str = ".refinery/extraction_ledger.jsonl"):
        full_config = load_refinery_config()
        self.config = full_config.get("extraction", {})
        
        self.fast_extractor = FastTextExtractor()
        self.layout_extractor = LayoutExtractor()
        self.vision_extractor = VisionExtractor(
            config=self.config.get("vision_extractor", {})
        )
        self.ledger_path = ledger_path
        
        self.thresholds = self.config.get("escalation", {
            "strategy_a_threshold": 0.5,
            "strategy_b_threshold": 0.7
        })
        
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)

    def route_and_extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        """
        Orchestrates full A -> B -> C escalation flow.
        """
        # Determine initial strategy
        initial_strategy = self._get_initial_strategy(profile)
        
        current_strategy = initial_strategy
        doc: Optional[ExtractedDocument] = None
        errors = []

        # Tiered Execution Loop
        while True:
            logger.info(f"Executing {current_strategy} for {pdf_path}")
            try:
                extractor = self._get_extractor(current_strategy)
                start_time = datetime.now()
                doc = extractor.extract(pdf_path, profile)
                duration = (datetime.now() - start_time).total_seconds()
                
                confidence = extractor.get_confidence_score(doc)
                self._log_to_ledger(profile.doc_id, current_strategy, confidence, duration, metadata=doc.metadata)

                # Check for escalation
                next_strategy = self._check_escalation(current_strategy, confidence)
                if next_strategy:
                    logger.warning(f"Low confidence ({confidence}) for {current_strategy}. Escalating to {next_strategy}.")
                    current_strategy = next_strategy
                    continue
                else:
                    return doc

            except Exception as e:
                logger.error(f"Strategy {current_strategy} failed: {e}")
                errors.append(f"{current_strategy}: {str(e)}")
                
                # Hard failure escalation
                next_strategy = self._get_next_tier(current_strategy)
                if next_strategy:
                    current_strategy = next_strategy
                    continue
                else:
                    # Final failure
                    return self._final_fallback(profile, errors)

    def _get_initial_strategy(self, profile: DocumentProfile) -> str:
        if profile.origin_type == OriginType.SCANNED_IMAGE or \
           profile.estimated_cost == ExtractionCost.NEEDS_VISION_MODEL:
            return "Strategy C"
        elif profile.estimated_cost == ExtractionCost.NEEDS_LAYOUT_MODEL:
            return "Strategy B"
        return "Strategy A"

    def _get_extractor(self, strategy: str):
        if strategy == "Strategy A": return self.fast_extractor
        if strategy == "Strategy B": return self.layout_extractor
        if strategy == "Strategy C": return self.vision_extractor
        raise ValueError(f"Unknown strategy: {strategy}")

    def _check_escalation(self, current: str, confidence: float) -> Optional[str]:
        if current == "Strategy A" and confidence < self.thresholds.get("strategy_a_threshold", 0.5):
            return "Strategy B"
        if current == "Strategy B" and confidence < self.thresholds.get("strategy_b_threshold", 0.7):
            return "Strategy C"
        return None

    def _get_next_tier(self, current: str) -> Optional[str]:
        if current == "Strategy A": return "Strategy B"
        if current == "Strategy B": return "Strategy C"
        return None

    def _final_fallback(self, profile: DocumentProfile, errors: List[str]) -> ExtractedDocument:
        logger.critical(f"All extraction strategies failed for {profile.doc_id}")
        return ExtractedDocument(
            doc_id=profile.doc_id,
            metadata={"errors": errors, "final_state": "failed"},
            confidence=ConfidenceMetadata(score=0.0, method="failure", warnings=errors)
        )

    def _log_to_ledger(self, doc_id: str, strategy: str, confidence: float, duration: float, metadata: Dict[str, Any] = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "strategy": strategy,
            "confidence": confidence,
            "duration": duration,
            "metadata_summary": {k: v for k, v in (metadata or {}).items() if k not in ["blocks", "tables"]}
        }
        try:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to ledger: {e}")
