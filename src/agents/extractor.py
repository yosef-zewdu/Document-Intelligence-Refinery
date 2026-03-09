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
        
        # Load extraction rules from rubric/extraction_rules.yaml
        self.extraction_rules = full_config.get("extraction_rules", {})
        
        self.fast_extractor = FastTextExtractor()
        self.layout_extractor = LayoutExtractor()
        self.vision_extractor = VisionExtractor(
            config=self.config.get("vision_extractor", {})
        )
        self.ledger_path = ledger_path
        
        # Get thresholds from extraction_rules if available, fallback to old config
        escalation_config = self.extraction_rules.get("escalation", {})
        self.thresholds = {
            "strategy_a_threshold": escalation_config.get("from_strategy_a", {}).get("threshold", 
                                                          self.config.get("escalation", {}).get("strategy_a_threshold", 0.5)),
            "strategy_b_threshold": escalation_config.get("from_strategy_b", {}).get("threshold",
                                                          self.config.get("escalation", {}).get("strategy_b_threshold", 0.7))
        }
        
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)

    def route_and_extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        """
        Orchestrates full A → B → C escalation flow with cost tracking.
        """
        initial_strategy = self._get_initial_strategy(profile)
        current_strategy = initial_strategy
        doc: Optional[ExtractedDocument] = None
        errors = []
        total_cost = 0.0
        escalation_history = []

        # Tiered Execution Loop
        while True:
            logger.info(f"Executing {current_strategy} for {pdf_path}")
            try:
                extractor = self._get_extractor(current_strategy)
                start_time = datetime.now()
                doc = extractor.extract(pdf_path, profile)
                duration = (datetime.now() - start_time).total_seconds()
                
                confidence = extractor.get_confidence_score(doc)
                
                # Estimate cost for this strategy
                cost = self._estimate_cost(current_strategy, profile, doc)
                total_cost += cost
                
                # Determine escalation reason if applicable
                escalation_reason = None
                if escalation_history:
                    escalation_reason = escalation_history[-1]
                
                # Log to ledger with cost and confidence object
                self._log_to_ledger(
                    profile.doc_id, 
                    current_strategy, 
                    confidence, 
                    duration, 
                    doc.metadata,
                    cost_estimate=cost,
                    escalation_reason=escalation_reason,
                    confidence_obj=doc.confidence
                )

                # Check for escalation
                next_strategy = self._check_escalation(current_strategy, confidence)
                if next_strategy:
                    reason = f"Low confidence ({confidence:.2f}) - escalating from {current_strategy} to {next_strategy}"
                    escalation_history.append(reason)
                    logger.warning(reason)
                    current_strategy = next_strategy
                    continue
                else:
                    # Success - add cost summary to metadata
                    doc.metadata["total_cost_usd"] = round(total_cost, 4)
                    doc.metadata["escalation_history"] = escalation_history
                    return doc

            except Exception as e:
                logger.error(f"Strategy {current_strategy} failed: {e}")
                errors.append(f"{current_strategy}: {str(e)}")
                
                # Hard failure escalation
                next_strategy = self._get_next_tier(current_strategy)
                if next_strategy:
                    reason = f"Strategy {current_strategy} failed with error - escalating to {next_strategy}"
                    escalation_history.append(reason)
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

    def _estimate_cost(self, strategy: str, profile: DocumentProfile, doc: ExtractedDocument) -> float:
        """
        Estimate extraction cost based on strategy and document.
        Returns cost in USD.
        """
        if strategy == "Strategy A":
            return 0.0  # Free (local processing)
        elif strategy == "Strategy B":
            # Estimate based on pages processed
            pages = int(profile.metadata.get("total_pages", 1))
            cost_per_page = 0.01  # $0.01 per page for Docling
            return pages * cost_per_page
        elif strategy == "Strategy C":
            # Get actual spend from vision extractor
            return float(doc.metadata.get("spend_usd", 0.0))
        return 0.0

    def _log_to_ledger(self, doc_id: str, strategy: str, confidence: float, duration: float, 
                       metadata: Dict[str, Any] = None, cost_estimate: float = 0.0, 
                       escalation_reason: Optional[str] = None, confidence_obj = None):
        """
        Log extraction attempt to ledger with all required fields from spec.
        """
        # Extract confidence signals and warnings from the confidence object
        confidence_signals = {}
        warnings = []
        
        if confidence_obj:
            if hasattr(confidence_obj, 'signals'):
                confidence_signals = confidence_obj.signals or {}
            if hasattr(confidence_obj, 'warnings'):
                warnings = confidence_obj.warnings or []
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "strategy_used": strategy,
            "confidence_score": round(confidence, 4),
            "processing_time": round(duration, 3),
            "cost_estimate": round(cost_estimate, 4),
            "escalation_reason": escalation_reason,
            "confidence_signals": confidence_signals,
            "warnings": warnings
        }
        try:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to ledger: {e}")
