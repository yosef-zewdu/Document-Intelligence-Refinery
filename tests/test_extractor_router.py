import os
import json
import pytest
from unittest.mock import MagicMock
from src.agents.extractor import ExtractionRouter
from src.models import (
    DocumentProfile, 
    OriginType, 
    LayoutComplexity, 
    DomainHint, 
    ExtractionCost,
    ExtractedDocument,
    TextBlock,
    BBox
)

@pytest.fixture
def extraction_router():
    path = "/tmp/test_extraction_ledger.jsonl"
    if os.path.exists(path):
        os.remove(path)
    return ExtractionRouter(ledger_path=path)

def test_router_fast_path(extraction_router):
    pdf_path = "data/interim_report.pdf" 
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    profile = DocumentProfile(
        doc_id="router_fast_test",
        filename=os.path.basename(pdf_path),
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        domain_hint=DomainHint.GENERAL,
        estimated_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
        metadata={"total_pages": 5}
    )
    
    doc = extraction_router.route_and_extract(pdf_path, profile)
    
    # Standard check
    assert "FastTextExtractor" in doc.metadata["strategy"]
    assert doc.confidence.score >= 0.8

def test_router_escalation_path(extraction_router, monkeypatch):
    # Force Strategy A -> Strategy B escalation
    pdf_path = "tests/test_data/interim_report.pdf" 
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    profile = DocumentProfile(
        doc_id="router_escalation_test",
        filename=os.path.basename(pdf_path),
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        domain_hint=DomainHint.GENERAL,
        estimated_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
        metadata={"total_pages": 5}
    )

    # Mock get_confidence_score to return 0.1 for Strategy A
    # and 0.9 for Strategy B to stop there.
    def mock_step_conf(doc):
        strategy = doc.metadata.get("strategy", "")
        if "FastTextExtractor" in strategy:
            return 0.1
        return 0.9

    monkeypatch.setattr(extraction_router.fast_extractor, "get_confidence_score", mock_step_conf)
    monkeypatch.setattr(extraction_router.layout_extractor, "get_confidence_score", mock_step_conf)
    
    doc = extraction_router.route_and_extract(pdf_path, profile)
    
    # Should have escalated to Strategy B and stopped there
    assert "LayoutExtractor" in doc.metadata["strategy"]
    
    # Check ledger for both attempts
    with open(extraction_router.ledger_path, "r") as f:
        log_lines = [json.loads(line) for line in f if line.strip()]
        strategies = [entry["strategy"] for entry in log_lines if entry["doc_id"] == "router_escalation_test"]
        assert "Strategy A" in strategies
        assert "Strategy B" in strategies

def test_router_vision_path(extraction_router):
    # Set mock env var for vision extractor
    os.environ["MOCK_VLM"] = "true"
    
    pdf_path = "tests/test_data/interim_report.pdf" 
    profile = DocumentProfile(
        doc_id="router_vision_test",
        filename=os.path.basename(pdf_path),
        origin_type=OriginType.SCANNED_IMAGE,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        domain_hint=DomainHint.GENERAL,
        estimated_cost=ExtractionCost.NEEDS_VISION_MODEL,
        metadata={"total_pages": 5}
    )
    
    doc = extraction_router.route_and_extract(pdf_path, profile)
    
    assert "VisionExtractor" in doc.metadata["strategy"]
    assert doc.confidence.score > 0.4 # Mock VLM should have some confidence
