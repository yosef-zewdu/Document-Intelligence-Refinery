import os
import pytest
from src.strategies.fast_text import FastTextExtractor
from src.models import (
    DocumentProfile, 
    OriginType, 
    LayoutComplexity, 
    DomainHint, 
    ExtractionCost
)

@pytest.fixture
def fast_extractor():
    return FastTextExtractor()

@pytest.fixture
def delivery_challenge_profile():
    return DocumentProfile(
        doc_id="delivery_challenge_test",
        filename="Delivery Challenge Week 2.pdf",
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.SINGLE_COLUMN,
        domain_hint=DomainHint.GENERAL,
        estimated_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
        metadata={}
    )

def test_fast_text_confidence(fast_extractor, delivery_challenge_profile):
    pdf_path = "tests/test_data/Delivery Challenge Week 2.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    extracted = fast_extractor.extract(pdf_path, delivery_challenge_profile)
    confidence = fast_extractor.get_confidence_score(extracted)
    assert confidence > 0.9 # Should be high for native digital text

def test_fast_text_extraction(fast_extractor, delivery_challenge_profile):
    pdf_path = "tests/test_data/Delivery Challenge Week 2.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    extracted = fast_extractor.extract(pdf_path, delivery_challenge_profile)
    
    assert len(extracted.blocks) > 0
    assert extracted.metadata["strategy"] == "FastTextExtractor"
    
    # Check for specific expected content
    content_str = " ".join([b.content for b in extracted.blocks])
    assert "Delivery Challenge Week 2" in content_str
    assert "Structured Question Generation" in content_str

def test_fast_text_scanned_confidence(fast_extractor, delivery_challenge_profile):
    # A scanned PDF should have significantly lower confidence
    pdf_path = "tests/test_data/Audit Report - 2023.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    extracted = fast_extractor.extract(pdf_path, delivery_challenge_profile)
    confidence = fast_extractor.get_confidence_score(extracted)
    assert confidence < 0.3 # Scanned image should trigger low confidence
