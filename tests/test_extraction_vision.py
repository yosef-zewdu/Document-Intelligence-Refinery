import os
import pytest
from src.strategies.vision_augmented import VisionExtractor
from src.models import (
    DocumentProfile, 
    OriginType, 
    LayoutComplexity, 
    DomainHint, 
    ExtractionCost
)

@pytest.fixture
def vision_extractor():
    # Set mock env var for testing
    os.environ["MOCK_VLM"] = "true"
    return VisionExtractor(max_pages_per_doc=2)

@pytest.fixture
def scanned_doc_profile():
    return DocumentProfile(
        doc_id="scanned_vision_test",
        filename="2022_Audited_Financial_Statement_Report.pdf", # Known scanned-like/complex
        origin_type=OriginType.SCANNED_IMAGE,
        layout_complexity=LayoutComplexity.MULTI_COLUMN,
        domain_hint=DomainHint.FINANCIAL,
        estimated_cost=ExtractionCost.NEEDS_VISION_MODEL,
        metadata={"total_pages": 40}
    )

def test_vision_extractor_page_selection(vision_extractor, scanned_doc_profile):
    pages = vision_extractor._choose_pages(scanned_doc_profile)
    # Default selection for 40 pages with max_pages_per_doc=2
    # Should be [0, 1] because of sorted(set(pages))[:self.max_pages_per_doc]
    assert len(pages) == 2
    assert 0 in pages

def test_vision_extraction_pipeline(vision_extractor, scanned_doc_profile):
    pdf_path = "tests/test_data/2022_Audited_Financial_Statement_Report.pdf" # Using a smaller PDF for faster rendering
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    extracted = vision_extractor.extract(pdf_path, scanned_doc_profile)
    
    assert len(extracted.blocks) > 0
    assert "VisionExtractor" in extracted.metadata["strategy"]
    assert extracted.confidence.signals["spend_usd"] > 0
    
    # Check that mock data was returned
    assert any("Sample text" in b.content for b in extracted.blocks)
    assert len(extracted.tables) > 0
    assert extracted.tables[0].headers == ["ID", "Summary", "Value"]

def test_vision_confidence_calculation(vision_extractor):
    # Test confidence helper directly
    mock_parsed = {
        "blocks": [{"content": "a" * 1000}], # > 600 chars
        "tables": [{"headers": ["A", "B"], "rows": [["1", "2"], ["3", "4"]]}] # 1 good table
    }
    # score = 0.5 (base) + 0.2 (text) + 0.1 (table) = 0.8
    conf = vision_extractor._compute_page_confidence(mock_parsed)
    assert conf >= 0.8

def test_vision_budget_check(vision_extractor):
    vision_extractor.budget_cap = 0.01 # Very low budget
    with pytest.raises(Exception) as excinfo:
        vision_extractor._budget_check(10) # 10 * 0.03 = 0.3 > 0.01
    assert "budget exceeded" in str(excinfo.value)
