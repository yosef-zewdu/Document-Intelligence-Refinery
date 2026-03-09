import os
import pytest
from src.strategies.layout_aware import LayoutExtractor
from src.models import (
    DocumentProfile, 
    OriginType, 
    LayoutComplexity, 
    DomainHint, 
    ExtractionCost
)

# Mark all tests in this module as slow
pytestmark = pytest.mark.slow

@pytest.fixture
def layout_extractor():
    return LayoutExtractor()

@pytest.fixture
def complex_doc_profile():
    return DocumentProfile(
        doc_id="complex_layout_test",
        filename="interim_report.pdf",
        origin_type=OriginType.NATIVE_DIGITAL,
        layout_complexity=LayoutComplexity.MULTI_COLUMN,
        domain_hint=DomainHint.GENERAL,
        estimated_cost=ExtractionCost.NEEDS_LAYOUT_MODEL,
        metadata={"total_pages": 5}
    )

def test_layout_extractor_confidence(layout_extractor, complex_doc_profile):
    pdf_path = "data/interim_report.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    extracted = layout_extractor.extract(pdf_path, complex_doc_profile)
    confidence = layout_extractor.get_confidence_score(extracted)
    assert 0.0 <= confidence <= 1.0

def test_layout_extractor_extraction(layout_extractor, complex_doc_profile):
    pdf_path = "data/tax_expenditure_ethiopia_2021_22.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    extracted = layout_extractor.extract(pdf_path, complex_doc_profile)
    
    print(f"DEBUG: Warnings: {extracted.confidence.warnings}")
    print(f"DEBUG: Metadata: {extracted.metadata}")
    print(f"DEBUG: Blocks count: {len(extracted.blocks)}")
    print(f"DEBUG: Tables count: {len(extracted.tables)}")

    assert len(extracted.blocks) > 0
    assert "LayoutExtractor" in extracted.metadata["strategy"]
    
    # Verify we got some tables (this doc is known to have many)
    assert len(extracted.tables) > 0
    
    # Check a table structure
    first_table = extracted.tables[0]
    assert len(first_table.headers) > 0
    assert len(first_table.rows) > 0
    assert first_table.page_num >= 1

def test_layout_extractor_provenance(layout_extractor, complex_doc_profile):
    pdf_path = "data/interim_report.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"{pdf_path} not found")
        
    extracted = layout_extractor.extract(pdf_path, complex_doc_profile)
    
    # Docling blocks should have non-zero bboxes if provenance works
    # Note: Some meta blocks might have 0,0,0,0, so we check if at least some have data
    bboxes = [b.bbox for b in extracted.blocks if b.bbox.x1 > 0]
    assert len(bboxes) > 0
