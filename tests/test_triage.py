import pytest
import os
from src.agents.triage import TriageAgent
from src.models import OriginType, LayoutComplexity, DomainHint

@pytest.fixture
def triage_agent():
    return TriageAgent()

@pytest.mark.parametrize("pdf_path, expected_origin, expected_layout, expected_domain", [
    ("tests/test_data/2022_Audited_Financial_Statement_Report.pdf", OriginType.SCANNED_IMAGE, LayoutComplexity.FIGURE_HEAVY, DomainHint.GENERAL),
    ("tests/test_data/fta_performance_survey_final_report_2022.pdf", OriginType.NATIVE_DIGITAL, LayoutComplexity.TABLE_HEAVY, DomainHint.FINANCIAL),
    ("tests/test_data/tax_expenditure_ethiopia_2021_22.pdf", OriginType.NATIVE_DIGITAL, LayoutComplexity.TABLE_HEAVY, DomainHint.FINANCIAL),
])
def test_triage_classification(triage_agent, pdf_path, expected_origin, expected_layout, expected_domain):
    # Check if the file exists before running the test
    if not os.path.exists(pdf_path):
        pytest.skip(f"Test file {pdf_path} not found")
        
    profile = triage_agent.classify(pdf_path)
    
    assert profile.origin_type == expected_origin
    assert profile.layout_complexity == expected_layout
    assert profile.domain_hint == expected_domain
    assert profile.filename == os.path.basename(pdf_path)
    
    # Verify language detection fields are present
    assert hasattr(profile, 'language')
    assert hasattr(profile, 'language_confidence')
    assert 0.0 <= profile.language_confidence <= 1.0


def test_triage_custom_thresholds():
    custom_config = {
        "thresholds": {
            "scanned_density_max": 1.0, # Unrealistically high to force scanned classification
        }
    }
    agent = TriageAgent(config=custom_config)
    
    # Even a digital file should be classified as scanned with this threshold
    # Assuming 'tests/test_data/CBE ANNUAL REPORT 2023-24.pdf' exists and has density < 1.0
    doc_path = "tests/test_data/Delivery Challenge Week 2.pdf"
    if os.path.exists(doc_path):
        profile = agent.classify(doc_path)
        assert profile.origin_type == OriginType.SCANNED_IMAGE


def test_triage_metadata_fields(triage_agent):
    """Test that all expected metadata fields are present."""
    doc_path = "tests/test_data/fta_performance_survey_final_report_2022.pdf"
    if not os.path.exists(doc_path):
        pytest.skip(f"Test file {doc_path} not found")
    
    profile = triage_agent.classify(doc_path)
    
    # Check metadata fields
    assert "avg_char_density" in profile.metadata
    assert "avg_image_ratio" in profile.metadata
    assert "avg_x_offsets" in profile.metadata
    assert "avg_table_count" in profile.metadata
    assert "total_pages" in profile.metadata
    assert "sampled_pages" in profile.metadata


def test_language_detection(triage_agent):
    """Test language detection functionality."""
    doc_path = "tests/test_data/fta_performance_survey_final_report_2022.pdf"
    if not os.path.exists(doc_path):
        pytest.skip(f"Test file {doc_path} not found")
    
    profile = triage_agent.classify(doc_path)
    
    # Language should be detected (likely 'en' for English documents)
    assert profile.language is not None
    assert len(profile.language) >= 2  # Language codes are at least 2 chars
    assert profile.language_confidence >= 0.0
    assert profile.language_confidence <= 1.0

