import pytest
import os
from src.agents.triage import TriageAgent
from src.models import OriginType, LayoutComplexity, DomainHint

@pytest.fixture
def triage_agent():
    return TriageAgent()

@pytest.mark.parametrize("pdf_path, expected_origin, expected_layout, expected_domain", [
    ("data/CBE ANNUAL REPORT 2023-24.pdf", OriginType.NATIVE_DIGITAL, LayoutComplexity.MULTI_COLUMN, DomainHint.GENERAL),
    ("data/Audit Report - 2023.pdf", OriginType.SCANNED_IMAGE, LayoutComplexity.SINGLE_COLUMN, DomainHint.FINANCIAL),
    ("data/fta_performance_survey_final_report_2022.pdf", OriginType.NATIVE_DIGITAL, LayoutComplexity.MULTI_COLUMN, DomainHint.FINANCIAL),
    ("data/tax_expenditure_ethiopia_2021_22.pdf", OriginType.NATIVE_DIGITAL, LayoutComplexity.MULTI_COLUMN, DomainHint.FINANCIAL),
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

def test_triage_invalid_file(triage_agent):
    with pytest.raises(Exception):
        triage_agent.classify("non_existent_file.pdf")

def test_triage_custom_thresholds():
    custom_thresholds = {
        "scanned_density_max": 1.0, # Unrealistically high to force scanned classification
    }
    agent = TriageAgent(thresholds=custom_thresholds)
    
    # Even a digital file should be classified as scanned with this threshold
    # Assuming 'data/CBE ANNUAL REPORT 2023-24.pdf' exists and has density < 1.0
    doc_path = "data/CBE ANNUAL REPORT 2023-24.pdf"
    if os.path.exists(doc_path):
        profile = agent.classify(doc_path)
        assert profile.origin_type == OriginType.SCANNED_IMAGE
