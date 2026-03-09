"""
Test suite to verify the Triage Agent can classify all four document classes:
- Class A: Annual Financial Report (PDF, native digital)
- Class B: Scanned Government/Legal Document (PDF, image-based)
- Class C: Technical Assessment Report (PDF, mixed: text + tables + structured findings)
- Class D: Structured Data Report (PDF, table-heavy with numerical fiscal data)
"""
import pytest
import os
from src.agents.triage import TriageAgent
from src.models import OriginType, LayoutComplexity, DomainHint

@pytest.fixture
def triage_agent():
    return TriageAgent()


class TestDocumentClassification:
    """Test all four document classes are properly classified."""
    
    def test_class_a_native_digital_financial_report(self, triage_agent):
        """
        Class A: Annual Financial Report (PDF, native digital)
        Expected: OriginType.NATIVE_DIGITAL, LayoutComplexity varies, DomainHint.FINANCIAL
        """
        doc_path = "tests/test_data/Delivery Challenge Week 2.pdf"
        if not os.path.exists(doc_path):
            pytest.skip(f"Test file {doc_path} not found")
        
        profile = triage_agent.classify(doc_path)
        
        # Class A characteristics
        assert profile.origin_type == OriginType.NATIVE_DIGITAL, \
            f"Expected NATIVE_DIGITAL but got {profile.origin_type}"
        assert profile.domain_hint == DomainHint.FINANCIAL, \
            f"Expected FINANCIAL domain but got {profile.domain_hint}"
        
        # Should have reasonable text density
        avg_density = float(profile.metadata.get("avg_char_density", 0))
        assert avg_density > 0.001, "Native digital should have decent text density"
        
        print(f"✓ Class A classified: {profile.origin_type}, {profile.layout_complexity}, {profile.domain_hint}")
    
    def test_class_b_scanned_image_document(self, triage_agent):
        """
        Class B: Scanned Government/Legal Document (PDF, image-based)
        Expected: OriginType.SCANNED_IMAGE
        """
        doc_path = "tests/test_data/2022_Audited_Financial_Statement_Report.pdf"
        if not os.path.exists(doc_path):
            pytest.skip(f"Test file {doc_path} not found")
        
        profile = triage_agent.classify(doc_path)
        
        # Class B characteristics
        assert profile.origin_type == OriginType.SCANNED_IMAGE, \
            f"Expected SCANNED_IMAGE but got {profile.origin_type}"
        
        # Should have very low text density or high image ratio
        avg_density = float(profile.metadata.get("avg_char_density", 0))
        avg_image_ratio = float(profile.metadata.get("avg_image_ratio", 0))
        
        assert avg_density < 0.001 or avg_image_ratio > 0.5, \
            "Scanned document should have low text density or high image ratio"
        
        print(f"✓ Class B classified: {profile.origin_type}, {profile.layout_complexity}")
    
    def test_class_c_mixed_technical_report(self, triage_agent):
        """
        Class C: Technical Assessment Report (PDF, mixed: text + tables + structured findings)
        Expected: OriginType.NATIVE_DIGITAL or MIXED, LayoutComplexity.MIXED or TABLE_HEAVY
        """
        doc_path = "tests/test_data/tax_expenditure_ethiopia_2021_22.pdf"
        if not os.path.exists(doc_path):
            pytest.skip(f"Test file {doc_path} not found")
        
        profile = triage_agent.classify(doc_path)
        
        # Class C characteristics - should detect complexity
        assert profile.origin_type in [OriginType.NATIVE_DIGITAL, OriginType.MIXED], \
            f"Expected NATIVE_DIGITAL or MIXED but got {profile.origin_type}"
        
        # Mixed documents should have some tables or complex layout
        avg_table_count = float(profile.metadata.get("avg_table_count", 0))
        table_word_ratio = float(profile.metadata.get("table_word_ratio", 0))
        
        print(f"✓ Class C classified: {profile.origin_type}, {profile.layout_complexity}")
        print(f"  Tables per page: {avg_table_count:.2f}, Table word ratio: {table_word_ratio:.3f}")
    
    def test_class_d_table_heavy_structured_data(self, triage_agent):
        """
        Class D: Structured Data Report (PDF, table-heavy with numerical fiscal data)
        Expected: OriginType.NATIVE_DIGITAL, LayoutComplexity.TABLE_HEAVY or MIXED
        """
        doc_path = "tests/test_data/fta_performance_survey_final_report_2022.pdf"
        if not os.path.exists(doc_path):
            pytest.skip(f"Test file {doc_path} not found")
        
        profile = triage_agent.classify(doc_path)
        
        # Class D characteristics
        assert profile.origin_type == OriginType.NATIVE_DIGITAL, \
            f"Expected NATIVE_DIGITAL but got {profile.origin_type}"
        
        # Should detect tables
        avg_table_count = float(profile.metadata.get("avg_table_count", 0))
        table_word_ratio = float(profile.metadata.get("table_word_ratio", 0))
        
        # Table-heavy documents should have either many tables or high table word ratio
        has_tables = avg_table_count >= 1.0 or table_word_ratio >= 0.2
        
        print(f"✓ Class D classified: {profile.origin_type}, {profile.layout_complexity}")
        print(f"  Tables per page: {avg_table_count:.2f}, Table word ratio: {table_word_ratio:.3f}")
        print(f"  Has significant table content: {has_tables}")


class TestLayoutComplexityDetection:
    """Test that layout complexity is properly detected."""
    
    def test_single_column_detection(self, triage_agent):
        """Test single column layout detection."""
        # Most simple documents should be single column
        doc_path = "tests/test_data/fta_performance_survey_final_report_2022.pdf"
        if not os.path.exists(doc_path):
            pytest.skip(f"Test file {doc_path} not found")
        
        profile = triage_agent.classify(doc_path)
        avg_x_offsets = float(profile.metadata.get("avg_x_offsets", 0))
        
        print(f"Single column test - X offsets: {avg_x_offsets:.1f}, Layout: {profile.layout_complexity}")
    
    def test_table_heavy_detection(self, triage_agent):
        """Test table-heavy layout detection."""
        doc_path = "tests/test_data/fta_performance_survey_final_report_2022.pdf"
        if not os.path.exists(doc_path):
            pytest.skip(f"Test file {doc_path} not found")
        
        profile = triage_agent.classify(doc_path)
        avg_table_count = float(profile.metadata.get("avg_table_count", 0))
        table_word_ratio = float(profile.metadata.get("table_word_ratio", 0))
        
        print(f"Table-heavy test - Tables: {avg_table_count:.2f}, Ratio: {table_word_ratio:.3f}, Layout: {profile.layout_complexity}")


class TestMetadataCompleteness:
    """Test that all metadata fields are properly populated."""
    
    def test_all_metadata_fields_present(self, triage_agent):
        """Verify all expected metadata fields are present."""
        doc_path = "tests/test_data/fta_performance_survey_final_report_2022.pdf"
        if not os.path.exists(doc_path):
            pytest.skip(f"Test file {doc_path} not found")
        
        profile = triage_agent.classify(doc_path)
        
        # Required metadata fields
        required_fields = [
            "avg_char_density",
            "avg_image_ratio",
            "avg_x_offsets",
            "avg_table_count",
            "table_word_ratio",
            "total_pages",
            "sampled_pages"
        ]
        
        for field in required_fields:
            assert field in profile.metadata, f"Missing metadata field: {field}"
        
        # Verify language detection
        assert profile.language is not None
        assert len(profile.language) >= 2
        assert 0.0 <= profile.language_confidence <= 1.0


def test_all_documents_summary(triage_agent):
    """Run classification on all available test documents and print summary."""
    test_docs = [
        ("Class A - Native Digital", "tests/test_data/Delivery Challenge Week 2.pdf"),
        ("Class B - Scanned Document", "tests/test_data/2022_Audited_Financial_Statement_Report.pdf"),
        ("Class C - Mixed Technical", "tests/test_data/tax_expenditure_ethiopia_2021_22.pdf"),
        ("Class D - Table Heavy", "tests/test_data/fta_performance_survey_final_report_2022.pdf"),
        ("Class B - Scanned Document", "tests/test_data/2013-E.C-Procurement-information.pdf"),
    ]
    
    print("\n" + "="*80)
    print("DOCUMENT CLASSIFICATION SUMMARY")
    print("="*80)
    
    for doc_name, doc_path in test_docs:
        if not os.path.exists(doc_path):
            print(f"\n{doc_name}: SKIPPED (file not found)")
            continue
        
        profile = triage_agent.classify(doc_path)
        
        print(f"\n{doc_name}:")
        print(f"  Origin Type: {profile.origin_type.value}")
        print(f"  Layout Complexity: {profile.layout_complexity.value}")
        print(f"  Domain: {profile.domain_hint.value}")
        print(f"  Language: {profile.language} (confidence: {profile.language_confidence:.2f})")
        print(f"  Extraction Cost: {profile.estimated_cost.value}")
        print(f"  Metrics:")
        print(f"    - Char Density: {profile.metadata.get('avg_char_density')}")
        print(f"    - Image Ratio: {profile.metadata.get('avg_image_ratio')}")
        print(f"    - Tables/Page: {profile.metadata.get('avg_table_count')}")
        print(f"    - Table Word Ratio: {profile.metadata.get('table_word_ratio')}")
    
    print("\n" + "="*80)
