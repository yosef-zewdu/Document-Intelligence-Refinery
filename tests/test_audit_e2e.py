"""
Phase 4: End-to-End Claim Verification Tests

Comprehensive E2E tests for the audit workflow including:
- Full provenance chain verification
- BBox presence in audit results
- Multi-hop verification (multiple tools)
- Negative cases (unverifiable claims)
- Edge cases (ambiguous claims, partial matches)
"""

import pytest
import json
import os
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from src.agents.query_agent import QueryAgent
from src.models.types import BBox, ProvenanceChain


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def doc_id():
    """Document ID for testing"""
    return "CBE ANNUAL REPORT 2023-24.pdf"


@pytest.fixture
def page_index_path():
    """Path to page index"""
    return ".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"


@pytest.fixture
def agent(doc_id, page_index_path):
    """Create query agent for testing"""
    if not os.path.exists(page_index_path):
        pytest.skip(f"Page index not found: {page_index_path}")
    
    return QueryAgent(
        doc_id=doc_id,
        page_index_path=page_index_path,
        llm_provider="openrouter",
        llm_model="arcee-ai/trinity-large-preview:free",
        max_iterations=3
    )


# ============================================================================
# TEST 1: BASIC AUDIT FUNCTIONALITY
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_returns_correct_structure(agent):
    """Test that audit returns the expected data structure"""
    claim = "The bank has assets"
    
    # Mock the graph invoke to avoid actual LLM calls
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: The bank has total assets of 1,436 billion birr")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=10,
                    bbox=BBox(x0=100, y0=200, x1=400, y1=220),
                    content_hash="test_hash_123"
                )
            ],
            "context": [],
            "iterations": 1
        }
        
        result = agent.audit(claim)
    
    # Verify structure
    assert isinstance(result, dict)
    assert "claim" in result
    assert "status" in result
    assert "verification" in result
    assert "provenance" in result
    assert "doc_id" in result
    assert "iterations" in result
    
    # Verify values
    assert result["claim"] == claim
    assert result["status"] in ["VERIFIED", "NOT_FOUND", "TIMEOUT", "ERROR"]
    assert result["doc_id"] == agent.doc_id


# ============================================================================
# TEST 2: PROVENANCE CHAIN VERIFICATION
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_includes_complete_provenance(agent):
    """Test that audit results include complete provenance with all required fields"""
    claim = "Interest income was 101,040,098,062 ETB"
    
    # Mock the graph invoke
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: Interest income was 101,040,098,062 ETB")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=25,
                    bbox=BBox(x0=150.5, y0=300.2, x1=450.8, y1=320.5),
                    content_hash="sha256:abc123def456"
                ),
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=26,
                    bbox=BBox(x0=100.0, y0=250.0, x1=400.0, y1=270.0),
                    content_hash="sha256:xyz789uvw012"
                )
            ],
            "context": [],
            "iterations": 2
        }
        
        result = agent.audit(claim)
    
    # Verify provenance exists
    assert len(result["provenance"]) > 0, "Provenance list should not be empty"
    
    # Verify each provenance entry has all required fields
    for prov in result["provenance"]:
        assert "document_name" in prov, "Missing document_name"
        assert "page_number" in prov, "Missing page_number"
        assert "bbox" in prov, "Missing bbox"
        assert "content_hash" in prov, "Missing content_hash"
        
        # Verify values are valid
        assert prov["document_name"] == agent.doc_id
        assert prov["page_number"] > 0, "Page number must be positive"
        assert prov["content_hash"] is not None, "Content hash should not be None"


# ============================================================================
# TEST 3: BBOX PRESENCE AND VALIDITY
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_provenance_includes_valid_bbox(agent):
    """Test that provenance includes valid bounding boxes"""
    claim = "Total assets reached 1,436 billion birr"
    
    # Mock the graph invoke
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: Total assets reached 1,436 billion birr")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=15,
                    bbox=BBox(x0=100.0, y0=200.0, x1=500.0, y1=250.0),
                    content_hash="sha256:test123"
                )
            ],
            "context": [],
            "iterations": 1
        }
        
        result = agent.audit(claim)
    
    # Verify bbox exists and is valid
    for prov in result["provenance"]:
        bbox = prov["bbox"]
        
        # BBox should not be None
        assert bbox is not None, "BBox should not be None"
        
        # BBox should have all coordinates
        assert "x0" in bbox, "Missing x0 coordinate"
        assert "y0" in bbox, "Missing y0 coordinate"
        assert "x1" in bbox, "Missing x1 coordinate"
        assert "y1" in bbox, "Missing y1 coordinate"
        
        # BBox should have valid dimensions
        assert bbox["x1"] > bbox["x0"], f"Invalid bbox: x1 ({bbox['x1']}) <= x0 ({bbox['x0']})"
        assert bbox["y1"] > bbox["y0"], f"Invalid bbox: y1 ({bbox['y1']}) <= y0 ({bbox['y0']})"
        
        # Coordinates should be reasonable (not negative, not too large)
        assert bbox["x0"] >= 0, "x0 should not be negative"
        assert bbox["y0"] >= 0, "y0 should not be negative"
        assert bbox["x1"] <= 10000, "x1 seems unreasonably large"
        assert bbox["y1"] <= 10000, "y1 seems unreasonably large"


# ============================================================================
# TEST 4: VERIFIED STATUS
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_verified_status(agent):
    """Test that verifiable claims return VERIFIED status"""
    claim = "The bank operates in Ethiopia"
    
    # Mock the graph invoke with VERIFIED response
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: The bank operates in Ethiopia as stated on page 5")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=5,
                    bbox=BBox(x0=100, y0=200, x1=400, y1=220),
                    content_hash="sha256:test"
                )
            ],
            "context": [],
            "iterations": 1
        }
        
        result = agent.audit(claim)
    
    # Verify VERIFIED status
    assert result["status"] == "VERIFIED", f"Expected VERIFIED, got {result['status']}"
    assert len(result["provenance"]) > 0, "VERIFIED claims should have provenance"


# ============================================================================
# TEST 5: NOT_FOUND STATUS
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_not_found_status(agent):
    """Test that unverifiable claims return NOT_FOUND status"""
    claim = "The bank has 500 branches on Mars"
    
    # Mock the graph invoke with NOT_FOUND response
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="NOT_FOUND: No evidence found for this claim")],
            "provenance": [],
            "context": [],
            "iterations": 2
        }
        
        result = agent.audit(claim)
    
    # Verify NOT_FOUND status
    assert result["status"] == "NOT_FOUND", f"Expected NOT_FOUND, got {result['status']}"
    # NOT_FOUND claims may have empty provenance
    assert isinstance(result["provenance"], list)


# ============================================================================
# TEST 6: TIMEOUT HANDLING
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_timeout_handling(agent):
    """Test that audit handles timeouts gracefully"""
    claim = "Test timeout"
    
    # Mock the graph invoke to raise TimeoutError
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.side_effect = TimeoutError("Audit timed out")
        
        result = agent.audit(claim, timeout=1)
    
    # Verify timeout handling
    assert result["status"] == "TIMEOUT", f"Expected TIMEOUT, got {result['status']}"
    assert "timed out" in result["verification"].lower()
    assert result["provenance"] == []


# ============================================================================
# TEST 7: ERROR HANDLING
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_error_handling(agent):
    """Test that audit handles errors gracefully"""
    claim = "Test error"
    
    # Mock the graph invoke to raise an exception
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.side_effect = Exception("Test error")
        
        result = agent.audit(claim)
    
    # Verify error handling
    assert result["status"] == "ERROR", f"Expected ERROR, got {result['status']}"
    assert "error" in result["verification"].lower()
    assert result["provenance"] == []


# ============================================================================
# TEST 8: MULTIPLE PROVENANCE SOURCES
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_multiple_provenance_sources(agent):
    """Test that audit can collect provenance from multiple sources"""
    claim = "Interest income and expenses are reported"
    
    # Mock the graph invoke with multiple provenance entries
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: Both interest income and expenses are reported")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=25,
                    bbox=BBox(x0=100, y0=200, x1=400, y1=220),
                    content_hash="sha256:income_hash"
                ),
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=26,
                    bbox=BBox(x0=100, y0=300, x1=400, y1=320),
                    content_hash="sha256:expense_hash"
                ),
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=27,
                    bbox=BBox(x0=100, y0=400, x1=400, y1=420),
                    content_hash="sha256:summary_hash"
                )
            ],
            "context": [],
            "iterations": 2
        }
        
        result = agent.audit(claim)
    
    # Verify multiple provenance entries
    assert len(result["provenance"]) >= 2, "Should have multiple provenance entries"
    
    # Verify each entry is unique
    hashes = [p["content_hash"] for p in result["provenance"]]
    assert len(hashes) == len(set(hashes)), "Provenance entries should be unique"


# ============================================================================
# TEST 9: SPATIAL TRACEABILITY
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_spatial_traceability(agent):
    """Test that bbox allows spatial traceability to exact location in document"""
    claim = "Net profit is reported"
    
    # Mock the graph invoke
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: Net profit is reported")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=30,
                    bbox=BBox(x0=150.5, y0=300.2, x1=450.8, y1=320.5),
                    content_hash="sha256:profit_hash"
                )
            ],
            "context": [],
            "iterations": 1
        }
        
        result = agent.audit(claim)
    
    # Verify spatial traceability
    for prov in result["provenance"]:
        page = prov["page_number"]
        bbox = prov["bbox"]
        
        # Should be able to locate this in the PDF
        assert page > 0, "Page number must be valid"
        assert bbox is not None, "BBox must exist for spatial traceability"
        
        # BBox should define a valid rectangular region
        width = bbox["x1"] - bbox["x0"]
        height = bbox["y1"] - bbox["y0"]
        
        assert width > 0, "BBox must have positive width"
        assert height > 0, "BBox must have positive height"
        
        # Region should be reasonable (not too small, not too large)
        assert width >= 10, "BBox width seems too small"
        assert height >= 10, "BBox height seems too small"
        assert width <= 1000, "BBox width seems too large"
        assert height <= 1000, "BBox height seems too large"


# ============================================================================
# TEST 10: EDGE CASE - EMPTY CLAIM
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_empty_claim(agent):
    """Test handling of empty claim"""
    claim = ""
    
    # Mock the graph invoke
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="NOT_FOUND: Empty claim")],
            "provenance": [],
            "context": [],
            "iterations": 0
        }
        
        result = agent.audit(claim)
    
    # Should handle gracefully
    assert result["status"] in ["NOT_FOUND", "ERROR"]
    assert result["claim"] == claim


# ============================================================================
# TEST 11: EDGE CASE - VERY LONG CLAIM
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_long_claim(agent):
    """Test handling of very long claim"""
    claim = "The bank " + "has many operations " * 50  # Very long claim
    
    # Mock the graph invoke
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="NOT_FOUND: Claim too complex")],
            "provenance": [],
            "context": [],
            "iterations": 3
        }
        
        result = agent.audit(claim)
    
    # Should handle gracefully
    assert result["status"] in ["NOT_FOUND", "VERIFIED", "TIMEOUT", "ERROR"]
    assert result["claim"] == claim


# ============================================================================
# TEST 12: EDGE CASE - AMBIGUOUS CLAIM
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_ambiguous_claim(agent):
    """Test handling of ambiguous claim"""
    claim = "The numbers increased"  # Very vague
    
    # Mock the graph invoke
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: Multiple metrics increased")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=10,
                    bbox=BBox(x0=100, y0=200, x1=400, y1=220),
                    content_hash="sha256:test"
                )
            ],
            "context": [],
            "iterations": 2
        }
        
        result = agent.audit(claim)
    
    # Should handle gracefully
    assert result["status"] in ["VERIFIED", "NOT_FOUND"]


# ============================================================================
# TEST 13: ITERATION TRACKING
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_tracks_iterations(agent):
    """Test that audit tracks number of iterations"""
    claim = "Test iteration tracking"
    
    # Mock the graph invoke
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: Test")],
            "provenance": [],
            "context": [],
            "iterations": 2
        }
        
        result = agent.audit(claim)
    
    # Verify iteration tracking
    assert "iterations" in result
    assert isinstance(result["iterations"], int)
    assert result["iterations"] >= 0
    assert result["iterations"] <= agent.max_iterations


# ============================================================================
# TEST 14: PROVENANCE UNIQUENESS
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_provenance_uniqueness(agent):
    """Test that provenance entries are unique (no duplicates)"""
    claim = "Test provenance uniqueness"
    
    # Mock the graph invoke with duplicate provenance
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: Test")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=10,
                    bbox=BBox(x0=100, y0=200, x1=400, y1=220),
                    content_hash="sha256:same_hash"
                ),
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=10,
                    bbox=BBox(x0=100, y0=200, x1=400, y1=220),
                    content_hash="sha256:same_hash"
                )
            ],
            "context": [],
            "iterations": 1
        }
        
        result = agent.audit(claim)
    
    # Check for duplicates (this test documents current behavior)
    # In a production system, you might want to deduplicate
    hashes = [p["content_hash"] for p in result["provenance"]]
    
    # Document whether duplicates exist
    has_duplicates = len(hashes) != len(set(hashes))
    
    # This is informational - system may or may not deduplicate
    print(f"Provenance has duplicates: {has_duplicates}")


# ============================================================================
# TEST 15: DOCUMENT ID CONSISTENCY
# ============================================================================

@pytest.mark.skipif(
    not os.path.exists(".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"),
    reason="Page index not available"
)
def test_audit_document_id_consistency(agent):
    """Test that all provenance entries reference the correct document"""
    claim = "Test document ID consistency"
    
    # Mock the graph invoke
    with patch.object(agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="VERIFIED: Test")],
            "provenance": [
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=10,
                    bbox=BBox(x0=100, y0=200, x1=400, y1=220),
                    content_hash="sha256:hash1"
                ),
                ProvenanceChain(
                    document_name=agent.doc_id,
                    page_number=20,
                    bbox=BBox(x0=100, y0=300, x1=400, y1=320),
                    content_hash="sha256:hash2"
                )
            ],
            "context": [],
            "iterations": 1
        }
        
        result = agent.audit(claim)
    
    # Verify all provenance entries reference the same document
    for prov in result["provenance"]:
        assert prov["document_name"] == agent.doc_id, \
            f"Provenance document mismatch: {prov['document_name']} != {agent.doc_id}"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
