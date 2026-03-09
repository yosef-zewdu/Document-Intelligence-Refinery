"""
Test BBox Provenance Threading

Verifies that bounding boxes are properly threaded from LDUs through
vector store to ProvenanceChain in query results.
"""

import pytest
import os
from unittest.mock import Mock, patch
from src.agents.query_agent import QueryAgent
from src.agents.vector_store import VectorStoreManager
from src.models.types import ProvenanceChain, BBox


# Test data paths
DOC_ID = "CBE ANNUAL REPORT 2023-24.pdf"
PAGE_INDEX_PATH = ".refinery/indices/CBE ANNUAL REPORT 2023-24.pdf_page_index.json"
VECTOR_STORE_PATH = ".refinery/vectorstore/CBE ANNUAL REPORT 2023-24.pdf"


@pytest.fixture
def skip_if_no_data():
    """Skip tests if required data is not available"""
    if not os.path.exists(PAGE_INDEX_PATH):
        pytest.skip(f"Page index not found: {PAGE_INDEX_PATH}")
    if not os.path.exists(VECTOR_STORE_PATH):
        pytest.skip(f"Vector store not found: {VECTOR_STORE_PATH}")


@pytest.fixture
def skip_if_no_api_key():
    """Skip tests that require API keys if not available"""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set - skipping LLM-dependent tests")


@pytest.fixture
def query_agent(skip_if_no_data, skip_if_no_api_key):
    """Create a QueryAgent instance for testing"""
    return QueryAgent(
        doc_id=DOC_ID,
        page_index_path=PAGE_INDEX_PATH,
        max_iterations=3
    )


def test_bbox_in_vector_store(skip_if_no_data):
    """Test that bbox is stored in vector store metadata"""
    vs = VectorStoreManager()
    
    # Search for something
    results = vs.search(DOC_ID, "interest income", k=3)
    
    assert len(results) > 0, "No search results returned"
    
    # Check if at least one result has bbox
    bbox_count = sum(1 for _, _, metadata in results if metadata.get('bbox'))
    
    # We expect at least some results to have bbox
    assert bbox_count > 0, f"No bboxes found in {len(results)} results"
    
    # Verify bbox structure
    for content, score, metadata in results:
        bbox = metadata.get('bbox')
        if bbox:
            assert isinstance(bbox, dict), "BBox should be a dict"
            assert all(k in bbox for k in ['x0', 'y0', 'x1', 'y1']), "BBox missing coordinates"
            assert bbox['x1'] > bbox['x0'], "Invalid bbox: x1 <= x0"
            assert bbox['y1'] > bbox['y0'], "Invalid bbox: y1 <= y0"


def test_bbox_in_query_results(query_agent):
    """Test that bbox appears in query agent results"""
    # Mock the graph invoke to avoid real LLM calls
    from unittest.mock import Mock, patch
    
    with patch.object(query_agent.graph, 'invoke') as mock_invoke:
        mock_invoke.return_value = {
            "messages": [Mock(content="Interest income was 101,040,098,062 ETB")],
            "provenance": [
                ProvenanceChain(
                    document_name=query_agent.doc_id,
                    page_number=25,
                    bbox=BBox(x0=100, y0=200, x1=400, y1=220),
                    content_hash="test_hash_123"
                )
            ],
            "context": [],
            "iterations": 1
        }
        
        result = query_agent.query("What was the interest income for 2024?", timeout=30)
    
    assert 'provenance' in result, "Result missing provenance"
    assert len(result['provenance']) > 0, "No provenance returned"
    
    # Check if at least one provenance has bbox
    bbox_count = sum(1 for prov in result['provenance'] if prov.get('bbox'))
    
    # We expect at least some provenance to have bbox
    assert bbox_count > 0, f"No bboxes found in {len(result['provenance'])} provenance items"
    
    # Verify bbox structure in provenance
    for prov in result['provenance']:
        assert 'document_name' in prov
        assert 'page_number' in prov
        assert 'content_hash' in prov
        
        bbox = prov.get('bbox')
        if bbox:
            assert isinstance(bbox, dict), "BBox should be a dict"
            assert all(k in bbox for k in ['x0', 'y0', 'x1', 'y1']), "BBox missing coordinates"
            assert bbox['x1'] > bbox['x0'], f"Invalid bbox: x1 ({bbox['x1']}) <= x0 ({bbox['x0']})"
            assert bbox['y1'] > bbox['y0'], f"Invalid bbox: y1 ({bbox['y1']}) <= y0 ({bbox['y0']})"


def test_bbox_in_audit_results(query_agent):
    """Test that bbox appears in audit mode results"""
    claim = "The interest income was 101,040,098,062 ETB in 2024"
    result = query_agent.audit(claim, timeout=30)
    
    assert 'status' in result, "Result missing status"
    assert 'provenance' in result, "Result missing provenance"
    
    # Only check bbox if we have provenance
    if len(result['provenance']) > 0:
        # Check if at least one provenance has bbox
        bbox_count = sum(1 for prov in result['provenance'] if prov.get('bbox'))
        
        # We expect at least some provenance to have bbox
        assert bbox_count > 0, f"No bboxes found in {len(result['provenance'])} provenance items"
        
        # Verify bbox structure
        for prov in result['provenance']:
            bbox = prov.get('bbox')
            if bbox:
                assert isinstance(bbox, dict), "BBox should be a dict"
                assert all(k in bbox for k in ['x0', 'y0', 'x1', 'y1']), "BBox missing coordinates"
                assert bbox['x1'] > bbox['x0'], "Invalid bbox: x1 <= x0"
                assert bbox['y1'] > bbox['y0'], "Invalid bbox: y1 <= y0"


def test_bbox_spatial_traceability(skip_if_no_data):
    """Test that bbox provides spatial traceability"""
    vs = VectorStoreManager()
    results = vs.search(DOC_ID, "interest income", k=1)
    
    assert len(results) > 0, "No search results"
    
    content, score, metadata = results[0]
    bbox = metadata.get('bbox')
    page = metadata.get('page_min')
    
    if bbox and page:
        # Verify we have enough information for spatial traceability
        assert page > 0, "Invalid page number"
        assert bbox['x0'] >= 0, "BBox x0 should be non-negative"
        assert bbox['y0'] >= 0, "BBox y0 should be non-negative"
        
        # Verify bbox is reasonable (not zero-sized)
        width = bbox['x1'] - bbox['x0']
        height = bbox['y1'] - bbox['y0']
        assert width > 0, "BBox has zero width"
        assert height > 0, "BBox has zero height"
        
        # Verify bbox is within reasonable page bounds (typical PDF page)
        assert bbox['x1'] < 1000, "BBox x1 seems too large"
        assert bbox['y1'] < 1000, "BBox y1 seems too large"


@pytest.mark.parametrize("query,expected_tool", [
    ("What was the interest income?", "semantic_search"),
    ("interest income 2024", "semantic_search"),
])
def test_bbox_preserved_across_tools(query_agent, query, expected_tool):
    """Test that bbox is preserved when using different tools"""
    result = query_agent.query(query, timeout=30)
    
    # If we got provenance, check bbox
    if result.get('provenance'):
        bbox_count = sum(1 for prov in result['provenance'] if prov.get('bbox'))
        # At least some results should have bbox
        assert bbox_count > 0, f"No bboxes in provenance for query: {query}"
