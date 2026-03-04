from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"

class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"

class DomainHint(str, Enum):
    FINANCIAL = "financial"
    LEGAL = "legal"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    GENERAL = "general"

class ExtractionCost(str, Enum):
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"

class DocumentProfile(BaseModel):
    doc_id: str
    filename: str
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: str = "en"
    language_confidence: float = 1.0
    domain_hint: DomainHint
    estimated_cost: ExtractionCost
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

class TextBlock(BaseModel):
    content: str
    bbox: BBox
    page_num: int

class TableStructure(BaseModel):
    headers: List[str]
    rows: List[List[Any]]
    bbox: Optional[BBox] = None
    page_num: int

class ExtractedDocument(BaseModel):
    doc_id: str
    blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[TableStructure] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LDU(BaseModel):
    content: str
    chunk_type: str  # e.g., "text", "table", "figure"
    page_refs: List[int]
    bounding_box: Optional[BBox] = None
    parent_section: Optional[str] = None
    token_count: int
    content_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SectionNode(BaseModel):
    title: str
    page_start: int
    page_end: int
    summary: Optional[str] = None
    data_types_present: List[str] = Field(default_factory=list)
    child_sections: List["SectionNode"] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)

class PageIndex(BaseModel):
    doc_id: str
    root: SectionNode

class ProvenanceChain(BaseModel):
    document_name: str
    page_number: int
    bbox: Optional[BBox] = None
    content_hash: str
