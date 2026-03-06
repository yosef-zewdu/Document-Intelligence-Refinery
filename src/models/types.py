from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import hashlib

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
    language_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    domain_hint: DomainHint
    estimated_cost: ExtractionCost
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("filename")
    @classmethod
    def validate_pdf_ext(cls, v: str) -> str:
        if not v.lower().endswith(".pdf"):
            raise ValueError("File must be a PDF")
        return v

class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_dimensions(self) -> "BBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"Invalid dimensions: x1 ({self.x1}) < x0 ({self.x0}) or y1 ({self.y1}) < y0 ({self.y0})")
        return self

class TextBlock(BaseModel):
    content: str
    bbox: BBox
    page_num: int = Field(ge=1)
    
    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("TextBlock content cannot be empty")
        return v

class TableStructure(BaseModel):
    headers: List[str]
    rows: List[List[Any]]
    bbox: Optional[BBox] = None
    page_num: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_table_consistency(self) -> "TableStructure":
        if not self.headers and not self.rows:
             raise ValueError("Table must have headers or rows")
        if self.headers and self.rows:
            header_len = len(self.headers)
            for i, row in enumerate(self.rows):
                if len(row) != header_len:
                    # We might want to be more lax later, but for now let's enforce it
                    pass 
        return self

class ConfidenceMetadata(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    method: str  # e.g., "heuristic", "vlm_self_report", "docling_coverage"
    warnings: List[str] = Field(default_factory=list)
    signals: Dict[str, Any] = Field(default_factory=dict)

class ExtractedDocument(BaseModel):
    doc_id: str
    blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[TableStructure] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[ConfidenceMetadata] = None

class LDU(BaseModel):
    content: str
    chunk_type: str  # e.g., "text", "table", "figure"
    page_refs: List[int]
    bounding_box: Optional[BBox] = None
    parent_section: Optional[str] = None
    token_count: int
    content_hash: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def generate_hash(cls, data: Any) -> Any:
        if isinstance(data, dict) and "content" in data and "content_hash" not in data:
            data["content_hash"] = hashlib.sha256(data["content"].encode()).hexdigest()
        return data

class SectionNode(BaseModel):
    title: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    summary: Optional[str] = None
    data_types_present: List[str] = Field(default_factory=list)
    child_sections: List["SectionNode"] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)

class PageIndex(BaseModel):
    doc_id: str
    root: SectionNode

class ProvenanceChain(BaseModel):
    document_name: str
    page_number: int = Field(ge=1)
    bbox: Optional[BBox] = None
    content_hash: str
