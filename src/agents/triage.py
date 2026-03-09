import os
import pdfplumber
from typing import Dict, Any, List, Tuple
from src.models import (
    DocumentProfile, 
    OriginType, 
    LayoutComplexity, 
    DomainHint, 
    ExtractionCost
)
import logging
from src.utils.config_loader import load_refinery_config
from src.utils.doc_id_generator import generate_doc_id
from fast_langdetect import detect_language

# Define the logger for this file
logger = logging.getLogger(__name__)

class TriageAgent:
    def __init__(self, config: Dict[str, Any] = None):
        if not config:
            full_config = load_refinery_config()
            self.config = full_config.get("triage", {})
        else:
            self.config = config
            
        self.thresholds = self.config.get("thresholds", {
            "scanned_density_max": 0.0005,
            "digital_density_min": 0.001,
            "multi_column_x_offsets": 200,
            "table_heavy_word_ratio": 0.4,
            "figure_heavy_image_ratio": 0.3,
            "form_fillable_field_ratio": 0.1
        })
        self.domain_keywords = self.config.get("domain_keywords", {})

    def classify(self, pdf_path: str) -> DocumentProfile:
        # Generate clean document ID (human-readable + unique hash)
        doc_id = generate_doc_id(pdf_path, strategy="filename_with_hash")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    return self._default_profile(doc_id, pdf_path, "Empty PDF")
                
                # Sample pages for analysis (first 5 pages or all if fewer)
                sample_size = min(5, total_pages)
                sample_pages = pdf.pages[:sample_size]
                
                # Collect metrics
                densities = []
                image_ratios = []
                max_x_offsets = []
                table_counts = []
                table_word_counts = []
                total_word_counts = []
                form_field_counts = []
                all_text = []
                
                for page in sample_pages:
                    text = page.extract_text() or ""
                    all_text.append(text)
                    char_count = len(text)
                    page_area = float(page.width * page.height)
                    density = char_count / page_area if page_area > 0 else 0
                    densities.append(density)
                    
                    # Image analysis
                    image_area = sum([float(img.get("width", 0) * img.get("height", 0)) for img in page.images])
                    image_ratio = image_area / page_area if page_area > 0 else 0
                    image_ratios.append(image_ratio)
                    
                    # Column detection via x-offsets
                    words = page.extract_words()
                    total_words = len(words) if words else 0
                    total_word_counts.append(total_words)
                    
                    if words:
                        x_starts = sorted(list(set([round(w["x0"], 0) for w in words])))
                        max_x_offsets.append(len(x_starts))
                    else:
                        max_x_offsets.append(0)
                    
                    # Enhanced table detection
                    tables = page.find_tables()
                    num_tables = len(tables) if tables else 0
                    table_counts.append(num_tables)
                    
                    # Count words inside tables (if tables found)
                    table_words = 0
                    if tables and words:
                        for table in tables:
                            table_bbox = table.bbox
                            if table_bbox:
                                x0, y0, x1, y1 = table_bbox
                                # Count words within table boundaries
                                for word in words:
                                    wx0, wy0, wx1, wy1 = word['x0'], word['top'], word['x1'], word['bottom']
                                    # Check if word center is inside table
                                    word_center_x = (wx0 + wx1) / 2
                                    word_center_y = (wy0 + wy1) / 2
                                    if x0 <= word_center_x <= x1 and y0 <= word_center_y <= y1:
                                        table_words += 1
                    
                    # Fallback: if no tables detected but we have structured data patterns
                    # Look for grid-like patterns in word positions
                    if num_tables == 0 and words and len(words) > 20:
                        # Check for aligned columns (words with similar x positions)
                        x_positions = [w['x0'] for w in words]
                        y_positions = [w['top'] for w in words]
                        
                        # Count unique x positions (columns) and y positions (rows)
                        unique_x = len(set([round(x, -1) for x in x_positions]))  # Round to nearest 10
                        unique_y = len(set([round(y, -1) for y in y_positions]))
                        
                        # If we have grid-like structure (many rows and columns)
                        if unique_x >= 3 and unique_y >= 5:
                            # Estimate this is table-like content
                            table_words = int(len(words) * 0.5)  # Assume 50% are in table-like structures
                    
                    table_word_counts.append(table_words)
                    
                    # Form field detection (heuristic: look for form annotations)
                    annots = page.annots if hasattr(page, 'annots') else []
                    form_field_counts.append(len([a for a in annots if a.get('Subtype') == '/Widget']))

                # Calculate averages
                avg_density = sum(densities) / len(densities)
                avg_image_ratio = sum(image_ratios) / len(image_ratios)
                avg_x_offsets = sum(max_x_offsets) / len(max_x_offsets)
                avg_table_count = sum(table_counts) / len(table_counts)
                total_form_fields = sum(form_field_counts)
                
                # Calculate table word ratio (words in tables / total words)
                total_words = sum(total_word_counts)
                total_table_words = sum(table_word_counts)
                table_word_ratio = total_table_words / total_words if total_words > 0 else 0.0
                
                # 1. Determine Origin Type
                origin_type = self._detect_origin_type(
                    avg_density, avg_image_ratio, total_form_fields, sample_size
                )
                
                # 2. Determine Layout Complexity
                layout_complexity = self._detect_layout_complexity(
                    avg_x_offsets, avg_table_count, avg_image_ratio, table_word_ratio
                )
                
                # 3. Detect Language
                combined_text = " ".join(all_text)
                language, language_confidence = self._detect_language(combined_text)
                
                # 4. Detect Domain
                domain_hint = self._detect_domain(sample_pages)
                
                # 5. Estimate Extraction Cost
                estimated_cost = self._estimate_extraction_cost(
                    origin_type, layout_complexity
                )

                return DocumentProfile(
                    doc_id=doc_id,
                    filename=os.path.basename(pdf_path),
                    origin_type=origin_type,
                    layout_complexity=layout_complexity,
                    language=language,
                    language_confidence=language_confidence,
                    domain_hint=domain_hint,
                    estimated_cost=estimated_cost,
                    metadata={
                        "avg_char_density": f"{avg_density:.6f}",
                        "avg_image_ratio": f"{avg_image_ratio:.6f}",
                        "avg_x_offsets": f"{avg_x_offsets:.1f}",
                        "avg_table_count": f"{avg_table_count:.2f}",
                        "table_word_ratio": f"{table_word_ratio:.3f}",
                        "total_pages": total_pages,
                        "sampled_pages": sample_size
                    }
                )
        except Exception as e:
            logger.error(f"Failed to triage {pdf_path}: {e}")
            return self._default_profile(doc_id, pdf_path, str(e))

    def _default_profile(self, doc_id: str, pdf_path: str, error: str) -> DocumentProfile:
        return DocumentProfile(
            doc_id=doc_id,
            filename=os.path.basename(pdf_path),
            origin_type=OriginType.NATIVE_DIGITAL,
            layout_complexity=LayoutComplexity.SINGLE_COLUMN,
            language="en",
            language_confidence=0.5,
            domain_hint=DomainHint.GENERAL,
            estimated_cost=ExtractionCost.FAST_TEXT_SUFFICIENT,
            metadata={"error": error, "is_fallback": True}
        )
    
    def _detect_origin_type(
        self, 
        avg_density: float, 
        avg_image_ratio: float, 
        total_form_fields: int,
        sample_size: int
    ) -> OriginType:
        """Detect document origin type based on multiple signals."""
        scanned_threshold = self.thresholds.get("scanned_density_max", 0.0005)
        form_field_ratio = total_form_fields / sample_size if sample_size > 0 else 0
        form_threshold = self.thresholds.get("form_fillable_field_ratio", 0.1)
        
        # Form-fillable PDFs have interactive fields
        if form_field_ratio >= form_threshold:
            return OriginType.FORM_FILLABLE
        
        # Scanned documents have very low text density or very high image ratio
        if avg_density < scanned_threshold or avg_image_ratio > 0.9:
            return OriginType.SCANNED_IMAGE
        
        # Mixed documents have moderate image ratio with decent text
        if 0.3 < avg_image_ratio < 0.9 and avg_density > scanned_threshold:
            return OriginType.MIXED
        
        return OriginType.NATIVE_DIGITAL
    
    def _detect_layout_complexity(
        self,
        avg_x_offsets: float,
        avg_table_count: float,
        avg_image_ratio: float,
        table_word_ratio: float
    ) -> LayoutComplexity:
        """Detect layout complexity based on multiple signals.
        
        Priority order for financial documents:
        1. Table-heavy: Documents with many tables or high table content
        2. Figure-heavy: Documents with many images/charts
        3. Multi-column: Academic papers, newspapers (rare in financial docs)
        4. Mixed: Multiple complexity indicators
        5. Single-column: Standard single-flow documents
        """
        table_word_threshold = self.thresholds.get("table_heavy_word_ratio", 0.4)
        figure_threshold = self.thresholds.get("figure_heavy_image_ratio", 0.3)
        
        # Table-heavy: either many tables OR high ratio of words in tables
        # This is the most common complexity in financial documents
        is_table_heavy = avg_table_count >= 2 or table_word_ratio >= table_word_threshold
        
        # Figure-heavy: high image ratio (charts, diagrams)
        is_figure_heavy = avg_image_ratio > figure_threshold
        
        # Multi-column: Only flag if x-offsets are VERY high (academic papers)
        # Most financial docs have varied indentation but aren't truly multi-column
        # We use a much higher threshold to avoid false positives
        is_multi_column = avg_x_offsets > 300  # Increased from 200
        
        # Count complexity signals
        complexity_signals = sum([is_multi_column, is_table_heavy, is_figure_heavy])
        
        # Priority-based classification
        if complexity_signals >= 2:
            return LayoutComplexity.MIXED
        elif is_table_heavy:
            # Most financial documents fall here
            return LayoutComplexity.TABLE_HEAVY
        elif is_figure_heavy:
            return LayoutComplexity.FIGURE_HEAVY
        elif is_multi_column:
            # Rare for financial docs - mostly academic papers
            return LayoutComplexity.MULTI_COLUMN
        else:
            return LayoutComplexity.SINGLE_COLUMN
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """Detect document language using fast-langdetect if available."""
        if  not text.strip():
            return "en", 0.5
        
        try:
            # Use fast-langdetect for language detection
            # Note: fast_langdetect returns just the language code as a string
            lang_code = detect_language(text[:5000])  # Sample first 5000 chars for speed
            
            if lang_code:
                # Convert to lowercase for consistency
                lang_code = lang_code.lower()
                # fast_langdetect is very accurate, so we use high confidence
                return lang_code, 0.95
            
            return "en", 0.5
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en", 0.5
    
    def _estimate_extraction_cost(
        self,
        origin_type: OriginType,
        layout_complexity: LayoutComplexity
    ) -> ExtractionCost:
        """Estimate extraction cost based on origin and layout."""
        if origin_type == OriginType.SCANNED_IMAGE:
            return ExtractionCost.NEEDS_VISION_MODEL
        
        if layout_complexity in [
            LayoutComplexity.MULTI_COLUMN,
            LayoutComplexity.TABLE_HEAVY,
            LayoutComplexity.FIGURE_HEAVY,
            LayoutComplexity.MIXED
        ]:
            return ExtractionCost.NEEDS_LAYOUT_MODEL
        
        return ExtractionCost.FAST_TEXT_SUFFICIENT

    def _detect_domain(self, pages) -> DomainHint:
        import re
        text = ""
        for page in pages:
            text += (page.extract_text() or "").lower()
            
        scores = {domain: 0 for domain in self.domain_keywords}
        
        # Track best domain based on weighted keyword hits
        for domain_key, data in self.domain_keywords.items():
            stems = data.get("stems", {})
            for pattern, weight in stems.items():
                matches = re.findall(pattern, text)
                # Map yaml string key to DomainHint enum
                try:
                    enum_val = DomainHint(domain_key)
                    scores[domain_key] += len(matches) * weight
                except ValueError:
                    continue
        
        if not any(scores.values()):
            return DomainHint.GENERAL
            
        best_key = max(scores, key=scores.get)
        best_score = scores[best_key]
        
        # Confidence threshold from config
        required_score = self.domain_keywords.get(best_key, {}).get("confidence_threshold", 5)
        
        if best_score < required_score:
            return DomainHint.GENERAL
            
        return DomainHint(best_key)
