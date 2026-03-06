import os
import json
import pdfplumber
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling_core.types.doc import ImageRefMode

def analyze_document_with_pdfplumber(pdf_path):
    """
    Analyzes a PDF using pdfplumber to determine character density, 
    image area, and basic layout complexity.
    """
    print(f"Loading {pdf_path}...")
    results = {
        "filename": os.path.basename(pdf_path),
        "pages": []
    }
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            char_count = len(text)
            page_area = float(page.width * page.height)
            char_density = char_count / page_area if page_area > 0 else 0
            
            # Analyze images
            image_count = len(page.images)
            image_area = sum([float(img["width"] * img["height"]) for img in page.images])
            image_ratio = image_area / page_area if page_area > 0 else 0
            
            # Basic layout complexity: word count and spatial distribution
            words = page.extract_words()
            word_count = len(words)
            
            results["pages"].append({
                "page_num": i + 1,
                "char_count": char_count,
                "word_count": word_count,
                "char_density": f"{char_density:.6f}",
                "image_count": image_count,
                "image_ratio": f"{image_ratio:.6f}",
            })
            
    return results


# def analyze_document_with_docling(pdf_path):
#     # Initialize with default options to probe the file
#     converter = DocumentConverter()
#     result = converter.convert(pdf_path)
#     doc = result.document

#     # 1. Origin Type: Check if text is selectable or needs OCR
#     # Docling's 'origin' usually indicates if it was parsed or OCR'd
#     is_scanned = any(page.cells == [] for page in doc.pages) # Simplified check
#     origin_type = "scanned_image" if is_scanned else "native_digital"

#     # 2. Layout Complexity: Ratio-based logic
#     table_ratio = len(doc.tables) / doc.num_pages if doc.num_pages > 0 else 0
#     picture_ratio = len(doc.pictures) / doc.num_pages if doc.num_pages > 0 else 0
    
#     layout_complexity = "mixed"
#     if table_ratio > 0.5: layout_complexity = "table_heavy"
#     elif picture_ratio > 0.5: layout_complexity = "figure_heavy"
#     # Note: Column detection usually requires checking bounding boxes of text elements

#     # 3. Language & Domain Hint
#     # Docling often extracts language in metadata
#     detected_lang = doc.meta.language if doc.meta and doc.meta.language else "unknown"
    
#     # Simple keyword-based Domain Hint
#     text_sample = " ".join([t.text for t in doc.texts[:20]]).lower()
#     domain_hint = "general"
#     if any(k in text_sample for k in ["tax", "expenditure", "revenue", "fiscal"]):
#         domain_hint = "financial"
#     elif any(k in text_sample for k in ["article", "regulation", "law", "decree"]):
#         domain_hint = "legal"

#     # 4. Estimated Extraction Cost
#     cost_est = "fast_text_sufficient"
#     if is_scanned or table_ratio > 0.3:
#         cost_est = "needs_layout_model"
#     if len(doc.pictures) > 5:
#         cost_est = "needs_vision_model"

#     return {
#         "filename": os.path.basename(pdf_path),
#         "origin_type": origin_type,
#         "layout_complexity": layout_complexity,
#         "language": detected_lang,
#         "domain_hint": domain_hint,
#         "estimated_extraction_cost": cost_est,
#         "num_tables": len(doc.tables),
#         "num_pictures": len(doc.pictures)
#     }


def analyze_document_with_docling(pdf_path):
    """
    Analyzes a PDF using Docling and compares the output quality.
    """
    print(f"Running Docling on {pdf_path}...")
    # 1. Enable image extraction and scaling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True  # Required for charts/images
    pipeline_options.images_scale = 2.0  
    
    # Define image save path
    image_dir = os.path.join(".refinery/exploration", f"{os.path.basename(pdf_path)}_images")
    os.makedirs(image_dir, exist_ok=True)
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                image_ref_mode=ImageRefMode.EMBEDDED # Try embedded first
            )
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document
    md = doc.export_to_markdown()

    # 1. Origin Type
    is_scanned = len(doc.texts) < 5 * len(doc.pages)
    origin_type = "image_heavy_or_scanned" if is_scanned else "native_digital"

    # 2. Layout Complexity
    table_ratio = len(doc.tables) / len(doc.pages) if len(doc.pages) > 0 else 0
    picture_ratio = len(doc.pictures) / len(doc.pages) if len(doc.pages) > 0 else 0
    
    layout_complexity = "mixed"
    if table_ratio > 0.5: layout_complexity = "table_heavy"
    elif picture_ratio > 0.5: layout_complexity = "figure_heavy"

    # 3. Language & Domain Hint: Already detected by fallback above
    # Simplified Domain Hint (based on text preview)
    text_sample = " ".join([t.text for t in doc.texts[:20]]).lower()
    domain_hint = "general"
    if any(k in text_sample for k in ["tax", "expenditure", "revenue", "fiscal", "report", "audit"]):
        domain_hint = "financial"
    elif any(k in text_sample for k in ["article", "regulation", "law", "decree", "contract", "agreement"]):
        domain_hint = "legal"
    
    # 4. Estimated Extraction Cost
    cost_est = "fast_text_sufficient"
    if is_scanned or table_ratio > 0.3:
        cost_est = "needs_layout_model"
    if len(doc.pictures) > 5:
        cost_est = "needs_vision_model"

    return {
        "filename": os.path.basename(pdf_path),
        "num_pages": len(doc.pages) if hasattr(doc, 'pages') else 0,
        "num_tables": len(doc.tables) if hasattr(doc, 'tables') and isinstance(doc.tables, list) else 0,
        "num_pictures": len(doc.pictures) if hasattr(doc, 'pictures') and isinstance(doc.pictures, list) else 0,
        "origin_type": origin_type,
        "layout_complexity": layout_complexity,
        "domain_hint": domain_hint,
        "estimated_extraction_cost": cost_est,
    } , md

if __name__ == "__main__":
    test_docs = [
        # "data/CBE ANNUAL REPORT 2023-24.pdf",
        # "data/Audit Report - 2023.pdf",
        # "data/fta_performance_survey_final_report_2022.pdf",
        # "data/tax_expenditure_ethiopia_2021_22.pdf",
        "data/interim_report.pdf"
    ]
    
    os.makedirs(".refinery/exploration", exist_ok=True)
    
    for doc in test_docs:
        if not os.path.exists(doc):
            print(f"Skipping {doc}, file not found.")
            continue
            
        print(f"Analyzing {doc} with pdfplumber...")
        try:
            plumber_results = analyze_document_with_pdfplumber(doc)
            plumber_output = f".refinery/exploration/{os.path.basename(doc)}_plumber.json"
            with open(plumber_output, "w") as f:
                json.dump(plumber_results, f, indent=2)
            print(f"Results saved to {plumber_output}")
        except Exception as e:
            print(f"pdfplumber analysis failed for {doc}: {e}")
        
        # docling analysis (optional)
        try:
            docling_results = analyze_document_with_docling(doc)
            docling_json = f".refinery/exploration/{os.path.basename(doc)}_1_docling.json"
            docling_md = f".refinery/exploration/{os.path.basename(doc)}_1_docling.md"
            with open(docling_json, "w") as f:
                json.dump(docling_results[0], f, indent=2)
            with open(docling_md, "w") as f:
                f.write(docling_results[1])
            print(f"Docling results saved to {docling_json}")
            print(f"Docling markdown saved to {docling_md}")
        except ImportError:
            print(f"Docling not yet installed, skipping.")
        except Exception as e:
            print(f"Docling analysis failed for {doc}: {e}")
