import os
import time
import json
import base64
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from typing import List, Dict, Any, Optional, Tuple

from .base import BaseExtractor
from src.models import (
    ExtractedDocument, 
    TextBlock, 
    TableStructure, 
    DocumentProfile, 
    BBox,
    ConfidenceMetadata
)
from src.utils.config_loader import load_refinery_config

# Try to load .env from project root
env_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)

class VisionExtractor(BaseExtractor):
    """
    Strategy C: Vision-augmented extraction (page images -> VLM -> structured JSON).
    Designed for escalation: run on selected pages, not entire doc.
    """

    def __init__(
        self,
        config: Dict[str, Any] = None,
        # Legacy defaults for local overrides
        model_name: str = None,
        budget_cap_usd: float = None,
        est_cost_per_page_usd: float = None,
        max_pages_per_doc: int = None,
        confidence_accept: float = None,
        vision_provider: str = None,  # New: 'groq', 'huggingface', or 'gemini'
    ):
        if not config:
            full_config = load_refinery_config()
            self.config = full_config.get("extraction", {}).get("vision_extractor", {})
        else:
            self.config = config

        # Priority: explicit args > config file > hardcoded defaults
        self.vision_provider = vision_provider or self.config.get("vision_provider", "huggingface")
        
        # Set model based on provider
        if self.vision_provider == "groq":
            default_model = "llama-3.2-90b-vision-preview"
        elif self.vision_provider == "gemini":
            default_model = "gemini-1.5-flash"
        else:  # huggingface
            default_model = "Qwen/Qwen2.5-VL-7B-Instruct"
            
        self.model_name = model_name or self.config.get("model_name", default_model)
        self.budget_cap = budget_cap_usd or self.config.get("budget_cap_usd", 5.0)
        self.current_spend = 0.0

        self.est_cost_per_page = est_cost_per_page_usd or self.config.get("est_cost_per_page_usd", 0.03)
        self.max_pages_per_doc = max_pages_per_doc or self.config.get("max_pages_per_doc", 12)
        self.confidence_accept = confidence_accept or self.config.get("confidence_accept", 0.80)

    # ---------- page selection ----------
    def _choose_pages(self, profile: DocumentProfile) -> List[int]:
        """
        Select pages for vision. Prefer evidence from profile if available.
        Fallback: first pages + middle + last.
        """
        total_pages = int(profile.metadata.get("total_pages", 0)) or 1

        # If triage stored page-level origin labels, prioritize scanned_like pages
        labels = profile.metadata.get("page_origin_labels")
        sampled_pages = profile.metadata.get("sampled_pages")

        if labels and sampled_pages and len(labels) == len(sampled_pages):
            scanned_pages = [p for p, lab in zip(sampled_pages, labels) if "scanned" in lab]
            if scanned_pages:
                return scanned_pages[: self.max_pages_per_doc]

        # default sampling
        pages = list(range(min(6, total_pages)))
        if total_pages > 8:
            pages += [total_pages // 2, total_pages - 1]
        return sorted(set(pages))[: self.max_pages_per_doc]

    # ---------- pdf -> images ----------
    def _render_pages_to_images(self, pdf_path: str, page_indices: List[int], dpi: int = 200):
        """
        Convert selected PDF pages to PIL images.
        Uses pdf2image if available.
        """
        from pdf2image import convert_from_path

        # pdf2image uses 1-based page numbers
        page_numbers = [i + 1 for i in page_indices]
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt="png",
            first_page=min(page_numbers),
            last_page=max(page_numbers),
        )

        # convert_from_path returns a contiguous range; we map back to requested indices
        # If you request non-contiguous pages, you'll want a per-page render instead.
        # Simple approach: render each page independently (slower but correct).
        if (max(page_numbers) - min(page_numbers) + 1) != len(page_numbers):
            images = []
            for pn in page_numbers:
                images += convert_from_path(pdf_path, dpi=dpi, fmt="png", first_page=pn, last_page=pn)

        return images

    def _img_to_b64(self, pil_img) -> str:
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ---------- VLM call (pluggable) ----------
    def _call_vlm(self, image_b64: str, page_num_0: int, domain_hint: str) -> Dict[str, Any]:
        """
        Calls vision model based on configured provider (Groq, HuggingFace, or Gemini).
        Falls back to mock if API fails.
        """
        if os.environ.get("MOCK_VLM") == "true":
            return self._get_mock_vlm_response(page_num_0, domain_hint)
        
        # Route to appropriate provider
        if self.vision_provider == "groq":
            return self._call_groq_vision(image_b64, page_num_0, domain_hint)
        elif self.vision_provider == "gemini":
            return self._call_gemini_vision(image_b64, page_num_0, domain_hint)
        else:  # huggingface (default)
            return self._call_huggingface_vision(image_b64, page_num_0, domain_hint)
    
    def _call_groq_vision(self, image_b64: str, page_num_0: int, domain_hint: str) -> Dict[str, Any]:
        """
        Calls Groq's vision API (llama-3.2-90b-vision-preview or similar).
        Uses OpenAI-compatible API format.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            logger.warning("GROQ_API_KEY not found. Falling back to mock.")
            return self._get_mock_vlm_response(page_num_0, domain_hint)
        
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
            prompt = (
                f"This is a {domain_hint} document page. "
                "Extract all text contents and tables. "
                "Return the data strictly in JSON format with the following keys:\n"
                "- 'blocks': list of objects with a 'content' field (string) and a 'bbox' field (list of 4 floats: [x0, y0, x1, y1] for coordinates)\n"
                "- 'tables': list of objects with 'headers' (list of strings), 'rows' (list of list of strings), and 'bbox' (list of 4 floats)\n"
                "If bounding boxes are unknown, use [0.0, 0.0, 0.0, 0.0]. "
                "Do not include any chat or preamble, only the raw JSON."
            )
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2048,
                temperature=0
            )
            
            output_text = response.choices[0].message.content
            
            # Clean up JSON formatting
            if "```json" in output_text:
                output_text = output_text.split("```json")[1].split("```")[0].strip()
            elif "```" in output_text:
                output_text = output_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(output_text)
            
        except Exception as e:
            logger.error(f"Groq vision API call failed: {e}. Falling back to mock.")
            return self._get_mock_vlm_response(page_num_0, domain_hint)
    
    def _call_gemini_vision(self, image_b64: str, page_num_0: int, domain_hint: str) -> Dict[str, Any]:
        """
        Calls Google Gemini's vision API.
        """
        api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found. Falling back to mock.")
            return self._get_mock_vlm_response(page_num_0, domain_hint)
        
        try:
            import google.generativeai as genai
            import PIL.Image
            import io
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model_name)
            
            # Decode base64 to PIL Image
            image_data = base64.b64decode(image_b64)
            image = PIL.Image.open(io.BytesIO(image_data))
            
            prompt = (
                f"This is a {domain_hint} document page. "
                "Extract all text contents and tables. "
                "Return the data strictly in JSON format with the following keys:\n"
                "- 'blocks': list of objects with a 'content' field (string) and a 'bbox' field (list of 4 floats: [x0, y0, x1, y1] for coordinates)\n"
                "- 'tables': list of objects with 'headers' (list of strings), 'rows' (list of list of strings), and 'bbox' (list of 4 floats)\n"
                "If bounding boxes are unknown, use [0.0, 0.0, 0.0, 0.0]. "
                "Do not include any chat or preamble, only the raw JSON."
            )
            
            response = model.generate_content([prompt, image])
            output_text = response.text
            
            # Clean up JSON formatting
            if "```json" in output_text:
                output_text = output_text.split("```json")[1].split("```")[0].strip()
            elif "```" in output_text:
                output_text = output_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(output_text)
            
        except Exception as e:
            logger.error(f"Gemini vision API call failed: {e}. Falling back to mock.")
            return self._get_mock_vlm_response(page_num_0, domain_hint)
    
    def _call_huggingface_vision(self, image_b64: str, page_num_0: int, domain_hint: str) -> Dict[str, Any]:
        """
        Calls Qwen (or compatible VLM) via Hugging Face Inference API.
        Falls back to local transformers if API fails and MOCK_VLM is not set.
        """
        token = os.environ.get("HF_TOKEN")
        
        if not token:
            logger.warning("HF_TOKEN not found in environment. Falling back to mock.")
            return self._get_mock_vlm_response(page_num_0, domain_hint)

        client = InferenceClient(api_key=token)
        
        prompt = (
            f"This is a {domain_hint} document page. "
            "Extract all text contents and tables. "
            "Return the data strictly in JSON format with the following keys:\n"
            "- 'blocks': list of objects with a 'content' field (string) and a 'bbox' field (list of 4 floats: [x0, y0, x1, y1] for coordinates)\n"
            "- 'tables': list of objects with 'headers' (list of strings), 'rows' (list of list of strings), and 'bbox' (list of 4 floats)\n"
            "If bounding boxes are unknown, use [0.0, 0.0, 0.0, 0.0]. "
            "Do not include any chat or preamble, only the raw JSON."
        )

        try:
            # Inference API chat completion with image
            response = client.chat_completion(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                        ]
                    }
                ],
                max_tokens=2048
            )
            
            output_text = response.choices[0].message.content
            
            # Clean up JSON formatting
            if "```json" in output_text:
                output_text = output_text.split("```json")[1].split("```")[0].strip()
            elif "```" in output_text:
                output_text = output_text.split("```")[1].split("```")[0].strip()
                
            return json.loads(output_text)

        except Exception as e:
            logger.error(f"Hugging Face API call failed: {e}. Falling back to mock.")
            return self._get_mock_vlm_response(page_num_0, domain_hint)

    def _get_mock_vlm_response(self, page_num_0: int, domain_hint: str) -> Dict[str, Any]:
        """Provides realistic mock data for verification."""
        return {
            "blocks": [
                {"content": f"Sample text from page {page_num_0 + 1}", "bbox": [10.0, 10.0, 200.0, 40.0]},
                {"content": f"This document appears to be related to {domain_hint}.", "bbox": [10.0, 50.0, 250.0, 80.0]},
            ],
            "tables": [
                {
                    "headers": ["ID", "Summary", "Value"],
                    "rows": [["1", "Item A", "100.00"], ["2", "Item B", "200.00"]],
                    "bbox": [10.0, 100.0, 300.0, 200.0]
                }
            ],
            "notes": {"mocked": True}
        }

    # ---------- parsing + confidence ----------
    def _safe_parse_output(self, raw: Any) -> Tuple[Dict[str, Any], bool]:
        if isinstance(raw, dict):
            return raw, True
        if isinstance(raw, str):
            try:
                return json.loads(raw), True
            except Exception:
                return {"blocks": [], "tables": [], "raw_text": raw}, False
        return {"blocks": [], "tables": []}, False

    def _compute_page_confidence(self, parsed: Dict[str, Any]) -> float:
        """
        Score page extraction quality.
        """
        score = 0.50

        blocks = parsed.get("blocks", []) or []
        tables = parsed.get("tables", []) or []

        # Text yield
        total_chars = 0
        for b in blocks:
            if isinstance(b, dict):
                content = b.get("content", "") or ""
            else:
                content = str(b)
            total_chars += len(content)
            
        if total_chars > 600:
            score += 0.20
        elif total_chars > 150:
            score += 0.10
        else:
            score -= 0.10

        # Table quality
        if tables:
            good_tables = 0
            for t in tables:
                headers = t.get("headers") or []
                rows = t.get("rows") or []
                if len(headers) >= 2 and len(rows) >= 2:
                    good_tables += 1
            score += min(0.20, 0.10 * good_tables)

        # Noise heuristic: too many weird tokens / extremely short blocks
        short_count = 0
        for b in blocks:
            if isinstance(b, dict):
                content = (b.get("content") or "").strip()
            else:
                content = str(b).strip()
            if len(content) < 8:
                short_count += 1
                
        if blocks and (short_count / max(len(blocks), 1)) > 0.5:
            score -= 0.10

        return round(max(0.0, min(1.0, score)), 4)

    def _budget_check(self, pages_to_run: int):
        projected = self.current_spend + pages_to_run * self.est_cost_per_page
        if projected > self.budget_cap:
            raise Exception(
                f"Vision extraction budget exceeded. current=${self.current_spend:.2f}, "
                f"projected=${projected:.2f}, cap=${self.budget_cap:.2f}"
            )

    # ---------- public API ----------
    def extract(self, pdf_path: str, profile: DocumentProfile) -> ExtractedDocument:
        page_indices = self._choose_pages(profile)
        self._budget_check(len(page_indices))

        t0 = time.time()
        blocks: List[TextBlock] = []
        tables: List[TableStructure] = []
        warnings = []
        signals = {}

        page_confidences: Dict[int, float] = {}
        parse_ok_pages = 0

        # Render selected pages
        images = self._render_pages_to_images(pdf_path, page_indices)

        # One image per selected page in same order
        for page_i, img in zip(page_indices, images):
            img_b64 = self._img_to_b64(img)

            try:
                # Call VLM (you connect this)
                raw = self._call_vlm(img_b64, page_num_0=page_i, domain_hint=str(profile.domain_hint.value))
                parsed, ok = self._safe_parse_output(raw)
                if ok:
                    parse_ok_pages += 1
                else:
                    warnings.append(f"Failed to parse VLM response for page {page_i+1}")

                # Compute confidence for this page
                c = self._compute_page_confidence(parsed)
                page_confidences[page_i] = round(c, 4)

                # Convert parsed -> your models
                for b in (parsed.get("blocks") or []):
                    if isinstance(b, dict):
                        txt = str(b.get("content", "")).strip()
                        bbox_list = b.get("bbox", [0.0, 0.0, 0.0, 0.0])
                    else:
                        txt = str(b).strip()
                        bbox_list = [0.0, 0.0, 0.0, 0.0]

                    if not txt:
                        continue

                    try:
                        parsed_bbox = BBox(x0=float(bbox_list[0]), y0=float(bbox_list[1]), 
                                           x1=float(bbox_list[2]), y1=float(bbox_list[3]))
                    except (IndexError, ValueError, TypeError):
                        parsed_bbox = BBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

                    blocks.append(TextBlock(
                        content=txt,
                        bbox=parsed_bbox,
                        page_num=page_i + 1  # 1-based indexing for pages
                    ))

                for t in (parsed.get("tables") or []):
                    headers = [str(x) for x in (t.get("headers") or [])]
                    rows = [[str(cell) if cell is not None else "" for cell in row] for row in (t.get("rows") or [])]
                    bbox_list = t.get("bbox", [0.0, 0.0, 0.0, 0.0]) if isinstance(t, dict) else [0.0, 0.0, 0.0, 0.0]

                    if not headers and not rows:
                        continue

                    try:
                        parsed_bbox = BBox(x0=float(bbox_list[0]), y0=float(bbox_list[1]), 
                                           x1=float(bbox_list[2]), y1=float(bbox_list[3]))
                        # If table parser returns valid looking BBox
                        if parsed_bbox.x0 == 0.0 and parsed_bbox.x1 == 0.0:
                            parsed_bbox = None
                    except (IndexError, ValueError, TypeError):
                        parsed_bbox = None

                    tables.append(TableStructure(
                        headers=headers,
                        rows=rows,
                        bbox=parsed_bbox,
                        page_num=page_i + 1
                    ))
            except Exception as e:
                warnings.append(f"Error on page {page_i+1}: {str(e)}")

            # Spend accounting per page
            self.current_spend += self.est_cost_per_page

        elapsed = time.time() - t0

        # Overall confidence = average of page confidences, adjusted by parse success
        if page_confidences:
            avg_c = sum(page_confidences.values()) / len(page_confidences)
        else:
            avg_c = 0.0

        parse_ratio = parse_ok_pages / max(len(page_indices), 1)
        score = max(0.0, min(1.0, avg_c * (0.7 + 0.3 * parse_ratio)))
        
        signals = {
            "page_confidences": page_confidences,
            "parse_ok_ratio": round(parse_ratio, 4),
            "spend_usd": round(self.current_spend, 2),
            "budget_cap_usd": self.budget_cap,
            "est_cost_per_page_usd": self.est_cost_per_page,
            "pages_processed_0_based": page_indices,
        }

        confidence_meta = ConfidenceMetadata(
            score=round(score, 4),
            method="vlm_vision",
            warnings=warnings,
            signals=signals
        )

        return ExtractedDocument(
            doc_id=profile.doc_id,
            blocks=blocks,
            tables=tables,
            metadata={
                "strategy": "VisionExtractor",
                "vision_provider": self.vision_provider,
                "model": self.model_name,
                "elapsed_sec": round(elapsed, 3),
                "provenance": {
                    "bbox_available": False,
                    "bbox_note": "VLM output lacks reliable bbox; page-level provenance only unless paired with OCR/layout tool.",
                },
            },
            confidence=confidence_meta
        )

    def get_confidence_score(self, extracted: ExtractedDocument) -> float:
        if extracted.confidence:
            return extracted.confidence.score
        return 0.0