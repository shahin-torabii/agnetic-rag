from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import fitz
import numpy as np
from PIL import Image
from paddleocr import PPStructure

from langchain_text_splitters import RecursiveCharacterTextSplitter
from word_handler import (
    Chunk, DocType, DOC_TYPE_SIGNALS, _token_count,
    DOC_TYPE_PROFILES, IMAGE_DIR,
)

os.makedirs(IMAGE_DIR, exist_ok=True)

RASTER_DPI = 300
CONTEXT_WINDOW = 3
CAPTION_MAX_TOKENS = 25
CAPTION_MAX_DIST_PX = 80


STYLE_CHUNK_PARAMS: Dict[str, Tuple[int, int]] = {
    "Heading 1": (128, 0),
    "Heading 2": (128, 0),
    "Heading 3": (128, 0),
    "Footnote": (100, 0),
    "Normal": (500, 75),
}

HEADING_STYLES = {"Heading 1", "Heading 2", "Heading 3"}

# PPStructure type strings → our internal type
PP_TYPE_MAP = {
    "title": "title",
    "text": "text",
    "figure": "figure",
    "figure_caption": "figure_caption",
    "table": "table",
    "table_caption": "table_caption",
    "reference": "text",
    "equation": "text",
}



@dataclass
class PdfMeta:
    doc_id: str
    title: str = ""
    doc_type: str = "GENERAL"
    num_pages: int = 0
    needs_ocr: bool = False


@dataclass
class TextElement:
    text: str
    page: int
    bbox: Tuple[float, float, float, float]
    confidence: float = 0.9
    font_size: float = 12.0
    is_bold: bool = False
    style: str = "Normal"
    section_path: List[str] = field(default_factory=list)
    element_id: int = 0
    doc_id: str = ""


@dataclass
class ImageElement:
    image_path: str
    page: int
    bbox: Tuple[float, float, float, float]
    caption: str = ""
    section_path: List[str] = field(default_factory=list)
    element_id: int = 0
    doc_id: str = ""
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)


@dataclass
class TableElement:
    table: str  # HTML from PPStructure
    page: int
    bbox: Tuple[float, float, float, float]
    section_path: List[str] = field(default_factory=list)
    element_id: int = 0
    doc_id: str = ""
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)


engine: Optional[PPStructure] = None


def get_engine() -> PPStructure:
    """
    lang="en"  for English documents.
    lang="ar"  for better Persian/Arabic support
    """
    global engine
    if engine is None:
        engine = PPStructure(
            table=True,
            ocr=True,
            show_log=False,
            lang="en",
        )
    return engine


def infer_font_size(bbox: tuple) -> float:
    h = bbox[3] - bbox[1]
    return max(6.0, min(40.0, h / 1.5))


def infer_bold(text: str, confidence: float, font_size: float) -> bool:
    if text.isupper() and 2 <= len(text.split()) <= 8:
        return True
    if confidence > 0.9 and font_size >= 16:
        return True
    return False


def infer_style(pp_type: str, font_size: float, is_bold: bool) -> str:

    if pp_type == "title":
        return "Heading 1"
    if font_size >= 16 and is_bold:
        return "Heading 2"
    if font_size >= 13 and is_bold:
        return "Heading 3"
    if font_size < 9:
        return "Footnote"
    return "Normal"


def extract_text_from_res(res: Any) -> Tuple[str, float]:

    if not res or not isinstance(res, list):
        return "", 0.0
    texts, scores = [], []
    for item in res:
        if isinstance(item, dict):
            t = item.get("transcription", "")
            s = item.get("score", 0.9)
            if t:
                texts.append(t)
                scores.append(float(s))
    text = " ".join(texts).strip()
    confidence = float(np.mean(scores)) if scores else 0.0
    return text, confidence


def extract_table_html(item: dict) -> str:

    res = item.get("res", {})
    if isinstance(res, dict):
        html = res.get("html", "")
        if html:
            return html
    if isinstance(res, list):
        parts = [r.get("transcription", "") for r in res if isinstance(r, dict)]
        return " | ".join(p for p in parts if p)
    return str(res) if res else ""


def crop_and_save(page_image: Image.Image, bbox: tuple, save_path: str) -> str:

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    page_image.crop(bbox).save(save_path)
    return save_path


def add_context_windows(body_elements: list, window: int = CONTEXT_WINDOW):
    for idx, element in enumerate(body_elements):
        if not isinstance(element, (ImageElement, TableElement)):
            continue
        before, after = [], []
        j = idx - 1
        while j >= 0 and len(before) < window:
            if isinstance(body_elements[j], TextElement):
                before.insert(0, body_elements[j].text)
            j -= 1
        j = idx + 1
        while j < len(body_elements) and len(after) < window:
            if isinstance(body_elements[j], TextElement):
                after.append(body_elements[j].text)
            j += 1
        element.context_before = before
        element.context_after = after


def ocr_extract(pdf_path: str) -> Tuple[List, PdfMeta]:



    doc = fitz.open(pdf_path)
    doc_id = Path(pdf_path).name

    try:
        meta_title = doc.metadata.get("title", "").strip() or doc_id
    except Exception:
        meta_title = doc_id

    pdf_meta = PdfMeta(
        doc_id=doc_id,
        title=meta_title,
        num_pages=len(doc),
        needs_ocr=False,
    )

    engine = get_engine()
    elements: list = []
    element_id = 0
    image_count = 0
    section_stack: List[str] = []

    doc_image_dir = os.path.join(IMAGE_DIR, f"ocr_{doc_id}")
    os.makedirs(doc_image_dir, exist_ok=True)

    for page_num, page in enumerate(doc, start=1):


        pix = page.get_pixmap(dpi=RASTER_DPI)
        page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_bgr = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)


        result = engine(img_bgr)
        result.sort(key=lambda r: r.get("bbox", [0, 0, 0, 0])[1])

        page_elements: list = []
        last_image_el: Optional[ImageElement] = None

        for item in result:
            raw_type = item.get("type", "text").lower()
            pp_type = PP_TYPE_MAP.get(raw_type, "text")
            bbox = tuple(item.get("bbox", [0, 0, 0, 0]))


            if pp_type == "figure_caption":
                text, _ = extract_text_from_res(item.get("res", []))
                if text and last_image_el is not None:
                    last_image_el.caption = text
                continue


            if pp_type == "table_caption":
                text, _ = extract_text_from_res(item.get("res", []))
                if text:
                    for el in reversed(page_elements):
                        if isinstance(el, TableElement):
                            el.table = f"Caption: {text}\n{el.table}"
                            break
                continue


            if pp_type in ("text", "title"):
                text, confidence = extract_text_from_res(item.get("res", []))
                if not text:
                    continue

                font_size = infer_font_size(bbox)
                is_bold = infer_bold(text, confidence, font_size)
                style = infer_style(pp_type, font_size, is_bold)

                if style in HEADING_STYLES:
                    level = int(style[-1])  # "Heading 1" → 1
                    section_stack = section_stack[:level - 1]
                    section_stack.append(text)

                page_elements.append(TextElement(
                    text=text,
                    page=page_num,
                    bbox=bbox,
                    confidence=confidence,
                    font_size=font_size,
                    is_bold=is_bold,
                    style=style,
                    section_path=section_stack.copy(),
                    element_id=element_id,
                    doc_id=doc_id,
                ))
                element_id += 1


            elif pp_type == "figure":
                image_count += 1
                save_path = os.path.join(
                    doc_image_dir,
                    f"fig_page{page_num}_{image_count}.png"
                )
                crop_and_save(page_image, bbox, save_path)

                img_el = ImageElement(
                    image_path=save_path,
                    page=page_num,
                    bbox=bbox,
                    section_path=section_stack.copy(),
                    element_id=element_id,
                    doc_id=doc_id,
                )
                page_elements.append(img_el)
                last_image_el = img_el
                element_id += 1


            elif pp_type == "table":
                table_str = extract_table_html(item)
                if not table_str:
                    continue
                page_elements.append(TableElement(  # was: page_el.append(elements)
                    table=table_str,
                    page=page_num,
                    bbox=bbox,
                    section_path=section_stack.copy(),
                    element_id=element_id,
                    doc_id=doc_id,
                ))
                element_id += 1

        elements.extend(page_elements)


    claimed = set()
    for idx, el in enumerate(elements):
        if not isinstance(el, ImageElement) or el.caption:
            continue
        best_text = None
        best_dist = float("inf")
        for j, other in enumerate(elements):
            if j in claimed or not isinstance(other, TextElement):
                continue
            if _token_count(other.text) > CAPTION_MAX_TOKENS:
                continue
            if other.page != el.page:
                continue
            if other.bbox[1] < el.bbox[3]:  # must be below image
                continue
            dist = other.bbox[1] - el.bbox[3]
            if dist < best_dist:
                best_dist = dist
                best_text = (j, other)
        if best_text and best_dist < CAPTION_MAX_DIST_PX:
            el.caption = best_text[1].text
            claimed.add(best_text[0])

    body_elements = [e for i, e in enumerate(elements) if i not in claimed]
    add_context_windows(body_elements)

    return body_elements, pdf_meta
