from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from word_handler import (
    Chunk, DocType, DOC_TYPE_SIGNALS, _token_count, IMAGE_DIR
)

os.makedirs(IMAGE_DIR, exist_ok=True)

OCR_TEXT_RATIO_THRESHOLD = 0.2

# Minimum characters on a page to count it as "has text"
MIN_TEXT_CHARS = 50

# Font-size thresholds for heading detection
# Calibrated against typical academic/technical PDFs
HEADING_FONT_THRESHOLDS = [
    (18.0, "Heading 1"),
    (15.0, "Heading 2"),
    (13.0, "Heading 3"),
]


STYLE_CHUNK_PARAMS: Dict[str, Tuple[int, int]] = {
    "Heading 1": (128, 0),
    "Heading 2": (128, 0),
    "Heading 3": (128, 0),
    "Normal": (500, 75),
}

CONTEXT_WINDOW = 3

COLUMN_X_TOLERANCE = 50.0


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
    table: str
    page: int
    bbox: Tuple[float, float, float, float]
    section_path: List[str] = field(default_factory=list)
    element_id: int = 0
    doc_id: str = ""
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)

def bbox_overlaps(bbox, tb):
    pass

def infer_style():
    pass

def add_context_windows():
    pass

def table_to_string():
    pass

def reading_order_sort():
    pass


def check_needs_ocr(doc: fitz.Document) -> bool:

    pages_with_text = sum(
        1 for page in doc
        if len(page.get_text().strip()) >= MIN_TEXT_CHARS
    )
    ratio = pages_with_text / max(len(doc), 1)
    return ratio < OCR_TEXT_RATIO_THRESHOLD


def extract_pdf(pdf_path: str) -> Tuple[List, PdfMeta]:

    doc = fitz.open(pdf_path)
    doc_id = Path(pdf_path).name

    needs_ocr = check_needs_ocr(doc)

    try:
        meta_title = doc.metadata.get("title", "").strip() or doc_id
    except Exception:
        meta_title = doc_id

    pdf_meta = PdfMeta(
        doc_id=doc_id,
        title=meta_title,
        num_pages=len(doc),
        needs_ocr=needs_ocr,
    )

    if needs_ocr:
        return [], pdf_meta

    body_elements: list = []
    heading_stack: List[str] = []
    element_counter = 0
    image_counter = 0

    for page_num, page in enumerate(doc, start=1):
        page_elems: list = []

        table_bboxes = []
        try:
            tables = page.find_tables()
            for table in tables.tables:
                table_bboxes.append(tuple(table.bbox))
        except Exception:
            tables = None


        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue

            block_bbox = tuple(block["bbox"])


            if any(bbox_overlaps(block_bbox, tb) for tb in table_bboxes):
                continue


            all_spans = [
                span
                for line in block["lines"]
                for span in line["spans"]
            ]
            if not all_spans:
                continue

            rep_span = max(all_spans, key=lambda s: s["size"])
            font_size = rep_span["size"]
            is_bold = "bold" in rep_span["font"].lower()
            style = infer_style(font_size, is_bold)


            lines_text = []
            for line in block["lines"]:
                line_text = " ".join(s["text"] for s in line["spans"]).strip()
                if line_text:
                    lines_text.append(line_text)
            text = "\n".join(lines_text).strip()

            if not text:
                continue


            if style.startswith("Heading"):
                level = int(style[-1])
                heading_stack = heading_stack[:level - 1]
                heading_stack.append(text)

            element_counter += 1
            page_elems.append(TextElement(
                text=text,
                page=page_num,
                bbox=block_bbox,
                font_size=font_size,
                is_bold=is_bold,
                style=style,
                section_path=heading_stack.copy(),
                element_id=element_counter,
                doc_id=doc_id,
            ))
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                img_data = doc.extract_image(xref)
            except Exception:
                continue

            image_counter += 1
            ext = img_data["ext"]
            image_path = os.path.join(
                IMAGE_DIR, f"{doc_id}_page{page_num}_img{image_counter}.{ext}"
            )
            with open(image_path, "wb") as f:
                f.write(img_data["image"])

            # Get bbox from image info on the page
            img_rects = page.get_image_rects(xref)
            bbox = tuple(img_rects[0]) if img_rects else (0, 0, 0, 0)

            element_counter += 1
            page_elems.append(ImageElement(
                image_path=image_path,
                page=page_num,
                bbox=bbox,
                section_path=heading_stack.copy(),
                element_id=element_counter,
                doc_id=doc_id,
            ))


        if tables:
            for table in tables.tables:
                table_data = table.extract()
                if not table_data:
                    continue

                table_str = table_to_string(table_data)
                element_counter += 1
                page_elems.append(TableElement(
                    table=table_str,
                    page=page_num,
                    bbox=tuple(table.bbox),
                    section_path=heading_stack.copy(),
                    element_id=element_counter,
                    doc_id=doc_id,
                ))


        page_elems = reading_order_sort(page_elems)
        body_elements.extend(page_elems)

    claimed = set()
    for idx, el in enumerate(body_elements):
        if not isinstance(el, ImageElement):
            continue

        best_text = None
        best_dist = float("inf")

        for j, other in enumerate(body_elements):
            if j in claimed:
                continue
            if not isinstance(other, TextElement):
                continue
            if _token_count(other.text) > 25:
                continue

            if other.page != el.page:
                continue
            if other.bbox[1] < el.bbox[3]:
                continue

            dist = abs(other.bbox[1] - el.bbox[3])
            if dist < best_dist:
                best_dist = dist
                best_text = (j, other)

        if best_text and best_dist < 50:
            j, caption_el = best_text
            el.caption = caption_el.text
            claimed.add(j)

    body_elements = [
        e for i, e in enumerate(body_elements) if i not in claimed
    ]

    add_context_windows(body_elements)


    return body_elements, pdf_meta