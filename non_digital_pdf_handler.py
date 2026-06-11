from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "false"
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import fitz
import numpy as np
from PIL import Image

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


engine = None


def get_engine() :
    from paddleocr import PPStructure
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


def classify_pdf(meta: PdfMeta, body_elements: list) -> DocType:
    high_weight = meta.title.lower()
    body_sample = " ".join(
        el.text for el in body_elements[:20] if isinstance(el, TextElement)
    ).lower()
    heading_blob = " ".join(
        el.text for el in body_elements
        if isinstance(el, TextElement) and el.style in HEADING_STYLES
    ).lower()

    scores: Dict[DocType, int] = {dt: 0 for dt in DocType}
    for doc_type, keywords in DOC_TYPE_SIGNALS.items():
        for kw in keywords:
            if kw in high_weight:   scores[doc_type] += 2
            if kw in body_sample:   scores[doc_type] += 1
            if kw in heading_blob:  scores[doc_type] += 2

    best = max(scores, key=lambda dt: scores[dt])
    best_score = scores[best]
    return best if best_score > 0 else DocType.GENERAL


def pick_chunk_params(style: str, doc_type: DocType) -> Tuple[int, int]:
    profile = DOC_TYPE_PROFILES.get(doc_type, {})
    if style in profile:
        return profile[style]
    return STYLE_CHUNK_PARAMS.get(style, (500, 75))


def build_section_text(elements: List[TextElement]) -> str:
    SECTION_MARKERS = {1: "[SECTION]", 2: "[SUBSECTION]", 3: "[SUBSUBSECTION]"}
    lines, last_style = [], None
    for el in elements:
        if el.style in HEADING_STYLES:
            level = int(el.style[-1])
            marker = SECTION_MARKERS.get(level, "[SECTION]")
            if el.style != last_style:
                lines.append(marker)
                last_style = el.style
            lines.append(el.text)
        else:
            lines.append(el.text)
    return "\n\n".join(lines)


def chunk(
        body_elements: list,
        pdf_meta: PdfMeta,
        doc_type: DocType,
) -> Tuple[List[Chunk], Dict[tuple, List[Chunk]], Dict[tuple, List[Chunk]]]:
    chunks: List[Chunk] = []
    image_to_chunks: Dict[tuple, List[Chunk]] = {}
    table_to_chunks: Dict[tuple, List[Chunk]] = {}
    chunk_index = 0
    doc_id = pdf_meta.doc_id
    i = 0

    while i < len(body_elements):
        el = body_elements[i]

        if isinstance(el, ImageElement):
            section_prefix = " > ".join(el.section_path)
            full_text = "\n".join(filter(None, [
                f"[{section_prefix}]" if section_prefix else f"[Page {el.page}]",
                f"[IMAGE_{el.element_id}]",
                f"Caption: {el.caption}" if el.caption else "",
                " ".join(el.context_before),
                " ".join(el.context_after),
            ])).strip()
            c = Chunk(
                text=full_text, doc_id=doc_id, chunk_index=chunk_index,
                chunk_type="image_context", style="Normal",
                section_path=el.section_path,
                start_element_id=el.element_id, end_element_id=el.element_id,
                image_refs={el.element_id: el.image_path},
                token_count=_token_count(full_text),
            )
            chunks.append(c)
            image_to_chunks.setdefault((doc_id, el.element_id), []).append(c)
            chunk_index += 1
            i += 1

        elif isinstance(el, TableElement):
            rows = el.table.split("\n")
            section_prefix = " > ".join(el.section_path)
            header_lines = "\n".join(filter(None, [
                f"[{section_prefix}]" if section_prefix else f"[Page {el.page}]",
                " ".join(el.context_before),
                f"[TABLE_{el.element_id}]",
            ]))
            if len(rows) <= 15:
                full_text = f"{header_lines}\n{el.table}".strip()
                c = Chunk(
                    text=full_text, doc_id=doc_id, chunk_index=chunk_index,
                    chunk_type="table", section_path=el.section_path,
                    start_element_id=el.element_id, end_element_id=el.element_id,
                    table_refs={el.element_id: el.table},
                    token_count=_token_count(full_text),
                )
                chunks.append(c)
                table_to_chunks.setdefault((doc_id, el.element_id), []).append(c)
                chunk_index += 1
            else:
                header_row = rows[0]
                data_rows = rows[1:]
                window, step = 10, 8
                for w in range(0, len(data_rows), step):
                    batch = [header_row] + data_rows[w:w + window]
                    prefix = header_lines if w == 0 else f"[TABLE_{el.element_id} continued]"
                    full_text = f"{prefix}\n" + "\n".join(batch)
                    c = Chunk(
                        text=full_text.strip(), doc_id=doc_id, chunk_index=chunk_index,
                        chunk_type="table", section_path=el.section_path,
                        start_element_id=el.element_id, end_element_id=el.element_id,
                        table_refs={el.element_id: el.table},
                        token_count=_token_count(full_text),
                    )
                    chunks.append(c)
                    table_to_chunks.setdefault((doc_id, el.element_id), []).append(c)
                    chunk_index += 1
            i += 1

        elif isinstance(el, TextElement):
            section_els: List[TextElement] = []
            while i < len(body_elements) and isinstance(body_elements[i], TextElement):
                cur = body_elements[i]
                if cur.style == "Heading 1" and section_els:
                    break
                section_els.append(cur)
                i += 1

            sub_groups: List[List[TextElement]] = []
            current: List[TextElement] = []
            for elem in section_els:
                if elem.style in HEADING_STYLES and current:
                    sub_groups.append(current)
                    current = []
                current.append(elem)
            if current:
                sub_groups.append(current)

            for group in sub_groups:
                dominant_style = group[0].style
                last_heading = next(
                    (e for e in reversed(group) if e.style in HEADING_STYLES), None
                )
                dominant_path = (
                    last_heading.section_path if last_heading
                    else group[-1].section_path
                )
                section_text = build_section_text(group)
                size, overlap = pick_chunk_params(dominant_style, doc_type)

                if _token_count(section_text) <= size:
                    chunks.append(Chunk(
                        text=section_text, doc_id=doc_id, chunk_index=chunk_index,
                        chunk_type="text", style=dominant_style,
                        section_path=dominant_path,
                        start_element_id=group[0].element_id,
                        end_element_id=group[-1].element_id,
                        token_count=_token_count(section_text),
                    ))
                    chunk_index += 1
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=size * 5, chunk_overlap=overlap * 5,
                        separators=["[SECTION]", "[SUBSECTION]", "[SUBSUBSECTION]",
                                    "\n\n", "\n", ". ", " ", ""],
                    )
                    for sc in splitter.split_text(section_text):
                        chunks.append(Chunk(
                            text=sc, doc_id=doc_id, chunk_index=chunk_index,
                            chunk_type="text", style=dominant_style,
                            section_path=dominant_path,
                            start_element_id=group[0].element_id,
                            end_element_id=group[-1].element_id,
                            token_count=_token_count(sc),
                        ))
                        chunk_index += 1
        else:
            i += 1

    return chunks, image_to_chunks, table_to_chunks


def process_scanned_pdf(pdf_path: str):
    body_elements, pdf_meta = ocr_extract(pdf_path)
    doc_type = classify_pdf(pdf_meta, body_elements)
    pdf_meta.doc_type = doc_type.value
    chunks, image_to_chunks, table_to_chunks = chunk(body_elements, pdf_meta, doc_type)
    return chunks, pdf_meta, image_to_chunks, table_to_chunks


if __name__ == "__main__":


    path = r"D:\rag\ztm\ZeroToMastery - AI Engineering Retrieval Augmented Generation (RAG) for LLMs 2025-8\code\GenAI\RAG\RAG with OpenAI\The chinese cookbook.pdf"

    chunks, meta, img_idx, tbl_idx = process_scanned_pdf(path)

    print(f"\n File      : {meta.doc_id}")
    print(f"   Doc type  : {meta.doc_type}")
    print(f"   Pages     : {meta.num_pages}")
    print(f"   Chunks    : {len(chunks)}")
    print(f"   img→chunk : { {k: len(v) for k, v in img_idx.items()} }")
    print(f"   tbl→chunk : { {k: len(v) for k, v in tbl_idx.items()} }\n")

    for c in chunks:
        print(f"[{c.chunk_index:03d}] type={c.chunk_type:<14} style={c.style:<14} "
              f"tokens={c.token_count:<4} path={c.section_path}")
        print(f"       preview : {c.text[:120].strip()}")
        if c.image_refs:
            print(f"       images  : {c.image_refs}")
        if c.table_refs:
            print(f"       tables  : {list(c.table_refs.keys())}")
        print()
