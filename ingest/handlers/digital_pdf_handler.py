from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.manager import get_config
from core.constants import DOC_TYPE_PROFILES, DOC_TYPE_SIGNALS, IMAGE_DIR, DocType
from core.types import BaseMeta, Chunk, _token_count

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
class PdfMeta(BaseMeta):
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


def bbox_contains(outer: tuple, inner: tuple) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def bbox_overlaps(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def infer_style(font_size: float, is_bold: bool) -> str:

    for threshold, style in HEADING_FONT_THRESHOLDS:
        if font_size >= threshold:
            return style
    if is_bold and font_size >= 11.5:
        return "Heading 3"
    return "Normal"


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


def table_to_string(table_data: List[List]) -> str:
    rows = []
    for row in table_data:
        cells = [str(c).strip() if c is not None else "" for c in row]
        rows.append("|".join(cells))
    return "\n".join(rows)


def reading_order_sort(elements: list) -> list:

    left_xs = sorted(
        set(
            round(e.bbox[0] / COLUMN_X_TOLERANCE) * COLUMN_X_TOLERANCE for e in elements
        )
    )

    if len(left_xs) <= 1 or np.abs(left_xs[0] - left_xs[-1]) <= COLUMN_X_TOLERANCE * 2:
        return sorted(elements, key=lambda x: (x.bbox[1], x.bbox[0]))

    mid_x = (left_xs[0] + left_xs[-1]) / 2

    left_col = [e for e in elements if e.bbox[0] < mid_x]
    right_col = [e for e in elements if e.bbox[0] >= mid_x]

    left_col.sort(key=lambda e: e.bbox[1])
    right_col.sort(key=lambda e: e.bbox[1])

    return left_col + right_col


def check_needs_ocr(doc: fitz.Document) -> bool:

    pages_with_text = sum(
        1 for page in doc if len(page.get_text().strip()) >= MIN_TEXT_CHARS
    )
    ratio = pages_with_text / max(len(doc), 1)
    return ratio < OCR_TEXT_RATIO_THRESHOLD


def classify_pdf(meta: PdfMeta, body_elements: list) -> DocType:
    high_weight = meta.title.lower()

    body_sample = " ".join(
        el.text for el in body_elements[:20] if isinstance(el, TextElement)
    ).lower()

    heading_blob = " ".join(
        el.text
        for el in body_elements
        if isinstance(el, TextElement) and el.style.startswith("Heading")
    ).lower()

    scores: Dict[DocType, int] = {dt: 0 for dt in DocType}
    for doc_type, keywords in DOC_TYPE_SIGNALS.items():
        for kw in keywords:
            if kw in high_weight:
                scores[doc_type] += 2
            if kw in body_sample:
                scores[doc_type] += 1
            if kw in heading_blob:
                scores[doc_type] += 2

    best = max(scores, key=lambda dt: scores[dt])
    best_score = scores[best]
    return best if best_score > 0 else DocType.GENERAL


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
        source_type="pdf",
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

            all_spans = [span for line in block["lines"] for span in line["spans"]]
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
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(text)

            element_counter += 1
            page_elems.append(
                TextElement(
                    text=text,
                    page=page_num,
                    bbox=block_bbox,
                    font_size=font_size,
                    is_bold=is_bold,
                    style=style,
                    section_path=heading_stack.copy(),
                    element_id=element_counter,
                    doc_id=doc_id,
                )
            )

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

            img_rects = page.get_image_rects(xref)
            bbox = tuple(img_rects[0]) if img_rects else (0, 0, 0, 0)

            element_counter += 1
            page_elems.append(
                ImageElement(
                    image_path=image_path,
                    page=page_num,
                    bbox=bbox,
                    section_path=heading_stack.copy(),
                    element_id=element_counter,
                    doc_id=doc_id,
                )
            )

        if tables:
            for table in tables.tables:
                table_data = table.extract()
                if not table_data:
                    continue

                table_str = table_to_string(table_data)
                element_counter += 1
                page_elems.append(
                    TableElement(
                        table=table_str,
                        page=page_num,
                        bbox=tuple(table.bbox),
                        section_path=heading_stack.copy(),
                        element_id=element_counter,
                        doc_id=doc_id,
                    )
                )

        page_elems = reading_order_sort(page_elems)
        body_elements.extend(page_elems)

    claimed = set()
    for idx, el in enumerate(body_elements):
        if not isinstance(el, ImageElement):
            continue

        best_text = None
        best_dist = np.inf

        for j, other in enumerate(body_elements):
            if j in claimed:
                continue
            if not isinstance(other, TextElement):
                continue
            if _token_count(other.text) > 25:
                continue

            if other.page != el.page:
                continue

            dist = abs(other.bbox[1] - el.bbox[3])
            if dist < best_dist:
                best_dist = dist
                best_text = (j, other)

        if best_text and best_dist < 50:
            j, caption_el = best_text
            el.caption = caption_el.text
            claimed.add(j)

    body_elements = [e for i, e in enumerate(body_elements) if i not in claimed]

    add_context_windows(body_elements)

    return body_elements, pdf_meta


def pick_chunk_params(style: str, doc_type: DocType) -> Tuple[int, int]:

    profile = DOC_TYPE_PROFILES.get(doc_type, {})
    if style in profile:
        return profile[style]
    return STYLE_CHUNK_PARAMS.get(style, (500, 75))


def build_section_text(elements: List[TextElement]) -> str:

    SECTION_MARKERS = {1: "[SECTION]", 2: "[SUBSECTION]", 3: "[SUBSUBSECTION]"}
    lines, last_style = [], None

    for el in elements:
        if el.style.startswith("Heading"):
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
) -> Tuple[
    List[Chunk], Dict[tuple, Set[Tuple[str, int]]], Dict[tuple, Set[Tuple[str, int]]]
]:

    chunks: List[Chunk] = []
    image_to_chunks: Dict[tuple, Set[Tuple[str, int]]] = {}
    table_to_chunks: Dict[tuple, Set[Tuple[str, int]]] = {}
    chunk_index = 0
    doc_id = pdf_meta.doc_id

    i = 0
    while i < len(body_elements):
        el = body_elements[i]

        if isinstance(el, ImageElement):
            before_text = " ".join(el.context_before)
            after_text = " ".join(el.context_after)
            caption_text = f"Caption: {el.caption}" if el.caption else ""
            section_prefix = " > ".join(el.section_path)

            full_text = "\n".join(
                filter(
                    None,
                    [
                        f"[{section_prefix}]"
                        if section_prefix
                        else f"[Page {el.page}]",
                        f"[IMAGE_{el.element_id}]",
                        caption_text,
                        before_text,
                        after_text,
                    ],
                )
            ).strip()

            c = Chunk(
                text=full_text,
                doc_id=doc_id,
                chunk_index=chunk_index,
                chunk_type="image_context",
                style="Normal",
                section_path=el.section_path,
                start_element_id=el.element_id,
                end_element_id=el.element_id,
                image_refs={el.element_id: el.image_path},
                token_count=_token_count(full_text),
            )
            chunks.append(c)
            image_to_chunks.setdefault((doc_id, el.element_id), []).append(
                (doc_id, chunk_index)
            )
            chunk_index += 1
            i += 1

        elif isinstance(el, TableElement):
            rows = el.table.split("\n")
            context = " ".join(el.context_before)
            section_prefix = " > ".join(el.section_path)

            header_lines = "\n".join(
                filter(
                    None,
                    [
                        f"[{section_prefix}]"
                        if section_prefix
                        else f"[Page {el.page}]",
                        context,
                        f"[TABLE_{el.element_id}]",
                    ],
                )
            )

            if len(rows) <= 15:
                full_text = f"{header_lines}\n{el.table}".strip()
                c = Chunk(
                    text=full_text,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    chunk_type="table",
                    section_path=el.section_path,
                    start_element_id=el.element_id,
                    end_element_id=el.element_id,
                    table_refs={el.element_id: el.table},
                    token_count=_token_count(full_text),
                )
                chunks.append(c)
                table_to_chunks.setdefault((doc_id, el.element_id), []).append(
                    (doc_id, chunk_index)
                )
                chunk_index += 1

            else:
                header_row = rows[0]
                data_rows = rows[1:]
                window, step = 10, 8

                for w in range(0, len(data_rows), step):
                    batch = [header_row] + data_rows[w : w + window]
                    prefix = (
                        header_lines if w == 0 else f"[TABLE_{el.element_id} continued]"
                    )
                    full_text = f"{prefix}\n" + "\n".join(batch)

                    c = Chunk(
                        text=full_text.strip(),
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        chunk_type="table",
                        section_path=el.section_path,
                        start_element_id=el.element_id,
                        end_element_id=el.element_id,
                        table_refs={el.element_id: el.table},
                        token_count=_token_count(full_text),
                    )
                    chunks.append(c)
                    table_to_chunks.setdefault((doc_id, el.element_id), []).append(
                        (doc_id, chunk_index)
                    )
                    chunk_index += 1

            i += 1

        elif isinstance(el, TextElement):
            section_els: List[TextElement] = []
            start_eid = el.element_id

            while i < len(body_elements) and isinstance(body_elements[i], TextElement):
                cur = body_elements[i]
                if cur.style == "Heading 1" and section_els:
                    break
                section_els.append(cur)
                i += 1

            end_eid = section_els[-1].element_id
            dominant_style = section_els[0].style
            dominant_path = section_els[0].section_path
            section_text = build_section_text(section_els)
            size, overlap = pick_chunk_params(dominant_style, doc_type)

            if _token_count(section_text) <= size:
                chunks.append(
                    Chunk(
                        text=section_text,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        chunk_type="text",
                        style=dominant_style,
                        section_path=dominant_path,
                        start_element_id=start_eid,
                        end_element_id=end_eid,
                        token_count=_token_count(section_text),
                    )
                )
                chunk_index += 1

            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=get_config().chunking.max_tokens,
                    chunk_overlap=get_config().chunking.overlap_tokens,
                    separators=[
                        "[SECTION]",
                        "[SUBSECTION]",
                        "[SUBSUBSECTION]",
                        "\n\n",
                        "\n",
                        ". ",
                        " ",
                        "",
                    ],
                )
                for sc in splitter.split_text(section_text):
                    chunks.append(
                        Chunk(
                            text=sc,
                            doc_id=doc_id,
                            chunk_index=chunk_index,
                            chunk_type="text",
                            style=dominant_style,
                            section_path=dominant_path,
                            start_element_id=start_eid,
                            end_element_id=end_eid,
                            token_count=_token_count(sc),
                        )
                    )
                    chunk_index += 1

        else:
            i += 1

    return chunks, image_to_chunks, table_to_chunks


def process_pdf(pdf_path: str):
    body_elements, pdf_meta = extract_pdf(pdf_path)
    if pdf_meta.needs_ocr:
        return [], pdf_meta, {}, {}

    doc_type = classify_pdf(pdf_meta, body_elements)
    pdf_meta.doc_type = doc_type.value
    chunks, image_to_chunks, table_to_chunks = chunk(body_elements, pdf_meta, doc_type)
    return chunks, pdf_meta, image_to_chunks, table_to_chunks


if __name__ == "__main__":
    path = r"F:\university\os\03_os_dualmode.pdf"

    chunks, meta, img_idx, tbl_idx = process_pdf(path)

    if meta.needs_ocr:
        print(f"⚠️  {meta.doc_id} needs OCR — no text extracted")
    else:
        print(f"\n File      : {meta.doc_id}")
        print(f"   Doc type  : {meta.doc_type}")
        print(f"   Pages     : {meta.num_pages}")
        print(f"   Chunks    : {len(chunks)}")
        print(f"   img→chunk : { {k: len(v) for k, v in img_idx.items()} }")
        print(f"   tbl→chunk : { {k: len(v) for k, v in tbl_idx.items()} }\n")

        for c in chunks:
            print(
                f"[{c.chunk_index:03d}] type={c.chunk_type:<14} "
                f"style={c.style:<14} tokens={c.token_count:<4} "
                f"path={c.section_path}"
            )
            print(f"       preview : {c.text[:120].strip()}")
            if c.image_refs:
                print(f"       images  : {c.image_refs}")
            if c.table_refs:
                print(f"       tables  : {list(c.table_refs.keys())}")
            print()
