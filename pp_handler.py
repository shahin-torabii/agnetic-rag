from __future__ import annotations

from dataclasses import Field, dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE_TYPE

import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter

from word_handler import DocType, DOC_TYPE_SIGNALS, _token_count, Chunk, IMAGE_DIR

IMAGE_DIR ="images"

SECTION_HEADER_LAYOUTS = {
    "section header",
    "section",
    "divider",
    "chapter",
}

SKIP_LAYOUTS = {
    "title slide",
}

SLIDE_TOKEN_LIMIT = 300


@dataclass
class PPTX_META:
    doc_id: str
    title: str
    doc_type:str = "GENERAL"
    slide_count :int = 0

@dataclass
class SlideElement:
    doc_id: str
    element_id :int
    slide_number:int
    style:str
    type:str
    caption:str = ""
    text: str = ""
    image_path: str = ""
    x:int = 0
    y :int = 0
    height:int = 0
    width:int = 0

@dataclass
class SlideData:
    doc_id:str
    slide_number:int
    prev_slide: int
    next_slide: int
    title:str
    section_name:str
    layout_name:str
    elements: List[SlideElement] = field(default_factory=list)


def is_title(shape) -> bool:
    if not shape.is_placeholder:
        return False
    else:
        pp_type = shape.placeholder_format.type
        return pp_type in (
            PP_PLACEHOLDER.TITLE,
            PP_PLACEHOLDER.CENTER_TITLE
        )

def extract_text(shape) -> str:
    lines = []
    if not shape.has_text_frame:
        return " "


    for para in shape.text_frame.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)

    return "\n".join(lines)


def shape_center(x: int, y: int, width: int, height: int) -> Tuple[float, float]:

    return (x + width / 2, y + height / 2)


def table_to_string(table) ->str:
    rows = []

    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        cells_ = "|".join(cells)
        rows.append(cells_)
    table_str = "\n".join(rows)
    return table_str

def find_nearest_text(
    image_el: "SlideElement",
    body_elements: List["SlideElement"],
    max_tokens:    int = 30,
) -> Optional["SlideElement"]:

    if not body_elements:
        return None
    img_cx, img_cy = shape_center(image_el.x, image_el.y, image_el.width, image_el.height)

    best_el = None
    best_dist = np.inf

    for el in body_elements:
        if _token_count(el.text) > max_tokens:
            continue
        el_cx, el_cy = shape_center(el.x, el.y, el.width, el.height)
        distance = np.linalg.norm(img_cx- el_cx, img_cy - el_cy)
        if distance < best_dist:
            best_dist = distance
            best_el = el

    return best_el


def classify_presentation(meta: PPTX_META, slides: List[SlideData]) -> DocType:

    high_weight = meta.title.lower()

    body_weights = " ".join(
        s.title for s in slides[:7]
    ).lower()

    body_weights+=" " + " ".join(
        el.text for slide in slides[:7] for el in slide.elements if el.type =="body"
    ).lower()

    scores: Dict[DocType, int] = {dt: 0 for dt in DocType}

    for doc_type, keywords in DOC_TYPE_SIGNALS.items():
        for kw in keywords:
            if kw in high_weight:
                scores[doc_type] += 2
            if kw in body_weights:
                scores[doc_type] += 1

    best = max(scores, key=lambda dt: scores[dt])
    best_score = scores[best]
    return best if best_score > 0 else DocType.GENERAL


def extract_slides(pptx_path:str) -> Tuple[List[SlideData], PPTX_META]:

    path = Path(pptx_path)
    doc_id = path.name

    pp_file = Presentation(pptx_path)

    try:
        title = pp_file.core_properties.title or " "
    except Exception:
        title =" "

    pp_meta = PPTX_META(
        doc_id=doc_id,
        title= title,
        slide_count=len(pp_file.slides)
    )

    slides: List[SlideData] = []
    current_section = ""
    element_counter = 0
    image_counter = 0

    for slide_index,slide in enumerate(pp_file.slides):
        slide_number = slide_index +1
        layout_names= slide.slide_layout.name.lower().strip()

        if any(s in layout_names for s in SECTION_HEADER_LAYOUTS):
            for shape in slide.shapes:
                if is_title(shape):
                    section = extract_text(shape).strip()
                    if section:
                        current_section = section
                        break

        skip = any(s in layout_names for s in SKIP_LAYOUTS)



        slide_elements = List[SlideElement] = []
        title:str = ""
        if not skip:
            for shape in slide.shapes:
                if is_title(shape):
                    title = extract_text(shape).strip()
                    continue

                if shape.has_table:
                    element_counter+=1
                    table_str = table_to_string(shape.table)

                    tb_element = SlideElement(
                        doc_id=doc_id,
                        element_id=element_counter,
                        text=table_str,
                        slide_number=slide_number,
                        type="table"
                    )
                    slide_elements.append(tb_element)
                    continue

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    element_counter+=1
                    image_counter+=1

                    image_bytes = shape.image.blob
                    image_ext = shape.image.ext

                    file_name = f"{doc_id}_slide{slide_number}_image {image_counter}"
                    image_path = os.path.join(IMAGE_DIR, file_name)

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    im_element = SlideElement(
                        doc_id=doc_id,
                        element_id=element_counter,
                        image_path=image_path,
                        slide_number=slide_number,
                        type="image",
                        x=shape.left or 0,
                        y=shape.top or 0,
                        width=shape.width or 0,
                        height=shape.height or 0,
                    )
                    slide_elements.append(im_element)

                if shape.has_text_frame:
                    element_counter+=1
                    text = extract_text(shape).strip()

                    body_element = SlideElement(
                            type         = "body",
                            text         = text,
                            slide_number = slide_number,
                            element_id   = element_counter,
                            doc_id       = doc_id,
                            x            = shape.left   or 0,
                            y            = shape.top    or 0,
                            width        = shape.width  or 0,
                            height       = shape.height or 0,
                        )
                    slide_elements.append(body_element)

        if slide.has_notes_slide:
            note_text = slide.notes_slide.notes_text_frame.text.strip()
            element_counter+=1
            note_el = SlideElement(
                    type         = "notes",
                    text         = note_text,
                    slide_number = slide_number,
                    element_id   = element_counter,
                    doc_id       = doc_id,
                )

            slide_elements.append(note_el)



        slide_el = SlideData(title = title,
                             slide_number=slide_number,
                             slide_elements = slide_elements,
                             section_name=current_section,
                             layout_name=layout_names)

        body_els = [e for e in slide_elements if e.type == "body"]
        claimed = set()

        for el in slide_elements:
            if el.type != "image":
                continue
            candidate = find_nearest_text(
                el,
                [b for b in body_els if b.element_id not in claimed],
            )
            if candidate:
                el.caption = candidate.text
                claimed.add(candidate.element_id)


        elements = [e for e in elements if e.element_id not in claimed]
        slides.append(slide)

    for index,slide in enumerate(slides):
        slide.prev_slide = slides[index - 1].slide_number if index> 0 else 0
        slide.next_slide = slides[index+1].slide_number  if index < len(slides) else 0

    return slides, pp_meta



