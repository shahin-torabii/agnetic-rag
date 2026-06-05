from dataclasses import dataclass, field
from enum import Enum
import docx
from docx.opc.oxml import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.document import Document as DocxDocument
from docx.opc.constants import RELATIONSHIP_TYPE as rt
import os
from pathlib import Path
from typing import List, Optional, Tuple , Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


@dataclass
class DocMeta:

    doc_id: str
    title: str = ""
    doc_type: str = "GENERAL"
    headers: List[str] = field(default_factory=list)
    footers: List[str] = field(default_factory=list)


@dataclass
class TextElement:
    text: str
    style: str = "Normal"
    section_path: List[str] = field(default_factory=list)
    element_id: int = None
    doc_id: str = None


@dataclass
class ImageElement:
    image_path: str
    caption: str = ""
    section_path: List[str] = field(default_factory=list)
    element_id: int = None
    doc_id: str = None
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)


@dataclass
class TableElement:
    table: str
    section_path: List[str] = field(default_factory=list)
    element_id: int = None
    doc_id: str = None
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)


@dataclass
class SectionNode:
    """
    One node in the section tree.
    heading_text is "" for the implicit root node (pre-first-heading content).
    depth: 0=root, 1=Heading1, 2=Heading2, …
    """
    heading_text: str
    depth: int
    section_path: List[str]
    elements: list = field(default_factory=list)  # TextElement/ImageElement/TableElement
    children: list = field(default_factory=list)  # List[SectionNode]


@dataclass
class Chunk:

    text: str
    doc_id: str
    chunk_index: int
    chunk_type: str
    style: str = "Normal"
    section_path: List[str] = field(default_factory=list)
    start_element_id: int = None
    end_element_id: int = None
    image_refs: Dict[int, str] = field(default_factory=dict)
    table_refs: Dict[int, str] = field(default_factory=dict)
    token_count: int = 0


class DocType(str, Enum):
    RESUME = "RESUME"
    EMAIL = "EMAIL"
    LEGAL = "LEGAL"
    ACADEMIC = "ACADEMIC"
    TECHNICAL = "TECHNICAL"
    REPORT = "REPORT"
    GENERAL = "GENERAL"

DOC_TYPE_SIGNALS : Dict[DocType, List[str]] ={
    DocType.RESUME: [
        "resume", "curriculum vitae", "cv", "work experience",
        "education", "skills", "objective", "references",
        "employment history", "professional experience",
        "سوابق شغلی", "تجربه کاری", "مهارت‌ها", "توانایی‌ها",
        "تحصیلات", "رزومه", "زندگی‌نامه", "اهداف شغلی",
    ],

    DocType.EMAIL: [
        "from:", "to:", "cc:", "subject:", "dear", "regards",
        "sincerely", "forwarded message", "reply", "attachment",
        "با احترام", "از طرف", "به:", "موضوع:",
        "ارادتمند", "پیام فوروارد شده",
    ],

    DocType.LEGAL: [
        "whereas", "hereinafter", "party", "agreement", "contract",
        "clause", "terms and conditions", "liability", "jurisdiction",
        "witnesseth", "indemnity",
        "قرارداد", "ماده", "تبصره", "طرفین", "تعهدات",
        "شرایط و ضوابط", "مسئولیت", "صلاحیت قضایی",
    ],

    DocType.ACADEMIC: [
        "abstract", "introduction", "methodology", "results",
        "conclusion", "references", "literature review", "hypothesis",
        "discussion", "experiment", "analysis",
        "چکیده", "مقدمه", "روش‌شناسی", "نتایج",
        "نتیجه‌گیری", "منابع", "بررسی ادبیات",
        "فرضیه", "بحث", "تحلیل",
    ],

    DocType.TECHNICAL: [
        "architecture", "implementation", "algorithm", "api",
        "specification", "module", "interface", "configuration",
        "deployment", "system design", "performance",
        "fpga", "routing", "hardware", "firmware", "protocol",
        "database", "backend", "frontend",
        "پیاده‌سازی", "معماری", "الگوریتم",
        "پیکربندی", "رابط", "ماژول",
        "پروتکل", "سیستم", "کارایی",
    ],

    DocType.REPORT: [
        "executive summary", "findings", "recommendations",
        "overview", "background", "scope", "appendix",
        "methodology", "results", "discussion",
        "خلاصه اجرایی", "یافته‌ها", "پیشنهادات",
        "بررسی کلی", "پیش‌زمینه", "دامنه",
        "ضمیمه", "گزارش", "نتایج",
    ],
}


DOC_TYPE_PROFILES : Dict[DocType, Dict[str, Tuple[int, int]]]={
 DocType.RESUME: {

        "Normal":    (150, 20),
        "Heading 1": (80,  10),
        "Heading 2": (80,  10),
    },
    DocType.EMAIL: {
        "Normal":    (300, 20),
    },
    DocType.LEGAL: {

        "Normal":    (400, 100),
        "Heading 1": (128, 20),
        "Heading 2": (128, 20),
    },
    DocType.ACADEMIC: {
        "Normal":    (600, 80),
        "Heading 1": (128, 20),
        "Heading 2": (128, 20),
    },
    DocType.TECHNICAL: {
        # Technical: code/specs need precision, smaller chunks
        "Normal":    (350, 50),
        "Heading 1": (128, 10),
        "Heading 2": (128, 10),
    },
    DocType.REPORT: {
        "Normal":    (500, 75),
        "Heading 1": (128, 20),
        "Heading 2": (128, 20),
    },
    DocType.GENERAL: {
        "Normal":    (500, 75),
        "Heading 1": (128, 10),
        "Heading 2": (128, 10),
    },
}

BASE_STYLE_PARAMS: Dict[str, Tuple[int, int]] = {
    "Title":          (128, 10),
    "Heading 1":      (128, 10),
    "Heading 2":      (128, 10),
    "Heading 3":      (128, 10),
    "Heading 4":      (128, 10),
    "Caption":        (100, 10),
    "Figure Caption": (100, 10),
    "Table Caption":  (100, 10),
    "Normal":         (500, 75),
}

def classify_document(doc_meta: DocMeta, body_elements:list) -> DocType:

    main_concepts = " ".join(filter(None, [ doc_meta.title,
        *doc_meta.headers,
        *doc_meta.footers,
    ])).lower()

    secondary_concepts = " ".join(el.text for el in body_elements[:20] if isinstance(el , TextElement)).lower()

    heading_texts = [
        el.text.lower() for el in body_elements
        if isinstance(el, TextElement) and el.style in ("Heading 1", "Heading 2")
    ]
    heading_blob = " ".join(heading_texts)
    scores: Dict[DocType, int] = {dt: 0 for dt in DocType}

    for doctype, keywords in DOC_TYPE_SIGNALS.items():
        for keyword in keywords:
            if keyword in main_concepts:
                scores[doctype] +=2

            if keyword in secondary_concepts:
                scores[doctype]+=1

            if keyword in heading_blob:
                scores[doctype] += 2

    category = max(scores, key=lambda dt:scores[dt])
    max_score = scores[category]
    return category if max_score > 0 else DocType.GENERAL


def pick_chunk_size(doc_type:DocType, chunk_type:str , style:str) -> Tuple[int, int]:
    if chunk_type == "image_context":
        return 400, 60
    elif chunk_type == "table":
        return None, None
    else:
        profile = DOC_TYPE_PROFILES.get(doc_type, {})
        if style in profile:
            return profile[style]
        else:
            return BASE_STYLE_PARAMS.get(style, (500, 75))

IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)


HEADING_STYLES : Dict[str, int]= {
    "title":0,
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Heading 4": 4,
}

CAPTION_STYLES = {"caption", "Figure Caption", "Table Caption"}

SECTION_MARKERS = {
    0: "[DOCUMENT]",
    1: "[SECTION]",
    2: "[SUBSECTION]",
    3: "[SUBSUBSECTION]",
    4: "[SUBSUBSUBSECTION]",
}



def iter_block_items(parent):



    parent_elm = parent.element.body

    for child in parent_elm.iterchildren():

        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)

        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)

def extract_element(docx_path):

    doc = docx.Document(docx_path)

    elements = []
    image_counter = 0
    image_map = {}

    path = Path(docx_path)
    file_name = path.name
    elements.append(file_name)

    for rel in doc.part.rels.values():

        if rel.reltype == rt.IMAGE:
            image_counter+=1

            image_part = rel.target_part
            format = image_part.content_type.split("/")[-1]

            image_path = os.path.join(IMAGE_DIR, f"image_{image_counter}.{format}")
            with open(image_path, "wb") as f:
                f.write(image_part.blob)

            image_map[rel.rId] = image_path

    for block in iter_block_items(doc):

        if isinstance(block, Paragraph):
            text = block.text.strip()

            if text:
                elements.append(TextElement(text))

            blips = block._element.xpath(".//a:blip")
            for blip in blips:

                embed_id = blip.get(qn("r:embed"))

                if embed_id in image_map:
                    elements.append(ImageElement( image_path = image_map[embed_id]))


        elif isinstance(block, Table):

            rows = []

            for row in block.rows:

                row_content = [cell.text.strip() for cell in row.cells]
                row_content = "|".join(row_content)
                rows.append(row_content)

            table = "\n".join(rows)

            elements.append(TableElement(table=table))


    return elements



def create_block(elements: list):
    doc_id = elements[0]
    block = []
    for index, element in enumerate(elements[1:]):
        element.element_id = index + 1
        element.doc_id = doc_id

        if isinstance(element, ImageElement) or isinstance(element, TableElement):
            before = []
            after = []
            window_size = 4
            j = index - 1
            while j>=0 and len(before)<= window_size:
                if isinstance(elements[j], TextElement):
                    before.insert(0, elements[j])
                j-=1

            j = index + 1
            while j <len(elements) and len(after) <= window_size:
                if isinstance(elements[j], TextElement):
                    after.insert(0, elements[j])
                j += 1

            element.context_before = before
            element.context_after = after
        block.append(element)

    return block


def build_single_doc(blocks):
    image_lookup = {}
    table_lookup = {}
    final_doc = []

    for block in blocks:
        if isinstance(block, TextElement):
            final_doc.append(block.text)
        elif isinstance(block, ImageElement):
            placeholder = f"Image {block.element_id}"
            final_doc.append(placeholder)
            image_lookup[block.element_id] = block.image_path
        elif isinstance(block, TableElement):
            placeholder = f"Table {block.element_id}"
            final_doc.append(placeholder)
            table_lookup[block.element_id] = block.table

    return "\n".join(final_doc), image_lookup, table_lookup


def split_and_chunk():
    pass




###################TEST######################
path_of_word = r"F:\university\az e riz\گزارش.docx"
elements = extract_element(path_of_word)
block = create_block(elements)
for el in block:
    print(el)
    print("\n")