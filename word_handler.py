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


def _token_count(text: str) -> int:
    return len(text.split())


def iter_block_items(parent):

    parent_elm = parent.element.body

    for child in parent_elm.iterchildren():

        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)

        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)


def extract_headers_footers(doc) -> Tuple[List[str], List[str]]:
    seen_h, seen_f = set(), set()
    headers, footers = [], []
    for section in doc.sections:
        if section.header and not section.header.is_linked_to_previous:
            for para in section.header.paragraphs:
                t = para.text.strip()
                if t and t not in seen_h:
                    seen_h.add(t);
                    headers.append(t)
        if section.footer and not section.footer.is_linked_to_previous:
            for para in section.footer.paragraphs:
                t = para.text.strip()
                if t and t not in seen_f:
                    seen_f.add(t);
                    footers.append(t)
    return headers, footers


def extract_element(docx_path: str) -> Tuple[list, DocMeta]:

    doc = docx.Document(docx_path)
    path = Path(docx_path)
    doc_id = path.name

    headers, footers = extract_headers_footers(doc)

    try:
        title = doc.core_properties.title
    except Exception:
        title = ""

    doc_meta = DocMeta(doc_id = doc_id, title= title, headers=headers, footers= footers)

    image_counter = 0
    image_map = Dict[str, str] = {}

    body_elements = []
    heading_stack = List[str] = []

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
            style = block.style.name
            if style in HEADING_STYLES:
                depth = HEADING_STYLES[style]
                heading_stack= heading_stack[:depth]
                if text:
                    heading_stack.append(text)

            elif style in CAPTION_STYLES:
                if body_elements and isinstance(body_elements[-1], ImageElement):
                    body_elements[-1].caption = text

                    continue
                elif body_elements and isinstance(body_elements[-1], TableElement):
                    body_elements[-1].table = f"Caption: {text}\n{body_elements[-1].table}"
                    continue
            if text:
                text_el = TextElement(
                    text         = text,
                    style        = style,
                    section_path = heading_stack.copy(),
                )
                body_elements.append(text_el)

            blips = block._element.xpath(".//a:blip")
            for blip in blips:

                embed_id = blip.get(qn("r:embed"))

                if embed_id in image_map:
                    body_elements.append(ImageElement( image_path = image_map[embed_id],  section_path = heading_stack.copy(),))


        elif isinstance(block, Table):

            rows = []

            for row in block.rows:

                row_content = [cell.text.strip() for cell in row.cells]
                row_content = "|".join(row_content)
                rows.append(row_content)

            table = "\n".join(rows)

            body_elements.append(TableElement(table=table,  section_path = heading_stack.copy(),))


    return body_elements, doc_meta



def create_block(body_elements: list, doc_meta:DocMeta) -> list:
    doc_id = doc_meta.doc_id
    block = []
    for index, element in enumerate(body_elements):
        element.element_id = index + 1
        element.doc_id = doc_id

        if isinstance(element, ImageElement) or isinstance(element, TableElement):
            before ,after = [], []
            window_size = 3
            j = index - 1
            while j>=0 and len(before)<= window_size:
                if isinstance(body_elements[j], TextElement):
                    before.insert(0,body_elements[j].text)
                j-=1

            j = index + 1
            while j <len(elements) and len(after) <= window_size:
                if isinstance(body_elements[j], TextElement):
                    after.insert(0, body_elements[j].text)
                j += 1

            element.context_before = before
            element.context_after = after
        block.append(element)

    return block

def build_section_tree(bodyelements: list) -> SectionNode:

    root = SectionNode(heading_text="", depth=0, section_path=[])
    stack : List[SectionNode] = [root]

    for element in bodyelements:
        if isinstance(element, TextElement) and element.style in HEADING_STYLES:
            depth = HEADING_STYLES[element.style]

            new_node = SectionNode(
                 heading_text=element.text,
                depth=depth,
                section_path= element.section_path.copy()
             )

            new_node.elements.append(element)

            while len(stack) >1 and stack[-1].depth >=depth:
                stack.pop()

            stack[-1].children.append(new_node)
            stack.append(new_node)

        else:
            # Non-heading element → belongs to current section
            stack[-1].elements.append(element)

    return root


def _flatten_section_tree(node: SectionNode) -> List[SectionNode]:

    leaves = []

    direct_non_heading = [
        e for e in node.elements
        if not (isinstance(e, TextElement) and e.style in HEADING_STYLES)
    ]

    if node.children:

        if direct_non_heading:
            synthetic = SectionNode(
                heading_text=node.heading_text,
                depth=node.depth,
                section_path=node.section_path,
                elements=direct_non_heading,
            )
            leaves.append(synthetic)
        for child in node.children:
            leaves.extend(_flatten_section_tree(child))
    else:

        if node.elements or node.heading_text:
            leaves.append(node)

    return leaves




def build_single_doc(blocks:list) -> str:
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
#
#
# def split_and_chunk():
#     pass




###################TEST######################
path_of_word = r"F:\university\az e riz\گزارش.docx"
elements = extract_element(path_of_word)
block = create_block(elements)
for el in block:
    print(el)
    print("\n")