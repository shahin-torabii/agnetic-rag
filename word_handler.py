from dataclasses import dataclass, field
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