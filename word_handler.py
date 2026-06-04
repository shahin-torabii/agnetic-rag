from dataclasses import dataclass
import docx
from docx.opc.oxml import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.document import Document as DocxDocument
from docx.opc.constants import RELATIONSHIP_TYPE as rt
import os
from pathlib import Path

@dataclass
class TextElement:
    text:str


@dataclass
class ImageElement:
    image_path: str


@dataclass
class TableElement:
    table:str

IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)


def iter_block_items(parent):


    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc

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

path_of_word = r"F:\university\az e riz\گزارش.docx"
elements = extract_element(path_of_word)
for el in elements:
    print(el)
    print("\n")