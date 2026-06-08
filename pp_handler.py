from __future__ import annotations

from dataclasses import Field, dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

from pptx import presentation
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter

from word_handler import DocType, DOC_TYPE_SIGNALS, _token_count, Chunk, IMAGE_DIR

IMAGE_DIR ="images/pp"

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
    image_path : str
    caption:str
    text: str = ""
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

def extract_element(pptx_path:str) -> Tuple[List[SlideData], PPTX_META]

