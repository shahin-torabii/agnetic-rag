from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
CAPTION_MAX_DIST_PX = 80  # pixels — valid because bboxes are NOT normalised

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


# ══════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════

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

