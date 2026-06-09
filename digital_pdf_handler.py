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
