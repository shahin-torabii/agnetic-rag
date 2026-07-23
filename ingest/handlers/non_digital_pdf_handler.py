from __future__ import annotations

"""
scanned_pdf_handler.py  —  Handles ALL non-digital PDF categories.

TOP 5 CATEGORIES COVERED
─────────────────────────
CAT 1  MRC / Internet Archive
       JPX background + JBIG2 smask text layer.
       Strategy: pdfplumber direct (zero OCR cost) → smask Tesseract fallback.

CAT 2  Clean scanned print
       Single raster image per page, good quality.
       Strategy: script detection → Tesseract.

CAT 3  Degraded scanned print
       Skewed, noisy, low-contrast, inverted.
       Strategy: preprocess (denoise → deskew → binarize → inversion fix) → Tesseract.

CAT 4  Non-Latin script
       Arabic, Persian, Chinese, etc. at any quality level.
       Strategy: OSD script detection → correct Tesseract lang pack.
       Note: lang packs must be installed on the OS.
             sudo apt install tesseract-ocr-ara tesseract-ocr-fas tesseract-ocr-chi-sim

CAT 5  Handwritten / truly messy
       No free local model reliably handles handwriting.
       Strategy: preprocess → Tesseract → if confidence < threshold → VLM flag.
       VLM-flagged chunks have chunk_type="vlm_pending" with the page image path.

DECISION: Tesseract (not PaddleOCR)
  PaddleOCR is unavailable in the target environment and conflicts with PyTorch.
  Tesseract 5 LSTM with proper preprocessing covers all 5 categories adequately.
  For truly handwritten content, no free local model beats a VLM anyway.

RETURN SIGNATURE (identical to process_docx / process_pptx / process_pdf):
  chunks, pdf_meta, image_to_chunks, table_to_chunks

VLM-pending chunks:
  chunk.chunk_type == "vlm_pending"
  chunk.image_refs == {element_id: page_image_path}
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image, ImageOps

from config.manager import get_config
from core.constants import DOC_TYPE_PROFILES, DOC_TYPE_SIGNALS, DocType
from core.types import Chunk, _token_count

os.makedirs(get_config().storage.image_dir, exist_ok=True)


MIN_CHARS_FOR_TEXT = 20


MIN_REAL_WORD_RATIO = 0.40


OCR_RENDER_SCALE = 3.0

VLM_CONFIDENCE_THRESHOLD = 40.0


SMASK_MEAN_THRESHOLD = 100


INVERSION_THRESHOLD = 100


DESKEW_MIN_ANGLE = 0.5

SCRIPT_TO_LANG: Dict[str, str] = {
    "Latin": "eng",
    "Arabic": "ara",
    "Han": "chi_sim",
    "Cyrillic": "rus",
    "Devanagari": "hin",
    "Korean": "kor",
    "Japanese": "jpn",
    "Bengali": "ben",
    "Greek": "ell",
}

FALLBACK_LANG = "eng"

STYLE_CHUNK_PARAMS: Dict[str, Tuple[int, int]] = {
    "Heading 1": (128, 0),
    "Heading 2": (128, 0),
    "Heading 3": (128, 0),
    "Normal": (500, 75),
}
HEADING_STYLES = {"Heading 1", "Heading 2", "Heading 3"}
CONTEXT_WINDOW = 3


@dataclass
class PdfMeta:
    doc_id: str
    title: str = ""
    doc_type: str = "GENERAL"
    num_pages: int = 0
    needs_ocr: bool = True


@dataclass
class TextElement:
    text: str
    page: int
    style: str = "Normal"
    section_path: List[str] = field(default_factory=list)
    element_id: int = 0
    doc_id: str = ""
    source: str = ""  # "pdfplumber" | "tesseract" | "tesseract_smask"


@dataclass
class ImageElement:
    image_path: str
    page: int
    caption: str = ""
    section_path: List[str] = field(default_factory=list)
    element_id: int = 0
    doc_id: str = ""
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    flag_for_vlm: bool = False


def fix_inversion(gray: np.ndarray) -> np.ndarray:

    if gray.mean() < INVERSION_THRESHOLD:
        return cv2.bitwise_not(gray)
    return gray


def deskew(gray: np.ndarray) -> np.ndarray:

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) < DESKEW_MIN_ANGLE:
        return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def denoise(gray: np.ndarray) -> np.ndarray:

    return cv2.fastNlMeansDenoising(
        gray, h=10, templateWindowSize=7, searchWindowSize=21
    )


def binarize(gray: np.ndarray) -> np.ndarray:

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return adaptive


def preprocess(pil_img: Image.Image) -> Image.Image:
    """
    Full preprocessing pipeline for degraded scans (CAT 3):
      1. Inversion fix   — white-on-black → black-on-white
      2. Denoise         — remove scanner noise
      3. Deskew          — correct rotation
      4. Binarize        — adaptive threshold for uneven lighting

    Returns a clean binary PIL image ready for Tesseract.
    """
    gray = np.array(pil_img.convert("L"))
    gray = fix_inversion(gray)
    gray = denoise(gray)
    gray = deskew(gray)
    gray = binarize(gray)
    return Image.fromarray(gray)


def detect_script(pil_img: Image.Image) -> Tuple[str, str, int]:
    """
    Use Tesseract OSD (Orientation and Script Detection) to identify:
      - script name  (e.g. "Arabic", "Han", "Latin")
      - tesseract lang code  (e.g. "ara", "chi_sim", "eng")
      - rotation in degrees  (0, 90, 180, 270)

    Falls back to ("Latin", "eng", 0) on any error.
    """
    try:
        osd = pytesseract.image_to_osd(
            pil_img,
            config="--psm 0 -c min_characters_to_try=5",
            nice=0,
        )
        fields = {}
        for line in osd.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()

        script = fields.get("Script", "Latin")
        rotate = int(fields.get("Rotate", "0"))
        lang = SCRIPT_TO_LANG.get(script, FALLBACK_LANG)
        return script, lang, rotate

    except Exception:
        return "Latin", FALLBACK_LANG, 0


def correct_rotation(pil_img: Image.Image, degrees: int) -> Image.Image:
    if degrees == 0:
        return pil_img
    return pil_img.rotate(degrees, expand=True)


def run_ocr(pil_img: Image.Image, lang: str = "eng") -> Tuple[str, float]:

    def _ocr(img, lg):
        data = pytesseract.image_to_data(
            img,
            lang=lg,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        confs = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
        words = [
            data["text"][i]
            for i, c in enumerate(data["conf"])
            if str(c).isdigit() and int(c) >= 0 and data["text"][i].strip()
        ]
        text = " ".join(words)
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        return text, mean_conf

    try:
        return _ocr(pil_img, lang)
    except pytesseract.TesseractError:
        try:
            return _ocr(pil_img, FALLBACK_LANG)
        except Exception:
            return "", 0.0
    except Exception:
        return "", 0.0


def extract_smask(page_pdfium) -> Optional[Image.Image]:

    for obj in page_pdfium.get_objects():
        if not isinstance(obj, pdfium.PdfImage):
            continue
        try:
            bm = obj.get_bitmap(render=True)
            pi = bm.to_pil()
            if pi.mode != "RGBA":
                continue
            _, _, _, alpha = pi.split()
            arr = np.array(alpha)
            if arr.mean() < SMASK_MEAN_THRESHOLD:
                return ImageOps.invert(alpha.convert("L"))
        except Exception:
            continue
    return None


def render_page(page_pdfium, scale: float = OCR_RENDER_SCALE) -> Image.Image:

    return page_pdfium.render(scale=scale).to_pil().convert("L")


def text_is_readable(text: str) -> bool:

    if not text or len(text.strip()) < 10:
        return False
    words = text.split()
    if not words:
        return False
    real = [w for w in words if re.match(r"^[a-zA-Z]{2,}", w)]
    return (len(real) / len(words)) >= MIN_REAL_WORD_RATIO


def infer_style(line: str) -> str:

    s = line.strip()
    if not s:
        return "Normal"
    words = s.split()
    if len(words) <= 6 and (s.isupper() or s.istitle()) and s[-1] not in ".,:;?!)":
        return "Heading 1" if len(words) <= 3 else "Heading 2"
    return "Normal"


def lines_to_elements(
    text: str, page: int, doc_id: str, counter: list, stack: list, source: str
) -> List[TextElement]:

    elements = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        style = infer_style(line)
        if style in HEADING_STYLES:
            level = int(style[-1])
            stack[:] = stack[: level - 1]
            stack.append(line)
        elements.append(
            TextElement(
                text=line,
                page=page,
                style=style,
                section_path=list(stack),
                element_id=counter[0],
                doc_id=doc_id,
                source=source,
            )
        )
        counter[0] += 1
    return elements


def save_page_image(
    img: Image.Image, doc_id: str, page_num: int, image_dir: str
) -> str:
    os.makedirs(image_dir, exist_ok=True)
    path = os.path.join(image_dir, f"page_{page_num}.png")
    img.save(path)
    return path


def add_context_windows(body_elements: list, window: int = CONTEXT_WINDOW):
    for idx, el in enumerate(body_elements):
        if not isinstance(el, ImageElement):
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
        el.context_before = before
        el.context_after = after


def ocr_image(img_pil: Image.Image) -> Tuple[str, float]:
    """
    Full OCR pipeline for a single page image.
    Handles CAT 2 (clean), CAT 3 (degraded), CAT 4 (non-Latin).

    Order:
      1. Script + orientation detection (OSD)
      2. Rotation correction
      3. Preprocessing (inversion, denoise, deskew, binarize)
      4. Tesseract with detected lang
      5. If confidence still low, retry with preprocessed image
    """
    # Step 1: detect script and rotation
    script, lang, rotate = detect_script(img_pil)

    # Step 2: fix rotation
    img_pil = correct_rotation(img_pil, rotate)

    # Step 3: preprocess
    img_preprocessed = preprocess(img_pil)

    # Step 4: OCR on preprocessed
    text, conf = run_ocr(img_preprocessed, lang=lang)

    # Step 5: if confidence is poor, also try raw
    if conf < VLM_CONFIDENCE_THRESHOLD:
        raw_gray = np.array(img_pil.convert("L"))
        raw_gray = fix_inversion(raw_gray)
        text_raw, conf_raw = run_ocr(Image.fromarray(raw_gray), lang=lang)
        if conf_raw > conf:
            text, conf = text_raw, conf_raw

    return text, conf


def extract_pages(pdf_path: str) -> Tuple[List, PdfMeta]:
    """
    Per-page extraction.

    Priority per page:
      1. pdfplumber direct  — zero cost, handles valid MRC text (CAT 1)
      2. Readability check  — catch garbage-encoded MRC → route to OCR
      3. Smask extraction   — clean JBIG2 text layer (CAT 1 fallback)
      4. Full OCR pipeline  — CAT 2/3/4/5
      5. VLM flag           — confidence < threshold (CAT 5, heavily degraded)
    """
    path = Path(pdf_path)
    doc_id = path.name

    pdf_pdfium = pdfium.PdfDocument(pdf_path)

    try:
        with pdfplumber.open(pdf_path) as pdf_pl:
            meta_title = pdf_pl.metadata.get("Title", "").strip() or doc_id
    except Exception:
        meta_title = doc_id

    pdf_meta = PdfMeta(
        doc_id=doc_id,
        title=meta_title,
        num_pages=len(pdf_pdfium),
        needs_ocr=True,
    )

    body_elements: list = []
    section_stack: list = []
    counter: list = [0]
    image_counter: int = 0
    doc_image_dir = os.path.join(get_config().storage.image_dir, f"scan_{doc_id}")

    try:
        with pdfplumber.open(pdf_path) as pdf_pl:
            for page_idx, pl_page in enumerate(pdf_pl.pages):
                page_num = page_idx + 1
                chars = pl_page.chars or []
                pl_images = pl_page.images or []
                has_text = len(chars) >= MIN_CHARS_FOR_TEXT

                # ── CAT 1 FAST PATH: pdfplumber readable ─────
                if has_text:
                    raw = pl_page.extract_text() or ""
                    if text_is_readable(raw):
                        els = lines_to_elements(
                            raw,
                            page_num,
                            doc_id,
                            counter,
                            section_stack,
                            source="pdfplumber",
                        )
                        body_elements.extend(els)
                        continue

                pdfium_page = pdf_pdfium[page_idx]

                smask = extract_smask(pdfium_page)
                if smask:
                    text, conf = run_ocr(smask, lang="eng")
                    if conf >= VLM_CONFIDENCE_THRESHOLD and text.strip():
                        els = lines_to_elements(
                            text,
                            page_num,
                            doc_id,
                            counter,
                            section_stack,
                            source="tesseract_smask",
                        )
                        body_elements.extend(els)
                        continue

                rendered = render_page(pdfium_page)
                text, conf = ocr_image(rendered)

                if conf >= VLM_CONFIDENCE_THRESHOLD and text.strip():
                    els = lines_to_elements(
                        text,
                        page_num,
                        doc_id,
                        counter,
                        section_stack,
                        source="tesseract",
                    )
                    body_elements.extend(els)

                else:
                    image_counter += 1
                    img_path = save_page_image(
                        rendered, doc_id, page_num, doc_image_dir
                    )
                    body_elements.append(
                        ImageElement(
                            image_path=img_path,
                            page=page_num,
                            section_path=list(section_stack),
                            element_id=counter[0],
                            doc_id=doc_id,
                            flag_for_vlm=True,
                        )
                    )
                    counter[0] += 1

    except Exception as e:
        print(f"[warn] extraction error on {doc_id}: {e}")

    finally:
        pdf_pdfium.close()

    add_context_windows(body_elements)
    return body_elements, pdf_meta


def classify_pdf(meta: PdfMeta, body_elements: list) -> DocType:
    high = meta.title.lower()
    sample = " ".join(
        e.text for e in body_elements[:20] if isinstance(e, TextElement)
    ).lower()
    headings = " ".join(
        e.text
        for e in body_elements
        if isinstance(e, TextElement) and e.style in HEADING_STYLES
    ).lower()

    scores: Dict[DocType, int] = {dt: 0 for dt in DocType}
    for dt, keywords in DOC_TYPE_SIGNALS.items():
        for kw in keywords:
            if kw in high:
                scores[dt] += 2
            if kw in sample:
                scores[dt] += 1
            if kw in headings:
                scores[dt] += 2

    best = max(scores, key=lambda dt: scores[dt])
    return best if scores[best] > 0 else DocType.GENERAL


def pick_chunk_params(style: str, doc_type: DocType) -> Tuple[int, int]:
    profile = DOC_TYPE_PROFILES.get(doc_type, {})
    if style in profile:
        return profile[style]
    return STYLE_CHUNK_PARAMS.get(style, (500, 75))


def build_section_text(elements: List[TextElement]) -> str:
    MARKERS = {1: "[SECTION]", 2: "[SUBSECTION]", 3: "[SUBSUBSECTION]"}
    lines, last = [], None
    for el in elements:
        if el.style in HEADING_STYLES:
            level = int(el.style[-1])
            if el.style != last:
                lines.append(MARKERS.get(level, "[SECTION]"))
                last = el.style
            lines.append(el.text)
        else:
            lines.append(el.text)
    return "\n\n".join(lines)


def split_and_chunk(
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
            prefix = " > ".join(el.section_path) or f"Page {el.page}"

            if el.flag_for_vlm:
                full_text = f"[{prefix}]\n[VLM_PENDING_{el.element_id}]"
                ctype = "vlm_pending"
            else:
                full_text = "\n".join(
                    filter(
                        None,
                        [
                            f"[{prefix}]",
                            f"[IMAGE_{el.element_id}]",
                            f"Caption: {el.caption}" if el.caption else "",
                            " ".join(el.context_before),
                            " ".join(el.context_after),
                        ],
                    )
                ).strip()
                ctype = "image_context"

            c = Chunk(
                text=full_text,
                doc_id=doc_id,
                chunk_index=chunk_index,
                chunk_type=ctype,
                style="Normal",
                section_path=el.section_path,
                start_element_id=el.element_id,
                end_element_id=el.element_id,
                image_refs={el.element_id: el.image_path},
                token_count=_token_count(full_text),
            )
            chunks.append(c)
            image_to_chunks.setdefault((doc_id, el.element_id), []).append(c)
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
                    last_heading.section_path
                    if last_heading
                    else group[-1].section_path
                )
                section_text = build_section_text(group)
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
                            start_element_id=group[0].element_id,
                            end_element_id=group[-1].element_id,
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
                                start_element_id=group[0].element_id,
                                end_element_id=group[-1].element_id,
                                token_count=_token_count(sc),
                            )
                        )
                        chunk_index += 1
        else:
            i += 1

    return chunks, image_to_chunks, table_to_chunks


def process_scanned_pdf(pdf_path: str):
    """
    Full pipeline for any non-digital PDF.

    VLM-pending chunks (handwritten, messy, OCR-failed):
        chunk.chunk_type == "vlm_pending"
        chunk.image_refs == {element_id: "/path/to/page_N.png"}

    Usage in retrieval.py:
        chunks, meta, img_idx, tbl_idx = process_scanned_pdf(path)
        doc_meta_store[meta.doc_id] = meta

        ready_chunks = [c for c in chunks if c.chunk_type != "vlm_pending"]
        vlm_chunks   = [c for c in chunks if c.chunk_type == "vlm_pending"]

        index_chunks(ready_chunks)
        index_images(ready_chunks)
        # route vlm_chunks to your VLM pipeline when budget allows
    """
    body_elements, pdf_meta = extract_pages(pdf_path)
    doc_type = classify_pdf(pdf_meta, body_elements)
    pdf_meta.doc_type = doc_type.value
    chunks, img_idx, tbl_idx = split_and_chunk(body_elements, pdf_meta, doc_type)
    return chunks, pdf_meta, img_idx, tbl_idx


if __name__ == "__main__":
    path = r"C:\Users\shahin\Desktop\jozve.pdf"

    chunks, meta, img_idx, tbl_idx = process_scanned_pdf(path)

    text_c = [c for c in chunks if c.chunk_type == "text"]
    img_c = [c for c in chunks if c.chunk_type == "image_context"]
    vlm_c = [c for c in chunks if c.chunk_type == "vlm_pending"]

    print(f"\n📄  {meta.doc_id}")
    print(f"    doc_type : {meta.doc_type}")
    print(f"    pages    : {meta.num_pages}")
    print(f"    chunks   : {len(chunks)}")
    print(f"      text   : {len(text_c)}")
    print(f"      images : {len(img_c)}")
    print(f"      vlm    : {len(vlm_c)}  ← flagged for VLM")

    print("\n── Text chunks (first 5) ──────────────────────────")
    for c in text_c[:20]:
        src = next((e.source for e in [] if hasattr(e, "source")), "")
        print(
            f"  [{c.chunk_index:03d}] tokens={c.token_count:4d}  path={c.section_path}"
        )
        print(f"         {c.text[:100].strip()}")

    if vlm_c:
        print("\n── VLM-pending pages ──────────────────────────────")
        for c in vlm_c[:5]:
            print(f"  [{c.chunk_index:03d}] page_img={list(c.image_refs.values())}")
