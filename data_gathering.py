from dataclasses import  dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

@dataclass
class BaseMeta:
    doc_id: str
    title: str = ""
    source_type: str = ""
    doc_type: str = "General"


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

@dataclass
class UserRequest:
    query: str
    images: list = None
    documents: list = None
    audio:list = None


ChunkKey = Tuple[str, int]
MediaKey = Tuple[str, int]

class Data:

    docs: Dict[str, BaseMeta] = {}


    chunks: List[Chunk] = []


    chunk_by_key: Dict[ChunkKey, Chunk] = {}


    image_to_chunks: Dict[MediaKey, List[ChunkKey]] = {}


    table_to_chunks: Dict[MediaKey, List[ChunkKey]] = {}



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

IMAGE_DIR = "images"

def _token_count(text: str) -> int:
    return len(text.split())



def ingest(
    chunks: List[Chunk],
    doc_meta: BaseMeta,
    img_to_ch: Dict[MediaKey, List[ChunkKey]],
    tbl_to_ch: Dict[MediaKey, List[ChunkKey]]
) -> None:

    Data.docs[doc_meta.doc_id] = doc_meta

    for chunk in chunks:
        chunk_key = (chunk.doc_id, chunk.chunk_index)

        Data.chunks.append(chunk)

        Data.chunk_by_key[chunk_key] = chunk

    for image_key, chunk_keys in img_to_ch.items():
        Data.image_to_chunks.setdefault(
            image_key,
            []
        ).extend(chunk_keys)

    for table_key, chunk_keys in tbl_to_ch.items():
        Data.table_to_chunks.setdefault(
            table_key,
            []
        ).extend(chunk_keys)