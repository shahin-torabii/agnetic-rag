from enum import Enum
from typing import Dict, List, Literal, Tuple

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5
TOKENS_PER_BATCH = 3000
GROUP_SIZE = 5

UPLOAD_DIR = "uploads"


BLEND_LOW_WEIGHTS = (0.8, 0.2)
BLEND_HIGH_WEIGHTS = (0.2, 0.8)

IGNORE_IMAGE_THRESHOLD = 0.1
HIGH_IMAGE_THRESHOLD = 0.4

REFERENTIAL_PATTERN = r"\b(it|its|that|this|them|those|again|the (first|second|third|last|previous|other) one)\b"


Route = Literal[
    "document",
    "image",
    "audio",
    "multimodal",
    "general",
]


class DocType(str, Enum):
    RESUME = "RESUME"
    EMAIL = "EMAIL"
    LEGAL = "LEGAL"
    ACADEMIC = "ACADEMIC"
    TECHNICAL = "TECHNICAL"
    REPORT = "REPORT"
    GENERAL = "GENERAL"


STRUCTURAL_TERMS = {
    # Persian
    "بخش",
    "فصل",
    "مرحله",
    "گام",
    "اول",
    "اولین",
    "آخر",
    "آخرین",
    "پایانی",
    "نتیجه",
    "نتیجه گیری",
    "جمع بندی",
    "مقدمه",
    # English
    "chapter",
    "section",
    "part",
    "step",
    "first",
    "last",
    "final",
    "conclusion",
    "summary",
    "introduction",
}

DOC_TYPE_SIGNALS: Dict[DocType, List[str]] = {
    DocType.RESUME: [
        "resume",
        "curriculum vitae",
        "cv",
        "work experience",
        "education",
        "skills",
        "objective",
        "references",
        "employment history",
        "professional experience",
        "سوابق شغلی",
        "تجربه کاری",
        "مهارت‌ها",
        "توانایی‌ها",
        "تحصیلات",
        "رزومه",
        "زندگی‌نامه",
        "اهداف شغلی",
    ],
    DocType.EMAIL: [
        "from:",
        "to:",
        "cc:",
        "subject:",
        "dear",
        "regards",
        "sincerely",
        "forwarded message",
        "reply",
        "attachment",
        "با احترام",
        "از طرف",
        "به:",
        "موضوع:",
        "ارادتمند",
        "پیام فوروارد شده",
    ],
    DocType.LEGAL: [
        "whereas",
        "hereinafter",
        "party",
        "agreement",
        "contract",
        "clause",
        "terms and conditions",
        "liability",
        "jurisdiction",
        "witnesseth",
        "indemnity",
        "قرارداد",
        "ماده",
        "تبصره",
        "طرفین",
        "تعهدات",
        "شرایط و ضوابط",
        "مسئولیت",
        "صلاحیت قضایی",
    ],
    DocType.ACADEMIC: [
        "abstract",
        "introduction",
        "methodology",
        "results",
        "conclusion",
        "references",
        "literature review",
        "hypothesis",
        "discussion",
        "experiment",
        "analysis",
        "چکیده",
        "مقدمه",
        "روش‌شناسی",
        "نتایج",
        "نتیجه‌گیری",
        "منابع",
        "بررسی ادبیات",
        "فرضیه",
        "بحث",
        "تحلیل",
    ],
    DocType.TECHNICAL: [
        "architecture",
        "implementation",
        "algorithm",
        "api",
        "specification",
        "module",
        "interface",
        "configuration",
        "deployment",
        "system design",
        "performance",
        "fpga",
        "routing",
        "hardware",
        "firmware",
        "protocol",
        "database",
        "backend",
        "frontend",
        "پیاده‌سازی",
        "معماری",
        "الگوریتم",
        "پیکربندی",
        "رابط",
        "ماژول",
        "پروتکل",
        "سیستم",
        "کارایی",
    ],
    DocType.REPORT: [
        "executive summary",
        "findings",
        "recommendations",
        "overview",
        "background",
        "scope",
        "appendix",
        "methodology",
        "results",
        "discussion",
        "خلاصه اجرایی",
        "یافته‌ها",
        "پیشنهادات",
        "بررسی کلی",
        "پیش‌زمینه",
        "دامنه",
        "ضمیمه",
        "گزارش",
        "نتایج",
    ],
}

DOC_TYPE_PROFILES: Dict[DocType, Dict[str, Tuple[int, int]]] = {
    DocType.RESUME: {
        "Normal": (150, 20),
        "Heading 1": (80, 10),
        "Heading 2": (80, 10),
    },
    DocType.EMAIL: {
        "Normal": (300, 20),
    },
    DocType.LEGAL: {
        "Normal": (400, 100),
        "Heading 1": (128, 20),
        "Heading 2": (128, 20),
    },
    DocType.ACADEMIC: {
        "Normal": (600, 80),
        "Heading 1": (128, 20),
        "Heading 2": (128, 20),
    },
    DocType.TECHNICAL: {
        "Normal": (350, 50),
        "Heading 1": (128, 10),
        "Heading 2": (128, 10),
    },
    DocType.REPORT: {
        "Normal": (500, 75),
        "Heading 1": (128, 20),
        "Heading 2": (128, 20),
    },
    DocType.GENERAL: {
        "Normal": (500, 75),
        "Heading 1": (128, 10),
        "Heading 2": (128, 10),
    },
}

BASE_STYLE_PARAMS: Dict[str, Tuple[int, int]] = {
    "Title": (128, 10),
    "Heading 1": (128, 10),
    "Heading 2": (128, 10),
    "Heading 3": (128, 10),
    "Heading 4": (128, 10),
    "Caption": (100, 10),
    "Figure Caption": (100, 10),
    "Table Caption": (100, 10),
    "Normal": (500, 75),
}

IMAGE_DIR = "images"

FA_EXPLAIN = [
    "توضیح",
    "توضیح بده",
    "شرح بده",
    "تفسیر کن",
]

FA_SUMMARIZE = [
    "خلاصه",
    "خلاصه کن",
    "جمع بندی",
    "جمع‌بندی",
]

FA_SEARCH = [
    "پیدا کن",
    "جستجو",
    "بگرد",
    "کجا",
    "کجاست",
]

FA_COMPARE = [
    "مقایسه",
    "فرق",
    "تفاوت",
]

FA_SECTION = [
    "بخش",
    "فصل",
    "قسمت",
    "مقدمه",
    "نتایج",
    "نتیجه گیری",
    "نتیجه‌گیری",
    "روش",
    "روش شناسی",
    "روش‌شناسی",
]

FA_ACTION = [
    "ترجمه",
    "یادداشت",
    "فلش کارت",
    "فلش‌کارت",
    "سوال",
    "تمرین",
    "quiz",
]

EN_OVERVIEW = [
    "what is this about",
    "what is this pdf about",
    "what is this file about",
    "overview",
    "give me an overview",
    "high level overview",
    "main idea",
    "main topic",
    "topic",
    "subject",
    "what does it discuss",
    "what does this discuss",
    "what is being discussed",
]

FA_OVERVIEW = [
    "درباره چیست",
    "در مورد چیست",
    "موضوع چیست",
    "موضوع فایل چیست",
    "موضوع این فایل چیست",
    "موضوع این سند چیست",
    "موضوع این pdf چیست",
    "این فایل درباره چیست",
    "این سند درباره چیست",
    "این مقاله درباره چیست",
    "این صوت درباره چیست",
    "مرور کلی",
    "نمای کلی",
    "دید کلی",
    "ایده اصلی",
    "موضوع اصلی",
]

EN_EXPLAIN = [
    "explain",
    "describe",
    "interpret",
]

EN_SUMMARIZE = [
    "summarize",
    "summary",
    "tldr",
]

EN_SEARCH = [
    "find",
    "search",
    "locate",
    "show me",
]

EN_COMPARE = [
    "compare",
    "difference",
]

EN_SECTION = [
    "section",
    "chapter",
    "part",
    "methodology",
    "results",
    "discussion",
    "conclusion",
    "introduction",
    "background",
    "related work",
]
EN_ACTION = [
    "translate",
    "notes",
    "flashcards",
    "quiz",
    "questions",
]

EN_AUDIO_TRANSCRIBE = [
    "transcribe",
    "speech to text",
    "convert audio to text",
]

FA_AUDIO_TRANSCRIBE = [
    "رونویسی",
    "تبدیل صوت به متن",
    "پیاده سازی صوت",
]

EN_AUDIO_SUMMARIZE = [
    "summarize audio",
    "summarize recording",
    "summarize meeting",
]

FA_AUDIO_SUMMARIZE = [
    "خلاصه فایل صوتی",
    "خلاصه جلسه",
    "خلاصه کن",
]

CURRENT_FILE_SIGNALS = {
    # English
    "this",
    "these",
    "attached",
    "uploaded",
    "this file",
    "this document",
    "this pdf",
    # Persian
    "این",
    "این فایل",
    "این سند",
    "این پی دی اف",
    "این pdf",
    "فایل آپلود شده",
    "سند آپلود شده",
    "فایلی که آپلود کردم",
    "فایلی که فرستادم",
}

ALL_FILES_SIGNALS = {
    # English
    "all uploaded files",
    "all files",
    "all documents",
    "every file",
    "every document",
    # Persian
    "همه فایل ها",
    "همه فایل‌ها",
    "تمام فایل ها",
    "تمام فایل‌ها",
    "همه اسناد",
    "تمام اسناد",
}

ORDINAL_SIGNALS = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "اول": 0,
    "دوم": 1,
    "سوم": 2,
    "چهارم": 3,
    "پنجم": 4,
}

LAST_SIGNALS = {"last", "latest", "آخر", "آخری", "آخرین"}



IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
