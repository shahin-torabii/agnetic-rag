from pathlib import Path
import puremagic

from core.logger import get_logger
from ingest.handlers.digital_pdf_handler import process_pdf

logger = get_logger(__name__)
from ingest.handlers.non_digital_pdf_handler import process_scanned_pdf
from ingest.handlers.word_handler import process_docx
from ingest.handlers.excel_handler import process_excel
from ingest.handlers.pp_handler import process_pptx


def detect_file_type(file_path):
    path = Path(file_path)

    suffix, mime = None, None

    if path.exists():
        suffix = path.suffix
        mime = puremagic.from_file(str(path), mime=True)

    return mime, suffix, path


def get_doc_chunks(file_path):
    mime, suffix, path = detect_file_type(file_path)

    if mime == "application/pdf":
        logger.info("Processing digital PDF", extra={"path": str(path)})
        chunks, meta, img_idx, tbl_idx = process_pdf(file_path)
        if chunks == []:
            logger.info("Falling back to scanned PDF processing", extra={"path": str(path)})
            return process_scanned_pdf(file_path)

        return chunks, meta, img_idx, tbl_idx

    if mime in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        #word
        return process_docx(file_path)

    if mime in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel"
    ]:
        #excel
        return process_excel(file_path)

    if mime in [
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint"
    ]:
        #pptx
        return process_pptx(file_path)
    # if mime.startswith("video/"):
    #     return "Video"

    return "invalid"
