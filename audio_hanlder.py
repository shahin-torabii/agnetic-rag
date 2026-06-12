from faster_whisper import WhisperModel
from dataclasses import dataclass
from typing import Dict, Tuple, List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from word_handler import Chunk, DocType, DOC_TYPE_SIGNALS, _token_count



@dataclass
class TranscriptMeta:
    doc_id: str
    len_audio: float
    text_type: str = ""


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: List[Segment]



DOC_TYPE_PROFILES: Dict[DocType, Tuple[int, int]] = {
    DocType.RESUME: (150, 20),
    DocType.EMAIL: (300, 20),
    DocType.LEGAL: (400, 100),
    DocType.ACADEMIC: (600, 80),
    DocType.TECHNICAL: (350, 50),
    DocType.REPORT: (500, 75),
    DocType.GENERAL: (500, 75),
}




def create_transcriptions(audio_path: str) -> Tuple[Transcript, TranscriptMeta]:

    path = Path(audio_path)
    doc_id = path.name

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    segments_iter, info = model.transcribe(
        audio_path,
        language="en",
        vad_filter=True
    )

    segments: List[Segment] = []
    text_parts = []

    for seg in segments_iter:
        segment = Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip()
        )
        segments.append(segment)
        text_parts.append(segment.text)

    text = " ".join(text_parts)

    len_audio = segments[-1].end - segments[0].start if segments else 0.0

    meta = TranscriptMeta(
        doc_id=doc_id,
        len_audio=len_audio
    )

    transcript = Transcript(
        text=text,
        segments=segments
    )

    return transcript, meta


def classify_doc(meta: TranscriptMeta, text: str) -> DocType:

    title = meta.doc_id.lower().strip()
    body = text.lower().strip()

    scores: Dict[DocType, int] = {dt: 0 for dt in DocType}

    for doc_type, keywords in DOC_TYPE_SIGNALS.items():
        for kw in keywords:
            if kw in title or kw in body:
                scores[doc_type] += 1

    best = max(scores, key=lambda dt: scores[dt])

    return best if scores[best] > 0 else DocType.GENERAL


def pick_chunk_params(doc_type: DocType) -> Tuple[int, int]:
    return DOC_TYPE_PROFILES.get(doc_type, (500, 75))



def chunk(transcript: Transcript, meta: TranscriptMeta) -> Tuple[List[Chunk], TranscriptMeta]:

    doc_type = classify_doc(meta, transcript.text)
    meta.text_type = doc_type

    chunk_size, overlap = pick_chunk_params(doc_type)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 5,
        chunk_overlap=overlap * 5,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    timeline_text = []

    for seg in transcript.segments:
        timeline_text.append(
            f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}"
        )

    full_text = transcript.text + "\n\nTIMELINE:\n" + "\n".join(timeline_text)

    chunks: List[Chunk] = []
    chunk_idx = 0

    for token in splitter.split_text(full_text):

        chunks.append(
            Chunk(
                text=token,
                doc_id=meta.doc_id,
                chunk_index=chunk_idx,
                chunk_type="text",
                token_count=_token_count(token),
            )
        )

        chunk_idx += 1

    return chunks, meta



def process_audio(audio_path: str):

    transcript, meta = create_transcriptions(audio_path)
    chunks, meta = chunk(transcript, meta)

    return chunks, transcript, meta



if __name__ == "__main__":

    audio_path = r"C:\Users\shahin\Desktop\1530-Tahia-UK-Physical-Health.mp3"

    chunks, transcript, meta = process_audio(audio_path)

    print("Doc ID:", meta.doc_id)
    print("Audio length:", meta.len_audio)
    print("Doc type:", meta.text_type)
    print("Chunks:", len(chunks))