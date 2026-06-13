from query_router import route_query, Intent
from data_gathering import UserRequest
from LLM import HF_LLM, encode_image_to_base64
from index_embedd import index_images, VectorStore, get_image_id, index_chunks, index_chunk_images
from typing import List
from retreival import retrieval_image, retrieval
from pathlib import Path
from chunking_handlers.audio_hanlder import process_audio
from data_gathering import ingest, Chunk


def send_images_to_vlm(image_paths: list[str],query: str):
    content = [
        {
            "type": "text",
            "text": query
        }
    ]

    for image_path in image_paths:
        img_b64 = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            }
        )
    response = HF_LLM.client.chat.completions.create(
        model=HF_LLM.vision_model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content":
                "You are an assistant that answers questions about images."
            },
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.choices[0].message.content


def ingest_image(image_paths:List[str]):
    not_indexed_images = []
    for im_path in image_paths:
        p = Path(im_path)

        image_id = get_image_id(p)
        doc_id = "upload"
        key = (doc_id, image_id)
        if not key in VectorStore.indexed_images:
            not_indexed_images.append(im_path)

    index_images(not_indexed_images)

def index_to_faiss(chunks:List[Chunk]):
    index_chunks(chunks)
    index_chunk_images(chunks)


def handle_image(intent, request):
    image_paths = [image_path for image_path in request.images]
    ingest_image(image_paths)

    match intent:
        case Intent.IMAGE_SEARCH:
            if len(request.images) <4:
                response = send_images_to_vlm(image_paths, request)
                print(response)
            else:
                result = retrieval_image(request.query)
                ##TODO send results along query to llm for final result
        case Intent.IMAGE_UNDERSTANDING:
            response = send_images_to_vlm(image_paths, request)
            print(response)


def summarize_chunks(chunks, meta_audio):
    pass



def handle_audio(intent, request):

    audios_chunks = []
    transcripts = []
    meta_audio = []

    for audio_path in request.audio:

        chunks, transcript, meta = process_audio(audio_path)

        audios_chunks.extend(chunks)
        transcripts.append(transcript)
        meta_audio.append(meta)

        ingest(
            chunks=chunks,
            doc_meta=meta,
            img_to_ch=None,
            tbl_to_ch=None
        )

        index_to_faiss(chunks)

    match intent:

        case Intent.AUDIO_TRANSCRIBE:
            return "\n".join(transcripts)

        case Intent.AUDIO_QA:
            result = retrieval(query=request.query,k=10,is_doc=True)

        case Intent.AUDIO_SUMMARIZE:
            summary  = summarize_chunks(chunks=chunks,meta=meta_audio)
        case _:
            return "\n".join(transcripts)


def handle_general(intent, request):
    pass


def handle_document(intent, request):
    pass


def handle_query(request:UserRequest):
    intent = route_query(request)

    match intent:
        case Intent.IMAGE_SEARCH | Intent.IMAGE_UNDERSTANDING:
            handle_image(intent, request)
        case Intent.AUDIO_QA | Intent.AUDIO_SUMMARIZE | Intent.AUDIO_TRANSCRIBE :
            handle_audio(intent, request)
        case Intent.GENERAL_CHAT | Intent.UNKNOWN:
            handle_general(intent, request)
        case _:
            handle_document(intent, request)
            