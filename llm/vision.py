import base64
import time
from langchain_core.messages import SystemMessage, HumanMessage
from llm.client import HF_LLM
from config.manager import get_config


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_description_cache = {}

def send_images_to_vlm(image_paths, query, context=""):

    full_text = query if not context else f"{query}\n\nContext:\n{context}"
    content = [{"type": "text", "text": full_text}]
    for image_path in image_paths:
        img_b64 = encode_image_to_base64(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    messages = [
        SystemMessage(content="You are an assistant that answers questions about images."),
        HumanMessage(content=content),
    ]

    _tuning = get_config().tuning
    last_error = None
    for attempt in range(_tuning.max_retries):
        try:
            response = HF_LLM.vision_llm.invoke(messages)
            return response.content
        except Exception as e:
            last_error = e
            print(f"VLM call failed (attempt {attempt + 1}/{_tuning.max_retries}): {e}")
            time.sleep(_tuning.retry_wait_seconds)
    raise last_error


#
# def send_images_to_vlm(image_paths, query, context=""):
#
#     full_text = query if not context else f"{query}\n\nContext:\n{context}"
#
#     content = [{"type": "text", "text": full_text}]
#
#     for image_path in image_paths:
#         img_b64 = encode_image_to_base64(image_path)
#         content.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{img_b64}"
#             },
#         })
#
#     messages = [
#         {
#             "role": "system",
#             "content": "You are an assistant that answers questions about images."
#         },
#         {
#             "role": "user",
#             "content": content
#         }
#     ]
#
#     last_error = None
#
#     for attempt in range(MAX_RETRIES):
#         try:
#             response = HF_LLM.vision_llm.invoke(messages)
#             return response
#         except Exception as e:
#             last_error = e
#             print(f"VLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
#             time.sleep(RETRY_WAIT_SECONDS)
#
#     raise last_error


def describe_images(image_paths, context=""):
    descriptions = []
    for path in image_paths:
        if path not in image_description_cache:
            image_description_cache[path] = send_images_to_vlm(
                [path], "Explain this image using the provided context.", context
            )
        descriptions.append(image_description_cache[path])
    return descriptions
