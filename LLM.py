import os
from openai import OpenAI
from dotenv import load_dotenv
import base64

class HF_LLM:
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    vision_model_name = "Qwen/Qwen3-VL-8B-Instruct"
    api_key = None
    base_url="https://router.huggingface.co/v1"
    client = None

def set_environ():
    os.environ["HF_HOME"] = r"D:\models\huggingface"
    os.environ["TORCH_HOME"] = r"D:\models\torch_models"


def load_api_key():
    load_dotenv()
    hf_api_key = os.getenv("HUGGIN_FACE_API")
    HF_LLM.api_key = hf_api_key


def creat_HF_client():
    set_environ()
    load_api_key()
    hf_client = OpenAI(
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url
    )
    HF_LLM.client = hf_client


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def initialize_hf_llm():
    set_environ()
    load_api_key()
    creat_HF_client()