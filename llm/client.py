import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


from openai import OpenAI

from config.manager import get_config


class HF_LLM:
    api_key = None

    client = None
    strong_llm = None
    fast_llm = None
    vision_llm = None

    strong_model_name = "poolside/laguna-xs-2.1:free"
    fast_model_name = "poolside/laguna-xs-2.1:free"
    vision_model_name = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    base_url = "https://openrouter.ai/api/v1"



def set_environ():
    os.environ["HF_HOME"] = r"D:\models\huggingface"
    os.environ["TORCH_HOME"] = r"D:\models\torch_models"


def load_api_key():
    load_dotenv()
    hf_api_key = os.getenv("HUGGIN_FACE_API")
    open_router_api_key = os.getenv("OPEN_ROUter_API_KEY")
    HF_LLM.api_key = open_router_api_key


def create_HF_client():
    set_environ()
    load_api_key()
    cfg = get_config().llm

    HF_LLM.strong_model_name = cfg.strong_model
    HF_LLM.fast_model_name = cfg.fast_model
    HF_LLM.vision_model_name = cfg.vision_model
    HF_LLM.base_url = cfg.server_url

    HF_LLM.client = OpenAI(
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url,
    )

    HF_LLM.strong_llm = ChatOpenAI(
        model=HF_LLM.strong_model_name,
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url,
        temperature=cfg.temperature,
    )
    HF_LLM.fast_llm = ChatOpenAI(
        model=HF_LLM.fast_model_name,
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url,
        temperature=cfg.temperature,
    ).with_retry(
        stop_after_attempt=3
    )
    HF_LLM.vision_llm = ChatOpenAI(
        model=HF_LLM.vision_model_name,
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url,
        temperature=cfg.temperature,
    )



def initialize_hf_llm():
    set_environ()
    load_api_key()
    create_HF_client()
