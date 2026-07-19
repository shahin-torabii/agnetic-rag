import os

from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import base64
from huggingface_hub import InferenceClient

class HF_LLM:
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # vision_model_name = "Qwen/Qwen3-VL-8B-Instruct"
    api_key = None

    client = None
    strong_llm = None
    fast_llm = None
    vision_llm  = None

    # vision_model_name = "Qwen/Qwen3-VL-8B-Instruct"
    #vision_model_name =  "meta-llama/Llama-3.2-11B-Vision-Instruct"
    #vision_model_name = "CohereLabs/aya-vision-32b"
    # vision_model_name = "HuggingFaceM4/idefics2-8b"
    # fast_model_name= "Qwen/Qwen2.5-7B-Instruct"
    #strong_model_name = "google/gemma-4-31b-it:free"
    # base_url="https://router.huggingface.co/v1"

    #strong_model_name = "meta-llama/llama-3.3-70b-instruct:free"
    #fast_model_name = "meta-llama/llama-3.3-70b-instruct:free"
    strong_model_name = "poolside/laguna-xs-2.1:free"
    #fast_model_name = "liquid/lfm-2.5-1.2b-instruct:free"
    #strong_model_name = "liquid/lfm-2.5-1.2b-instruct:free"
    fast_model_name = "poolside/laguna-xs-2.1:free"
    #vision_model_name = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
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

    HF_LLM.strong_llm = ChatOpenAI(
        model=HF_LLM.strong_model_name,
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url,
        temperature=0,
    )
    HF_LLM.fast_llm = ChatOpenAI(
        model=HF_LLM.fast_model_name,
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url,
        temperature=0,
    ).with_retry(
        stop_after_attempt=3
    )
    HF_LLM.vision_llm = ChatOpenAI(
        model=HF_LLM.vision_model_name,
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url,
        temperature=0.2,
    )

    # HF_LLM.fast_llm = ChatHuggingFace(
    #     llm=HuggingFaceEndpoint(
    #         repo_id=HF_LLM.fast_model_name,
    #         huggingfacehub_api_token=HF_LLM.api_key,
    #         temperature=0,
    #         # Force the backend away from third-party partners like Together AI
    #         extra_body={"provider": "hf-inference"}
    #     )
    # )
    #
    # HF_LLM.strong_llm = ChatHuggingFace(
    #     llm=HuggingFaceEndpoint(
    #         repo_id=HF_LLM.strong_model_name,
    #         huggingfacehub_api_token=HF_LLM.api_key,
    #         temperature=0,
    #         extra_body={"provider": "hf-inference"}
    #     )
    # )
    #
    # client = InferenceClient(provider="hf-inference", api_key=HF_LLM.api_key)
    #
    # def vision_invoke(messages):
    #     # 2. Swap to a vision model fully supported on HF's free serverless architecture
    #     response = client.chat.completions.create(
    #         model=HF_LLM.vision_model_name,
    #         messages=messages,
    #     )
    #     return response.choices[0].message.content
    #
    # HF_LLM.vision_llm = RunnableLambda(vision_invoke)
    #

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def initialize_hf_llm():
    set_environ()
    load_api_key()
    create_HF_client()

if __name__ == "__main__":

    set_environ()
    load_api_key()
    client = ChatOpenAI(
        base_url=HF_LLM.base_url,
        api_key=HF_LLM.api_key
    )

    # 2. Make the API Call using a free model variant
    try:
        completion = client.chat.completions.create(
            model=HF_LLM.model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Tell me about influence of michael jackson on the new generation gen z"
                }
            ],

        )

        # 3. Print out the response text
        print("--- OpenRouter Response ---")
        print(completion.choices[0].message.content)

    except Exception as e:
        print(f"An error occurred: {e}")