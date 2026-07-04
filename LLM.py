import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import base64

class HF_LLM:
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # vision_model_name = "Qwen/Qwen3-VL-8B-Instruct"
    api_key = None
    #base_url="https://router.huggingface.co/v1"
    client = None
    strong_llm = None
    fast_llm = None
    vision_llm  = None

    strong_model_name = "qwen/qwen3-8b"
    fast_model_name = "qwen/qwen-2.5-7b-instruct"
    vision_model_name = "qwen/qwen3-vl-8b-instruct"
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
    )
    HF_LLM.vision_llm = ChatOpenAI(
        model=HF_LLM.vision_model_name,
        api_key=HF_LLM.api_key,
        base_url=HF_LLM.base_url,
        temperature=0.2,
    )


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