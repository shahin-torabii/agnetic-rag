import os
from openai import OpenAI
from dotenv import load_dotenv
import base64

class HF_LLM:
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # vision_model_name = "Qwen/Qwen3-VL-8B-Instruct"
    api_key = None
    #base_url="https://router.huggingface.co/v1"
    client = None

    model_name ="qwen/qwen3.5-9b"
    vision_model_name = "qwen/qwen3-vl-8b-instruct",
    base_url = "https://openrouter.ai/api/v1"


def set_environ():
    os.environ["HF_HOME"] = r"D:\models\huggingface"
    os.environ["TORCH_HOME"] = r"D:\models\torch_models"


def load_api_key():
    load_dotenv()
    hf_api_key = os.getenv("HUGGIN_FACE_API")
    open_router_api_key = os.getenv("OPEN_ROUter_API_KEY")
    HF_LLM.api_key = open_router_api_key


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

if __name__ == "__main__":

    set_environ()
    load_api_key()
    client = OpenAI(
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