FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .[dev] 2>/dev/null || pip install --no-cache-dir \
    fastapi uvicorn langgraph langchain-core langchain-openai openai \
    sqlalchemy faiss-cpu numpy streamlit pydantic python-multipart requests \
    python-dotenv pyyaml faster-whisper fitz pypdfium2 pytesseract \
    python-docx openpyxl python-pptx pillow opencv-python puremagic \
    passlib bcrypt python-jose

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]
