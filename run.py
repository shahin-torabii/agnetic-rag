import subprocess
import sys
import time


def main():
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "chatbot:app", "--host", "127.0.0.1", "--port", "8000", "--timeout-keep-alive" ,"600"]
    )
    time.sleep(2)

    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    )

    try:
        frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        backend.terminate()
        frontend.terminate()


if __name__ == "__main__":
    main()