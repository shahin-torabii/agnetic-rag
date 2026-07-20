import socket
import subprocess
import sys
import time


def _kill_port(port: int):
    """Kill any process already listening on the given port."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.split()[-1]
                    subprocess.run(f"taskkill /PID {pid} /F",
                                   shell=True, capture_output=True)
        else:
            subprocess.run(f"fuser -k {port}/tcp",
                           shell=True, capture_output=True)
    except Exception:
        pass


def _wait_for_port(port: int, timeout: int = 120, label: str = ""):
    """Block until something is accepting connections on the port."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                print(f"  {label or f'port {port}'} is ready.")
                return True
        except OSError:
            time.sleep(1)
    print(f"  WARNING: {label or f'port {port}'} did not become ready in {timeout}s.")
    return False


def main():
    # Kill stale processes from any previous run
    print("Clearing ports 8000 and 8001...")
    for port in (8000, 8001):
        _kill_port(port)
    time.sleep(1)

    # 1. Model server — embeddings + reranking (port 8001)
    print("Starting model server on port 8001...")
    model_server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.model_server:app",
         "--host", "127.0.0.1", "--port", "8001"]
    )
    _wait_for_port(8001, timeout=120, label="Model server")

    # 2. Backend API — FastAPI chatbot (port 8000)
    print("Starting backend API on port 8000...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "127.0.0.1", "--port", "8000",
         "--timeout-keep-alive", "600"]
    )
    _wait_for_port(8000, timeout=120, label="Backend API")

    # 3. Frontend — Streamlit UI (opens browser automatically)
    print("Starting Streamlit UI...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"]
    )

    try:
        frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        frontend.terminate()
        backend.terminate()
        model_server.terminate()


if __name__ == "__main__":
    main()
