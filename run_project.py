"""
Hotel Revenue Prediction System - Unified Project Launcher
Starts both the FastAPI Backend Service and the Streamlit Web Dashboard.
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "hotel_boooking"
FRONTEND_DIR = ROOT_DIR / "web_app_dashboards" / "web_app"

def main():
    print("=" * 60)
    print("Starting Hotel Revenue Prediction System")
    print("=" * 60)

    # 1. Start FastAPI Backend
    print("\n[1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BACKEND_DIR)
    )

    time.sleep(2)

    # 2. Start Streamlit Frontend
    print("[2/2] Launching Streamlit Dashboard on http://localhost:8501 ...")
    frontend_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
        cwd=str(FRONTEND_DIR)
    )

    print("\nBoth services are running!")
    print("  - Streamlit Dashboard: http://localhost:8501")
    print("  - FastAPI API Docs:   http://127.0.0.1:8000/docs")
    print("Press CTRL+C to stop both services.\n")

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend service stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend service stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nShutting down services...")
    finally:
        for proc in (backend_proc, frontend_proc):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("All services stopped.")

if __name__ == "__main__":
    main()
