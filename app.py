#!/usr/bin/env python3
"""
Development startup script for Emergency Lighting Detection API
Helps start all required services for development
"""

import subprocess
import sys
import time
import os
import signal
import redis
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def check_redis():
    """Check if Redis is running"""
    redis_url = os.getenv('REDIS_URL', "redis://localhost:6379/0")
    try:
        r = redis.from_url(redis_url)
        return r.ping()
    except Exception:
        return False

def start_redis():
    """Start Redis server"""
    redis_url = os.getenv('REDIS_URL', "redis://localhost:6379/0")
    try:
        r = redis.from_url(redis_url)
        r.ping()
        print("✅ Connected to Redis")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please ensure Redis is running and accessible at {redis_url}")
        return False



def start_celery():
    """Start Celery worker"""
    print("Starting Celery worker...")
    try:
        celery_process = subprocess.Popen([
            sys.executable, '-m', 'celery', '-A', 'backend.worker', 'worker', '--loglevel=info'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)
        if celery_process.poll() is None:
            print("[OK] Celery worker started successfully")
            return celery_process
        else:
            stderr_output = celery_process.stderr.read().decode().strip()
            print(f"[FAIL] Failed to start Celery worker. Error: {stderr_output}")
            return None
    except FileNotFoundError:
        print("[FAIL] Celery not found. Run: pip install celery")
        return None

def start_fastapi():
    """Start FastAPI server"""
    print("Starting FastAPI server...")
    try:
        fastapi_process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 'backend.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        if fastapi_process.poll() is None:
            print("[OK] FastAPI server started successfully")
            print("API available at: http://localhost:8000")
            print("API docs at: http://localhost:8000/docs")
            return fastapi_process
        else:
            stderr_output = fastapi_process.stderr.read().decode().strip()
            print(f"[FAIL] Failed to start FastAPI server. Error: {stderr_output}")
            return None
    except FileNotFoundError:
        print("[FAIL] Uvicorn not found. Run: pip install uvicorn")
        return None

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'celery': 'Distributed task queue',
        'redis': 'Redis client',
        'PyMuPDF': 'PDF processing (requires Visual Studio Build Tools on Windows)',
        'camelot': 'PDF table extraction (requires Poppler and Ghostscript)',
        'pandas': 'Data manipulation library'
    }
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package.replace('-', '_').lower())
        except ImportError:
            missing_packages.append(f"{package} ({description})")
    
    if missing_packages:
        print("[FAIL] Some Python packages are missing or failed to install:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease run 'pip install -r requirements.txt' to install them.")
        print("\nNOTE: On Windows, PyMuPDF and Camelot require additional system dependencies:")
        print("  - PyMuPDF: Visual Studio Build Tools (with 'Desktop development with C++' workload)")
        print("  - Camelot: Poppler (add to PATH) and Ghostscript.")
        print("Please install these manually if you encounter further installation errors.")
        return False
    else:
        print("[OK] All Python dependencies installed")
        return True

def create_directories():
    """Create necessary directories"""
    directories = [
        Path("output/uploads"),
        Path("output/images"),
        Path("output/results"),
        Path("output/visualizations"),
        Path("output/annotations")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("[OK] Created necessary directories")

def main():
    """Main startup function"""
    print("Emergency Lighting Detection API Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check/Start Redis
    if not check_redis():
        print("Redis is not running. Attempting to start...")
        if not start_redis():
            print("Failed to connect to or start Redis. Please ensure Redis is running and accessible.")
            sys.exit(1)
    else:
        print("[OK] Redis is already running")
    
    # Start services
    celery_process = start_celery()
    if not celery_process:
        sys.exit(1)
    
    fastapi_process = start_fastapi()
    if not fastapi_process:
        if celery_process:
            celery_process.terminate()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("All services started successfully!")
    print("=" * 50)
    print("\nPress Ctrl+C to stop all services")
    
    def signal_handler(sig, frame):
        print("\n\nShutting down services...")
        if celery_process:
            celery_process.terminate()
            print("[OK] Celery worker stopped")
        if fastapi_process:
            fastapi_process.terminate()
            print("[OK] FastAPI server stopped")
        print("Goodbye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()