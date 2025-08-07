import os
import subprocess
import sys

def setup_competition_environment():
    """Setup environment for competition submission."""
    
    print("🏆 Setting up Emergency Lighting Detection for Competition")
    print("=" * 60)
    
    # Create necessary directories
    directories = [
        "output/uploads",
        "output/images", 
        "output/annotations",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install requirements
    print("\n📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    # Setup environment file
    env_content = """
# Competition Environment Variables
REDIS_URL=redis://localhost:6379/0
GOOGLE_API_KEY=AIzaSyDDyTIsgMHpFOouUrP3lmkwUk1GRUMa9mI
UPLOAD_DIR=output/uploads
OUTPUT_DIR=output
DB_PATH=emergency_lighting.db
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✅ Created .env file")
    
    # Create start script
    start_script = """#!/bin/bash
# Start script for competition demo

echo "🚀 Starting Emergency Lighting Detection System"

# Start Redis in background (if not running)
redis-server --daemonize yes --port 6379 || echo "Redis already running"

# Start Celery worker in background
celery -A backend.worker worker --loglevel=info --concurrency=1 &
CELERY_PID=$!

# Start FastAPI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo "🌐 API running at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "echo 'Stopping services...'; kill $CELERY_PID $API_PID; exit" INT
wait
"""
    
    with open("start_competition.sh", "w") as f:
        f.write(start_script)
    
    # Make executable on Unix systems
    try:
        os.chmod("start_competition.sh", 0o755)
    except:
        pass
    
    print("✅ Created start_competition.sh")
    
    print("\n" + "=" * 60)
    print("🎯 Competition Setup Complete!")
    print("\nNext Steps:")
    print("1. Add your OpenAI API key to .env file")
    print("2. Start Redis: redis-server --port 6379")
    print("3. Run: ./start_competition.sh (or python -m uvicorn backend.main:app)")
    print("4. Test with the Postman collection")
    print("5. Deploy to Render.com using render.yaml")
    print("\nFor Render deployment:")
    print("- Push code to GitHub")
    print("- Connect to Render.com") 
    print("- Set OPENAI_API_KEY environment variable")
    print("- Deploy!")

if __name__ == "__main__":
    setup_competition_environment()
