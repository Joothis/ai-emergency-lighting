import subprocess
import sys
import os

def install_system_dependencies():
    """Install system dependencies for different platforms."""
    import platform
    
    system = platform.system().lower()
    
    if system == "linux":
        print("Installing system dependencies for Linux...")
        try:
            # Update package list
            subprocess.run(["sudo", "apt", "update"], check=True)
            
            # Install required packages
            packages = [
                "python3-dev",
                "python3-pip", 
                "libgl1-mesa-glx",
                "libglib2.0-0",
                "libsm6",
                "libxext6",
                "libxrender-dev",
                "libgomp1",
                "libgtk-3-0",
                "poppler-utils",
                "ghostscript"
            ]
            
            subprocess.run(["sudo", "apt", "install", "-y"] + packages, check=True)
            print("System dependencies installed")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to install system dependencies: {e}")
            return False
            
    elif system == "darwin":  # macOS
        print("Installing system dependencies for macOS...")
        try:
            # Install Homebrew packages
            packages = ["poppler", "ghostscript"]
            subprocess.run(["brew", "install"] + packages, check=True)
            print("System dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install system dependencies: {e}")
            print("Please install Homebrew and the required packages manually")
            return False
    
    return True

def install_python_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    try:
        # Upgrade pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        
        print("Python dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Python dependencies: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("Setting up directories...")
    
    directories = [
        "output/uploads",
        "output/images", 
        "output/visualizations",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return True

def main():
    """Main installation script."""
    print("Emergency Lighting Detection System Setup")
    print("=" * 50)
    
    # Install system dependencies
    if not install_system_dependencies():
        print("System dependencies installation failed. Some features might not work.")
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("Python dependencies installation failed. Cannot continue.")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start Redis: docker-compose up -d redis")
    print("2. Start services: python scripts/start_services.py")
    print("3. Or start manually:")
    print("   - Celery worker: celery -A backend.enhanced_worker worker --loglevel=info")
    print("   - FastAPI server: uvicorn backend.enhanced_main:app --reload")

if __name__ == "__main__":
    main()