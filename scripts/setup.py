#!/usr/bin/env python3
"""
Setup script for Coral TPU Crypto Trading System
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is suitable"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old. Requires Python 3.8+")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if Path("requirements.txt").exists():
        success = run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")
        if not success:
            print("⚠️ Some packages failed to install. You may need to install them manually.")
    else:
        print("❌ requirements.txt not found")
        return False
    
    return True

def setup_coral_tpu():
    """Setup Coral TPU runtime"""
    print("🚀 Setting up Coral TPU runtime...")
    
    import platform
    os_name = platform.system().lower()
    
    if os_name == "linux":
        print("🐧 Detected Linux - installing Edge TPU runtime...")
        commands = [
            "echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",
            "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
            "sudo apt update",
            "sudo apt install libedgetpu1-std"
        ]
        
        for cmd in commands:
            if not run_command(cmd, f"Running: {cmd}"):
                print("⚠️ Failed to install Edge TPU runtime. Please install manually.")
                break
    
    elif os_name == "darwin":
        print("🍎 Detected macOS - please install Edge TPU runtime manually from:")
        print("   https://coral.ai/software/#edgetpu-runtime")
    
    elif os_name == "windows":
        print("🪟 Detected Windows - please install Edge TPU runtime manually from:")
        print("   https://coral.ai/software/#edgetpu-runtime")
    
    else:
        print(f"❓ Unknown OS: {os_name}")
        print("   Please install Edge TPU runtime manually from:")
        print("   https://coral.ai/software/#edgetpu-runtime")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ["data", "logs", "models", "notebooks"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_environment():
    """Setup environment file"""
    print("⚙️ Setting up environment configuration...")
    
    if not Path(".env").exists():
        if Path(".env.template").exists():
            # Copy template to .env
            with open(".env.template", "r") as template:
                content = template.read()
            
            with open(".env", "w") as env_file:
                env_file.write(content)
            
            print("✅ Created .env file from template")
            print("📝 Please edit .env file and add your API keys")
        else:
            print("❌ .env.template not found")
    else:
        print("✅ .env file already exists")

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test basic imports
        import numpy
        import pandas
        print("✅ NumPy and Pandas imported successfully")
        
        # Test Coral TPU (optional)
        try:
            from pycoral.utils import edgetpu
            devices = edgetpu.list_edge_tpus()
            if devices:
                print(f"✅ Found {len(devices)} Coral TPU device(s)")
            else:
                print("⚠️ No Coral TPU devices detected")
        except ImportError:
            print("⚠️ PyCoral not available (install manually if needed)")
        
        # Test TensorFlow Lite
        try:
            import tflite_runtime.interpreter as tflite
            print("✅ TensorFlow Lite runtime available")
        except ImportError:
            try:
                import tensorflow as tf
                print("✅ TensorFlow available")
            except ImportError:
                print("⚠️ Neither TFLite runtime nor TensorFlow available")
        
        print("✅ Installation test completed")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Coral TPU Crypto Trading System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup Coral TPU
    setup_coral_tpu()
    
    # Setup environment
    setup_environment()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📚 Next steps:")
        print("1. Edit .env file with your API keys")
        print("2. Connect your Coral TPU device")
        print("3. Run: python demo.py --test")
        print("4. Run: python demo.py")
        print("5. Run: python main.py")
    else:
        print("\n⚠️ Setup completed with warnings. Some features may not work.")
        print("Please check the error messages above and install missing components.")

if __name__ == "__main__":
    main()
