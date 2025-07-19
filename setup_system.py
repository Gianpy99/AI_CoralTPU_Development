#!/usr/bin/env python3
"""
Automatic setup script for Coral TPU Trading System
Handles installation and configuration
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🔧 {title}")
    print("=" * 60)


def print_status(message, status="info"):
    """Print status message with icon"""
    icons = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️",
        "error": "❌",
        "working": "🔄"
    }
    print(f"{icons.get(status, 'ℹ️')} {message}")


def check_admin():
    """Check if running as administrator on Windows"""
    if platform.system() == "Windows":
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    return True


def install_python_packages():
    """Install required Python packages"""
    print_header("Installing Python Packages")
    
    packages = [
        "loguru",
        "ccxt", 
        "python-dotenv",
        "numpy",
        "pandas",
        "fastapi",
        "uvicorn",
        "requests",
        "aiofiles",
        "websockets",
        "plotly",
        "psutil"
    ]
    
    for package in packages:
        print_status(f"Installing {package}...", "working")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print_status(f"✓ {package} installed", "success")
            else:
                print_status(f"✗ Failed to install {package}: {result.stderr}", "error")
        except Exception as e:
            print_status(f"✗ Error installing {package}: {e}", "error")


def install_coral_packages():
    """Install Coral TPU specific packages"""
    print_header("Installing Coral TPU Packages")
    
    # Try to install TensorFlow Lite runtime
    print_status("Installing TensorFlow Lite runtime...", "working")
    tflite_urls = {
        "cp38": "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-win_amd64.whl",
        "cp39": "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp39-cp39-win_amd64.whl", 
        "cp310": "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp310-cp310-win_amd64.whl"
    }
    
    # Try different Python versions
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    
    if python_version in tflite_urls:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", tflite_urls[python_version]
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print_status("✓ TensorFlow Lite runtime installed", "success")
            else:
                print_status("Using alternative installation method...", "warning")
        except:
            print_status("TensorFlow Lite installation failed", "warning")
    
    # Try to install PyCoral
    print_status("Installing PyCoral...", "working")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--extra-index-url", "https://google-coral.github.io/py-repo/",
            "pycoral~=2.0"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print_status("✓ PyCoral installed", "success")
        else:
            print_status(f"PyCoral installation had issues: {result.stderr}", "warning")
            print_status("This is normal if Coral TPU hardware is not connected", "info")
    except Exception as e:
        print_status(f"PyCoral installation failed: {e}", "warning")


def download_coral_runtime():
    """Download and guide installation of Coral TPU runtime"""
    print_header("Coral TPU Runtime Setup")
    
    if platform.system() != "Windows":
        print_status("This installer is for Windows only", "error")
        return False
    
    print_status("Coral TPU Runtime installation requires manual steps:", "info")
    print_status("1. Go to: https://coral.ai/software/#edgetpu-runtime", "info")
    print_status("2. Download 'Edge TPU runtime library (Windows)'", "info")
    print_status("3. Extract the ZIP file", "info")
    print_status("4. Run install.bat as Administrator", "info")
    print_status("5. Restart your computer", "info")
    
    # Check if user wants to open the URL
    try:
        import webbrowser
        response = input("\n🌐 Open download page in browser? (y/n): ")
        if response.lower() in ['y', 'yes']:
            webbrowser.open("https://coral.ai/software/#edgetpu-runtime")
            print_status("Opened browser to download page", "success")
    except:
        pass
    
    return True


def create_env_file():
    """Create .env configuration file"""
    print_header("Creating Configuration File")
    
    env_file = Path(".env")
    template_file = Path(".env.template")
    
    if env_file.exists():
        print_status(".env file already exists", "info")
        return True
    
    if not template_file.exists():
        print_status("Creating .env.template...", "working")
        template_content = """# Coral TPU Trading System Configuration

# Exchange API Keys (IMPORTANT: Use test keys first!)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET_KEY=your_coinbase_secret_key_here

# Trading Configuration  
TRADING_MODE=simulation
TRADING_PAIR=BTCUSDT
DEFAULT_SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT
MAX_POSITION_SIZE=0.02
STOP_LOSS_PERCENTAGE=0.02
TAKE_PROFIT_PERCENTAGE=0.05

# AI Model Configuration
MODEL_PATH=models/price_predictor.tflite
CONFIDENCE_THRESHOLD=0.7
PREDICTION_HORIZON=5

# System Configuration
LOOP_INTERVAL=30
ENABLE_DASHBOARD=true
DASHBOARD_PORT=8000
LOG_LEVEL=INFO

# Risk Management
MAX_DAILY_LOSS=0.05
MAX_DRAWDOWN=0.10
POSITION_SIZING_METHOD=kelly

# Notifications (Optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ENABLE_NOTIFICATIONS=false

# Database
DATABASE_URL=sqlite:///data/trading_data.db
"""
        with open(template_file, 'w') as f:
            f.write(template_content)
        print_status("✓ Created .env.template", "success")
    
    # Copy template to .env
    shutil.copy(template_file, env_file)
    print_status("✓ Created .env file from template", "success")
    print_status("⚠️  Edit .env file with your actual API keys!", "warning")
    
    return True


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directory Structure")
    
    directories = [
        "data",
        "logs", 
        "models",
        "notebooks",
        "scripts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_status(f"✓ Created directory: {directory}", "success")


def run_tests():
    """Run system tests"""
    print_header("Running System Tests")
    
    # Test basic system
    print_status("Testing basic system functionality...", "working")
    try:
        result = subprocess.run([
            sys.executable, "simple_test.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print_status("✓ Basic system test passed", "success")
        else:
            print_status("Basic system test had issues", "warning")
            print(result.stdout)
    except Exception as e:
        print_status(f"Basic system test failed: {e}", "error")
    
    # Test Coral TPU
    print_status("Testing Coral TPU detection...", "working")
    try:
        result = subprocess.run([
            sys.executable, "test_coral_tpu.py"
        ], capture_output=True, text=True, timeout=30)
        
        if "Coral TPU is ready for use" in result.stdout:
            print_status("✓ Coral TPU is fully functional!", "success")
        elif "TensorFlow Lite is available" in result.stdout:
            print_status("⚠️  TensorFlow Lite ready, but no Coral TPU detected", "warning")
        else:
            print_status("Coral TPU setup needs completion", "info")
            
    except Exception as e:
        print_status(f"Coral TPU test failed: {e}", "error")


def main():
    """Main setup function"""
    print("🚀 Coral TPU Trading System - Automatic Setup")
    print("This script will set up your trading environment\n")
    
    # Check system compatibility
    print_status(f"Detected: {platform.system()} {platform.release()}", "info")
    print_status(f"Python: {sys.version}", "info")
    
    if platform.system() != "Windows":
        print_status("This setup script is optimized for Windows", "warning")
        print_status("Manual installation may be required", "info")
    
    # Start installation process
    try:
        # 1. Create directories
        create_directories()
        
        # 2. Install Python packages
        install_python_packages()
        
        # 3. Try to install Coral packages
        install_coral_packages()
        
        # 4. Create configuration
        create_env_file()
        
        # 5. Coral runtime guidance
        download_coral_runtime()
        
        # 6. Run tests
        run_tests()
        
        # Final summary
        print_header("Setup Complete!")
        print_status("🎉 Basic system setup completed!", "success")
        print_status("📋 Next steps:", "info")
        print_status("   1. Install Coral TPU runtime (if not done)", "info")
        print_status("   2. Connect Coral TPU hardware", "info") 
        print_status("   3. Edit .env file with your API keys", "info")
        print_status("   4. Run: python test_coral_tpu.py", "info")
        print_status("   5. Run: python main.py", "info")
        
        print_status("\n💡 See CORAL_TPU_SETUP.md for detailed instructions", "info")
        
    except KeyboardInterrupt:
        print_status("\n❌ Setup interrupted by user", "error")
    except Exception as e:
        print_status(f"\n❌ Setup failed: {e}", "error")
        print_status("Check the error and try again", "info")


if __name__ == "__main__":
    main()
