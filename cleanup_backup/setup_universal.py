#!/usr/bin/env python3
"""
Setup script for Universal Coral TPU system
Installs additional dependencies and downloads sample models
"""

import sys
import subprocess
import os
from pathlib import Path
import urllib.request
import json
from loguru import logger

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🔧 {text}")
    print("="*60)

def install_package(package_name, pip_name=None):
    """Install a Python package"""
    if pip_name is None:
        pip_name = package_name
    
    try:
        __import__(package_name)
        logger.success(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        logger.info(f"📦 Installing {package_name}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", pip_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.success(f"✅ {package_name} installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install {package_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error installing {package_name}: {e}")
            return False

def install_opencv():
    """Install OpenCV for camera functionality"""
    logger.info("📹 Installing OpenCV for camera support...")
    
    # Try different OpenCV packages
    opencv_packages = [
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless"
    ]
    
    for package in opencv_packages:
        if install_package("cv2", package):
            return True
    
    logger.error("❌ Failed to install any OpenCV package")
    return False

def install_additional_dependencies():
    """Install additional dependencies for vision functionality"""
    print_header("Installing Additional Dependencies")
    
    dependencies = [
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("requests", "requests"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn")
    ]
    
    success_count = 0
    for import_name, pip_name in dependencies:
        if install_package(import_name, pip_name):
            success_count += 1
    
    logger.info(f"📊 Installed {success_count}/{len(dependencies)} packages")
    return success_count == len(dependencies)

def download_file(url, destination):
    """Download a file from URL"""
    try:
        logger.info(f"⬇️ Downloading {destination.name}...")
        urllib.request.urlretrieve(url, destination)
        logger.success(f"✅ Downloaded {destination.name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {destination.name}: {e}")
        return False

def download_sample_models():
    """Download sample models for testing"""
    print_header("Downloading Sample Models")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load config
    config_file = Path("inference_config.json")
    if not config_file.exists():
        logger.error("❌ Configuration file not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False
    
    # Download models with URLs
    downloaded = 0
    total = 0
    
    for model_name, model_config in config["models"].items():
        if "download_url" in model_config:
            total += 1
            model_path = Path(model_config["path"])
            
            if model_path.exists():
                logger.info(f"✅ {model_name} already exists")
                downloaded += 1
                continue
            
            # Download model
            if download_file(model_config["download_url"], model_path):
                downloaded += 1
            
            # Download labels if available
            if "labels_url" in model_config:
                labels_path = Path(model_config["labels"])
                if not labels_path.exists():
                    download_file(model_config["labels_url"], labels_path)
    
    logger.info(f"📊 Downloaded {downloaded}/{total} models")
    return downloaded > 0

def create_label_files():
    """Create label files for models"""
    print_header("Creating Label Files")
    
    # COCO labels (80 classes)
    coco_labels = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    
    # Create COCO labels file
    coco_labels_path = Path("models/coco_labels.txt")
    if not coco_labels_path.exists():
        with open(coco_labels_path, 'w') as f:
            for label in coco_labels:
                f.write(f"{label}\n")
        logger.success(f"✅ Created {coco_labels_path}")
    
    # Crypto labels
    crypto_labels = ["down", "sideways", "up"]
    crypto_labels_path = Path("models/crypto_labels.txt")
    if not crypto_labels_path.exists():
        with open(crypto_labels_path, 'w') as f:
            for label in crypto_labels:
                f.write(f"{label}\n")
        logger.success(f"✅ Created {crypto_labels_path}")
    
    # Face labels
    face_labels = ["face"]
    face_labels_path = Path("models/face_labels.txt")
    if not face_labels_path.exists():
        with open(face_labels_path, 'w') as f:
            for label in face_labels:
                f.write(f"{label}\n")
        logger.success(f"✅ Created {face_labels_path}")

def test_camera():
    """Test camera functionality"""
    print_header("Testing Camera")
    
    try:
        import cv2
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                logger.success(f"✅ Camera test successful: {frame.shape}")
                cap.release()
                return True
            else:
                logger.warning("⚠️ Camera opened but failed to read frame")
        else:
            logger.warning("⚠️ Failed to open camera")
        
        cap.release()
        return False
        
    except ImportError:
        logger.error("❌ OpenCV not available for camera test")
        return False
    except Exception as e:
        logger.error(f"❌ Camera test failed: {e}")
        return False

def test_coral_tpu():
    """Test Coral TPU functionality"""
    print_header("Testing Coral TPU")
    
    try:
        from pycoral.utils import edgetpu
        
        devices = edgetpu.list_edge_tpus()
        if devices:
            logger.success(f"✅ Found {len(devices)} Coral TPU device(s)")
            for i, device in enumerate(devices):
                logger.info(f"  Device {i}: {device}")
            return True
        else:
            logger.warning("⚠️ No Coral TPU devices found")
            return False
            
    except ImportError:
        logger.warning("⚠️ PyCoral not available")
        return False
    except Exception as e:
        logger.error(f"❌ Coral TPU test failed: {e}")
        return False

def create_example_scripts():
    """Create example usage scripts"""
    print_header("Creating Example Scripts")
    
    # Quick camera test script
    camera_test_script = '''#!/usr/bin/env python3
"""Quick camera test with AI"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from universal_app import CoralTPUApp

def main():
    app = CoralTPUApp()
    if app.initialize():
        print("Taking photo with AI analysis...")
        app.photo_mode()
    else:
        print("Failed to initialize app")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_camera_test.py", 'w') as f:
        f.write(camera_test_script)
    logger.success("✅ Created quick_camera_test.py")
    
    # Crypto test script
    crypto_test_script = '''#!/usr/bin/env python3
"""Quick crypto prediction test"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from universal_app import CoralTPUApp

def main():
    app = CoralTPUApp()
    if app.initialize():
        print("Running crypto predictions...")
        app.crypto_mode()
    else:
        print("Failed to initialize app")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_crypto_test.py", 'w') as f:
        f.write(crypto_test_script)
    logger.success("✅ Created quick_crypto_test.py")

def main():
    """Main setup function"""
    print_header("Universal Coral TPU Setup")
    
    logger.info("🚀 Setting up Universal Coral TPU system...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher required")
        sys.exit(1)
    
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    success_steps = 0
    total_steps = 6
    
    # Step 1: Install additional dependencies
    if install_additional_dependencies():
        success_steps += 1
    
    # Step 2: Create label files
    try:
        create_label_files()
        success_steps += 1
    except Exception as e:
        logger.error(f"❌ Failed to create label files: {e}")
    
    # Step 3: Download sample models
    if download_sample_models():
        success_steps += 1
    
    # Step 4: Test camera
    if test_camera():
        success_steps += 1
    
    # Step 5: Test Coral TPU
    if test_coral_tpu():
        success_steps += 1
    
    # Step 6: Create example scripts
    try:
        create_example_scripts()
        success_steps += 1
    except Exception as e:
        logger.error(f"❌ Failed to create example scripts: {e}")
    
    # Summary
    print_header("Setup Summary")
    logger.info(f"📊 Completed {success_steps}/{total_steps} setup steps")
    
    if success_steps == total_steps:
        logger.success("🎉 Setup completed successfully!")
        logger.info("\n🚀 You can now run:")
        logger.info("  python universal_app.py --mode menu")
        logger.info("  python universal_app.py --mode demo")
        logger.info("  python quick_camera_test.py")
        logger.info("  python quick_crypto_test.py")
    else:
        logger.warning("⚠️ Setup completed with some issues")
        logger.info("💡 Check the logs above for any missing dependencies")
    
    # Test basic functionality
    print_header("Testing Universal Inference")
    try:
        from src.models.inference.universal_inference import UniversalTPUInference
        engine = UniversalTPUInference()
        models = engine.get_available_models()
        logger.info(f"📊 Available models: {models}")
        
        if models:
            logger.success("✅ Universal inference engine working!")
        else:
            logger.warning("⚠️ No models loaded - you may need to download them manually")
    
    except Exception as e:
        logger.error(f"❌ Universal inference test failed: {e}")

if __name__ == "__main__":
    main()
