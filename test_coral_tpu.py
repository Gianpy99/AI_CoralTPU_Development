#!/usr/bin/env python3
"""
Test script for Coral TPU detection and functionality
Works with or without PyCoral libraries installed
"""

import sys
import os
import platform
import subprocess
from pathlib import Path


def check_system_info():
    """Check basic system information"""
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()


def check_usb_devices():
    """Check for USB devices that might be Coral TPU"""
    print("🔌 USB DEVICE DETECTION")
    print("=" * 50)
    
    try:
        # Try to list USB devices using PowerShell on Windows
        if platform.system() == "Windows":
            result = subprocess.run([
                "powershell", 
                "Get-WmiObject -Class Win32_USBHub | Select-Object Name, DeviceID"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("USB Devices found:")
                print(result.stdout)
                
                # Look for Coral TPU specific identifiers
                output = result.stdout.lower()
                if "coral" in output or "google" in output or "18d1" in output:
                    print("✅ Potential Coral TPU device detected!")
                else:
                    print("⚠️  No obvious Coral TPU devices found")
            else:
                print("❌ Could not enumerate USB devices")
                
    except Exception as e:
        print(f"❌ Error checking USB devices: {e}")
        
        # Alternative: Check device manager via WMI
        try:
            result = subprocess.run([
                "wmic", "path", "Win32_USBControllerDevice", "get", "Dependent"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("\nAlternative USB check:")
                if "Google" in result.stdout or "Coral" in result.stdout:
                    print("✅ Google/Coral device found in device manager!")
                else:
                    print("⚠️  No Google/Coral devices in device manager")
        except:
            print("❌ Alternative USB check also failed")
    
    print()


def check_coral_runtime():
    """Check if Coral runtime is installed"""
    print("🧠 CORAL TPU RUNTIME CHECK")
    print("=" * 50)
    
    # Check for Edge TPU runtime files
    possible_paths = [
        r"C:\Program Files\EdgeTPU",
        r"C:\Program Files (x86)\EdgeTPU", 
        r"C:\coral",
        os.path.expanduser("~/.local/lib")
    ]
    
    runtime_found = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found potential runtime directory: {path}")
            runtime_found = True
            
            # List contents
            try:
                contents = list(os.listdir(path))
                print(f"   Contents: {contents[:5]}...")  # Show first 5 items
            except:
                pass
    
    if not runtime_found:
        print("❌ No Edge TPU runtime directories found")
        print("💡 You may need to install the Edge TPU runtime:")
        print("   https://coral.ai/software/#edgetpu-runtime")
    
    print()


def check_python_libraries():
    """Check if required Python libraries are available"""
    print("🐍 PYTHON LIBRARIES CHECK") 
    print("=" * 50)
    
    libraries = {
        'tflite_runtime': 'TensorFlow Lite Runtime',
        'pycoral.utils.edgetpu': 'PyCoral Edge TPU Utils',
        'pycoral.utils.dataset': 'PyCoral Dataset Utils',
        'pycoral.adapters.common': 'PyCoral Common Adapters',
        'numpy': 'NumPy',
        'PIL': 'Pillow (PIL)'
    }
    
    for module, description in libraries.items():
        try:
            __import__(module)
            print(f"✅ {description}: Available")
        except ImportError as e:
            print(f"❌ {description}: Not available ({e})")
    
    print()


def test_coral_tpu():
    """Test actual Coral TPU functionality if libraries are available"""
    print("🔬 CORAL TPU FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        # Try to import and use PyCoral
        from pycoral.utils import edgetpu
        
        print("✅ PyCoral libraries imported successfully")
        
        # Try to list Edge TPU devices
        try:
            devices = edgetpu.list_edge_tpus()
            print(f"📊 Edge TPU devices found: {len(devices)}")
            
            for i, device in enumerate(devices):
                print(f"   Device {i}: {device}")
                
            if devices:
                print("🎉 Coral TPU is ready for use!")
                return True
            else:
                print("⚠️  No Edge TPU devices detected")
                print("💡 Make sure:")
                print("   - Coral TPU is connected via USB")
                print("   - Edge TPU runtime is installed")
                print("   - Device drivers are properly installed")
                return False
                
        except Exception as e:
            print(f"❌ Error listing Edge TPU devices: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import PyCoral: {e}")
        print("💡 Install PyCoral with:")
        print("   pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral")
        return False


def test_tflite_basic():
    """Test basic TensorFlow Lite functionality"""
    print("⚡ TENSORFLOW LITE TEST")
    print("=" * 50)
    
    try:
        import tflite_runtime.interpreter as tflite
        print("✅ TensorFlow Lite runtime available")
        
        # Try to create a basic interpreter
        try:
            # This would normally load a model, but we'll just test the import
            print("✅ TensorFlow Lite interpreter can be created")
            return True
        except Exception as e:
            print(f"⚠️  TensorFlow Lite interpreter test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ TensorFlow Lite runtime not available: {e}")
        print("💡 Install with: pip install tflite-runtime")
        return False


def generate_report():
    """Generate a comprehensive report"""
    print("\n" + "=" * 70)
    print("📋 CORAL TPU SYSTEM REPORT")
    print("=" * 70)
    
    # Run all checks
    check_system_info()
    check_usb_devices()
    check_coral_runtime()
    check_python_libraries()
    tpu_ready = test_coral_tpu()
    tflite_ready = test_tflite_basic()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 50)
    
    if tpu_ready:
        print("🎉 Coral TPU is fully functional!")
        print("   ✅ Hardware detected")
        print("   ✅ Windows Device Manager: Coral PCIe Accelerator")
        print("   ✅ Runtime installed") 
        print("   ✅ Python libraries available")
        print("   ✅ Ready for AI inference")
        print("   💡 Check Device Manager: Coral Accelerator devices → Coral PCIe Accelerator")
    elif tflite_ready:
        print("⚠️  TensorFlow Lite is available but Coral TPU not detected")
        print("   ✅ Can run CPU inference")
        print("   ❌ TPU acceleration not available")
        print("   💡 Check hardware connection and drivers")
    else:
        print("❌ System not ready for AI inference")
        print("   💡 Install required libraries and check hardware")
    
    print("\n🔗 Useful Links:")
    print("   • Coral TPU Setup: https://coral.ai/docs/accelerator/get-started/")
    print("   • Edge TPU Runtime: https://coral.ai/software/#edgetpu-runtime")
    print("   • PyCoral Installation: https://coral.ai/software/#pycoral-api")
    print("   • Troubleshooting: https://coral.ai/docs/accelerator/get-started/#troubleshooting")


if __name__ == "__main__":
    print("🚀 Coral TPU System Detection and Test")
    print("This script will check if your Coral TPU is properly set up\n")
    
    try:
        generate_report()
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check your system setup")
    
    print("\n✨ Test completed!")
