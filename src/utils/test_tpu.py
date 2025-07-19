"""
Test script to verify Coral TPU functionality
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required libraries can be imported"""
    logger.info("Testing imports...")
    
    try:
        import numpy as np
        logger.success("✓ NumPy imported successfully")
    except ImportError as e:
        logger.error(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        logger.success("✓ Pandas imported successfully")
    except ImportError as e:
        logger.error(f"✗ Pandas import failed: {e}")
        return False
    
    # Test Coral TPU libraries
    try:
        from pycoral.utils import edgetpu
        from pycoral.adapters import common
        import tflite_runtime.interpreter as tflite
        logger.success("✓ Coral TPU libraries imported successfully")
        coral_available = True
    except ImportError as e:
        logger.warning(f"⚠ Coral TPU libraries not available: {e}")
        coral_available = False
    
    # Test TensorFlow as fallback
    try:
        import tensorflow as tf
        logger.success("✓ TensorFlow imported successfully")
        tf_available = True
    except ImportError as e:
        logger.warning(f"⚠ TensorFlow not available: {e}")
        tf_available = False
    
    if not coral_available and not tf_available:
        logger.error("✗ Neither Coral TPU nor TensorFlow is available")
        return False
    
    return True


def test_tpu_device():
    """Test if Coral TPU device is detected"""
    logger.info("Testing TPU device detection...")
    
    try:
        from pycoral.utils import edgetpu
        
        # List available Edge TPU devices
        devices = edgetpu.list_edge_tpus()
        
        if devices:
            logger.success(f"✓ Found {len(devices)} Edge TPU device(s):")
            for i, device in enumerate(devices):
                logger.info(f"  Device {i}: {device}")
            return True
        else:
            logger.warning("⚠ No Edge TPU devices found")
            logger.info("Make sure:")
            logger.info("  1. Coral TPU is connected via USB")
            logger.info("  2. Edge TPU runtime is installed")
            logger.info("  3. You have proper permissions")
            return False
            
    except ImportError:
        logger.warning("⚠ Cannot test TPU device - pycoral not available")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing TPU device: {e}")
        return False


def test_model_inference():
    """Test basic model inference functionality"""
    logger.info("Testing model inference...")
    
    try:
        from src.models.inference.tpu_inference import TPUInferenceEngine
        
        # Test with dummy model path
        model_path = "models/dummy_model.tflite"
        
        engine = TPUInferenceEngine(model_path)
        logger.success("✓ TPU inference engine created successfully")
        
        # Test with dummy data
        import numpy as np
        import pandas as pd
        
        # Create dummy market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        dummy_data = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100),
            'sma_20': np.random.uniform(40000, 50000, 100),
            'rsi': np.random.uniform(30, 70, 100),
        }, index=dates)
        
        # Test preprocessing
        processed_data = engine.preprocess_data(dummy_data)
        logger.success(f"✓ Data preprocessing successful - shape: {processed_data.shape}")
        
        # Test prediction
        prediction = engine.predict(processed_data)
        logger.success(f"✓ Prediction successful: {prediction}")
        
        # Test performance
        engine.warmup(5)
        stats = engine.get_performance_stats()
        if stats:
            logger.success(f"✓ Performance stats: {stats['avg_inference_time_ms']:.2f}ms avg")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model inference test failed: {e}")
        return False


def test_data_collection():
    """Test crypto data collection"""
    logger.info("Testing data collection...")
    
    try:
        from src.config.settings import Settings
        from src.data.collectors.crypto_collector import CryptoDataCollector
        
        settings = Settings()
        collector = CryptoDataCollector(settings)
        
        logger.success("✓ Data collector created successfully")
        
        # Note: This would require API keys to test fully
        logger.info("ℹ Full data collection test requires API keys")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data collection test failed: {e}")
        return False


def test_trading_engine():
    """Test trading engine"""
    logger.info("Testing trading engine...")
    
    try:
        from src.config.settings import Settings
        from src.models.inference.tpu_inference import TPUInferenceEngine
        from src.trading.engine.trading_engine import TradingEngine
        
        settings = Settings()
        inference_engine = TPUInferenceEngine("models/dummy_model.tflite")
        trading_engine = TradingEngine(settings, inference_engine)
        
        logger.success("✓ Trading engine created successfully")
        
        # Test portfolio summary
        summary = trading_engine.get_portfolio_summary()
        logger.success(f"✓ Portfolio summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Trading engine test failed: {e}")
        return False


def test_system_requirements():
    """Test system requirements"""
    logger.info("Testing system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        logger.success(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        logger.error(f"✗ Python version too old: {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)")
        return False
    
    # Check OS
    import platform
    os_name = platform.system()
    logger.info(f"Operating System: {os_name} {platform.release()}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        logger.info(f"Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            logger.warning("⚠ Low RAM - recommend at least 4GB")
        else:
            logger.success("✓ Sufficient RAM available")
            
    except ImportError:
        logger.info("ℹ Cannot check memory - psutil not available")
    
    return True


def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting Coral TPU Trading System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("System Requirements", test_system_requirements),
        ("Library Imports", test_imports),
        ("TPU Device Detection", test_tpu_device),
        ("Model Inference", test_model_inference),
        ("Data Collection", test_data_collection),
        ("Trading Engine", test_trading_engine),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.success("🎉 All tests passed! System is ready.")
    elif passed >= total * 0.7:
        logger.warning("⚠ Most tests passed. Some features may not work.")
    else:
        logger.error("❌ Many tests failed. System may not work properly.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
