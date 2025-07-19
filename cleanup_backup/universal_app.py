#!/usr/bin/env python3
"""
Universal Coral TPU Application
Combines crypto trading and computer vision capabilities
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.models.inference.universal_inference import UniversalTPUInference
    from src.utils.camera_vision import CameraVision
except ImportError:
    # Try alternative imports
    try:
        import sys
        sys.path.append('src/models/inference')
        sys.path.append('src/utils')
        from universal_inference import UniversalTPUInference
        from camera_vision import CameraVision
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure all dependencies are installed")
        sys.exit(1)


class CoralTPUApp:
    """
    Universal Coral TPU Application
    Supports both cryptocurrency trading and computer vision tasks
    """
    
    def __init__(self):
        self.inference_engine = None
        self.camera_vision = None
        
    def initialize(self):
        """Initialize the application"""
        logger.info("🚀 Initializing Coral TPU Universal Application")
        
        try:
            # Initialize inference engine
            self.inference_engine = UniversalTPUInference()
            logger.success("✅ Universal inference engine initialized")
            
            # Initialize camera vision
            self.camera_vision = CameraVision()
            logger.success("✅ Camera vision initialized")
            
            # Show available capabilities
            models = self.inference_engine.get_available_models()
            cameras = self.camera_vision.list_cameras()
            
            logger.info(f"📊 Available models: {models}")
            logger.info(f"📹 Available cameras: {cameras}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def crypto_mode(self):
        """Run in cryptocurrency trading mode"""
        logger.info("💰 Starting Crypto Trading Mode")
        
        if "crypto_trading" not in self.inference_engine.get_available_models():
            logger.warning("No crypto trading model available")
            logger.info("You can still test with dummy crypto data")
        
        # Import crypto trading components
        try:
            import pandas as pd
            import numpy as np
            
            # Create sample crypto data for demo
            logger.info("📊 Creating sample crypto data...")
            dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
            sample_data = pd.DataFrame({
                'open': np.random.uniform(40000, 50000, 100),
                'high': np.random.uniform(40000, 50000, 100),
                'low': np.random.uniform(40000, 50000, 100),
                'close': np.random.uniform(40000, 50000, 100),
                'volume': np.random.uniform(100, 1000, 100),
                'sma_20': np.random.uniform(40000, 50000, 100),
                'ema_12': np.random.uniform(40000, 50000, 100),
                'ema_26': np.random.uniform(40000, 50000, 100),
                'macd': np.random.uniform(-100, 100, 100),
                'rsi': np.random.uniform(30, 70, 100),
                'bb_upper': np.random.uniform(41000, 51000, 100),
                'bb_lower': np.random.uniform(39000, 49000, 100),
                'bb_position': np.random.uniform(0, 1, 100),
                'volume_ratio': np.random.uniform(0.5, 2.0, 100),
                'price_momentum': np.random.uniform(-0.1, 0.1, 100),
                'volatility': np.random.uniform(0.01, 0.05, 100)
            }, index=dates)
            
            # Set crypto model if available
            if "crypto_trading" in self.inference_engine.get_available_models():
                self.inference_engine.set_active_model("crypto_trading")
                model_name = "crypto_trading"
            else:
                # Use any available model for demo
                available_models = self.inference_engine.get_available_models()
                if available_models:
                    model_name = available_models[0]
                    self.inference_engine.set_active_model(model_name)
                else:
                    logger.error("No models available")
                    return
            
            logger.info(f"🤖 Using model: {model_name}")
            
            # Run predictions
            logger.info("🔮 Running crypto predictions...")
            for i in range(5):
                try:
                    # Use last 60 data points for prediction
                    prediction_data = sample_data.iloc[-60:].copy()
                    
                    result = self.inference_engine.predict(prediction_data, model_name)
                    
                    logger.info(f"📈 Prediction {i+1}:")
                    logger.info(f"  Model: {result.get('model_name', 'Unknown')}")
                    logger.info(f"  Device: {result.get('device', 'Unknown')}")
                    logger.info(f"  Inference time: {result.get('inference_time_ms', 0):.2f}ms")
                    
                    if result.get('prediction_type') == 'classification':
                        top_pred = result.get('top_prediction', {})
                        logger.info(f"  Prediction: {top_pred.get('label', 'Unknown')} "
                                   f"({top_pred.get('confidence', 0):.2%})")
                    elif result.get('prediction_type') == 'direction':
                        direction = result.get('predicted_direction', 'unknown')
                        confidence = result.get('confidence', 0)
                        logger.info(f"  Direction: {direction} ({confidence:.2%})")
                    
                    time.sleep(1)  # Wait 1 second between predictions
                    
                except Exception as e:
                    logger.error(f"Prediction {i+1} failed: {e}")
            
            # Show performance stats
            stats = self.inference_engine.get_performance_stats()
            logger.info(f"📊 Performance stats: {stats}")
            
        except Exception as e:
            logger.error(f"Crypto mode failed: {e}")
    
    def vision_mode(self, mode: str = "classification"):
        """Run in computer vision mode"""
        logger.info(f"👁️ Starting Vision Mode: {mode}")
        
        # Check for vision models
        vision_models = [m for m in self.inference_engine.get_available_models() 
                        if "image" in m or "object" in m or "face" in m]
        
        if not vision_models:
            logger.warning("No vision models available")
            logger.info("Downloading sample models...")
            try:
                self.inference_engine.download_sample_models()
                # Reinitialize to load new models
                self.inference_engine._initialize_models()
                vision_models = [m for m in self.inference_engine.get_available_models() 
                               if "image" in m or "object" in m]
            except Exception as e:
                logger.error(f"Failed to download models: {e}")
                return
        
        if not vision_models:
            logger.error("Still no vision models available")
            return
        
        # Select appropriate model
        if mode == "classification" and "image_classification" in vision_models:
            model_name = "image_classification"
        elif mode == "detection" and "object_detection" in vision_models:
            model_name = "object_detection"
        elif mode == "face" and "face_detection" in vision_models:
            model_name = "face_detection"
        else:
            model_name = vision_models[0]  # Use first available
        
        logger.info(f"🤖 Using model: {model_name}")
        
        # Check if cameras are available
        cameras = self.camera_vision.list_cameras()
        if not cameras:
            logger.error("No cameras found!")
            return
        
        # Start camera stream
        logger.info("📹 Starting camera stream...")
        logger.info("Controls:")
        logger.info("  'q' - Quit")
        logger.info("  's' - Save current frame")
        logger.info("  'c' - Change model")
        logger.info("  'p' - Pause/Resume")
        
        try:
            self.camera_vision.start_live_stream(model_name, display=True)
        except KeyboardInterrupt:
            logger.info("Vision mode interrupted by user")
        except Exception as e:
            logger.error(f"Vision mode failed: {e}")
        finally:
            self.camera_vision.stop_stream()
    
    def photo_mode(self, model_name: str = "image_classification"):
        """Take a single photo with AI analysis"""
        logger.info("📸 Photo Mode - Taking photo with AI analysis")
        
        try:
            result = self.camera_vision.take_photo_with_ai(model_name)
            
            logger.success("✅ Photo captured and analyzed!")
            logger.info(f"📄 Analysis report: {result.get('image_file', 'report.json')}")
            
            # Show summary
            ai_result = result.get("ai_analysis", {})
            if ai_result.get("prediction_type") == "classification":
                top_pred = ai_result.get("top_prediction", {})
                logger.info(f"🎯 Main prediction: {top_pred.get('label', 'Unknown')} "
                           f"({top_pred.get('confidence', 0):.1%})")
                
                # Show top 3 predictions
                for i, pred in enumerate(ai_result.get("top_5_predictions", [])[:3]):
                    logger.info(f"  {i+1}. {pred.get('label', 'Unknown')} "
                               f"({pred.get('confidence', 0):.1%})")
            
            elif ai_result.get("prediction_type") == "detection":
                detections = ai_result.get("detections", [])
                logger.info(f"🎯 Found {len(detections)} objects:")
                for detection in detections[:5]:  # Show first 5
                    label = detection.get("label", "Unknown")
                    score = detection.get("score", 0)
                    logger.info(f"  - {label} ({score:.1%})")
            
        except Exception as e:
            logger.error(f"Photo mode failed: {e}")
    
    def demo_mode(self):
        """Run comprehensive demo of all capabilities"""
        logger.info("🎪 Starting Comprehensive Demo")
        
        try:
            # Demo 1: Crypto predictions
            logger.info("\n💰 Demo 1: Crypto Trading Predictions")
            self.crypto_mode()
            
            input("\nPress Enter to continue to vision demo...")
            
            # Demo 2: Computer vision
            logger.info("\n👁️ Demo 2: Computer Vision")
            logger.info("This will start the camera stream...")
            
            self.vision_mode("classification")
            
            # Demo 3: Photo analysis
            logger.info("\n📸 Demo 3: Photo Analysis")
            input("Press Enter to take a photo with AI analysis...")
            
            self.photo_mode()
            
            logger.success("🎉 Demo completed!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
    
    def interactive_menu(self):
        """Show interactive menu"""
        while True:
            print("\n" + "="*50)
            print("🚀 CORAL TPU UNIVERSAL APPLICATION")
            print("="*50)
            print("1. 💰 Crypto Trading Mode")
            print("2. 👁️  Computer Vision (Live)")
            print("3. 📸 Take Photo with AI")
            print("4. 🔍 Object Detection Mode")
            print("5. 🎪 Run Full Demo")
            print("6. 📊 Show Statistics")
            print("7. ⚙️  System Information")
            print("0. ❌ Exit")
            print("="*50)
            
            try:
                choice = input("Select option (0-7): ").strip()
                
                if choice == "0":
                    logger.info("👋 Goodbye!")
                    break
                elif choice == "1":
                    self.crypto_mode()
                elif choice == "2":
                    self.vision_mode("classification")
                elif choice == "3":
                    self.photo_mode()
                elif choice == "4":
                    self.vision_mode("detection")
                elif choice == "5":
                    self.demo_mode()
                elif choice == "6":
                    self.show_statistics()
                elif choice == "7":
                    self.show_system_info()
                else:
                    print("❌ Invalid choice. Please try again.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Menu error: {e}")
    
    def show_statistics(self):
        """Show system statistics"""
        logger.info("📊 System Statistics")
        
        if self.inference_engine:
            stats = self.inference_engine.get_performance_stats()
            logger.info("🤖 Inference Engine Stats:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        
        if self.camera_vision:
            cam_stats = self.camera_vision.get_stats()
            logger.info("📹 Camera Vision Stats:")
            for key, value in cam_stats.items():
                logger.info(f"  {key}: {value}")
    
    def show_system_info(self):
        """Show system information"""
        logger.info("⚙️ System Information")
        
        # Models
        if self.inference_engine:
            models = self.inference_engine.get_available_models()
            logger.info(f"📊 Available Models: {models}")
            
            # Check TPU status
            tpu_models = []
            for model_name in models:
                model = self.inference_engine.models.get(model_name, {})
                if model.get("is_tpu", False):
                    tpu_models.append(model_name)
            
            if tpu_models:
                logger.success(f"🎉 TPU Models: {tpu_models}")
            else:
                logger.warning("⚠️ No TPU models (running on CPU)")
        
        # Cameras
        if self.camera_vision:
            cameras = self.camera_vision.list_cameras()
            logger.info(f"📹 Available Cameras: {cameras}")
        
        # Dependencies
        dependencies = {
            "PyCoral": "pycoral",
            "TensorFlow Lite": "tflite_runtime", 
            "OpenCV": "cv2",
            "PIL": "PIL"
        }
        
        logger.info("📦 Dependencies:")
        for name, module in dependencies.items():
            try:
                __import__(module)
                logger.success(f"  ✅ {name}")
            except ImportError:
                logger.error(f"  ❌ {name}")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Universal Coral TPU Application")
    parser.add_argument("--mode", choices=["crypto", "vision", "photo", "demo", "menu"], 
                       default="menu", help="Application mode")
    parser.add_argument("--vision-type", choices=["classification", "detection", "face"],
                       default="classification", help="Type of computer vision")
    parser.add_argument("--model", help="Specific model to use")
    
    args = parser.parse_args()
    
    # Initialize application
    app = CoralTPUApp()
    
    if not app.initialize():
        logger.error("Failed to initialize application")
        sys.exit(1)
    
    # Run based on mode
    try:
        if args.mode == "crypto":
            app.crypto_mode()
        elif args.mode == "vision":
            app.vision_mode(args.vision_type)
        elif args.mode == "photo":
            app.photo_mode(args.model or "image_classification")
        elif args.mode == "demo":
            app.demo_mode()
        elif args.mode == "menu":
            app.interactive_menu()
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
