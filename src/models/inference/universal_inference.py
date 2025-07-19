"""
Universal Coral TPU Inference Engine
Supports multiple AI tasks: crypto trading, image recognition, object detection, etc.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from loguru import logger
import time
import json

try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
    from pycoral.adapters import classify
    from pycoral.adapters import detect
    import tflite_runtime.interpreter as tflite
    CORAL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Coral TPU libraries not available: {e}")
    logger.warning("Falling back to TensorFlow Lite CPU inference")
    try:
        import tensorflow as tf
        CORAL_AVAILABLE = False
    except ImportError:
        logger.error("Neither Coral TPU nor TensorFlow available")
        raise

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available - camera features disabled")
    OPENCV_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logger.warning("PIL not available - some image features disabled")
    PIL_AVAILABLE = False


class UniversalTPUInference:
    """
    Universal Coral TPU inference engine that supports multiple AI tasks:
    - Cryptocurrency trading predictions
    - Image classification
    - Object detection
    - Custom models
    """
    
    def __init__(self, config_file: str = "inference_config.json"):
        self.config_file = Path(config_file)
        self.models = {}
        self.active_model = None
        self.inference_times = []
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize models based on config
        self._initialize_models()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load inference configuration"""
        default_config = {
            "models": {
                "crypto_trading": {
                    "path": "models/crypto_predictor_edgetpu.tflite",
                    "labels": "models/crypto_labels.txt",
                    "type": "classification",
                    "input_shape": [1, 60, 16],
                    "preprocessing": "crypto_timeseries"
                },
                "image_classification": {
                    "path": "models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
                    "labels": "models/inat_bird_labels.txt",
                    "type": "classification", 
                    "input_shape": [1, 224, 224, 3],
                    "preprocessing": "image_224x224"
                },
                "object_detection": {
                    "path": "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
                    "labels": "models/coco_labels.txt",
                    "type": "detection",
                    "input_shape": [1, 300, 300, 3],
                    "preprocessing": "image_300x300"
                },
                "face_detection": {
                    "path": "models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
                    "labels": "models/face_labels.txt",
                    "type": "detection",
                    "input_shape": [1, 320, 320, 3],
                    "preprocessing": "image_320x320"
                }
            },
            "camera": {
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30
            },
            "display": {
                "show_confidence": True,
                "confidence_threshold": 0.5,
                "max_detections": 10
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        # Save default config
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _initialize_models(self):
        """Initialize all configured models"""
        for model_name, model_config in self.config["models"].items():
            try:
                self._load_model(model_name, model_config)
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
    
    def _load_model(self, model_name: str, model_config: Dict[str, Any]):
        """Load a single model"""
        model_path = Path(model_config["path"])
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return
        
        try:
            # Load labels if available
            labels = None
            if "labels" in model_config:
                labels_path = Path(model_config["labels"])
                if labels_path.exists():
                    with open(labels_path, 'r') as f:
                        labels = [line.strip() for line in f.readlines()]
            
            # Initialize interpreter
            if CORAL_AVAILABLE:
                # Try Coral TPU
                devices = edgetpu.list_edge_tpus()
                if devices:
                    interpreter = tflite.Interpreter(
                        model_path=str(model_path),
                        experimental_delegates=[edgetpu.make_interpreter_delegate()]
                    )
                    is_tpu = True
                else:
                    # Fall back to CPU
                    interpreter = tflite.Interpreter(model_path=str(model_path))
                    is_tpu = False
            else:
                interpreter = tflite.Interpreter(model_path=str(model_path))
                is_tpu = False
            
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            self.models[model_name] = {
                "interpreter": interpreter,
                "input_details": input_details,
                "output_details": output_details,
                "labels": labels,
                "config": model_config,
                "is_tpu": is_tpu
            }
            
            logger.success(f"Model {model_name} loaded successfully ({'TPU' if is_tpu else 'CPU'})")
            
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
    
    def set_active_model(self, model_name: str):
        """Set the active model for inference"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        self.active_model = model_name
        logger.info(f"Active model set to: {model_name}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def preprocess_crypto_data(self, market_data: pd.DataFrame, 
                              sequence_length: int = 60) -> np.ndarray:
        """Preprocess cryptocurrency market data"""
        # Use the original crypto preprocessing logic
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'ema_12', 'ema_26', 'macd', 'rsi',
            'bb_upper', 'bb_lower', 'bb_position',
            'volume_ratio', 'price_momentum', 'volatility'
        ]
        
        available_columns = [col for col in feature_columns if col in market_data.columns]
        
        if len(available_columns) < 5:
            raise ValueError(f"Insufficient features: {available_columns}")
        
        feature_data = market_data[available_columns].copy()
        feature_data = feature_data.fillna(method='ffill').fillna(0)
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(feature_data)
        
        # Take last sequence_length samples
        if len(normalized_data) < sequence_length:
            # Pad with zeros if not enough data
            padded_data = np.zeros((sequence_length, len(available_columns)))
            padded_data[-len(normalized_data):] = normalized_data
            normalized_data = padded_data
        else:
            normalized_data = normalized_data[-sequence_length:]
        
        # Reshape for model input
        input_data = normalized_data.reshape(1, sequence_length, len(available_columns))
        return input_data.astype(np.float32)
    
    def preprocess_image(self, image: Union[np.ndarray, str, Path], 
                        target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for inference"""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL not available for image preprocessing")
        
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Convert from BGR to RGB if needed (OpenCV format)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1] or [-1, 1] depending on model
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, input_data: Union[np.ndarray, pd.DataFrame, str, Path], 
                model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Universal prediction method
        
        Args:
            input_data: Input data (array for direct inference, DataFrame for crypto, 
                       string/Path for image files)
            model_name: Model to use (uses active model if None)
            
        Returns:
            Prediction results with metadata
        """
        if model_name is None:
            model_name = self.active_model
        
        if model_name is None:
            raise ValueError("No active model set")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        interpreter = model["interpreter"]
        input_details = model["input_details"]
        output_details = model["output_details"]
        config = model["config"]
        
        start_time = time.time()
        
        # Preprocess input based on model type
        if config["preprocessing"] == "crypto_timeseries":
            if isinstance(input_data, pd.DataFrame):
                processed_input = self.preprocess_crypto_data(input_data)
            else:
                processed_input = input_data
        elif config["preprocessing"].startswith("image_"):
            # Extract target size from preprocessing name
            size_str = config["preprocessing"].split("_")[1]
            if "x" in size_str:
                width, height = map(int, size_str.split("x"))
                target_size = (width, height)
            else:
                # Square image
                size = int(size_str)
                target_size = (size, size)
            
            processed_input = self.preprocess_image(input_data, target_size)
        else:
            # Direct input
            processed_input = input_data
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in output_details:
            output = interpreter.get_tensor(output_detail['index'])
            outputs.append(output)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Process output based on model type
        result = self._process_output(outputs, model, processed_input.shape)
        
        # Add metadata
        result.update({
            'inference_time_ms': inference_time * 1000,
            'model_name': model_name,
            'device': 'TPU' if model['is_tpu'] else 'CPU',
            'timestamp': time.time(),
            'input_shape': processed_input.shape
        })
        
        return result
    
    def _process_output(self, outputs: List[np.ndarray], model: Dict[str, Any], 
                       input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Process model output based on model type"""
        config = model["config"]
        labels = model["labels"]
        
        if config["type"] == "classification":
            return self._process_classification_output(outputs[0], labels)
        elif config["type"] == "detection":
            return self._process_detection_output(outputs, labels, input_shape)
        else:
            return {"raw_output": [output.tolist() for output in outputs]}
    
    def _process_classification_output(self, output: np.ndarray, 
                                     labels: Optional[List[str]]) -> Dict[str, Any]:
        """Process classification output"""
        # Get probabilities
        probabilities = output[0]
        
        # Get top prediction
        top_index = np.argmax(probabilities)
        confidence = float(probabilities[top_index])
        
        result = {
            "prediction_type": "classification",
            "top_prediction": {
                "index": int(top_index),
                "confidence": confidence
            }
        }
        
        if labels and top_index < len(labels):
            result["top_prediction"]["label"] = labels[top_index]
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_predictions = []
        for idx in top_indices:
            pred = {
                "index": int(idx),
                "confidence": float(probabilities[idx])
            }
            if labels and idx < len(labels):
                pred["label"] = labels[idx]
            top_predictions.append(pred)
        
        result["top_5_predictions"] = top_predictions
        
        return result
    
    def _process_detection_output(self, outputs: List[np.ndarray], 
                                 labels: Optional[List[str]], 
                                 input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Process object detection output"""
        # Standard SSD MobileNet output format:
        # outputs[0]: bounding boxes [1, num_detections, 4]
        # outputs[1]: class IDs [1, num_detections]
        # outputs[2]: scores [1, num_detections]
        # outputs[3]: number of detections [1]
        
        if len(outputs) >= 4:
            boxes = outputs[0][0]  # [num_detections, 4]
            class_ids = outputs[1][0].astype(int)  # [num_detections]
            scores = outputs[2][0]  # [num_detections]
            num_detections = int(outputs[3][0])
        else:
            # Alternative format
            boxes = outputs[0][0]
            scores = outputs[1][0]
            class_ids = outputs[2][0].astype(int) if len(outputs) > 2 else np.zeros_like(scores)
            num_detections = len(scores)
        
        detections = []
        threshold = self.config["display"]["confidence_threshold"]
        max_detections = self.config["display"]["max_detections"]
        
        for i in range(min(num_detections, max_detections)):
            if scores[i] >= threshold:
                detection = {
                    "bbox": boxes[i].tolist(),  # [ymin, xmin, ymax, xmax] normalized
                    "score": float(scores[i]),
                    "class_id": int(class_ids[i])
                }
                
                if labels and class_ids[i] < len(labels):
                    detection["label"] = labels[class_ids[i]]
                
                detections.append(detection)
        
        return {
            "prediction_type": "detection",
            "detections": detections,
            "num_detections": len(detections)
        }
    
    def start_camera_stream(self, model_name: str = "image_classification"):
        """Start real-time camera inference"""
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available for camera features")
        
        self.set_active_model(model_name)
        
        # Initialize camera
        cap = cv2.VideoCapture(self.config["camera"]["device_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["height"])
        cap.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])
        
        logger.info(f"Starting camera stream with model: {model_name}")
        logger.info("Press 'q' to quit, 's' to save frame, 'c' to change model")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                try:
                    result = self.predict(frame, model_name)
                    
                    # Draw results on frame
                    frame = self._draw_results(frame, result)
                    
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                
                # Show frame
                cv2.imshow('Coral TPU Camera Stream', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save frame
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved as {filename}")
                elif key == ord('c'):
                    # Cycle through models
                    models = self.get_available_models()
                    current_idx = models.index(model_name)
                    next_idx = (current_idx + 1) % len(models)
                    model_name = models[next_idx]
                    self.set_active_model(model_name)
                    logger.info(f"Switched to model: {model_name}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _draw_results(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw inference results on frame"""
        if result["prediction_type"] == "classification":
            # Draw classification result
            top_pred = result["top_prediction"]
            label = top_pred.get('label', f"Class {top_pred.get('index', '?')}")
            confidence = top_pred['confidence']
            text = f"{label} ({confidence:.2f})"
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # Show top 3 predictions
            for i, pred in enumerate(result["top_5_predictions"][:3]):
                y_pos = 70 + i * 30
                label = pred.get('label', f"Class {pred.get('index', '?')}")
                confidence = pred['confidence']
                text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
        
        elif result["prediction_type"] == "detection":
            # Draw detection results
            height, width = frame.shape[:2]
            
            for detection in result["detections"]:
                bbox = detection["bbox"]
                score = detection["score"]
                label = detection.get("label", f"Class {detection['class_id']}")
                
                # Convert normalized coordinates to pixel coordinates
                ymin = int(bbox[0] * height)
                xmin = int(bbox[1] * width)
                ymax = int(bbox[2] * height)
                xmax = int(bbox[3] * width)
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Draw label
                label_text = f"{label} ({score:.2f})"
                cv2.putText(frame, label_text, (xmin, ymin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw performance info
        if self.inference_times:
            avg_time = np.mean(self.inference_times[-10:]) * 1000  # Last 10 frames
            fps = 1000 / avg_time if avg_time > 0 else 0
            info_text = f"FPS: {fps:.1f}, Avg: {avg_time:.1f}ms"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_image_file(self, image_path: str, model_name: str = "image_classification") -> Dict[str, Any]:
        """Process a single image file"""
        self.set_active_model(model_name)
        return self.predict(image_path, model_name)
    
    def process_crypto_data(self, market_data: pd.DataFrame, 
                           model_name: str = "crypto_trading") -> Dict[str, Any]:
        """Process cryptocurrency data"""
        self.set_active_model(model_name)
        return self.predict(market_data, model_name)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {"message": "No inference times recorded"}
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        return {
            "total_inferences": len(times_ms),
            "avg_time_ms": np.mean(times_ms),
            "min_time_ms": np.min(times_ms),
            "max_time_ms": np.max(times_ms),
            "fps": 1000 / np.mean(times_ms) if times_ms else 0,
            "loaded_models": list(self.models.keys()),
            "tpu_models": [name for name, model in self.models.items() if model["is_tpu"]]
        }
    
    def download_sample_models(self):
        """Download sample models for testing"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        sample_models = [
            {
                "name": "Image Classification (Birds)",
                "url": "https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
                "filename": "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
            },
            {
                "name": "Object Detection (COCO)",
                "url": "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
                "filename": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
            }
        ]
        
        for model in sample_models:
            model_path = models_dir / model["filename"]
            if not model_path.exists():
                logger.info(f"Downloading {model['name']}...")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(model["url"], model_path)
                    logger.success(f"Downloaded {model['filename']}")
                except Exception as e:
                    logger.error(f"Failed to download {model['filename']}: {e}")


# Convenience functions for backward compatibility
def create_crypto_inference(model_path: str = "models/crypto_predictor_edgetpu.tflite"):
    """Create inference engine specifically for crypto trading"""
    engine = UniversalTPUInference()
    if "crypto_trading" in engine.get_available_models():
        engine.set_active_model("crypto_trading")
    return engine

def create_vision_inference(model_type: str = "classification"):
    """Create inference engine specifically for computer vision"""
    engine = UniversalTPUInference()
    
    if model_type == "classification" and "image_classification" in engine.get_available_models():
        engine.set_active_model("image_classification")
    elif model_type == "detection" and "object_detection" in engine.get_available_models():
        engine.set_active_model("object_detection")
    
    return engine


# Testing functions
if __name__ == "__main__":
    import sys
    
    def test_universal_inference():
        """Test the universal inference engine"""
        logger.info("Testing Universal TPU Inference Engine...")
        
        try:
            # Create engine
            engine = UniversalTPUInference()
            
            logger.info(f"Available models: {engine.get_available_models()}")
            
            # Test each loaded model
            for model_name in engine.get_available_models():
                logger.info(f"Testing model: {model_name}")
                
                try:
                    engine.set_active_model(model_name)
                    
                    # Create dummy input based on model type
                    model_config = engine.models[model_name]["config"]
                    input_shape = model_config["input_shape"]
                    
                    if model_config["preprocessing"] == "crypto_timeseries":
                        # Create dummy crypto data
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
                        
                        result = engine.predict(dummy_data)
                        
                    else:
                        # Create dummy image data
                        if len(input_shape) == 4:  # [batch, height, width, channels]
                            dummy_input = np.random.randint(0, 256, input_shape[1:], dtype=np.uint8)
                        else:
                            dummy_input = np.random.random(input_shape[1:]).astype(np.float32)
                        
                        result = engine.predict(dummy_input)
                    
                    logger.success(f"Model {model_name} test passed: {result.get('prediction_type', 'unknown')}")
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} test failed: {e}")
            
            # Show performance stats
            stats = engine.get_performance_stats()
            logger.info(f"Performance stats: {stats}")
            
        except Exception as e:
            logger.error(f"Universal inference test failed: {e}")
            raise
    
    def test_camera_stream():
        """Test camera stream"""
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available - skipping camera test")
            return
        
        logger.info("Testing camera stream (press 'q' to quit)...")
        
        try:
            engine = UniversalTPUInference()
            
            # Download sample models if needed
            engine.download_sample_models()
            
            # Start camera stream
            if "image_classification" in engine.get_available_models():
                engine.start_camera_stream("image_classification")
            elif "object_detection" in engine.get_available_models():
                engine.start_camera_stream("object_detection")
            else:
                logger.warning("No vision models available")
        
        except Exception as e:
            logger.error(f"Camera test failed: {e}")
    
    # Run tests
    if len(sys.argv) > 1:
        if sys.argv[1] == "camera":
            test_camera_stream()
        else:
            test_universal_inference()
    else:
        test_universal_inference()
