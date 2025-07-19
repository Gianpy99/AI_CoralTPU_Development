"""
Coral TPU Inference Engine for cryptocurrency price prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import time

try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
    from pycoral.adapters import classify
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


class TPUInferenceEngine:
    """
    High-performance inference engine optimized for Coral TPU
    Falls back to CPU if TPU is not available
    """
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.output_shape = None
        self.is_tpu = False
        self.inference_times = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the TensorFlow Lite model with Coral TPU if available"""
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            logger.info("Creating a dummy model for demonstration...")
            self._create_dummy_model()
            return
        
        try:
            if CORAL_AVAILABLE:
                # Try to use Coral TPU
                self._initialize_tpu()
            else:
                # Fall back to CPU
                self._initialize_cpu()
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            logger.info("Creating a dummy model for demonstration...")
            self._create_dummy_model()
    
    def _initialize_tpu(self):
        """Initialize with Coral TPU"""
        try:
            # Check for available Edge TPU devices
            devices = edgetpu.list_edge_tpus()
            if not devices:
                logger.warning("No Edge TPU devices found, falling back to CPU")
                self._initialize_cpu()
                return
            
            logger.info(f"Found {len(devices)} Edge TPU device(s)")
            
            # Initialize interpreter with Edge TPU
            self.interpreter = tflite.Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[
                    tflite.load_delegate('libedgetpu.so.1')  # Linux
                    # For Windows: tflite.load_delegate('edgetpu.dll')
                    # For macOS: tflite.load_delegate('libedgetpu.1.dylib')
                ]
            )
            
            self.interpreter.allocate_tensors()
            self.is_tpu = True
            
            logger.success("Model loaded on Coral TPU successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize TPU: {e}")
            logger.info("Falling back to CPU inference")
            self._initialize_cpu()
    
    def _initialize_cpu(self):
        """Initialize with CPU-only TensorFlow Lite"""
        try:
            self.interpreter = tflite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.is_tpu = False
            
            logger.info("Model loaded on CPU successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CPU interpreter: {e}")
            raise
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration when no real model is available"""
        logger.info("Creating dummy inference engine for demonstration")
        self.interpreter = None
        self.is_dummy = True
    
    def _get_model_info(self):
        """Get model input/output information"""
        if self.interpreter is None:
            return
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        self.output_shape = self.output_details[0]['shape']
        
        logger.info(f"Model input shape: {self.input_shape}")
        logger.info(f"Model output shape: {self.output_shape}")
        logger.info(f"Input dtype: {self.input_details[0]['dtype']}")
        logger.info(f"Output dtype: {self.output_details[0]['dtype']}")
    
    def preprocess_data(self, market_data: pd.DataFrame, 
                       sequence_length: int = 60) -> np.ndarray:
        """
        Preprocess market data for model input
        
        Args:
            market_data: DataFrame with OHLCV and technical indicators
            sequence_length: Number of time steps to use for prediction
            
        Returns:
            Preprocessed numpy array ready for inference
        """
        if len(market_data) < sequence_length:
            raise ValueError(f"Insufficient data: need {sequence_length}, got {len(market_data)}")
        
        # Select features for the model
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'ema_12', 'ema_26', 'macd', 'rsi',
            'bb_upper', 'bb_lower', 'bb_position',
            'volume_ratio', 'price_momentum', 'volatility'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in market_data.columns]
        
        if not available_columns:
            logger.warning("No technical indicators found, using OHLCV only")
            available_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Get the latest sequence
        data = market_data[available_columns].tail(sequence_length).values
        
        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)
        
        # Normalize the data (simple min-max scaling)
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data_range = data_max - data_min
        
        # Avoid division by zero
        data_range[data_range == 0] = 1.0
        
        normalized_data = (data - data_min) / data_range
        
        # Reshape for model input [batch_size, sequence_length, features]
        input_data = normalized_data.reshape(1, sequence_length, len(available_columns))
        
        return input_data.astype(np.float32)
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on the Coral TPU
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            Prediction results with confidence scores and metadata
        """
        start_time = time.time()
        
        if hasattr(self, 'is_dummy') and self.is_dummy:
            # Return dummy predictions for demonstration
            return self._dummy_prediction()
        
        if self.interpreter is None:
            raise RuntimeError("Model not initialized")
        
        if self.input_details is None:
            self._get_model_info()
        
        try:
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Process output based on model type
            predictions = self._process_output(output_data)
            
            # Add metadata
            predictions.update({
                'inference_time_ms': inference_time * 1000,
                'device': 'TPU' if self.is_tpu else 'CPU',
                'timestamp': time.time()
            })
            
            logger.debug(f"Inference completed in {inference_time*1000:.2f}ms on {'TPU' if self.is_tpu else 'CPU'}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _process_output(self, output_data: np.ndarray) -> Dict[str, Any]:
        """Process raw model output into meaningful predictions"""
        
        # Assuming the model outputs price movement predictions
        # This would depend on your specific model architecture
        
        predictions = {}
        
        if output_data.shape[-1] == 1:
            # Regression: price prediction
            predicted_price = float(output_data[0, 0])
            predictions.update({
                'prediction_type': 'price',
                'predicted_price': predicted_price,
                'confidence': 0.8  # You might calculate this based on model uncertainty
            })
            
        elif output_data.shape[-1] == 3:
            # Classification: up, down, sideways
            probabilities = output_data[0]
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            class_names = ['down', 'sideways', 'up']
            predictions.update({
                'prediction_type': 'direction',
                'predicted_direction': class_names[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'down': float(probabilities[0]),
                    'sideways': float(probabilities[1]),
                    'up': float(probabilities[2])
                }
            })
            
        else:
            # Multi-output or other format
            predictions.update({
                'prediction_type': 'raw',
                'raw_output': output_data.tolist(),
                'confidence': 0.5
            })
        
        return predictions
    
    def _dummy_prediction(self) -> Dict[str, Any]:
        """Generate dummy predictions for demonstration"""
        import random
        
        # Simulate realistic predictions
        direction_probs = np.random.dirichlet([1, 1, 1])  # Random probabilities for down, sideways, up
        predicted_class = np.argmax(direction_probs)
        confidence = float(np.max(direction_probs))
        
        class_names = ['down', 'sideways', 'up']
        
        return {
            'prediction_type': 'direction',
            'predicted_direction': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'down': float(direction_probs[0]),
                'sideways': float(direction_probs[1]),
                'up': float(direction_probs[2])
            },
            'inference_time_ms': random.uniform(1, 5),  # Simulate fast TPU inference
            'device': 'DUMMY_TPU',
            'timestamp': time.time()
        }
    
    def batch_predict(self, batch_data: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Run batch inference for multiple inputs"""
        results = []
        
        for input_data in batch_data:
            try:
                prediction = self.predict(input_data)
                results.append(prediction)
            except Exception as e:
                logger.error(f"Batch inference failed for one sample: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}
        
        times_ms = [t * 1000 for t in self.inference_times]
        
        return {
            'avg_inference_time_ms': np.mean(times_ms),
            'min_inference_time_ms': np.min(times_ms),
            'max_inference_time_ms': np.max(times_ms),
            'std_inference_time_ms': np.std(times_ms),
            'total_inferences': len(times_ms),
            'throughput_per_second': 1000 / np.mean(times_ms) if times_ms else 0
        }
    
    def warmup(self, num_warmup: int = 10):
        """Warm up the model with dummy data"""
        logger.info(f"Warming up model with {num_warmup} dummy inferences...")
        
        if hasattr(self, 'is_dummy') and self.is_dummy:
            # For dummy model, just simulate warmup
            for _ in range(num_warmup):
                self._dummy_prediction()
            logger.success("Dummy model warmed up")
            return
        
        if self.input_details is None:
            self._get_model_info()
        
        # Create dummy input matching the expected shape
        dummy_input = np.random.random(self.input_shape).astype(np.float32)
        
        for i in range(num_warmup):
            try:
                self.predict(dummy_input)
                if (i + 1) % 5 == 0:
                    logger.info(f"Warmup progress: {i + 1}/{num_warmup}")
            except Exception as e:
                logger.warning(f"Warmup iteration {i + 1} failed: {e}")
        
        logger.success("Model warmup completed")
        stats = self.get_performance_stats()
        if stats:
            logger.info(f"Average inference time: {stats['avg_inference_time_ms']:.2f}ms")


# Standalone script for testing TPU
if __name__ == "__main__":
    import sys
    sys.path.append("../../..")
    
    def test_tpu():
        """Test TPU functionality"""
        model_path = "models/price_predictor.tflite"
        
        logger.info("Testing Coral TPU inference engine...")
        
        try:
            engine = TPUInferenceEngine(model_path)
            
            # Create dummy market data
            import pandas as pd
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
            
            # Preprocess data
            processed_data = engine.preprocess_data(dummy_data)
            logger.info(f"Processed data shape: {processed_data.shape}")
            
            # Warm up
            engine.warmup(5)
            
            # Run predictions
            for i in range(10):
                prediction = engine.predict(processed_data)
                logger.info(f"Prediction {i+1}: {prediction}")
            
            # Show performance stats
            stats = engine.get_performance_stats()
            logger.info(f"Performance stats: {stats}")
            
        except Exception as e:
            logger.error(f"TPU test failed: {e}")
            raise
    
    test_tpu()
