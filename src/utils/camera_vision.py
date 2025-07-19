"""
Camera Vision Module for Coral TPU
Provides easy-to-use camera functionality for real-time AI inference
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from loguru import logger
import time
import json

try:
    from ..models.inference.universal_inference import UniversalTPUInference
except ImportError:
    try:
        from src.models.inference.universal_inference import UniversalTPUInference
    except ImportError:
        import sys
        sys.path.append("../models/inference")
        sys.path.append("src/models/inference")
        from universal_inference import UniversalTPUInference


class CameraVision:
    """
    Easy-to-use camera interface for Coral TPU AI inference
    Supports image classification, object detection, and custom models
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.inference_engine = UniversalTPUInference()
        self.is_streaming = False
        self.cap = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Results storage
        self.last_result = None
        self.results_history = []
        self.max_history = 100
        
        logger.info("Camera Vision initialized")
    
    def list_cameras(self) -> list:
        """List available cameras"""
        available_cameras = []
        
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        
        logger.info(f"Available cameras: {available_cameras}")
        return available_cameras
    
    def list_models(self) -> list:
        """List available AI models"""
        models = self.inference_engine.get_available_models()
        logger.info(f"Available models: {models}")
        return models
    
    def set_model(self, model_name: str):
        """Set the AI model to use"""
        try:
            self.inference_engine.set_active_model(model_name)
            logger.success(f"Model set to: {model_name}")
        except Exception as e:
            logger.error(f"Failed to set model {model_name}: {e}")
            raise
    
    def capture_single_image(self, model_name: str = "image_classification", 
                           save_image: bool = False) -> Dict[str, Any]:
        """
        Capture a single image and run AI inference
        
        Args:
            model_name: AI model to use for inference
            save_image: Whether to save the captured image
            
        Returns:
            Dict containing inference results and metadata
        """
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        try:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            
            # Set model
            self.set_model(model_name)
            
            # Run inference
            result = self.inference_engine.predict(frame, model_name)
            
            # Save image if requested
            if save_image:
                timestamp = int(time.time())
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                result["saved_image"] = filename
                logger.info(f"Image saved as {filename}")
            
            self.last_result = result
            
            return result
            
        finally:
            cap.release()
    
    def start_live_stream(self, model_name: str = "image_classification", 
                         display: bool = True, 
                         callback: Optional[Callable] = None):
        """
        Start live camera stream with real-time AI inference
        
        Args:
            model_name: AI model to use
            display: Whether to show visual display
            callback: Optional callback function to process results
        """
        self.set_model(model_name)
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_streaming = True
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        logger.info(f"Starting live stream with model: {model_name}")
        if display:
            logger.info("Controls: 'q'=quit, 's'=save, 'c'=change model, 'p'=pause")
        
        try:
            while self.is_streaming:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                self.frame_count += 1
                
                # Run inference
                try:
                    result = self.inference_engine.predict(frame, model_name)
                    
                    # Store result
                    self.last_result = result
                    self.results_history.append(result)
                    if len(self.results_history) > self.max_history:
                        self.results_history.pop(0)
                    
                    # Call callback if provided
                    if callback:
                        callback(frame, result)
                    
                    # Update FPS counter
                    self.fps_counter += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        fps = self.fps_counter / (current_time - self.last_fps_time)
                        self.last_fps_time = current_time
                        self.fps_counter = 0
                    
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    result = {"error": str(e)}
                
                # Display frame if requested
                if display:
                    display_frame = self._draw_results(frame.copy(), result)
                    cv2.imshow('Coral TPU Live Stream', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = int(time.time())
                        filename = f"stream_capture_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        logger.info(f"Frame saved as {filename}")
                    elif key == ord('c'):
                        self._cycle_model()
                        model_name = self.inference_engine.active_model
                    elif key == ord('p'):
                        logger.info("Stream paused - press any key to continue")
                        cv2.waitKey(0)
        
        finally:
            self.stop_stream()
    
    def _cycle_model(self):
        """Cycle through available models"""
        models = self.inference_engine.get_available_models()
        if len(models) <= 1:
            return
        
        current_model = self.inference_engine.active_model
        try:
            current_idx = models.index(current_model)
            next_idx = (current_idx + 1) % len(models)
            next_model = models[next_idx]
            self.set_model(next_model)
            logger.info(f"Switched to model: {next_model}")
        except (ValueError, IndexError):
            # If current model not in list, use first model
            self.set_model(models[0])
    
    def _draw_results(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Draw AI inference results on frame"""
        if "error" in result:
            cv2.putText(frame, f"Error: {result['error']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Draw model name
        model_name = result.get('model_name', 'Unknown')
        device = result.get('device', 'Unknown')
        cv2.putText(frame, f"Model: {model_name} ({device})", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw inference time
        inference_time = result.get('inference_time_ms', 0)
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw prediction results
        if result.get("prediction_type") == "classification":
            self._draw_classification(frame, result)
        elif result.get("prediction_type") == "detection":
            self._draw_detection(frame, result)
        
        # Draw frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _draw_classification(self, frame: np.ndarray, result: Dict[str, Any]):
        """Draw classification results"""
        top_pred = result.get("top_prediction", {})
        label = top_pred.get("label", f"Class {top_pred.get('index', '?')}")
        confidence = top_pred.get("confidence", 0)
        
        # Main prediction
        text = f"{label} ({confidence:.2%})"
        cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2)
        
        # Top predictions
        top_predictions = result.get("top_5_predictions", [])[:3]
        for i, pred in enumerate(top_predictions):
            y_pos = 130 + i * 25
            label = pred.get("label", f"Class {pred.get('index', '?')}")
            confidence = pred.get("confidence", 0)
            text = f"{i+1}. {label} ({confidence:.1%})"
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
    
    def _draw_detection(self, frame: np.ndarray, result: Dict[str, Any]):
        """Draw object detection results"""
        detections = result.get("detections", [])
        height, width = frame.shape[:2]
        
        for detection in detections:
            bbox = detection["bbox"]
            score = detection["score"]
            label = detection.get("label", f"Object {detection.get('class_id', '?')}")
            
            # Convert normalized coordinates to pixel coordinates
            ymin = int(bbox[0] * height)
            xmin = int(bbox[1] * width)
            ymax = int(bbox[2] * height)
            xmax = int(bbox[3] * width)
            
            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Draw label background
            label_text = f"{label} ({score:.1%})"
            (text_width, text_height), _ = cv2.getTextSize(label_text, 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (xmin, ymin - text_height - 10), 
                         (xmin + text_width, ymin), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label_text, (xmin, ymin - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw detection count
        num_detections = len(detections)
        cv2.putText(frame, f"Detections: {num_detections}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def stop_stream(self):
        """Stop the camera stream"""
        self.is_streaming = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera stream stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get camera and inference statistics"""
        stats = self.inference_engine.get_performance_stats()
        stats.update({
            "camera_id": self.camera_id,
            "total_frames": self.frame_count,
            "streaming": self.is_streaming,
            "results_history_length": len(self.results_history)
        })
        return stats
    
    def save_results_history(self, filename: str = None):
        """Save inference results history to file"""
        if not filename:
            filename = f"camera_results_{int(time.time())}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)
        
        logger.info(f"Results history saved to {filename}")
    
    def take_photo_with_ai(self, model_name: str = "image_classification") -> Dict[str, Any]:
        """
        Convenience method to take a photo and get AI analysis
        
        Returns:
            Complete analysis including image file path and AI results
        """
        timestamp = int(time.time())
        
        # Capture and analyze
        result = self.capture_single_image(model_name, save_image=True)
        
        # Create a comprehensive report
        report = {
            "timestamp": timestamp,
            "camera_id": self.camera_id,
            "model_used": model_name,
            "ai_analysis": result,
            "image_file": result.get("saved_image", "not_saved")
        }
        
        # Save report
        report_file = f"photo_analysis_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Photo analysis saved to {report_file}")
        return report


def demo_camera_vision():
    """Demo function to showcase camera vision capabilities"""
    logger.info("🎥 Coral TPU Camera Vision Demo")
    
    try:
        # Initialize camera vision
        camera = CameraVision()
        
        # List available cameras and models
        cameras = camera.list_cameras()
        models = camera.list_models()
        
        if not cameras:
            logger.error("No cameras found!")
            return
        
        if not models:
            logger.error("No AI models available!")
            return
        
        logger.info(f"Available cameras: {cameras}")
        logger.info(f"Available models: {models}")
        
        # Demo 1: Single photo with AI analysis
        logger.info("\n📸 Demo 1: Taking photo with AI analysis...")
        try:
            # Use first available vision model
            vision_models = [m for m in models if "image" in m or "object" in m]
            model = vision_models[0] if vision_models else models[0]
            
            report = camera.take_photo_with_ai(model)
            logger.success(f"Photo taken and analyzed with {model}")
            
            # Show results
            ai_result = report["ai_analysis"]
            if ai_result.get("prediction_type") == "classification":
                top_pred = ai_result.get("top_prediction", {})
                logger.info(f"🎯 Top prediction: {top_pred.get('label', 'Unknown')} "
                           f"({top_pred.get('confidence', 0):.1%})")
            elif ai_result.get("prediction_type") == "detection":
                num_detections = len(ai_result.get("detections", []))
                logger.info(f"🎯 Found {num_detections} objects")
            
        except Exception as e:
            logger.warning(f"Photo demo failed: {e}")
        
        # Demo 2: Live stream
        logger.info("\n🎥 Demo 2: Starting live camera stream...")
        logger.info("Press 'q' to quit, 's' to save frame, 'c' to change model")
        
        try:
            # Start live stream
            model = vision_models[0] if vision_models else models[0]
            camera.start_live_stream(model, display=True)
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.warning(f"Live stream demo failed: {e}")
        
        # Show statistics
        stats = camera.get_stats()
        logger.info(f"\n📊 Final statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Camera vision demo failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_camera_vision()
        elif sys.argv[1] == "photo":
            # Take a single photo
            camera = CameraVision()
            result = camera.take_photo_with_ai("image_classification")
            print(f"Photo analysis: {result}")
        elif sys.argv[1] == "stream":
            # Start live stream
            camera = CameraVision()
            camera.start_live_stream("image_classification")
    else:
        # Default: run demo
        demo_camera_vision()
