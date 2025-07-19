#!/usr/bin/env python3
"""
Camera AI REALE con Coral TPU
Questo script usa REALMENTE il Coral TPU per riconoscimento persone/oggetti
"""

import cv2
import numpy as np
import time
from pathlib import Path
import argparse

# Import Coral TPU
try:
    import tflite_runtime.interpreter as tflite
    from pycoral.utils import edgetpu
    from pycoral.utils import dataset
    from pycoral.adapters import common
    from pycoral.adapters import detect
    CORAL_AVAILABLE = True
    print("✅ Librerie Coral TPU caricate")
except ImportError as e:
    print(f"❌ Coral TPU non disponibile: {e}")
    CORAL_AVAILABLE = False

class RealCoralCamera:
    """Camera con AI REALE usando Coral TPU"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.interpreters = {}
        self.labels = {}
        self.current_model = "detection"
        
        # Verifica Coral TPU
        if CORAL_AVAILABLE:
            devices = edgetpu.list_edge_tpus()
            if devices:
                print(f"✅ Coral TPU rilevato: {len(devices)} dispositivi")
                self.use_coral = True
            else:
                print("⚠️ Coral TPU non rilevato, uso CPU")
                self.use_coral = False
        else:
            print("❌ PyCoral non disponibile, uso CPU")
            self.use_coral = False
        
        # Carica modelli
        self.load_models()
        
    def load_models(self):
        """Carica i modelli EdgeTPU"""
        models_config = {
            "detection": {
                "model": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
                "labels": "coco_labels.txt",
                "type": "detection"
            },
            "face": {
                "model": "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite", 
                "labels": "face_labels.txt",
                "type": "detection"
            }
        }
        
        for model_name, config in models_config.items():
            model_path = self.models_dir / config["model"]
            labels_path = self.models_dir / config["labels"]
            
            if model_path.exists():
                try:
                    # Carica modello EdgeTPU se disponibile
                    if self.use_coral:
                        interpreter = tflite.Interpreter(
                            model_path=str(model_path),
                            experimental_delegates=[edgetpu.make_edge_tpu_delegate()]
                        )
                        print(f"✅ Modello {model_name} caricato su CORAL TPU")
                    else:
                        interpreter = tflite.Interpreter(model_path=str(model_path))
                        print(f"✅ Modello {model_name} caricato su CPU")
                    
                    interpreter.allocate_tensors()
                    self.interpreters[model_name] = interpreter
                    
                    # Carica labels
                    if labels_path.exists():
                        with open(labels_path, 'r', encoding='utf-8') as f:
                            self.labels[model_name] = [line.strip() for line in f.readlines()]
                    else:
                        self.labels[model_name] = [f"class_{i}" for i in range(100)]
                        
                except Exception as e:
                    print(f"❌ Errore caricamento {model_name}: {e}")
            else:
                print(f"❌ Modello non trovato: {model_path}")
    
    def preprocess_frame(self, frame, size=(300, 300)):
        """Preprocessa frame per inferenza"""
        # Resize mantenendo aspect ratio
        height, width = frame.shape[:2]
        if height != size[0] or width != size[1]:
            frame_resized = cv2.resize(frame, size)
        else:
            frame_resized = frame
        
        # Converti in formato richiesto dal modello
        input_data = np.expand_dims(frame_resized, axis=0)
        return input_data, frame_resized
    
    def detect_objects(self, frame):
        """Rileva oggetti usando Coral TPU"""
        if self.current_model not in self.interpreters:
            return []
        
        interpreter = self.interpreters[self.current_model]
        labels = self.labels[self.current_model]
        
        # Preprocessa
        input_data, processed_frame = self.preprocess_frame(frame, (300, 300))
        
        # Inferenza
        start_time = time.time()
        common.set_input(interpreter, input_data)
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Estrai risultati
        boxes = common.output_tensor(interpreter, 0)
        classes = common.output_tensor(interpreter, 1)
        scores = common.output_tensor(interpreter, 2)
        
        # Filtra risultati
        detections = []
        height, width = frame.shape[:2]
        
        for i in range(len(scores)):
            if scores[i] > 0.3:  # Soglia confidenza
                # Converti coordinate normalizzate in pixel
                ymin = int(boxes[i][0] * height)
                xmin = int(boxes[i][1] * width)
                ymax = int(boxes[i][2] * height)
                xmax = int(boxes[i][3] * width)
                
                class_id = int(classes[i])
                class_name = labels[class_id] if class_id < len(labels) else f"class_{class_id}"
                
                detections.append({
                    'bbox': [xmin, ymin, xmax, ymax],
                    'class': class_name,
                    'confidence': scores[i],
                    'class_id': class_id
                })
        
        return detections, inference_time
    
    def draw_detections(self, frame, detections, inference_time):
        """Disegna rilevamenti sul frame"""
        # Info TPU
        tpu_status = "🔥 CORAL TPU" if self.use_coral else "💻 CPU"
        cv2.putText(frame, f"{tpu_status} - {inference_time:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Modello: {self.current_model}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Disegna rilevamenti
        person_count = 0
        for detection in detections:
            xmin, ymin, xmax, ymax = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Conta persone
            if 'person' in class_name.lower():
                person_count += 1
                color = (0, 255, 0)  # Verde per persone
            else:
                color = (255, 0, 0)  # Blu per altri oggetti
            
            # Rettangolo
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (xmin, ymin - label_size[1] - 10), 
                         (xmin + label_size[0], ymin), color, -1)
            cv2.putText(frame, label, (xmin, ymin - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Statistiche
        cv2.putText(frame, f"Persone rilevate: {person_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Totale oggetti: {len(detections)}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Controlli
        cv2.putText(frame, "Controlli: Q=esci, S=salva, C=cambia modello", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def switch_model(self):
        """Cambia modello AI"""
        models = list(self.interpreters.keys())
        if models:
            current_idx = models.index(self.current_model) if self.current_model in models else 0
            next_idx = (current_idx + 1) % len(models)
            self.current_model = models[next_idx]
            print(f"🔄 Cambiato a modello: {self.current_model}")
    
    def run_camera(self):
        """Avvia camera con AI in tempo reale"""
        print("🚀 AVVIANDO CAMERA AI REALE CON CORAL TPU")
        print("=" * 50)
        
        # Apri camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Impossibile aprire la camera")
            return
        
        # Configura camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera aperta - Risoluzione: 640x480")
        print("🎯 Iniziando rilevamento AI...")
        print("\nControlli:")
        print("  Q = Esci")
        print("  S = Salva screenshot")
        print("  C = Cambia modello AI")
        print("  R = Reset statistiche")
        
        frame_count = 0
        total_inference_time = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Errore lettura camera")
                    break
                
                frame_count += 1
                
                # AI Detection
                if self.interpreters:
                    detections, inference_time = self.detect_objects(frame)
                    total_inference_time += inference_time
                    
                    # Disegna risultati
                    frame = self.draw_detections(frame, detections, inference_time)
                else:
                    cv2.putText(frame, "❌ Nessun modello AI caricato", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostra frame
                cv2.imshow('🔥 Coral TPU Camera AI - REALE', frame)
                
                # Input utente
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"coral_tpu_detection_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot salvato: {filename}")
                elif key == ord('c'):
                    self.switch_model()
                elif key == ord('r'):
                    frame_count = 0
                    total_inference_time = 0
                    start_time = time.time()
                    print("🔄 Statistiche resettate")
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione utente")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Statistiche finali
            print("\n📊 STATISTICHE FINALI")
            print("=" * 30)
            print(f"Frame processati: {frame_count}")
            print(f"FPS medio: {fps:.1f}")
            if total_inference_time > 0:
                avg_inference = total_inference_time / frame_count
                print(f"Tempo inferenza medio: {avg_inference:.1f}ms")
                print(f"Hardware utilizzato: {'🔥 CORAL TPU' if self.use_coral else '💻 CPU'}")

def main():
    parser = argparse.ArgumentParser(description='Camera AI con Coral TPU REALE')
    parser.add_argument('--test', action='store_true', help='Test rapido sistema')
    args = parser.parse_args()
    
    if args.test:
        print("🧪 TEST SISTEMA CORAL TPU")
        print("=" * 30)
        
        # Test TPU
        if CORAL_AVAILABLE:
            devices = edgetpu.list_edge_tpus()
            print(f"Coral TPU: {len(devices)} dispositivi")
        else:
            print("Coral TPU: Non disponibile")
        
        # Test modelli
        models_dir = Path("models")
        models = list(models_dir.glob("*_edgetpu.tflite"))
        print(f"Modelli EdgeTPU: {len(models)}")
        for model in models:
            print(f"  - {model.name}")
        
        # Test camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Camera: ✅ Disponibile")
            cap.release()
        else:
            print("Camera: ❌ Non disponibile")
        
        return
    
    # Avvia sistema
    camera_ai = RealCoralCamera()
    camera_ai.run_camera()

if __name__ == "__main__":
    main()
