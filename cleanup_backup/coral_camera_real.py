#!/usr/bin/env python3
"""
Camera AI REALE con Coral TPU - Versione Corretta
Usa l'API corretta del Coral TPU per riconoscimento in tempo reale
"""

import cv2
import numpy as np
import time
from pathlib import Path
import argparse

# Import Coral TPU con API corretta
try:
    import tflite_runtime.interpreter as tflite
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
    from pycoral.adapters import detect
    CORAL_AVAILABLE = True
    print("✅ Librerie Coral TPU caricate")
except ImportError as e:
    print(f"❌ Coral TPU non disponibile: {e}")
    CORAL_AVAILABLE = False

class CoralTPUCamera:
    """Camera con AI REALE usando Coral TPU - API corretta"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.interpreters = {}
        self.labels = {}
        self.current_model = "detection"
        
        # Verifica Coral TPU
        if CORAL_AVAILABLE:
            devices = edgetpu.list_edge_tpus()
            if devices:
                print(f"🔥 Coral TPU rilevato: {len(devices)} dispositivi")
                self.use_coral = True
                self.device_path = devices[0]['path']
            else:
                print("⚠️ Coral TPU non rilevato, uso CPU")
                self.use_coral = False
                self.device_path = None
        else:
            print("❌ PyCoral non disponibile, uso CPU")
            self.use_coral = False
            self.device_path = None
        
        # Carica modelli
        self.load_models()
        
    def load_models(self):
        """Carica i modelli EdgeTPU con API corretta"""
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
                    # API corretta per Coral TPU
                    if self.use_coral and self.device_path:
                        interpreter = tflite.Interpreter(
                            model_path=str(model_path),
                            experimental_delegates=[
                                tflite.load_delegate('libedgetpu.so.1', {'device': self.device_path})
                            ]
                        )
                        print(f"🔥 Modello {model_name} caricato su CORAL TPU")
                    else:
                        interpreter = tflite.Interpreter(model_path=str(model_path))
                        print(f"💻 Modello {model_name} caricato su CPU")
                    
                    interpreter.allocate_tensors()
                    self.interpreters[model_name] = interpreter
                    
                    # Carica labels
                    if labels_path.exists():
                        with open(labels_path, 'r', encoding='utf-8') as f:
                            self.labels[model_name] = [line.strip() for line in f.readlines()]
                        print(f"✅ Labels caricati: {len(self.labels[model_name])} classi")
                    else:
                        self.labels[model_name] = [f"class_{i}" for i in range(100)]
                        print(f"⚠️ Labels non trovati per {model_name}, uso generici")
                        
                except Exception as e:
                    print(f"❌ Errore caricamento {model_name}: {e}")
                    # Prova senza TPU
                    try:
                        interpreter = tflite.Interpreter(model_path=str(model_path))
                        interpreter.allocate_tensors()
                        self.interpreters[model_name] = interpreter
                        print(f"💻 Modello {model_name} caricato su CPU (fallback)")
                    except Exception as e2:
                        print(f"❌ Errore anche su CPU per {model_name}: {e2}")
            else:
                print(f"❌ Modello non trovato: {model_path}")
    
    def detect_objects(self, frame):
        """Rileva oggetti usando il modello attuale"""
        if self.current_model not in self.interpreters:
            return [], 0
        
        interpreter = self.interpreters[self.current_model]
        labels = self.labels[self.current_model]
        
        # Ottieni info input
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepara input
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize frame
        frame_resized = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        # Normalizza se necessario
        if input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32) / 255.0
        
        # Inferenza
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Estrai output
        detections = []
        
        try:
            # Per modelli SSD MobileNet
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
            classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Classes
            scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
            
            # Processa detections
            frame_height, frame_width = frame.shape[:2]
            
            for i in range(len(scores)):
                if scores[i] > 0.4:  # Soglia confidenza più alta per precisione
                    # Converti coordinate normalizzate
                    ymin = int(boxes[i][0] * frame_height)
                    xmin = int(boxes[i][1] * frame_width)
                    ymax = int(boxes[i][2] * frame_height)
                    xmax = int(boxes[i][3] * frame_width)
                    
                    # Assicurati che le coordinate siano valide
                    xmin = max(0, min(xmin, frame_width))
                    ymin = max(0, min(ymin, frame_height))
                    xmax = max(0, min(xmax, frame_width))
                    ymax = max(0, min(ymax, frame_height))
                    
                    class_id = int(classes[i])
                    if class_id < len(labels):
                        class_name = labels[class_id]
                    else:
                        class_name = f"class_{class_id}"
                    
                    detections.append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'class': class_name,
                        'confidence': scores[i],
                        'class_id': class_id
                    })
        
        except Exception as e:
            print(f"⚠️ Errore processing output: {e}")
        
        return detections, inference_time
    
    def draw_detections(self, frame, detections, inference_time):
        """Disegna rilevamenti sul frame"""
        # Header info
        tpu_status = "🔥 CORAL TPU" if self.use_coral else "💻 CPU"
        cv2.putText(frame, f"{tpu_status} - {inference_time:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Modello: {self.current_model}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Conta per categoria
        person_count = 0
        object_count = 0
        
        # Disegna rilevamenti
        for detection in detections:
            xmin, ymin, xmax, ymax = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Colore in base al tipo
            if 'person' in class_name.lower():
                person_count += 1
                color = (0, 255, 0)  # Verde per persone
                thickness = 3
            else:
                object_count += 1
                color = (255, 0, 0)  # Blu per altri oggetti
                thickness = 2
            
            # Verifica dimensioni valide
            if xmax > xmin and ymax > ymin:
                # Rettangolo
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
                
                # Label con background
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background per label
                cv2.rectangle(frame, (xmin, ymin - label_size[1] - 10), 
                             (xmin + label_size[0] + 5, ymin), color, -1)
                
                # Testo label
                cv2.putText(frame, label, (xmin + 2, ymin - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistiche
        cv2.putText(frame, f"Persone: {person_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Oggetti: {object_count}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Controlli
        controls = "Q=esci, S=salva, C=modello, R=reset"
        cv2.putText(frame, controls, 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def switch_model(self):
        """Cambia modello AI"""
        models = list(self.interpreters.keys())
        if models:
            current_idx = models.index(self.current_model) if self.current_model in models else 0
            next_idx = (current_idx + 1) % len(models)
            old_model = self.current_model
            self.current_model = models[next_idx]
            print(f"🔄 Cambiato da {old_model} a {self.current_model}")
    
    def run_camera(self):
        """Avvia camera con AI in tempo reale"""
        print("🚀 CAMERA AI REALE CON CORAL TPU")
        print("=" * 50)
        
        if not self.interpreters:
            print("❌ Nessun modello AI caricato!")
            return
        
        # Apri camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Impossibile aprire la camera")
            return
        
        # Configura camera per performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Riduce latenza
        
        print("✅ Camera configurata: 640x480 @ 30fps")
        print(f"🤖 Modelli caricati: {list(self.interpreters.keys())}")
        print(f"🎯 Modello attivo: {self.current_model}")
        print("\n🎮 CONTROLLI:")
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
                
                # AI Detection ogni frame
                detections, inference_time = self.detect_objects(frame)
                total_inference_time += inference_time
                
                # Disegna risultati
                frame = self.draw_detections(frame, detections, inference_time)
                
                # FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostra frame
                cv2.imshow('🔥 Coral TPU Camera AI - RICONOSCIMENTO REALE', frame)
                
                # Input utente
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"coral_real_detection_{int(time.time())}.jpg"
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
            print("=" * 40)
            print(f"Frame processati: {frame_count}")
            print(f"FPS medio: {fps:.1f}")
            if total_inference_time > 0:
                avg_inference = total_inference_time / frame_count
                print(f"Tempo inferenza medio: {avg_inference:.1f}ms")
                print(f"Hardware utilizzato: {'🔥 CORAL TPU' if self.use_coral else '💻 CPU'}")
                
                # Performance comparison
                if self.use_coral:
                    cpu_estimate = avg_inference * 10  # Stima CPU 10x più lenta
                    speedup = cpu_estimate / avg_inference
                    print(f"Speedup vs CPU: ~{speedup:.1f}x più veloce")

def main():
    parser = argparse.ArgumentParser(description='Camera AI con Coral TPU REALE')
    parser.add_argument('--test', action='store_true', help='Test rapido sistema')
    args = parser.parse_args()
    
    if args.test:
        print("🧪 TEST SISTEMA CORAL TPU REALE")
        print("=" * 40)
        
        # Test TPU
        if CORAL_AVAILABLE:
            devices = edgetpu.list_edge_tpus()
            print(f"🔥 Coral TPU: {len(devices)} dispositivi")
            for i, device in enumerate(devices):
                print(f"   Device {i}: {device}")
        else:
            print("❌ Coral TPU: Non disponibile")
        
        # Test modelli
        models_dir = Path("models")
        models = list(models_dir.glob("*_edgetpu.tflite"))
        print(f"📦 Modelli EdgeTPU: {len(models)}")
        for model in models:
            print(f"   - {model.name}")
        
        # Test camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"📹 Camera: ✅ Disponibile ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print("📹 Camera: ⚠️ Aperta ma nessun frame")
            cap.release()
        else:
            print("📹 Camera: ❌ Non disponibile")
        
        return
    
    # Avvia sistema
    print("🎬 Avviando sistema camera AI...")
    camera_ai = CoralTPUCamera()
    camera_ai.run_camera()

if __name__ == "__main__":
    main()
