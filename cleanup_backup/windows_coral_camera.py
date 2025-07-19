#!/usr/bin/env python3
"""
Camera AI WINDOWS con Coral TPU
Versione specifica per Windows con delegate corretto
"""

import cv2
import numpy as np
import time
from pathlib import Path
import argparse
import platform

# Import Coral TPU
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

def get_edgetpu_delegate():
    """Ottieni il delegate EdgeTPU corretto per la piattaforma"""
    system = platform.system().lower()
    
    if system == "windows":
        # Su Windows, prova diverse opzioni
        delegate_options = [
            "edgetpu.dll",
            "libedgetpu.dll", 
            "C:\\Program Files\\Google\\EdgeTPU\\edgetpu.dll"
        ]
    elif system == "linux":
        delegate_options = ["libedgetpu.so.1"]
    else:  # macOS
        delegate_options = ["libedgetpu.dylib"]
    
    for delegate_lib in delegate_options:
        try:
            delegate = tflite.load_delegate(delegate_lib)
            print(f"✅ EdgeTPU delegate caricato: {delegate_lib}")
            return delegate
        except Exception as e:
            print(f"⚠️ Tentativo {delegate_lib} fallito: {e}")
    
    return None

class WindowsCoralCamera:
    """Camera AI per Windows con Coral TPU"""
    
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
                self.edgetpu_delegate = get_edgetpu_delegate()
            else:
                print("⚠️ Coral TPU non rilevato")
                self.use_coral = False
                self.edgetpu_delegate = None
        else:
            print("❌ PyCoral non disponibile")
            self.use_coral = False
            self.edgetpu_delegate = None
        
        # Carica modelli
        self.load_models()
        
    def load_models(self):
        """Carica modelli con fallback intelligente"""
        models_config = {
            "detection": {
                "model_edgetpu": "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
                "model_cpu": "ssd_mobilenet_v2_coco_quant_postprocess.tflite",  # Se disponibile
                "labels": "coco_labels.txt",
                "type": "detection"
            },
            "face": {
                "model_edgetpu": "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
                "model_cpu": "ssd_mobilenet_v2_face_quant_postprocess.tflite",  # Se disponibile
                "labels": "face_labels.txt",
                "type": "detection"
            }
        }
        
        for model_name, config in models_config.items():
            # Prova prima EdgeTPU, poi CPU
            model_loaded = False
            
            # Tentativo 1: EdgeTPU
            if self.use_coral and self.edgetpu_delegate:
                model_path = self.models_dir / config["model_edgetpu"]
                if model_path.exists():
                    try:
                        interpreter = tflite.Interpreter(
                            model_path=str(model_path),
                            experimental_delegates=[self.edgetpu_delegate]
                        )
                        interpreter.allocate_tensors()
                        self.interpreters[model_name] = interpreter
                        print(f"🔥 {model_name}: Caricato su CORAL TPU")
                        model_loaded = True
                    except Exception as e:
                        print(f"⚠️ EdgeTPU fallito per {model_name}: {e}")
            
            # Tentativo 2: CPU con modello EdgeTPU (molti funzionano anche su CPU)
            if not model_loaded:
                model_path = self.models_dir / config["model_edgetpu"]
                if model_path.exists():
                    try:
                        interpreter = tflite.Interpreter(model_path=str(model_path))
                        interpreter.allocate_tensors()
                        self.interpreters[model_name] = interpreter
                        print(f"💻 {model_name}: Caricato su CPU (da EdgeTPU)")
                        model_loaded = True
                    except Exception as e:
                        print(f"⚠️ CPU con EdgeTPU fallito per {model_name}: {e}")
            
            # Tentativo 3: CPU con modello standard (se disponibile)
            if not model_loaded and "model_cpu" in config:
                model_path = self.models_dir / config["model_cpu"]
                if model_path.exists():
                    try:
                        interpreter = tflite.Interpreter(model_path=str(model_path))
                        interpreter.allocate_tensors()
                        self.interpreters[model_name] = interpreter
                        print(f"💻 {model_name}: Caricato su CPU (standard)")
                        model_loaded = True
                    except Exception as e:
                        print(f"❌ CPU standard fallito per {model_name}: {e}")
            
            # Carica labels se modello caricato
            if model_loaded:
                labels_path = self.models_dir / config["labels"]
                if labels_path.exists():
                    try:
                        with open(labels_path, 'r', encoding='utf-8') as f:
                            labels = [line.strip() for line in f.readlines()]
                            # Filtra labels vuoti
                            self.labels[model_name] = [l for l in labels if l]
                        print(f"✅ Labels {model_name}: {len(self.labels[model_name])} classi")
                    except Exception as e:
                        print(f"⚠️ Errore labels {model_name}: {e}")
                        self.labels[model_name] = [f"class_{i}" for i in range(90)]
                else:
                    print(f"⚠️ Labels non trovati per {model_name}")
                    self.labels[model_name] = [f"class_{i}" for i in range(90)]
    
    def detect_objects(self, frame):
        """Rilevamento oggetti ottimizzato"""
        if self.current_model not in self.interpreters:
            return [], 0
        
        interpreter = self.interpreters[self.current_model]
        labels = self.labels[self.current_model]
        
        # Input details
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        
        # Prepara input (assumendo input 300x300 per SSD)
        if len(input_shape) == 4:
            _, height, width, _ = input_shape
        else:
            height, width = 300, 300
        
        # Resize frame
        frame_resized = cv2.resize(frame, (width, height))
        
        # Prepara tensor di input
        if input_details[0]['dtype'] == np.uint8:
            input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
        else:
            input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0
        
        # Inferenza
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        
        # Estrai output
        output_details = interpreter.get_output_details()
        detections = []
        
        try:
            # Output standard per SSD MobileNet
            if len(output_details) >= 4:
                # Formato: [locations, classes, scores, num_detections]
                locations = interpreter.get_tensor(output_details[0]['index'])
                classes = interpreter.get_tensor(output_details[1]['index'])
                scores = interpreter.get_tensor(output_details[2]['index'])
                num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
            elif len(output_details) >= 3:
                # Formato alternativo
                locations = interpreter.get_tensor(output_details[0]['index'])
                classes = interpreter.get_tensor(output_details[1]['index'])
                scores = interpreter.get_tensor(output_details[2]['index'])
                num_detections = len(scores[0])
            else:
                return detections, inference_time
            
            # Processa detections
            frame_height, frame_width = frame.shape[:2]
            
            for i in range(min(num_detections, len(scores[0]))):
                score = scores[0][i]
                if score > 0.3:  # Soglia confidenza
                    # Coordinate normalizzate
                    ymin, xmin, ymax, xmax = locations[0][i]
                    
                    # Converti a pixel
                    xmin = int(xmin * frame_width)
                    ymin = int(ymin * frame_height)
                    xmax = int(xmax * frame_width)
                    ymax = int(ymax * frame_height)
                    
                    # Clamp coordinates
                    xmin = max(0, min(xmin, frame_width - 1))
                    ymin = max(0, min(ymin, frame_height - 1))
                    xmax = max(xmin + 1, min(xmax, frame_width))
                    ymax = max(ymin + 1, min(ymax, frame_height))
                    
                    # Classe
                    class_id = int(classes[0][i])
                    if class_id < len(labels):
                        class_name = labels[class_id]
                    else:
                        class_name = f"unknown_{class_id}"
                    
                    detections.append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'class': class_name,
                        'confidence': float(score),
                        'class_id': class_id
                    })
        
        except Exception as e:
            print(f"⚠️ Errore processing detections: {e}")
        
        return detections, inference_time
    
    def draw_detections(self, frame, detections, inference_time):
        """Disegna rilevamenti con stile professionale"""
        # Header
        hw_type = "🔥 CORAL TPU" if (self.use_coral and self.edgetpu_delegate) else "💻 CPU"
        cv2.putText(frame, f"{hw_type} | {inference_time:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Modello: {self.current_model}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Contatori
        people = sum(1 for d in detections if 'person' in d['class'].lower())
        objects = len(detections) - people
        
        # Disegna detections
        for detection in detections:
            xmin, ymin, xmax, ymax = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Colori per categoria
            if 'person' in class_name.lower():
                color = (0, 255, 0)  # Verde per persone
                thickness = 3
            elif class_name.lower() in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                color = (255, 165, 0)  # Arancione per veicoli
                thickness = 2
            elif class_name.lower() in ['cat', 'dog', 'bird', 'horse']:
                color = (255, 0, 255)  # Magenta per animali
                thickness = 2
            else:
                color = (255, 0, 0)  # Rosso per oggetti generici
                thickness = 2
            
            # Rettangolo
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
            
            # Label con sfondo
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background label
            cv2.rectangle(frame, 
                         (xmin, ymin - label_height - baseline - 5),
                         (xmin + label_width + 5, ymin),
                         color, -1)
            
            # Testo label
            cv2.putText(frame, label, 
                       (xmin + 2, ymin - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistiche
        cv2.putText(frame, f"👥 Persone: {people}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"📦 Oggetti: {objects}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Controlli
        cv2.putText(frame, "Q=esci | S=salva | C=modello | R=reset", 
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def switch_model(self):
        """Cambia modello attivo"""
        models = list(self.interpreters.keys())
        if len(models) > 1:
            current_idx = models.index(self.current_model) if self.current_model in models else 0
            next_idx = (current_idx + 1) % len(models)
            old_model = self.current_model
            self.current_model = models[next_idx]
            print(f"🔄 Modello: {old_model} → {self.current_model}")
        else:
            print(f"ℹ️ Solo un modello disponibile: {self.current_model}")
    
    def run_camera(self):
        """Esegui camera AI con riconoscimento avanzato"""
        print("🎬 CAMERA AI WINDOWS + CORAL TPU")
        print("=" * 50)
        
        if not self.interpreters:
            print("❌ Nessun modello AI disponibile!")
            print("💡 Controlla che i file .tflite siano nella cartella models/")
            return
        
        # Apri camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        # Configura camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera: {actual_width}x{actual_height} @ {actual_fps}fps")
        print(f"🤖 Modelli: {list(self.interpreters.keys())}")
        print(f"🎯 Attivo: {self.current_model}")
        print(f"⚡ Hardware: {'🔥 CORAL TPU' if (self.use_coral and self.edgetpu_delegate) else '💻 CPU'}")
        
        frame_count = 0
        total_inference_time = 0
        start_time = time.time()
        
        print("\n🎮 CONTROLLI ATTIVI:")
        print("  Q = Esci")
        print("  S = Salva screenshot") 
        print("  C = Cambia modello")
        print("  R = Reset statistiche")
        print("\n🔥 SISTEMA PRONTO - Iniziando riconoscimento...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # AI Detection
                detections, inference_time = self.detect_objects(frame)
                total_inference_time += inference_time
                
                # Disegna overlay
                frame = self.draw_detections(frame, detections, inference_time)
                
                # FPS counter
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostra frame
                cv2.imshow('🔥 Windows Coral TPU AI Camera', frame)
                
                # Input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"coral_windows_ai_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Salvato: {filename}")
                elif key == ord('c'):
                    self.switch_model()
                elif key == ord('r'):
                    frame_count = 0
                    total_inference_time = 0
                    start_time = time.time()
                    print("🔄 Statistiche resettate")
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione da tastiera")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Report finale
            print("\n📊 REPORT FINALE")
            print("=" * 40)
            print(f"📈 Frame processati: {frame_count}")
            print(f"🎯 FPS medio: {fps:.1f}")
            if frame_count > 0:
                avg_inference = total_inference_time / frame_count
                print(f"⚡ Inferenza media: {avg_inference:.1f}ms")
                print(f"🔥 Hardware: {'CORAL TPU' if (self.use_coral and self.edgetpu_delegate) else 'CPU'}")
                
                if self.use_coral and self.edgetpu_delegate:
                    print(f"🚀 Performance: ~10x più veloce di CPU standard")

def main():
    parser = argparse.ArgumentParser(description='Camera AI Windows con Coral TPU')
    parser.add_argument('--test', action='store_true', help='Test sistema')
    args = parser.parse_args()
    
    if args.test:
        print("🧪 TEST SISTEMA WINDOWS + CORAL TPU")
        print("=" * 45)
        print(f"🖥️ OS: {platform.system()} {platform.release()}")
        
        # Test Coral TPU
        if CORAL_AVAILABLE:
            devices = edgetpu.list_edge_tpus()
            print(f"🔥 Coral TPU: {len(devices)} dispositivi")
            for i, device in enumerate(devices):
                print(f"   Device {i}: {device}")
            
            # Test delegate
            delegate = get_edgetpu_delegate()
            if delegate:
                print("✅ EdgeTPU delegate: Disponibile")
            else:
                print("❌ EdgeTPU delegate: Non disponibile")
        else:
            print("❌ PyCoral: Non installato")
        
        # Test modelli
        models_dir = Path("models")
        if models_dir.exists():
            models = list(models_dir.glob("*.tflite"))
            print(f"📦 Modelli: {len(models)} trovati")
            for model in models:
                print(f"   - {model.name}")
        else:
            print("❌ Cartella models non trovata")
        
        # Test camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"📹 Camera: ✅ OK ({w}x{h})")
            else:
                print("📹 Camera: ⚠️ Aperta ma no frame")
            cap.release()
        else:
            print("📹 Camera: ❌ Non disponibile")
        
        return
    
    # Avvia sistema
    print("🎬 Inizializzando Windows Coral Camera AI...")
    camera = WindowsCoralCamera()
    camera.run_camera()

if __name__ == "__main__":
    main()
