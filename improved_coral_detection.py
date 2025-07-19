#!/usr/bin/env python3
"""
Sistema MIGLIORATO Coral TPU - Detection Oggetti Ottimizzata
Migliori performance per rilevamento oggetti con controlli dinamici
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import argparse
from datetime import datetime
import pickle

# Import Coral TPU
try:
    import tflite_runtime.interpreter as tflite
    from pycoral.utils import edgetpu
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False

def get_edgetpu_delegate():
    """Ottieni il delegate EdgeTPU per Windows"""
    try:
        delegate = tflite.load_delegate("edgetpu.dll")
        return delegate
    except Exception as e:
        print(f"⚠️ EdgeTPU delegate fallito: {e}")
        return None

class ImprovedCoralDetection:
    """Sistema Coral TPU ottimizzato per detection oggetti"""
    
    def __init__(self):
        self.models_dir = Path("models")
        
        # Coral TPU setup
        self.coral_interpreter = None
        self.coral_labels = []
        self.use_coral = False
        
        # Configurazione ottimizzata
        self.config = {
            "coral_confidence_threshold": 0.25,  # Soglia base bassa
            "nms_threshold": 0.5,  # Non-Maximum Suppression
            "high_confidence_objects": ["person", "car", "dog", "cat", "bottle", "phone", "laptop"],
            "medium_confidence_objects": ["chair", "table", "tv", "keyboard", "mouse", "book"],
            "adaptive_thresholds": True,
            "frame_skip": 1,  # Processa ogni frame
            "input_size": (300, 300),  # Risoluzione input
            "detection_zones": True  # Zone di interesse
        }
        
        # Soglie dinamiche per classe
        self.class_thresholds = {
            "person": 0.5,
            "car": 0.4,
            "bicycle": 0.4,
            "motorcycle": 0.4,
            "bus": 0.4,
            "truck": 0.4,
            "dog": 0.5,
            "cat": 0.5,
            "bottle": 0.3,
            "cup": 0.3,
            "phone": 0.4,
            "laptop": 0.4,
            "mouse": 0.3,
            "keyboard": 0.3,
            "book": 0.3,
            "chair": 0.25,
            "table": 0.25,
            "tv": 0.4,
            "remote": 0.3,
            "clock": 0.3
        }
        
        # Statistiche detection
        self.detection_stats = {}
        self.frame_count = 0
        
        # Inizializza
        self.load_coral_tpu()
    
    def load_coral_tpu(self):
        """Carica modello Coral TPU ottimizzato"""
        if not CORAL_AVAILABLE:
            print("⚠️ PyCoral non disponibile")
            return
        
        # Verifica dispositivi TPU
        devices = edgetpu.list_edge_tpus()
        if not devices:
            print("⚠️ Coral TPU non rilevato")
            return
        
        # Carica modello detection
        model_path = self.models_dir / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
        labels_path = self.models_dir / "coco_labels.txt"
        
        if not model_path.exists():
            print(f"❌ Modello non trovato: {model_path}")
            return
        
        try:
            # Carica con delegate EdgeTPU
            delegate = get_edgetpu_delegate()
            if delegate:
                self.coral_interpreter = tflite.Interpreter(
                    model_path=str(model_path),
                    experimental_delegates=[delegate]
                )
                print("🔥 Modello Coral TPU caricato con EdgeTPU")
            else:
                self.coral_interpreter = tflite.Interpreter(model_path=str(model_path))
                print("💻 Modello Coral TPU caricato su CPU")
            
            self.coral_interpreter.allocate_tensors()
            self.use_coral = True
            
            # Carica labels
            if labels_path.exists():
                with open(labels_path, 'r', encoding='utf-8') as f:
                    self.coral_labels = [line.strip() for line in f.readlines()]
                print(f"✅ Labels caricati: {len(self.coral_labels)} classi")
            
            # Inizializza statistiche
            for label in self.coral_labels:
                self.detection_stats[label] = {
                    'count': 0,
                    'avg_confidence': 0.0,
                    'last_seen': None
                }
            
        except Exception as e:
            print(f"❌ Errore caricamento Coral TPU: {e}")
    
    def get_dynamic_threshold(self, class_name):
        """Ottieni soglia dinamica per classe"""
        if not self.config["adaptive_thresholds"]:
            return self.config["coral_confidence_threshold"]
        
        # Soglia specifica per classe
        if class_name in self.class_thresholds:
            return self.class_thresholds[class_name]
        
        # Soglia base
        return self.config["coral_confidence_threshold"]
    
    def apply_nms(self, detections):
        """Applica Non-Maximum Suppression"""
        if not detections:
            return detections
        
        # Raggruppa per classe
        class_groups = {}
        for det in detections:
            class_name = det['class']
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)
        
        filtered_detections = []
        
        for class_name, class_dets in class_groups.items():
            if len(class_dets) <= 1:
                filtered_detections.extend(class_dets)
                continue
            
            # Prepara dati per NMS
            boxes = []
            scores = []
            
            for det in class_dets:
                x1, y1, x2, y2 = det['bbox']
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(det['confidence'])
            
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            
            # Applica NMS OpenCV
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                scores.tolist(), 
                self.get_dynamic_threshold(class_name),
                self.config["nms_threshold"]
            )
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    filtered_detections.append(class_dets[idx])
        
        return filtered_detections
    
    def detect_objects_improved(self, frame):
        """Detection oggetti migliorata"""
        if not self.use_coral or not self.coral_interpreter:
            return []
        
        try:
            # Prepara input
            input_details = self.coral_interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # Resize frame mantenendo aspect ratio
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if aspect_ratio > 1:
                new_width = width
                new_height = int(width / aspect_ratio)
            else:
                new_height = height
                new_width = int(height * aspect_ratio)
            
            # Resize e pad
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            # Crea canvas con padding
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            y_offset = (height - new_height) // 2
            x_offset = (width - new_width) // 2
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame_resized
            
            input_data = np.expand_dims(canvas, axis=0).astype(np.uint8)
            
            # Inferenza
            start_time = time.time()
            self.coral_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.coral_interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000
            
            # Output
            output_details = self.coral_interpreter.get_output_details()
            locations = self.coral_interpreter.get_tensor(output_details[0]['index'])
            classes = self.coral_interpreter.get_tensor(output_details[1]['index'])
            scores = self.coral_interpreter.get_tensor(output_details[2]['index'])
            
            # Processa detections
            detections = []
            
            for i in range(len(scores[0])):
                score = scores[0][i]
                class_id = int(classes[0][i])
                
                if class_id >= len(self.coral_labels):
                    continue
                
                class_name = self.coral_labels[class_id]
                threshold = self.get_dynamic_threshold(class_name)
                
                if score > threshold:
                    # Coordinate con compensazione padding
                    ymin, xmin, ymax, xmax = locations[0][i]
                    
                    # Converti da coordinate normalizzate
                    xmin_canvas = int(xmin * width)
                    ymin_canvas = int(ymin * height)
                    xmax_canvas = int(xmax * width)
                    ymax_canvas = int(ymax * height)
                    
                    # Rimuovi padding offset
                    xmin_frame = max(0, int((xmin_canvas - x_offset) * frame_width / new_width))
                    ymin_frame = max(0, int((ymin_canvas - y_offset) * frame_height / new_height))
                    xmax_frame = min(frame_width, int((xmax_canvas - x_offset) * frame_width / new_width))
                    ymax_frame = min(frame_height, int((ymax_canvas - y_offset) * frame_height / new_height))
                    
                    # Valida bounding box
                    if xmax_frame > xmin_frame and ymax_frame > ymin_frame:
                        detections.append({
                            'bbox': [xmin_frame, ymin_frame, xmax_frame, ymax_frame],
                            'class': class_name,
                            'confidence': float(score),
                            'source': 'coral_improved',
                            'class_id': class_id
                        })
                        
                        # Aggiorna statistiche
                        self.detection_stats[class_name]['count'] += 1
                        self.detection_stats[class_name]['last_seen'] = datetime.now().isoformat()
                        
                        # Media mobile confidenza
                        current_avg = self.detection_stats[class_name]['avg_confidence']
                        count = self.detection_stats[class_name]['count']
                        new_avg = (current_avg * (count - 1) + score) / count
                        self.detection_stats[class_name]['avg_confidence'] = new_avg
            
            # Applica NMS
            if self.config.get("nms_threshold", 0) > 0:
                detections = self.apply_nms(detections)
            
            return detections, inference_time
        
        except Exception as e:
            print(f"⚠️ Errore detection migliorata: {e}")
            return [], 0
    
    def get_detection_color(self, class_name, confidence):
        """Ottieni colore basato su classe e confidenza"""
        # Colori specifici per categorie
        if class_name == "person":
            return (255, 165, 0)  # Arancione
        elif class_name in ["car", "bicycle", "motorcycle", "bus", "truck"]:
            return (0, 255, 255)  # Ciano per veicoli
        elif class_name in ["dog", "cat", "bird", "horse"]:
            return (255, 0, 255)  # Magenta per animali
        elif class_name in ["bottle", "cup", "wine glass", "fork", "knife", "spoon"]:
            return (0, 255, 0)  # Verde per oggetti tavola
        elif class_name in ["phone", "laptop", "mouse", "keyboard", "tv", "remote"]:
            return (255, 255, 0)  # Giallo per elettronica
        elif class_name in ["book", "clock", "chair", "table"]:
            return (128, 255, 128)  # Verde chiaro per casa
        else:
            # Colore basato su confidenza
            if confidence > 0.8:
                return (0, 255, 0)  # Verde alta confidenza
            elif confidence > 0.5:
                return (0, 255, 255)  # Ciano media confidenza
            else:
                return (0, 0, 255)  # Rosso bassa confidenza
    
    def run_improved_detection(self):
        """Detection live migliorata"""
        print("🚀 CORAL TPU DETECTION MIGLIORATA")
        print("=" * 50)
        print(f"🔥 Coral TPU: {'Attivo' if self.use_coral else 'Non disponibile'}")
        print(f"🎯 Classi supportate: {len(self.coral_labels)}")
        print(f"⚙️ Soglie dinamiche: {'Attive' if self.config['adaptive_thresholds'] else 'Disattive'}")
        print(f"🔍 NMS threshold: {self.config['nms_threshold']}")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        total_inference_time = 0
        detection_counts = {}
        
        print("\n🎮 CONTROLLI:")
        print("  Q = Esci")
        print("  S = Salva screenshot")
        print("  T = Cambia soglia generale")
        print("  N = Toggle NMS")
        print("  R = Reset statistiche")
        print("  I = Info dettagliate")
        print("  + = Aumenta soglia classe corrente")
        print("  - = Diminuisci soglia classe corrente")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                self.frame_count += 1
                
                # Detection migliorata
                detections, inference_time = self.detect_objects_improved(frame)
                total_inference_time += inference_time
                
                # Conta detections per classe
                frame_classes = set()
                for detection in detections:
                    class_name = detection['class']
                    frame_classes.add(class_name)
                    if class_name not in detection_counts:
                        detection_counts[class_name] = 0
                    detection_counts[class_name] += 1
                
                # Disegna detections
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    class_name = detection['class']
                    confidence = detection['confidence']
                    
                    color = self.get_detection_color(class_name, confidence)
                    thickness = 3 if confidence > 0.7 else 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Label con soglia dinamica
                    threshold = self.get_dynamic_threshold(class_name)
                    label = f"{class_name}: {confidence:.2f} (>{threshold:.2f})"
                    
                    # Background per label
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Info overlay
                cv2.putText(frame, f"Frame: {self.frame_count} | Inference: {inference_time:.1f}ms", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Detections: {len(detections)} | Classi: {len(frame_classes)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.frame_count > 0:
                    avg_inference = total_inference_time / self.frame_count
                    fps = 1000 / avg_inference if avg_inference > 0 else 0
                    cv2.putText(frame, f"FPS: {fps:.1f} | Avg: {avg_inference:.1f}ms", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Top classi rilevate
                sorted_classes = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                y_offset = 120
                for i, (class_name, count) in enumerate(sorted_classes):
                    threshold = self.get_dynamic_threshold(class_name)
                    text = f"{class_name}: {count} (>{threshold:.2f})"
                    cv2.putText(frame, text, (10, y_offset + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Controlli
                cv2.putText(frame, "Q=esci | S=salva | T=soglia | N=NMS | R=reset | I=info", 
                           (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow('🔥 Coral TPU Detection Migliorata', frame)
                
                # Input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"improved_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Salvato: {filename}")
                elif key == ord('t'):
                    current = self.config["coral_confidence_threshold"]
                    new_threshold = float(input(f"Nuova soglia generale (attuale: {current}): "))
                    self.config["coral_confidence_threshold"] = max(0.1, min(0.9, new_threshold))
                    print(f"🔧 Soglia generale: {self.config['coral_confidence_threshold']:.2f}")
                elif key == ord('n'):
                    self.config["nms_threshold"] = 0.0 if self.config["nms_threshold"] > 0 else 0.5
                    print(f"🔧 NMS: {'Attivo' if self.config['nms_threshold'] > 0 else 'Disattivo'}")
                elif key == ord('r'):
                    detection_counts.clear()
                    total_inference_time = 0
                    self.frame_count = 0
                    print("🔄 Statistiche resettate")
                elif key == ord('i'):
                    self.print_detailed_stats(detection_counts, total_inference_time)
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_stats(detection_counts, total_inference_time)
    
    def print_detailed_stats(self, detection_counts, total_time):
        """Stampa statistiche dettagliate"""
        print(f"\n📊 STATISTICHE DETTAGLIATE")
        print(f"Frame processati: {self.frame_count}")
        
        if self.frame_count > 0:
            avg_time = total_time / self.frame_count
            fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"Tempo medio inferenza: {avg_time:.1f}ms")
            print(f"FPS medio: {fps:.1f}")
        
        print(f"\nClassi rilevate: {len(detection_counts)}")
        sorted_classes = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_classes[:10]:
            threshold = self.get_dynamic_threshold(class_name)
            avg_conf = self.detection_stats.get(class_name, {}).get('avg_confidence', 0)
            print(f"  {class_name}: {count} detections (soglia: {threshold:.2f}, conf media: {avg_conf:.2f})")
    
    def print_final_stats(self, detection_counts, total_time):
        """Report finale"""
        print(f"\n📊 REPORT FINALE")
        print("=" * 40)
        self.print_detailed_stats(detection_counts, total_time)

def main():
    parser = argparse.ArgumentParser(description='Coral TPU Detection Migliorata')
    parser.add_argument('--run', action='store_true', help='Avvia detection migliorata')
    
    args = parser.parse_args()
    
    detector = ImprovedCoralDetection()
    
    if args.run:
        detector.run_improved_detection()
    else:
        print("\n🚀 CORAL TPU DETECTION MIGLIORATA")
        print("=" * 50)
        print("1. 🔍 Avvia detection migliorata")
        print("2. 📊 Info sistema")
        print("0. ❌ Esci")
        
        while True:
            try:
                choice = input("\nScegli: ").strip()
                if choice == "0":
                    break
                elif choice == "1":
                    detector.run_improved_detection()
                elif choice == "2":
                    print(f"\n🔥 Coral TPU: {'Attivo' if detector.use_coral else 'Non disponibile'}")
                    print(f"🎯 Classi supportate: {len(detector.coral_labels)}")
                    print(f"⚙️ Soglie dinamiche: {'Attive' if detector.config['adaptive_thresholds'] else 'Disattive'}")
                    print("\nClassi con soglie personalizzate:")
                    for class_name, threshold in detector.class_thresholds.items():
                        print(f"  {class_name}: {threshold}")
                else:
                    print("❌ Opzione non valida")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
