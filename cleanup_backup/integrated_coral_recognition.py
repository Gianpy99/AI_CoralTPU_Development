#!/usr/bin/env python3
"""
Sistema INTEGRATO: Coral TPU + Riconoscimento Personalizzato
Combina l'AI del Coral TPU con il riconoscimento persone personalizzato
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import argparse
from datetime import datetime
import pickle
import platform

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

class CoralPersonRecognition:
    """Sistema integrato Coral TPU + Riconoscimento Personalizzato"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.database_path = Path("people_database")
        self.database_path.mkdir(exist_ok=True)
        
        # Files database
        self.templates_file = "person_templates.pkl"
        self.config_file = "recognition_config.json"
        
        # Database persone personalizzate
        self.person_templates = {}
        self.recognition_stats = {}
        
        # Coral TPU setup
        self.coral_interpreter = None
        self.coral_labels = []
        self.use_coral = False
        
        # Face detector per riconoscimento personalizzato
        self.face_cascade = None
        
        # Configurazione
        self.config = {
            "template_size": (100, 100),
            "similarity_threshold": 0.6,
            "coral_confidence_threshold": 0.25,  # Abbassata per più detections
            "current_mode": "detection"  # "detection", "personal", "both"
        }
        
        # Inizializza componenti
        self.load_face_detector()
        self.load_coral_tpu()
        self.load_database()
    
    def load_face_detector(self):
        """Carica rilevatore volti OpenCV"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if not self.face_cascade.empty():
                print("✅ Face detector OpenCV caricato")
            else:
                print("❌ Face detector non valido")
        except Exception as e:
            print(f"❌ Errore face detector: {e}")
    
    def load_coral_tpu(self):
        """Carica modello Coral TPU per detection"""
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
            
        except Exception as e:
            print(f"❌ Errore caricamento Coral TPU: {e}")
    
    def load_database(self):
        """Carica database persone personalizzate"""
        if Path(self.templates_file).exists():
            try:
                with open(self.templates_file, 'rb') as f:
                    data = pickle.load(f)
                    self.person_templates = data.get('templates', {})
                    self.recognition_stats = data.get('stats', {})
                print(f"✅ Database persone: {len(self.person_templates)} persone")
            except Exception as e:
                print(f"⚠️ Errore caricamento database: {e}")
        
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config.update(json.load(f))
            except Exception as e:
                print(f"⚠️ Errore config: {e}")
    
    def save_database(self):
        """Salva database persone"""
        try:
            data = {
                'templates': self.person_templates,
                'stats': self.recognition_stats,
                'created': datetime.now().isoformat()
            }
            
            with open(self.templates_file, 'wb') as f:
                pickle.dump(data, f)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            return False
    
    def coral_detect_objects(self, frame):
        """Detection oggetti con Coral TPU"""
        if not self.use_coral or not self.coral_interpreter:
            return []
        
        try:
            # Prepara input
            input_details = self.coral_interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # Resize frame
            frame_resized = cv2.resize(frame, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
            
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
            frame_height, frame_width = frame.shape[:2]
            detections = []
            
            for i in range(len(scores[0])):
                score = scores[0][i]
                if score > self.config["coral_confidence_threshold"]:
                    # Coordinate
                    ymin, xmin, ymax, xmax = locations[0][i]
                    xmin = int(xmin * frame_width)
                    ymin = int(ymin * frame_height)
                    xmax = int(xmax * frame_width)
                    ymax = int(ymax * frame_height)
                    
                    # Classe
                    class_id = int(classes[0][i])
                    class_name = self.coral_labels[class_id] if class_id < len(self.coral_labels) else f"class_{class_id}"
                    
                    detections.append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'class': class_name,
                        'confidence': float(score),
                        'source': 'coral'
                    })
            
            return detections, inference_time
        
        except Exception as e:
            print(f"⚠️ Errore Coral detection: {e}")
            return [], 0
    
    def preprocess_face(self, face_img):
        """Preprocessa volto per template matching"""
        face_resized = cv2.resize(face_img, self.config["template_size"])
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
        face_eq = cv2.equalizeHist(face_gray)
        return cv2.GaussianBlur(face_eq, (3, 3), 0)
    
    def calculate_similarity(self, template1, template2):
        """Calcola similarità tra template"""
        result = cv2.matchTemplate(template1, template2, cv2.TM_CCOEFF_NORMED)
        return result[0][0]
    
    def recognize_person(self, face_img):
        """Riconosce persona specifica"""
        if not self.person_templates:
            return "Sconosciuto", 0.0
        
        input_template = self.preprocess_face(face_img)
        best_match = "Sconosciuto"
        best_score = 0.0
        
        for person_name, templates in self.person_templates.items():
            person_scores = []
            for template in templates:
                try:
                    similarity = self.calculate_similarity(input_template, template)
                    person_scores.append(similarity)
                except:
                    continue
            
            if person_scores:
                person_best_score = max(person_scores)
                if person_best_score > best_score:
                    best_score = person_best_score
                    best_match = person_name
        
        if best_score < self.config["similarity_threshold"]:
            best_match = "Sconosciuto"
        else:
            if best_match in self.recognition_stats:
                self.recognition_stats[best_match]['recognitions'] += 1
                self.recognition_stats[best_match]['last_seen'] = datetime.now().isoformat()
        
        return best_match, best_score
    
    def detect_personal_faces(self, frame):
        """Rileva e riconosce persone personalizzate"""
        if not self.face_cascade or self.face_cascade.empty():
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        
        personal_detections = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            name, confidence = self.recognize_person(face_roi)
            
            personal_detections.append({
                'bbox': [x, y, x+w, y+h],
                'class': name,
                'confidence': confidence,
                'source': 'personal'
            })
        
        return personal_detections
    
    def add_person_camera(self, person_name, num_photos=6):
        """Aggiungi persona tramite camera"""
        if not self.face_cascade or self.face_cascade.empty():
            print("❌ Face detector non disponibile")
            return False
        
        print(f"\n🎬 REGISTRAZIONE PERSONA: {person_name}")
        print(f"📸 Foto da raccogliere: {num_photos}")
        print("⚠️ SPAZIO=cattura, Q=esci")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return False
        
        captured_templates = []
        captured_photos = 0
        
        person_dir = self.database_path / person_name
        person_dir.mkdir(exist_ok=True)
        
        while captured_photos < num_photos:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Rileva volti
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            # Display
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Registrando: {person_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Foto: {captured_photos}/{num_photos}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if len(faces) == 1 else (0, 255, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                if len(faces) == 1:
                    cv2.putText(display_frame, "PRONTO - SPAZIO", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(f'Registrazione {person_name}', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and len(faces) == 1:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    template = self.preprocess_face(face_roi)
                    captured_templates.append(template)
                    
                    photo_path = person_dir / f"{person_name}_{captured_photos:03d}.jpg"
                    cv2.imwrite(str(photo_path), face_roi)
                    
                    captured_photos += 1
                    print(f"📸 Foto {captured_photos}/{num_photos}")
                    
                    # Flash feedback
                    flash_frame = frame.copy()
                    cv2.rectangle(flash_frame, (x, y), (x+w, y+h), (255, 255, 255), 5)
                    cv2.imshow(f'Registrazione {person_name}', flash_frame)
                    cv2.waitKey(300)
                    
                except Exception as e:
                    print(f"❌ Errore: {e}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Salva nel database
        if captured_templates:
            self.person_templates[person_name] = captured_templates
            self.recognition_stats[person_name] = {
                'added_date': datetime.now().isoformat(),
                'templates_count': len(captured_templates),
                'recognitions': 0,
                'last_seen': None
            }
            
            if self.save_database():
                print(f"✅ {person_name} aggiunto con {len(captured_templates)} foto")
                return True
        
        print(f"❌ Registrazione fallita")
        return False
    
    def run_integrated_recognition(self):
        """Riconoscimento integrato live"""
        print("🚀 SISTEMA INTEGRATO CORAL TPU + PERSONE")
        print("=" * 50)
        print(f"🔥 Coral TPU: {'Attivo' if self.use_coral else 'Non disponibile'}")
        print(f"👥 Database persone: {len(self.person_templates)} persone")
        print(f"🎯 Modalità: {self.config['current_mode']}")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        frame_count = 0
        total_coral_time = 0
        total_personal_time = 0
        
        print("\n🎮 CONTROLLI:")
        print("  Q = Esci")
        print("  S = Salva screenshot")
        print("  M = Cambia modalità (detection/personal/both)")
        print("  A = Aggiungi persona")
        print("  + = Aumenta soglia")
        print("  - = Diminuisci soglia")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                coral_detections = []
                personal_detections = []
                coral_time = 0
                personal_time = 0
                
                # Coral TPU Detection
                if self.config['current_mode'] in ['detection', 'both'] and self.use_coral:
                    coral_detections, coral_time = self.coral_detect_objects(frame)
                    total_coral_time += coral_time
                
                # Personal Recognition  
                if self.config['current_mode'] in ['personal', 'both'] and self.person_templates:
                    start_time = time.time()
                    personal_detections = self.detect_personal_faces(frame)
                    personal_time = (time.time() - start_time) * 1000
                    total_personal_time += personal_time
                
                # Disegna Coral detections
                for detection in coral_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    class_name = detection['class']
                    confidence = detection['confidence']
                    
                    # Colore speciale per persone rilevate da Coral
                    if 'person' in class_name.lower():
                        color = (255, 165, 0)  # Arancione per persone generiche
                        thickness = 3
                    else:
                        color = (255, 0, 0)  # Rosso per altri oggetti
                        thickness = 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    label = f"[CORAL] {class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Disegna Personal detections
                for detection in personal_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    name = detection['class']
                    confidence = detection['confidence']
                    
                    # Verde brillante per persone conosciute
                    color = (0, 255, 0) if name != "Sconosciuto" else (0, 0, 255)
                    thickness = 4 if name != "Sconosciuto" else 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    label = f"[PERSONAL] {name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Info overlay
                cv2.putText(frame, f"Modalità: {self.config['current_mode']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Coral: {coral_time:.1f}ms | Personal: {personal_time:.1f}ms", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Soglia: {self.config['similarity_threshold']:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Contatori
                coral_persons = len([d for d in coral_detections if 'person' in d['class'].lower()])
                known_persons = len([d for d in personal_detections if d['class'] != "Sconosciuto"])
                
                cv2.putText(frame, f"Coral persone: {coral_persons} | Conosciute: {known_persons}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Controlli
                cv2.putText(frame, "Q=esci | S=salva | M=modalità | A=aggiungi | +/-=soglia", 
                           (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow('🚀 Sistema Integrato Coral TPU + Persone', frame)
                
                # Input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"integrated_recognition_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Salvato: {filename}")
                elif key == ord('m'):
                    modes = ['detection', 'personal', 'both']
                    current_idx = modes.index(self.config['current_mode'])
                    next_idx = (current_idx + 1) % len(modes)
                    self.config['current_mode'] = modes[next_idx]
                    print(f"🔄 Modalità: {self.config['current_mode']}")
                elif key == ord('a'):
                    cv2.destroyAllWindows()
                    name = input("Nome persona da aggiungere: ").strip()
                    if name:
                        self.add_person_camera(name, 6)
                    cap = cv2.VideoCapture(0)
                elif key == ord('+') or key == ord('='):
                    self.config['similarity_threshold'] = min(0.95, self.config['similarity_threshold'] + 0.05)
                    print(f"🔧 Soglia: {self.config['similarity_threshold']:.2f}")
                elif key == ord('-'):
                    self.config['similarity_threshold'] = max(0.2, self.config['similarity_threshold'] - 0.05)
                    print(f"🔧 Soglia: {self.config['similarity_threshold']:.2f}")
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_database()
            
            # Report finale
            if frame_count > 0:
                print(f"\n📊 STATISTICHE FINALI")
                print(f"Frame processati: {frame_count}")
                if total_coral_time > 0:
                    avg_coral = total_coral_time / frame_count
                    print(f"Coral TPU medio: {avg_coral:.1f}ms")
                if total_personal_time > 0:
                    avg_personal = total_personal_time / frame_count
                    print(f"Personal medio: {avg_personal:.1f}ms")

def main():
    parser = argparse.ArgumentParser(description='Sistema Integrato Coral TPU + Persone')
    parser.add_argument('--run', action='store_true', help='Avvia riconoscimento integrato')
    parser.add_argument('--add', type=str, help='Aggiungi persona')
    
    args = parser.parse_args()
    
    system = CoralPersonRecognition()
    
    if args.add:
        system.add_person_camera(args.add)
    elif args.run:
        system.run_integrated_recognition()
    else:
        print("\n🚀 SISTEMA INTEGRATO CORAL TPU + PERSONE")
        print("=" * 50)
        print("1. 🔍 Avvia riconoscimento integrato")
        print("2. 👤 Aggiungi persona")
        print("3. 📊 Info sistema")
        print("0. ❌ Esci")
        
        while True:
            try:
                choice = input("\nScegli: ").strip()
                if choice == "0":
                    break
                elif choice == "1":
                    system.run_integrated_recognition()
                elif choice == "2":
                    name = input("Nome persona: ").strip()
                    if name:
                        system.add_person_camera(name)
                elif choice == "3":
                    print(f"\n🔥 Coral TPU: {'Attivo' if system.use_coral else 'Non disponibile'}")
                    print(f"👥 Persone database: {len(system.person_templates)}")
                    print(f"🎯 Modalità corrente: {system.config['current_mode']}")
                    for name in system.person_templates:
                        count = len(system.person_templates[name])
                        recognitions = system.recognition_stats.get(name, {}).get('recognitions', 0)
                        print(f"  - {name}: {count} foto, {recognitions} riconoscimenti")
                else:
                    print("❌ Opzione non valida")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
