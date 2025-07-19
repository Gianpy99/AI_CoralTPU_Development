#!/usr/bin/env python3
"""
SISTEMA UNIFICATO AI CORAL TPU - Tutte le Modalità
Combina: Coral TPU + Riconoscimento Persone + Oggetti Custom + Detection Ottimizzata
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import argparse
from datetime import datetime
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import shutil
from collections import defaultdict
import threading

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

class UnifiedAISystem:
    """Sistema AI unificato con tutte le modalità"""
    
    def __init__(self):
        self.models_dir = Path("models")
        
        # Database paths
        self.persons_database_path = Path("persons_database")
        self.objects_database_path = Path("custom_objects_database")
        self.persons_photos_path = Path("persons_photos")
        self.objects_photos_path = Path("custom_objects_photos")
        
        # Crea directories
        for path in [self.persons_database_path, self.objects_database_path, 
                     self.persons_photos_path, self.objects_photos_path]:
            path.mkdir(exist_ok=True)
        
        # Database files
        self.persons_templates_file = "persons_templates.pkl"
        self.objects_templates_file = "custom_objects_templates.pkl"
        self.config_file = "unified_system_config.json"
        
        # Databases
        self.person_templates = {}
        self.person_stats = {}
        self.object_templates = {}
        self.object_stats = {}
        
        # Coral TPU
        self.coral_interpreter = None
        self.coral_labels = []
        self.use_coral = False
        
        # OpenCV Face Detection
        self.face_cascade = None
        self.use_face_detection = False
        
        # SIFT per feature matching
        self.sift = cv2.SIFT_create(nfeatures=500)
        self.matcher = cv2.BFMatcher()
        
        # Configurazione unificata
        self.config = {
            # Modalità sistema
            "current_mode": "unified",  # "coral_only", "persons_only", "objects_only", "unified"
            "detection_mode": "all",    # "coral", "face", "custom", "all"
            
            # Coral TPU settings
            "coral_confidence_threshold": 0.25,
            "coral_enabled": True,
            "adaptive_thresholds": True,
            "nms_threshold": 0.5,
            
            # Person recognition settings
            "person_similarity_threshold": 0.7,
            "face_detection_enabled": True,
            "person_template_size": (100, 100),
            "face_scale_factor": 1.1,
            "face_min_neighbors": 5,
            
            # Custom objects settings
            "object_similarity_threshold": 0.6,
            "object_template_size": (100, 100),
            "multi_scale_matching": True,
            "rotation_invariant": False,
            
            # Performance settings
            "frame_skip": 1,
            "max_detections_per_frame": 20,
            "confidence_display_threshold": 0.3,
            
            # Display settings
            "show_confidence": True,
            "show_fps": True,
            "show_statistics": True,
            "overlay_opacity": 0.7
        }
        
        # Soglie dinamiche Coral per classe
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
        
        # Statistiche unificate
        self.unified_stats = {
            'coral_detections': 0,
            'person_recognitions': 0,
            'object_recognitions': 0,
            'total_frames': 0,
            'avg_inference_time': 0,
            'session_start': datetime.now()
        }
        
        # Inizializza componenti
        self.load_coral_tpu()
        self.load_face_detection()
        self.load_persons_database()
        self.load_objects_database()
        self.load_config()
    
    def load_coral_tpu(self):
        """Carica modello Coral TPU"""
        if not CORAL_AVAILABLE:
            print("⚠️ PyCoral non disponibile")
            return
        
        devices = edgetpu.list_edge_tpus()
        if not devices:
            print("⚠️ Coral TPU non rilevato")
            return
        
        model_path = self.models_dir / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
        labels_path = self.models_dir / "coco_labels.txt"
        
        if not model_path.exists():
            print(f"❌ Modello Coral non trovato: {model_path}")
            return
        
        try:
            delegate = get_edgetpu_delegate()
            if delegate:
                self.coral_interpreter = tflite.Interpreter(
                    model_path=str(model_path),
                    experimental_delegates=[delegate]
                )
                print("🔥 Coral TPU caricato con EdgeTPU")
            else:
                self.coral_interpreter = tflite.Interpreter(model_path=str(model_path))
                print("💻 Coral TPU caricato su CPU")
            
            self.coral_interpreter.allocate_tensors()
            self.use_coral = True
            
            if labels_path.exists():
                with open(labels_path, 'r', encoding='utf-8') as f:
                    self.coral_labels = [line.strip() for line in f.readlines()]
                print(f"✅ Coral labels: {len(self.coral_labels)} classi")
            
        except Exception as e:
            print(f"❌ Errore Coral TPU: {e}")
    
    def load_face_detection(self):
        """Carica detector visi OpenCV"""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if not self.face_cascade.empty():
                self.use_face_detection = True
                print("✅ Face detector OpenCV caricato")
            else:
                print("❌ Face detector non caricato")
        except Exception as e:
            print(f"❌ Errore face detection: {e}")
    
    def load_persons_database(self):
        """Carica database persone"""
        if Path(self.persons_templates_file).exists():
            try:
                with open(self.persons_templates_file, 'rb') as f:
                    data = pickle.load(f)
                    self.person_templates = data.get('templates', {})
                    self.person_stats = data.get('stats', {})
                print(f"✅ Database persone: {len(self.person_templates)} persone")
            except Exception as e:
                print(f"⚠️ Errore caricamento persone: {e}")
    
    def load_objects_database(self):
        """Carica database oggetti custom"""
        if Path(self.objects_templates_file).exists():
            try:
                with open(self.objects_templates_file, 'rb') as f:
                    data = pickle.load(f)
                    self.object_templates = data.get('templates', {})
                    self.object_stats = data.get('stats', {})
                print(f"✅ Database oggetti: {len(self.object_templates)} oggetti")
            except Exception as e:
                print(f"⚠️ Errore caricamento oggetti: {e}")
    
    def load_config(self):
        """Carica configurazione"""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                print(f"✅ Configurazione caricata")
            except Exception as e:
                print(f"⚠️ Errore config: {e}")
    
    def save_databases(self):
        """Salva tutti i database"""
        try:
            # Salva persone
            persons_data = {
                'templates': self.person_templates,
                'stats': self.person_stats,
                'updated': datetime.now().isoformat()
            }
            with open(self.persons_templates_file, 'wb') as f:
                pickle.dump(persons_data, f)
            
            # Salva oggetti
            objects_data = {
                'templates': self.object_templates,
                'stats': self.object_stats,
                'updated': datetime.now().isoformat()
            }
            with open(self.objects_templates_file, 'wb') as f:
                pickle.dump(objects_data, f)
            
            # Salva config
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            return False
    
    def coral_detect_objects(self, frame):
        """Detection Coral TPU ottimizzata"""
        if not self.use_coral or not self.coral_interpreter or not self.config["coral_enabled"]:
            return [], 0
        
        try:
            input_details = self.coral_interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # Preprocessing ottimizzato
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if aspect_ratio > 1:
                new_width = width
                new_height = int(width / aspect_ratio)
            else:
                new_height = height
                new_width = int(height * aspect_ratio)
            
            frame_resized = cv2.resize(frame, (new_width, new_height))
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
            
            # Output processing
            output_details = self.coral_interpreter.get_output_details()
            locations = self.coral_interpreter.get_tensor(output_details[0]['index'])
            classes = self.coral_interpreter.get_tensor(output_details[1]['index'])
            scores = self.coral_interpreter.get_tensor(output_details[2]['index'])
            
            detections = []
            
            for i in range(len(scores[0])):
                score = scores[0][i]
                class_id = int(classes[0][i])
                
                if class_id >= len(self.coral_labels):
                    continue
                
                class_name = self.coral_labels[class_id]
                threshold = self.get_coral_threshold(class_name)
                
                if score > threshold:
                    ymin, xmin, ymax, xmax = locations[0][i]
                    
                    # Coordinate con compensazione padding
                    xmin_canvas = int(xmin * width)
                    ymin_canvas = int(ymin * height)
                    xmax_canvas = int(xmax * width)
                    ymax_canvas = int(ymax * height)
                    
                    xmin_frame = max(0, int((xmin_canvas - x_offset) * frame_width / new_width))
                    ymin_frame = max(0, int((ymin_canvas - y_offset) * frame_height / new_height))
                    xmax_frame = min(frame_width, int((xmax_canvas - x_offset) * frame_width / new_width))
                    ymax_frame = min(frame_height, int((ymax_canvas - y_offset) * frame_height / new_height))
                    
                    if xmax_frame > xmin_frame and ymax_frame > ymin_frame:
                        detections.append({
                            'bbox': [xmin_frame, ymin_frame, xmax_frame, ymax_frame],
                            'class': class_name,
                            'confidence': float(score),
                            'source': 'coral',
                            'class_id': class_id
                        })
            
            # Applica NMS
            if self.config["nms_threshold"] > 0:
                detections = self.apply_nms(detections)
            
            self.unified_stats['coral_detections'] += len(detections)
            return detections, inference_time
        
        except Exception as e:
            print(f"⚠️ Errore Coral detection: {e}")
            return [], 0
    
    def get_coral_threshold(self, class_name):
        """Ottieni soglia Coral per classe"""
        if self.config["adaptive_thresholds"]:
            return self.class_thresholds.get(class_name, self.config["coral_confidence_threshold"])
        return self.config["coral_confidence_threshold"]
    
    def apply_nms(self, detections):
        """Applica Non-Maximum Suppression"""
        if not detections or len(detections) <= 1:
            return detections
        
        # Raggruppa per classe
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det['class']].append(det)
        
        filtered_detections = []
        
        for class_name, class_dets in class_groups.items():
            if len(class_dets) <= 1:
                filtered_detections.extend(class_dets)
                continue
            
            boxes = []
            scores = []
            
            for det in class_dets:
                x1, y1, x2, y2 = det['bbox']
                boxes.append([x1, y1, x2-x1, y2-y1])
                scores.append(det['confidence'])
            
            indices = cv2.dnn.NMSBoxes(
                boxes, scores,
                self.config["coral_confidence_threshold"],
                self.config["nms_threshold"]
            )
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    filtered_detections.append(class_dets[idx])
        
        return filtered_detections
    
    def detect_and_recognize_faces(self, frame):
        """Detection e riconoscimento visi"""
        if not self.use_face_detection or not self.config["face_detection_enabled"]:
            return [], 0
        
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config["face_scale_factor"],
            minNeighbors=self.config["face_min_neighbors"],
            minSize=(30, 30)
        )
        
        recognitions = []
        
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            
            # Riconosci persona
            person_name, confidence = self.recognize_person(face_region)
            
            recognitions.append({
                'bbox': [x, y, x+w, y+h],
                'class': person_name,
                'confidence': confidence,
                'source': 'face_recognition',
                'type': 'person'
            })
            
            if person_name != "Sconosciuto":
                self.unified_stats['person_recognitions'] += 1
        
        detection_time = (time.time() - start_time) * 1000
        return recognitions, detection_time
    
    def recognize_person(self, face_region):
        """Riconosce persona da regione viso"""
        if not self.person_templates:
            return "Sconosciuto", 0.0
        
        # Prepara template
        face_resized = cv2.resize(face_region, self.config["person_template_size"])
        face_normalized = cv2.equalizeHist(face_resized)
        
        best_match = "Sconosciuto"
        best_score = 0.0
        
        for person_name, templates in self.person_templates.items():
            scores = []
            
            for template in templates:
                # Template matching
                result = cv2.matchTemplate(face_normalized, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(result)
                scores.append(score)
            
            if scores:
                person_score = max(scores)
                if person_score > best_score:
                    best_score = person_score
                    best_match = person_name
        
        # Applica soglia
        if best_score < self.config["person_similarity_threshold"]:
            best_match = "Sconosciuto"
        else:
            # Aggiorna statistiche
            if best_match in self.person_stats:
                self.person_stats[best_match]['recognitions'] += 1
                self.person_stats[best_match]['last_seen'] = datetime.now().isoformat()
        
        return best_match, best_score
    
    def detect_custom_objects_optimized(self, frame):
        """Detection oggetti custom ottimizzata"""
        if not self.object_templates:
            return [], 0
        
        start_time = time.time()
        custom_detections = []
        
        # Parametri ottimizzati per performance
        window_scales = [0.7, 1.0, 1.3]  # Ridotto per performance
        step_size = 60  # Aumentato per performance
        
        frame_height, frame_width = frame.shape[:2]
        
        for scale in window_scales:
            window_w = int(self.config["object_template_size"][0] * scale)
            window_h = int(self.config["object_template_size"][1] * scale)
            
            if window_w > frame_width or window_h > frame_height:
                continue
            
            for y in range(0, frame_height - window_h, step_size):
                for x in range(0, frame_width - window_w, step_size):
                    window = frame[y:y+window_h, x:x+window_w]
                    
                    object_name, confidence = self.recognize_custom_object(window)
                    
                    if object_name != "Sconosciuto" and confidence > self.config["object_similarity_threshold"]:
                        custom_detections.append({
                            'bbox': [x, y, x+window_w, y+window_h],
                            'class': object_name,
                            'confidence': confidence,
                            'source': 'custom_object',
                            'type': 'custom'
                        })
        
        detection_time = (time.time() - start_time) * 1000
        
        # Applica NMS
        if custom_detections:
            custom_detections = self.apply_nms_custom(custom_detections)
            self.unified_stats['object_recognitions'] += len(custom_detections)
        
        return custom_detections, detection_time
    
    def recognize_custom_object(self, obj_region):
        """Riconosce oggetto custom"""
        # Preprocessing
        obj_resized = cv2.resize(obj_region, self.config["object_template_size"])
        obj_gray = cv2.cvtColor(obj_resized, cv2.COLOR_BGR2GRAY) if len(obj_resized.shape) == 3 else obj_resized
        obj_eq = cv2.equalizeHist(obj_gray)
        
        # Feature extraction
        keypoints, descriptors = self.sift.detectAndCompute(obj_gray, None)
        
        best_match = "Sconosciuto"
        best_score = 0.0
        
        for object_name, templates in self.object_templates.items():
            object_scores = []
            
            for template_data in templates:
                # Template matching
                template_sim = cv2.matchTemplate(obj_eq, template_data['template'], cv2.TM_CCOEFF_NORMED)[0][0]
                
                # SIFT matching
                sift_sim = 0.0
                if descriptors is not None and template_data.get('descriptors') is not None:
                    try:
                        matches = self.matcher.knnMatch(descriptors, template_data['descriptors'], k=2)
                        good_matches = []
                        for match_pair in matches:
                            if len(match_pair) == 2:
                                m, n = match_pair
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        
                        if len(descriptors) > 0:
                            sift_sim = len(good_matches) / len(descriptors)
                    except:
                        pass
                
                # Combina punteggi
                combined_score = (template_sim * 0.6) + (sift_sim * 0.4)
                object_scores.append(combined_score)
            
            if object_scores:
                object_best = max(object_scores)
                if object_best > best_score:
                    best_score = object_best
                    best_match = object_name
        
        # Aggiorna statistiche
        if best_match != "Sconosciuto" and best_match in self.object_stats:
            self.object_stats[best_match]['recognitions'] += 1
            self.object_stats[best_match]['last_seen'] = datetime.now().isoformat()
        
        return best_match, best_score
    
    def apply_nms_custom(self, detections):
        """NMS per oggetti custom"""
        if len(detections) <= 1:
            return detections
        
        class_groups = defaultdict(list)
        for det in detections:
            class_groups[det['class']].append(det)
        
        filtered = []
        for class_name, class_dets in class_groups.items():
            if len(class_dets) <= 1:
                filtered.extend(class_dets)
                continue
            
            boxes = []
            scores = []
            for det in class_dets:
                x1, y1, x2, y2 = det['bbox']
                boxes.append([x1, y1, x2-x1, y2-y1])
                scores.append(det['confidence'])
            
            indices = cv2.dnn.NMSBoxes(
                boxes, scores,
                self.config["object_similarity_threshold"],
                self.config["nms_threshold"]
            )
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    filtered.append(class_dets[idx])
        
        return filtered
    
    def get_detection_color_unified(self, detection):
        """Colori unificati per tipo detection"""
        source = detection['source']
        class_name = detection['class']
        confidence = detection['confidence']
        
        if source == 'coral':
            if class_name == "person":
                return (255, 165, 0)  # Arancione per persone Coral
            elif class_name in ["car", "bicycle", "motorcycle"]:
                return (0, 255, 255)  # Ciano per veicoli
            elif class_name in ["dog", "cat", "bird"]:
                return (255, 0, 255)  # Magenta per animali
            else:
                return (128, 255, 128)  # Verde chiaro per altri oggetti Coral
        
        elif source == 'face_recognition':
            if class_name == "Sconosciuto":
                return (0, 0, 255)  # Rosso per sconosciuti
            else:
                return (0, 255, 0)  # Verde per persone riconosciute
        
        elif source == 'custom_object':
            return (255, 255, 0)  # Giallo per oggetti custom
        
        else:
            # Fallback basato su confidenza
            if confidence > 0.8:
                return (0, 255, 0)
            elif confidence > 0.5:
                return (255, 255, 0)
            else:
                return (0, 0, 255)
    
    def run_unified_system(self):
        """Sistema unificato completo"""
        print("🚀 SISTEMA AI UNIFICATO - TUTTE LE MODALITÀ")
        print("=" * 70)
        print(f"🔥 Coral TPU: {'Attivo' if self.use_coral else 'Non disponibile'}")
        print(f"👤 Face Recognition: {'Attivo' if self.use_face_detection else 'Non disponibile'}")
        print(f"🎯 Persone database: {len(self.person_templates)} persone")
        print(f"📦 Oggetti custom: {len(self.object_templates)} oggetti")
        print(f"🎮 Modalità: {self.config['current_mode']}")
        print(f"🔍 Detection: {self.config['detection_mode']}")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        total_inference_time = 0
        frame_count = 0
        
        print("\n🎮 CONTROLLI UNIFICATI:")
        print("  Q = Esci")
        print("  S = Salva screenshot")
        print("  M = Cambia modalità sistema")
        print("  D = Cambia detection mode")
        print("  1 = Solo Coral TPU")
        print("  2 = Solo riconoscimento persone")
        print("  3 = Solo oggetti custom")
        print("  4 = Modalità unificata (tutto)")
        print("  A = Cattura persona dalla camera")
        print("  O = Cattura oggetto dalla camera")
        print("  F = Aggiungi persona da file")
        print("  G = Aggiungi oggetto da file")
        print("  R = Rimuovi persona/oggetto")
        print("  L = Lista database")
        print("  I = Info sistema")
        print("  + = Aumenta soglie")
        print("  - = Diminuisci soglie")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                self.unified_stats['total_frames'] = frame_count
                
                all_detections = []
                total_frame_time = 0
                
                # CORAL TPU DETECTION
                if self.config['detection_mode'] in ['coral', 'all']:
                    coral_detections, coral_time = self.coral_detect_objects(frame)
                    all_detections.extend(coral_detections)
                    total_frame_time += coral_time
                
                # FACE RECOGNITION
                if self.config['detection_mode'] in ['face', 'all']:
                    face_recognitions, face_time = self.detect_and_recognize_faces(frame)
                    all_detections.extend(face_recognitions)
                    total_frame_time += face_time
                
                # CUSTOM OBJECTS
                if self.config['detection_mode'] in ['custom', 'all']:
                    custom_detections, custom_time = self.detect_custom_objects_optimized(frame)
                    all_detections.extend(custom_detections)
                    total_frame_time += custom_time
                
                total_inference_time += total_frame_time
                
                # Limita detections per performance
                if len(all_detections) > self.config["max_detections_per_frame"]:
                    all_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)[:self.config["max_detections_per_frame"]]
                
                # RENDERING UNIFICATO
                self.render_unified_frame(frame, all_detections, total_frame_time, frame_count)
                
                cv2.imshow('🚀 Sistema AI Unificato - Tutte le Modalità', frame)
                
                # INPUT HANDLING
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"unified_system_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Salvato: {filename}")
                elif key == ord('m'):
                    self.cycle_system_mode()
                elif key == ord('d'):
                    self.cycle_detection_mode()
                elif key == ord('1'):
                    self.config['detection_mode'] = 'coral'
                    print("🔥 Modalità: Solo Coral TPU")
                elif key == ord('2'):
                    self.config['detection_mode'] = 'face'
                    print("👤 Modalità: Solo riconoscimento persone")
                elif key == ord('3'):
                    self.config['detection_mode'] = 'custom'
                    print("📦 Modalità: Solo oggetti custom")
                elif key == ord('4'):
                    self.config['detection_mode'] = 'all'
                    print("🚀 Modalità: Unificata (tutto)")
                elif key == ord('a'):
                    cv2.destroyAllWindows()
                    self.add_person_from_camera()
                    cap = cv2.VideoCapture(0)
                elif key == ord('o'):
                    cv2.destroyAllWindows()
                    self.add_object_from_camera()
                    cap = cv2.VideoCapture(0)
                elif key == ord('f'):
                    cv2.destroyAllWindows()
                    self.add_person_interactive()
                    cap = cv2.VideoCapture(0)
                elif key == ord('g'):
                    cv2.destroyAllWindows()
                    self.add_object_interactive()
                    cap = cv2.VideoCapture(0)
                elif key == ord('r'):
                    cv2.destroyAllWindows()
                    self.remove_item_interactive()
                    cap = cv2.VideoCapture(0)
                elif key == ord('l'):
                    self.list_all_databases()
                elif key == ord('i'):
                    self.print_unified_info()
                elif key == ord('+'):
                    self.adjust_thresholds(0.05)
                elif key == ord('-'):
                    self.adjust_thresholds(-0.05)
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_databases()
            self.print_final_unified_stats(total_inference_time, frame_count)
    
    def render_unified_frame(self, frame, detections, inference_time, frame_count):
        """Rendering unificato del frame"""
        # Disegna tutte le detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            source = detection['source']
            
            if confidence < self.config["confidence_display_threshold"]:
                continue
            
            color = self.get_detection_color_unified(detection)
            
            # Spessore basato su confidenza
            thickness = 3 if confidence > 0.7 else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label con source
            source_icons = {
                'coral': '🔥',
                'face_recognition': '👤',
                'custom_object': '📦'
            }
            icon = source_icons.get(source, '❓')
            
            label = f"{icon} {class_name}: {confidence:.2f}"
            
            # Background per label
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Info overlay
        if self.config["show_fps"]:
            fps = 1000 / inference_time if inference_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.config["show_statistics"]:
            # Conta detections per source
            coral_count = len([d for d in detections if d['source'] == 'coral'])
            face_count = len([d for d in detections if d['source'] == 'face_recognition'])
            custom_count = len([d for d in detections if d['source'] == 'custom_object'])
            
            cv2.putText(frame, f"🔥Coral: {coral_count} | 👤Face: {face_count} | 📦Custom: {custom_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Modalità: {self.config['detection_mode']} | Database: {len(self.person_templates)}P+{len(self.object_templates)}O", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controlli overlay
        cv2.putText(frame, "Q=esci | 1234=modalità | A=cattura persona | O=cattura oggetto | F/G=da file | R=rimuovi | L=lista", 
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def cycle_system_mode(self):
        """Cicla modalità sistema"""
        modes = ['unified', 'coral_only', 'persons_only', 'objects_only']
        current_idx = modes.index(self.config['current_mode'])
        next_idx = (current_idx + 1) % len(modes)
        self.config['current_mode'] = modes[next_idx]
        print(f"🔄 Modalità sistema: {self.config['current_mode']}")
    
    def cycle_detection_mode(self):
        """Cicla modalità detection"""
        modes = ['all', 'coral', 'face', 'custom']
        current_idx = modes.index(self.config['detection_mode'])
        next_idx = (current_idx + 1) % len(modes)
        self.config['detection_mode'] = modes[next_idx]
        print(f"🔄 Detection: {self.config['detection_mode']}")
    
    def adjust_thresholds(self, delta):
        """Regola soglie globalmente"""
        self.config['coral_confidence_threshold'] = max(0.1, min(0.9, self.config['coral_confidence_threshold'] + delta))
        self.config['person_similarity_threshold'] = max(0.1, min(0.95, self.config['person_similarity_threshold'] + delta))
        self.config['object_similarity_threshold'] = max(0.1, min(0.95, self.config['object_similarity_threshold'] + delta))
        
        print(f"🔧 Soglie: Coral={self.config['coral_confidence_threshold']:.2f} | "
              f"Persone={self.config['person_similarity_threshold']:.2f} | "
              f"Oggetti={self.config['object_similarity_threshold']:.2f}")
    
    def add_person_from_camera(self):
        """Cattura persona direttamente dalla camera"""
        name = input("Nome persona da catturare: ").strip()
        if not name:
            return
        
        print(f"\n� CATTURA PERSONA: {name}")
        print("🎮 CONTROLLI:")
        print("  SPAZIO = Cattura viso")
        print("  S = Salva tutte le catture")
        print("  Q = Annulla e torna")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        captured_faces = []
        capture_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                display_frame = frame.copy()
                
                # Detect faces in real-time
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.config["face_scale_factor"],
                    minNeighbors=self.config["face_min_neighbors"],
                    minSize=(50, 50)
                )
                
                # Disegna i visi rilevati
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "VISO RILEVATO", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Info overlay
                cv2.putText(display_frame, f"Catturando: {name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                cv2.putText(display_frame, f"Catturati: {len(captured_faces)} visi", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f"Visi rilevati: {len(faces)}", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Controlli
                cv2.putText(display_frame, "SPAZIO=cattura | S=salva | Q=annulla", 
                           (10, display_frame.shape[0] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(f'📹 Cattura Persona: {name}', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("❌ Cattura annullata")
                    break
                elif key == ord(' '):  # Spazio per catturare
                    if len(faces) > 0:
                        # Cattura tutti i visi nel frame
                        frame_faces = []
                        for i, (x, y, w, h) in enumerate(faces):
                            face_region = gray[y:y+h, x:x+w]
                            face_resized = cv2.resize(face_region, self.config["person_template_size"])
                            face_normalized = cv2.equalizeHist(face_resized)
                            frame_faces.append(face_normalized)
                        
                        captured_faces.extend(frame_faces)
                        capture_count += 1
                        
                        print(f"📸 Cattura {capture_count}: {len(frame_faces)} visi -> Totale: {len(captured_faces)}")
                        
                        # Feedback visivo
                        cv2.putText(display_frame, "CATTURATO!", 
                                   (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        cv2.imshow(f'📹 Cattura Persona: {name}', display_frame)
                        cv2.waitKey(500)  # Mostra feedback per 500ms
                    else:
                        print("⚠️ Nessun viso rilevato nel frame")
                elif key == ord('s'):
                    if captured_faces:
                        # Salva le catture
                        self.save_captured_person(name, captured_faces)
                        break
                    else:
                        print("⚠️ Nessun viso catturato")
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione cattura")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def save_captured_person(self, person_name, face_templates):
        """Salva persona catturata dalla camera"""
        try:
            # Crea directory persona
            person_dir = self.persons_photos_path / person_name
            person_dir.mkdir(exist_ok=True)
            
            # Salva nel database
            if person_name in self.person_templates:
                # Aggiungi ai template esistenti
                existing_count = len(self.person_templates[person_name])
                self.person_templates[person_name].extend(face_templates)
                self.person_stats[person_name]['templates_count'] = len(self.person_templates[person_name])
                self.person_stats[person_name]['updated_date'] = datetime.now().isoformat()
                
                print(f"✅ Aggiunti {len(face_templates)} template a '{person_name}'")
                print(f"📊 Template totali: {existing_count} + {len(face_templates)} = {len(self.person_templates[person_name])}")
            else:
                # Nuova persona
                self.person_templates[person_name] = face_templates
                self.person_stats[person_name] = {
                    'added_date': datetime.now().isoformat(),
                    'templates_count': len(face_templates),
                    'recognitions': 0,
                    'last_seen': None,
                    'photos_processed': 0,
                    'camera_captures': len(face_templates),
                    'category': 'person'
                }
                
                print(f"✅ Nuova persona '{person_name}' con {len(face_templates)} template dalla camera")
            
            # Salva template come immagini per verifica
            captures_dir = person_dir / "camera_captures"
            captures_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            for i, face_template in enumerate(face_templates):
                capture_path = captures_dir / f"capture_{timestamp}_{i:03d}.jpg"
                cv2.imwrite(str(capture_path), face_template)
            
            if self.save_databases():
                print(f"💾 Database persone salvato")
                print(f"📁 Template salvati in: {captures_dir}")
                return True
            
        except Exception as e:
            print(f"❌ Errore salvataggio persona: {e}")
            return False
    
    def add_object_from_camera(self):
        """Cattura oggetto direttamente dalla camera"""
        name = input("Nome oggetto da catturare: ").strip()
        if not name:
            return
        
        print(f"\n� CATTURA OGGETTO: {name}")
        print("🎮 CONTROLLI:")
        print("  SPAZIO = Cattura frame corrente")
        print("  S = Salva tutte le catture")
        print("  Q = Annulla e torna")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        captured_objects = []
        capture_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                display_frame = frame.copy()
                
                # Info overlay
                cv2.putText(display_frame, f"Catturando: {name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                cv2.putText(display_frame, f"Catturati: {len(captured_objects)} oggetti", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_frame, "Posiziona oggetto e premi SPAZIO", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Controlli
                cv2.putText(display_frame, "SPAZIO=cattura | S=salva | Q=annulla", 
                           (10, display_frame.shape[0] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(f'📹 Cattura Oggetto: {name}', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("❌ Cattura annullata")
                    break
                elif key == ord(' '):  # Spazio per catturare
                    # Cattura frame corrente e lascia selezionare ROI
                    captured_frame = frame.copy()
                    cv2.destroyWindow(f'📹 Cattura Oggetto: {name}')
                    
                    # Selezione ROI
                    cv2.putText(captured_frame, f"Seleziona {name} e premi ENTER", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    roi = cv2.selectROI(f"Seleziona {name} - Cattura {capture_count + 1}", 
                                       captured_frame, False, False)
                    cv2.destroyAllWindows()
                    
                    if roi[2] > 0 and roi[3] > 0:  # Se ROI valida
                        x, y, w, h = roi
                        object_roi = captured_frame[y:y+h, x:x+w]
                        
                        # Preprocessa oggetto
                        template = self.preprocess_object_template(object_roi)
                        keypoints, descriptors = self.extract_sift_features(object_roi)
                        
                        template_data = {
                            'template': template,
                            'original_roi': object_roi,
                            'keypoints': len(keypoints) if keypoints else 0,
                            'descriptors': descriptors,
                            'size': (w, h),
                            'source': 'camera_capture',
                            'capture_timestamp': datetime.now().isoformat()
                        }
                        
                        captured_objects.append(template_data)
                        capture_count += 1
                        
                        print(f"📸 Cattura {capture_count}: Oggetto {w}x{h}, SIFT: {len(keypoints) if keypoints else 0}")
                    else:
                        print("⚠️ Selezione annullata")
                elif key == ord('s'):
                    if captured_objects:
                        # Salva le catture
                        self.save_captured_object(name, captured_objects)
                        break
                    else:
                        print("⚠️ Nessun oggetto catturato")
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione cattura")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def save_captured_object(self, object_name, object_templates):
        """Salva oggetto catturato dalla camera"""
        try:
            # Crea directory oggetto
            object_dir = self.objects_photos_path / object_name
            object_dir.mkdir(exist_ok=True)
            
            # Salva nel database
            if object_name in self.object_templates:
                # Aggiungi ai template esistenti
                existing_count = len(self.object_templates[object_name])
                self.object_templates[object_name].extend(object_templates)
                self.object_stats[object_name]['templates_count'] = len(self.object_templates[object_name])
                self.object_stats[object_name]['updated_date'] = datetime.now().isoformat()
                
                print(f"✅ Aggiunti {len(object_templates)} template a '{object_name}'")
                print(f"📊 Template totali: {existing_count} + {len(object_templates)} = {len(self.object_templates[object_name])}")
            else:
                # Nuovo oggetto
                self.object_templates[object_name] = object_templates
                self.object_stats[object_name] = {
                    'added_date': datetime.now().isoformat(),
                    'templates_count': len(object_templates),
                    'recognitions': 0,
                    'last_seen': None,
                    'photos_processed': 0,
                    'camera_captures': len(object_templates),
                    'category': 'custom_object'
                }
                
                print(f"✅ Nuovo oggetto '{object_name}' con {len(object_templates)} template dalla camera")
            
            # Salva template come immagini per verifica
            captures_dir = object_dir / "camera_captures"
            captures_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            for i, template_data in enumerate(object_templates):
                # Salva immagine originale
                capture_path = captures_dir / f"capture_{timestamp}_{i:03d}.jpg"
                cv2.imwrite(str(capture_path), template_data['original_roi'])
                
                # Salva template processato
                template_path = captures_dir / f"template_{timestamp}_{i:03d}.jpg"
                cv2.imwrite(str(template_path), template_data['template'])
            
            if self.save_databases():
                print(f"💾 Database oggetti salvato")
                print(f"📁 Template salvati in: {captures_dir}")
                return True
            
        except Exception as e:
            print(f"❌ Errore salvataggio oggetto: {e}")
            return False
    
    def remove_item_interactive(self):
        """Rimuovi elemento interattivamente"""
        print("\n🗑️ RIMUOVI ELEMENTO")
        print("1. Rimuovi persona")
        print("2. Rimuovi oggetto custom")
        print("0. Annulla")
        
        choice = input("Scegli: ").strip()
        if choice == "1":
            self.remove_person_interactive()
        elif choice == "2":
            self.remove_object_interactive()
    
    def remove_person_interactive(self):
        """Rimuovi persona dal database"""
        if not self.person_templates:
            print("❌ Database persone vuoto")
            return
        
        print("\n🗑️ RIMUOVI PERSONA")
        persons_list = list(self.person_templates.keys())
        
        for i, name in enumerate(persons_list, 1):
            templates_count = len(self.person_templates[name])
            recognitions = self.person_stats.get(name, {}).get('recognitions', 0)
            print(f"  {i}. {name} ({templates_count} template, {recognitions} riconoscimenti)")
        
        print("  0. Annulla")
        
        try:
            choice = input("\nScegli persona da rimuovere: ").strip()
            if choice == "0":
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(persons_list):
                person_name = persons_list[choice_idx]
                
                # Conferma
                templates_count = len(self.person_templates[person_name])
                print(f"\n⚠️ Rimuovere '{person_name}' con {templates_count} template?")
                
                confirm = input("Conferma (si/no): ").strip().lower()
                if confirm in ['si', 'sì', 'yes', 'y']:
                    # Rimuovi
                    del self.person_templates[person_name]
                    if person_name in self.person_stats:
                        del self.person_stats[person_name]
                    
                    # Rimuovi directory
                    person_dir = self.persons_photos_path / person_name
                    if person_dir.exists():
                        shutil.rmtree(person_dir)
                    
                    self.save_databases()
                    print(f"✅ Persona '{person_name}' rimossa")
            else:
                print("❌ Scelta non valida")
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    def remove_object_interactive(self):
        """Rimuovi oggetto custom dal database"""
        if not self.object_templates:
            print("❌ Database oggetti vuoto")
            return
        
        print("\n🗑️ RIMUOVI OGGETTO CUSTOM")
        objects_list = list(self.object_templates.keys())
        
        for i, name in enumerate(objects_list, 1):
            templates_count = len(self.object_templates[name])
            recognitions = self.object_stats.get(name, {}).get('recognitions', 0)
            print(f"  {i}. {name} ({templates_count} template, {recognitions} riconoscimenti)")
        
        print("  0. Annulla")
        
        try:
            choice = input("\nScegli oggetto da rimuovere: ").strip()
            if choice == "0":
                return
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(objects_list):
                object_name = objects_list[choice_idx]
                
                # Conferma
                templates_count = len(self.object_templates[object_name])
                print(f"\n⚠️ Rimuovere '{object_name}' con {templates_count} template?")
                
                confirm = input("Conferma (si/no): ").strip().lower()
                if confirm in ['si', 'sì', 'yes', 'y']:
                    # Rimuovi
                    del self.object_templates[object_name]
                    if object_name in self.object_stats:
                        del self.object_stats[object_name]
                    
                    # Rimuovi directory
                    object_dir = self.objects_photos_path / object_name
                    if object_dir.exists():
                        shutil.rmtree(object_dir)
                    
                    self.save_databases()
                    print(f"✅ Oggetto '{object_name}' rimosso")
            else:
                print("❌ Scelta non valida")
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    def select_photos_gui(self, item_name, item_type="persona"):
        """Seleziona foto usando interfaccia grafica"""
        print(f"📁 Seleziona foto per {item_type} '{item_name}'...")
        
        root = tk.Tk()
        root.withdraw()
        
        file_paths = filedialog.askopenfilenames(
            title=f"Seleziona foto per {item_name}",
            filetypes=[
                ("Immagini", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
                ("Tutti i file", "*.*")
            ]
        )
        
        root.destroy()
        return list(file_paths) if file_paths else []
    
    def add_person_from_photos(self, person_name):
        """Aggiungi persona usando foto"""
        print(f"\n👤 AGGIUNGI PERSONA: {person_name}")
        
        # Seleziona foto
        photo_paths = self.select_photos_gui(person_name, "persona")
        if not photo_paths:
            print("❌ Nessuna foto selezionata")
            return False
        
        print(f"📁 Selezionate {len(photo_paths)} foto")
        
        # Crea directory persona
        person_dir = self.persons_photos_path / person_name
        person_dir.mkdir(exist_ok=True)
        
        all_templates = []
        processed_photos = 0
        
        for photo_path in photo_paths:
            try:
                # Copia foto nella directory persona
                photo_name = f"{person_name}_{processed_photos:03d}{Path(photo_path).suffix}"
                dest_path = person_dir / photo_name
                shutil.copy2(photo_path, dest_path)
                
                # Estrai visi
                face_templates = self.extract_faces_from_image(photo_path, person_name)
                if face_templates:
                    all_templates.extend(face_templates)
                    processed_photos += 1
                    print(f"✅ Processata: {Path(photo_path).name} -> {len(face_templates)} visi")
                else:
                    print(f"⚠️ Nessun viso trovato in: {Path(photo_path).name}")
                
            except Exception as e:
                print(f"❌ Errore processando {photo_path}: {e}")
        
        # Salva nel database
        if all_templates:
            if person_name in self.person_templates:
                # Aggiungi ai template esistenti
                existing_count = len(self.person_templates[person_name])
                self.person_templates[person_name].extend(all_templates)
                self.person_stats[person_name]['templates_count'] = len(self.person_templates[person_name])
                self.person_stats[person_name]['updated_date'] = datetime.now().isoformat()
                
                print(f"✅ Aggiunti {len(all_templates)} template a '{person_name}'")
                print(f"📊 Template totali: {existing_count} + {len(all_templates)} = {len(self.person_templates[person_name])}")
            else:
                # Nuova persona
                self.person_templates[person_name] = all_templates
                self.person_stats[person_name] = {
                    'added_date': datetime.now().isoformat(),
                    'templates_count': len(all_templates),
                    'recognitions': 0,
                    'last_seen': None,
                    'photos_processed': processed_photos,
                    'category': 'person'
                }
                
                print(f"✅ Nuova persona '{person_name}' con {len(all_templates)} template")
            
            if self.save_databases():
                print(f"💾 Database persone salvato")
                return True
        
        print(f"❌ Nessun template valido estratto")
        return False
    
    def extract_faces_from_image(self, image_path, person_name):
        """Estrae visi da immagine statica"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"❌ Impossibile caricare: {image_path}")
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config["face_scale_factor"],
                minNeighbors=self.config["face_min_neighbors"],
                minSize=(50, 50)
            )
            
            templates = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Estrai regione viso
                face_region = gray[y:y+h, x:x+w]
                
                # Resize al template size
                face_resized = cv2.resize(face_region, self.config["person_template_size"])
                
                # Normalizza
                face_normalized = cv2.equalizeHist(face_resized)
                
                templates.append(face_normalized)
                
                # Salva estratto per verifica
                face_dir = self.persons_photos_path / person_name / "extracted"
                face_dir.mkdir(parents=True, exist_ok=True)
                
                extract_path = face_dir / f"{Path(image_path).stem}_face_{i:02d}.jpg"
                cv2.imwrite(str(extract_path), face_region)
            
            print(f"✅ Estratti {len(templates)} visi da {Path(image_path).name}")
            return templates
            
        except Exception as e:
            print(f"❌ Errore estrazione visi: {e}")
            return []
    
    def add_custom_object_from_photos(self, object_name):
        """Aggiungi oggetto personalizzato usando foto"""
        print(f"\n📦 AGGIUNGI OGGETTO PERSONALIZZATO: {object_name}")
        
        # Seleziona foto
        photo_paths = self.select_photos_gui(object_name, "oggetto")
        if not photo_paths:
            print("❌ Nessuna foto selezionata")
            return False
        
        print(f"📁 Selezionate {len(photo_paths)} foto")
        
        # Crea directory oggetto
        object_dir = self.objects_photos_path / object_name
        object_dir.mkdir(exist_ok=True)
        
        all_templates = []
        processed_photos = 0
        
        for photo_path in photo_paths:
            try:
                # Copia foto nella directory oggetto
                photo_name = f"{object_name}_{processed_photos:03d}{Path(photo_path).suffix}"
                dest_path = object_dir / photo_name
                shutil.copy2(photo_path, dest_path)
                
                # Estrai oggetti
                object_templates = self.extract_objects_from_image(photo_path, object_name)
                if object_templates:
                    all_templates.extend(object_templates)
                    processed_photos += 1
                    print(f"✅ Processata: {Path(photo_path).name} -> {len(object_templates)} estratti")
                else:
                    print(f"⚠️ Nessun oggetto estratto da: {Path(photo_path).name}")
                
            except Exception as e:
                print(f"❌ Errore processando {photo_path}: {e}")
        
        # Salva nel database
        if all_templates:
            if object_name in self.object_templates:
                # Aggiungi ai template esistenti
                existing_count = len(self.object_templates[object_name])
                self.object_templates[object_name].extend(all_templates)
                self.object_stats[object_name]['templates_count'] = len(self.object_templates[object_name])
                self.object_stats[object_name]['updated_date'] = datetime.now().isoformat()
                
                print(f"✅ Aggiunti {len(all_templates)} template a '{object_name}'")
                print(f"📊 Template totali: {existing_count} + {len(all_templates)} = {len(self.object_templates[object_name])}")
            else:
                # Nuovo oggetto
                self.object_templates[object_name] = all_templates
                self.object_stats[object_name] = {
                    'added_date': datetime.now().isoformat(),
                    'templates_count': len(all_templates),
                    'recognitions': 0,
                    'last_seen': None,
                    'photos_processed': processed_photos,
                    'category': 'custom_object'
                }
                
                print(f"✅ Nuovo oggetto '{object_name}' con {len(all_templates)} template")
            
            if self.save_databases():
                print(f"💾 Database oggetti salvato")
                return True
        
        print(f"❌ Nessun template valido estratto")
        return False
    
    def extract_objects_from_image(self, image_path, object_name):
        """Estrae oggetti da immagine statica"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"❌ Impossibile caricare: {image_path}")
                return []
            
            print(f"📸 Processando: {Path(image_path).name}")
            
            # Mostra immagine per selezione manuale
            display_img = img.copy()
            cv2.putText(display_img, f"Seleziona oggetto: {object_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, "Trascina per selezionare, SPAZIO=conferma, Q=salta", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Selezione ROI manuale
            roi = cv2.selectROI(f"Seleziona {object_name}", display_img, False, False)
            cv2.destroyAllWindows()
            
            if roi[2] > 0 and roi[3] > 0:  # Se ROI valida
                x, y, w, h = roi
                object_roi = img[y:y+h, x:x+w]
                
                # Template base
                template = self.preprocess_object_template(object_roi)
                
                # Feature SIFT per matching avanzato
                keypoints, descriptors = self.extract_sift_features(object_roi)
                
                template_data = {
                    'template': template,
                    'original_roi': object_roi,
                    'keypoints': len(keypoints) if keypoints else 0,
                    'descriptors': descriptors,
                    'size': (w, h),
                    'source_image': Path(image_path).name
                }
                
                # Salva estratto per verifica
                object_dir = self.objects_photos_path / object_name / "extracted"
                object_dir.mkdir(parents=True, exist_ok=True)
                
                extract_path = object_dir / f"{Path(image_path).stem}_extract.jpg"
                cv2.imwrite(str(extract_path), object_roi)
                
                print(f"✅ Estratto oggetto da {Path(image_path).name}")
                print(f"   Dimensioni: {w}x{h}, Feature SIFT: {len(keypoints) if keypoints else 0}")
                
                return [template_data]
            else:
                print(f"❌ Nessun oggetto selezionato da {Path(image_path).name}")
                return []
            
        except Exception as e:
            print(f"❌ Errore estrazione oggetto: {e}")
            return []
    
    def preprocess_object_template(self, obj_img):
        """Preprocessa oggetto per template matching"""
        # Resize
        obj_resized = cv2.resize(obj_img, self.config["object_template_size"])
        
        # Converti in grayscale
        if len(obj_resized.shape) == 3:
            obj_gray = cv2.cvtColor(obj_resized, cv2.COLOR_BGR2GRAY)
        else:
            obj_gray = obj_resized
        
        # Equalizzazione istogramma
        obj_eq = cv2.equalizeHist(obj_gray)
        
        # Blur per ridurre rumore
        obj_blur = cv2.GaussianBlur(obj_eq, (3, 3), 0)
        
        return obj_blur
    
    def extract_sift_features(self, img):
        """Estrai feature SIFT per matching avanzato"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            return keypoints, descriptors
        except Exception as e:
            print(f"⚠️ Errore estrazione SIFT: {e}")
            return [], None
    
    def add_person_interactive(self):
        """Aggiungi persona da file (fallback)"""
        name = input("Nome persona: ").strip()
        if name:
            print(f"\n👤 AGGIUNGI PERSONA DA FILE: {name}")
            self.add_person_from_photos(name)
    
    def add_object_interactive(self):
        """Aggiungi oggetto da file (fallback)"""
        name = input("Nome oggetto: ").strip()
        if name:
            print(f"\n📦 AGGIUNGI OGGETTO DA FILE: {name}")
            self.add_custom_object_from_photos(name)
    
    def list_all_databases(self):
        """Lista tutti i database"""
        print(f"\n📊 DATABASE UNIFICATO")
        print("=" * 40)
        print(f"👤 Persone: {len(self.person_templates)}")
        for name, templates in self.person_templates.items():
            stats = self.person_stats.get(name, {})
            recognitions = stats.get('recognitions', 0)
            print(f"  - {name}: {len(templates)} template, {recognitions} riconoscimenti")
        
        print(f"\n📦 Oggetti Custom: {len(self.object_templates)}")
        for name, templates in self.object_templates.items():
            stats = self.object_stats.get(name, {})
            recognitions = stats.get('recognitions', 0)
            print(f"  - {name}: {len(templates)} template, {recognitions} riconoscimenti")
    
    def print_unified_info(self):
        """Info sistema unificato"""
        print(f"\n📊 INFO SISTEMA UNIFICATO")
        print("=" * 50)
        print(f"🔥 Coral TPU: {'Attivo' if self.use_coral else 'Non disponibile'}")
        print(f"👤 Face Detection: {'Attivo' if self.use_face_detection else 'Non disponibile'}")
        print(f"🎮 Modalità sistema: {self.config['current_mode']}")
        print(f"🔍 Modalità detection: {self.config['detection_mode']}")
        print(f"📊 Statistiche sessione:")
        print(f"  - Coral detections: {self.unified_stats['coral_detections']}")
        print(f"  - Person recognitions: {self.unified_stats['person_recognitions']}")
        print(f"  - Object recognitions: {self.unified_stats['object_recognitions']}")
        print(f"  - Frame totali: {self.unified_stats['total_frames']}")
    
    def print_final_unified_stats(self, total_time, frame_count):
        """Statistiche finali"""
        print(f"\n📊 REPORT FINALE SISTEMA UNIFICATO")
        print("=" * 60)
        print(f"⏱️ Sessione durata: {datetime.now() - self.unified_stats['session_start']}")
        print(f"🖼️ Frame processati: {frame_count}")
        
        if frame_count > 0:
            avg_time = total_time / frame_count
            fps = 1000 / avg_time if avg_time > 0 else 0
            print(f"⚡ Tempo medio: {avg_time:.1f}ms")
            print(f"📈 FPS medio: {fps:.1f}")
        
        print(f"\n🎯 DETECTIONS TOTALI:")
        print(f"  🔥 Coral TPU: {self.unified_stats['coral_detections']}")
        print(f"  👤 Riconoscimenti persone: {self.unified_stats['person_recognitions']}")
        print(f"  📦 Oggetti custom: {self.unified_stats['object_recognitions']}")
        
        total_detections = (self.unified_stats['coral_detections'] + 
                           self.unified_stats['person_recognitions'] + 
                           self.unified_stats['object_recognitions'])
        print(f"  📊 TOTALE: {total_detections}")

def main():
    parser = argparse.ArgumentParser(description='Sistema AI Unificato - Tutte le Modalità')
    parser.add_argument('--run', action='store_true', help='Avvia sistema unificato')
    parser.add_argument('--mode', choices=['unified', 'coral_only', 'persons_only', 'objects_only'], 
                       default='unified', help='Modalità sistema')
    parser.add_argument('--detection', choices=['all', 'coral', 'face', 'custom'], 
                       default='all', help='Modalità detection')
    
    args = parser.parse_args()
    
    system = UnifiedAISystem()
    
    if args.mode:
        system.config['current_mode'] = args.mode
    if args.detection:
        system.config['detection_mode'] = args.detection
    
    if args.run:
        system.run_unified_system()
    else:
        print("\n🚀 SISTEMA AI UNIFICATO - TUTTE LE MODALITÀ")
        print("=" * 70)
        print("1. 🔍 Avvia sistema unificato completo")
        print("2. 🔥 Solo Coral TPU")
        print("3. 👤 Solo riconoscimento persone")
        print("4. 📦 Solo oggetti custom")
        print("5. 📊 Info sistema")
        print("6. 📋 Lista database")
        print("0. ❌ Esci")
        
        while True:
            try:
                choice = input("\nScegli: ").strip()
                if choice == "0":
                    break
                elif choice == "1":
                    system.config['detection_mode'] = 'all'
                    system.run_unified_system()
                elif choice == "2":
                    system.config['detection_mode'] = 'coral'
                    system.run_unified_system()
                elif choice == "3":
                    system.config['detection_mode'] = 'face'
                    system.run_unified_system()
                elif choice == "4":
                    system.config['detection_mode'] = 'custom'
                    system.run_unified_system()
                elif choice == "5":
                    system.print_unified_info()
                elif choice == "6":
                    system.list_all_databases()
                else:
                    print("❌ Opzione non valida")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
