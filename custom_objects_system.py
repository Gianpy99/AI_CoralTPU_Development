#!/usr/bin/env python3
"""
Sistema PERSONALIZZATO: Coral TPU + Riconoscimento Oggetti Custom
Addestra e riconosce oggetti personalizzati usando template matching
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

class CustomObjectRecognition:
    """Sistema completo per oggetti personalizzati + Coral TPU"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.objects_database_path = Path("custom_objects_database")
        self.objects_photos_path = Path("custom_objects_photos")
        self.objects_database_path.mkdir(exist_ok=True)
        self.objects_photos_path.mkdir(exist_ok=True)
        
        # Files database oggetti
        self.objects_templates_file = "custom_objects_templates.pkl"
        self.objects_config_file = "custom_objects_config.json"
        
        # Database oggetti personalizzati
        self.object_templates = {}
        self.object_stats = {}
        
        # Coral TPU setup
        self.coral_interpreter = None
        self.coral_labels = []
        self.use_coral = False
        
        # Configurazione
        self.config = {
            "template_size": (100, 100),
            "similarity_threshold": 0.6,
            "coral_confidence_threshold": 0.25,
            "nms_threshold": 0.5,
            "current_mode": "both",  # "coral", "custom", "both"
            "adaptive_thresholds": True,
            "multi_scale_matching": True,
            "rotation_invariant": False
        }
        
        # Soglie dinamiche Coral
        self.class_thresholds = {
            "person": 0.5,
            "car": 0.4,
            "bicycle": 0.4,
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
        
        # Descrittori per matching avanzato
        self.sift = cv2.SIFT_create(nfeatures=500)
        self.matcher = cv2.BFMatcher()
        
        # Inizializza componenti
        self.load_coral_tpu()
        self.load_objects_database()
    
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
            print(f"❌ Modello non trovato: {model_path}")
            print(f"📍 Percorso modello: {model_path.absolute()}")
            return
        
        try:
            delegate = get_edgetpu_delegate()
            if delegate:
                self.coral_interpreter = tflite.Interpreter(
                    model_path=str(model_path),
                    experimental_delegates=[delegate]
                )
                print("🔥 Modello Coral TPU caricato con EdgeTPU")
                print(f"📍 Modello: {model_path.absolute()}")
            else:
                self.coral_interpreter = tflite.Interpreter(model_path=str(model_path))
                print("💻 Modello Coral TPU caricato su CPU")
            
            self.coral_interpreter.allocate_tensors()
            self.use_coral = True
            
            if labels_path.exists():
                with open(labels_path, 'r', encoding='utf-8') as f:
                    self.coral_labels = [line.strip() for line in f.readlines()]
                print(f"✅ Labels caricati: {len(self.coral_labels)} classi COCO")
            
        except Exception as e:
            print(f"❌ Errore caricamento Coral TPU: {e}")
    
    def load_objects_database(self):
        """Carica database oggetti personalizzati"""
        if Path(self.objects_templates_file).exists():
            try:
                with open(self.objects_templates_file, 'rb') as f:
                    data = pickle.load(f)
                    self.object_templates = data.get('templates', {})
                    self.object_stats = data.get('stats', {})
                print(f"✅ Database oggetti: {len(self.object_templates)} oggetti custom")
            except Exception as e:
                print(f"⚠️ Errore caricamento database oggetti: {e}")
        
        if Path(self.objects_config_file).exists():
            try:
                with open(self.objects_config_file, 'r') as f:
                    self.config.update(json.load(f))
            except Exception as e:
                print(f"⚠️ Errore config oggetti: {e}")
    
    def save_objects_database(self):
        """Salva database oggetti"""
        try:
            data = {
                'templates': self.object_templates,
                'stats': self.object_stats,
                'created': datetime.now().isoformat()
            }
            
            with open(self.objects_templates_file, 'wb') as f:
                pickle.dump(data, f)
            
            with open(self.objects_config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ Errore salvataggio database oggetti: {e}")
            return False
    
    def select_photos_gui(self, object_name):
        """Seleziona foto usando interfaccia grafica"""
        print(f"📁 Seleziona foto per oggetto '{object_name}'...")
        
        root = tk.Tk()
        root.withdraw()
        
        file_paths = filedialog.askopenfilenames(
            title=f"Seleziona foto per {object_name}",
            filetypes=[
                ("Immagini", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
                ("Tutti i file", "*.*")
            ]
        )
        
        root.destroy()
        return list(file_paths) if file_paths else []
    
    def preprocess_object_template(self, obj_img):
        """Preprocessa oggetto per template matching"""
        # Resize
        obj_resized = cv2.resize(obj_img, self.config["template_size"])
        
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
    
    def add_custom_object_from_photos(self, object_name):
        """Aggiungi oggetto personalizzato usando foto"""
        print(f"\n🎯 AGGIUNGI OGGETTO PERSONALIZZATO: {object_name}")
        
        # Seleziona foto
        photo_paths = self.select_photos_gui(object_name)
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
            
            if self.save_objects_database():
                print(f"💾 Database oggetti salvato")
                return True
        
        print(f"❌ Nessun template valido estratto")
        return False
    
    def calculate_template_similarity(self, template1, template2):
        """Calcola similarità tra template usando template matching"""
        try:
            result = cv2.matchTemplate(template1, template2, cv2.TM_CCOEFF_NORMED)
            return result[0][0]
        except:
            return 0.0
    
    def calculate_sift_similarity(self, desc1, desc2):
        """Calcola similarità usando descrittori SIFT"""
        if desc1 is None or desc2 is None:
            return 0.0
        
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Similarità basata sul numero di good matches
            if len(desc1) > 0:
                similarity = len(good_matches) / len(desc1)
                return min(similarity, 1.0)
            
            return 0.0
        except:
            return 0.0
    
    def recognize_custom_object(self, img_region):
        """Riconosce oggetto personalizzato in una regione"""
        if not self.object_templates:
            return "Sconosciuto", 0.0
        
        # Prepara template di input
        input_template = self.preprocess_object_template(img_region)
        input_kp, input_desc = self.extract_sift_features(img_region)
        
        best_match = "Sconosciuto"
        best_score = 0.0
        
        for object_name, templates in self.object_templates.items():
            object_scores = []
            
            for template_data in templates:
                # Template matching classico
                template_sim = self.calculate_template_similarity(
                    input_template, 
                    template_data['template']
                )
                
                # SIFT matching se disponibile
                sift_sim = 0.0
                if input_desc is not None and template_data.get('descriptors') is not None:
                    sift_sim = self.calculate_sift_similarity(
                        input_desc, 
                        template_data['descriptors']
                    )
                
                # Combina i punteggi
                if self.config.get('multi_scale_matching', True):
                    combined_score = (template_sim * 0.6) + (sift_sim * 0.4)
                else:
                    combined_score = template_sim
                
                object_scores.append(combined_score)
            
            if object_scores:
                object_best_score = max(object_scores)
                if object_best_score > best_score:
                    best_score = object_best_score
                    best_match = object_name
        
        # Applica soglia
        if best_score < self.config["similarity_threshold"]:
            best_match = "Sconosciuto"
        else:
            # Aggiorna statistiche
            if best_match in self.object_stats:
                self.object_stats[best_match]['recognitions'] += 1
                self.object_stats[best_match]['last_seen'] = datetime.now().isoformat()
        
        return best_match, best_score
    
    def coral_detect_objects(self, frame):
        """Detection oggetti con Coral TPU (metodo esistente)"""
        if not self.use_coral or not self.coral_interpreter:
            return []
        
        try:
            input_details = self.coral_interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            frame_resized = cv2.resize(frame, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0).astype(np.uint8)
            
            start_time = time.time()
            self.coral_interpreter.set_tensor(input_details[0]['index'], input_data)
            self.coral_interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000
            
            output_details = self.coral_interpreter.get_output_details()
            locations = self.coral_interpreter.get_tensor(output_details[0]['index'])
            classes = self.coral_interpreter.get_tensor(output_details[1]['index'])
            scores = self.coral_interpreter.get_tensor(output_details[2]['index'])
            
            frame_height, frame_width = frame.shape[:2]
            detections = []
            
            for i in range(len(scores[0])):
                score = scores[0][i]
                class_id = int(classes[0][i])
                
                if class_id >= len(self.coral_labels):
                    continue
                
                class_name = self.coral_labels[class_id]
                threshold = self.class_thresholds.get(class_name, self.config["coral_confidence_threshold"])
                
                if score > threshold:
                    ymin, xmin, ymax, xmax = locations[0][i]
                    xmin = int(xmin * frame_width)
                    ymin = int(ymin * frame_height)
                    xmax = int(xmax * frame_width)
                    ymax = int(ymax * frame_height)
                    
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
    
    def detect_custom_objects_sliding_window(self, frame):
        """Rileva oggetti personalizzati usando sliding window"""
        if not self.object_templates:
            return []
        
        start_time = time.time()
        custom_detections = []
        
        # Parametri sliding window
        window_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        step_size = 50
        
        frame_height, frame_width = frame.shape[:2]
        
        for scale in window_scales:
            # Dimensione finestra scalata
            window_w = int(self.config["template_size"][0] * scale)
            window_h = int(self.config["template_size"][1] * scale)
            
            if window_w > frame_width or window_h > frame_height:
                continue
            
            # Sliding window
            for y in range(0, frame_height - window_h, step_size):
                for x in range(0, frame_width - window_w, step_size):
                    # Estrai regione
                    window = frame[y:y+window_h, x:x+window_w]
                    
                    # Riconosci oggetto
                    object_name, confidence = self.recognize_custom_object(window)
                    
                    if object_name != "Sconosciuto" and confidence > self.config["similarity_threshold"]:
                        custom_detections.append({
                            'bbox': [x, y, x+window_w, y+window_h],
                            'class': object_name,
                            'confidence': confidence,
                            'source': 'custom'
                        })
        
        detection_time = (time.time() - start_time) * 1000
        
        # Applica NMS per rimuovere detection sovrapposte
        if custom_detections:
            custom_detections = self._apply_nms_custom(custom_detections)
        
        return custom_detections, detection_time
    
    def _apply_nms_custom(self, detections):
        """Applica NMS ai custom detections"""
        if len(detections) <= 1:
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
            
            # Prepara per NMS
            boxes = []
            scores = []
            
            for det in class_dets:
                x1, y1, x2, y2 = det['bbox']
                boxes.append([x1, y1, x2-x1, y2-y1])
                scores.append(det['confidence'])
            
            # Applica NMS
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, 
                self.config["similarity_threshold"], 
                self.config["nms_threshold"]
            )
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    filtered_detections.append(class_dets[idx])
        
        return filtered_detections
    
    def get_detection_color(self, class_name, source):
        """Ottieni colore basato su classe e sorgente"""
        if source == 'custom':
            # Colori speciali per oggetti custom
            return (0, 255, 255)  # Ciano brillante per oggetti custom
        else:  # source == 'coral'
            if class_name == "person":
                return (255, 165, 0)  # Arancione
            elif class_name in ["car", "bicycle", "motorcycle", "bus", "truck"]:
                return (0, 255, 255)  # Ciano per veicoli
            elif class_name in ["bottle", "cup", "wine glass"]:
                return (0, 255, 0)  # Verde per oggetti tavola
            elif class_name in ["phone", "laptop", "mouse", "keyboard"]:
                return (255, 255, 0)  # Giallo per elettronica
            else:
                return (128, 255, 128)  # Verde chiaro per altri
    
    def remove_custom_object(self, object_name=None):
        """Rimuovi oggetto personalizzato"""
        if not self.object_templates:
            print("❌ Database oggetti vuoto")
            return False
        
        if not object_name:
            print("\n🗑️ RIMUOVI OGGETTO PERSONALIZZATO")
            objects_list = list(self.object_templates.keys())
            
            for i, name in enumerate(objects_list, 1):
                templates_count = len(self.object_templates[name])
                recognitions = self.object_stats.get(name, {}).get('recognitions', 0)
                print(f"  {i}. {name} ({templates_count} template, {recognitions} riconoscimenti)")
            
            print("  0. Annulla")
            
            try:
                choice = input("\nScegli oggetto da rimuovere: ").strip()
                if choice == "0":
                    return False
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(objects_list):
                    object_name = objects_list[choice_idx]
                else:
                    print("❌ Scelta non valida")
                    return False
            except:
                return False
        
        if object_name not in self.object_templates:
            print(f"❌ Oggetto '{object_name}' non trovato")
            return False
        
        # Conferma
        templates_count = len(self.object_templates[object_name])
        print(f"\n⚠️ Rimuovere '{object_name}' con {templates_count} template?")
        
        confirm = input("Conferma (si/no): ").strip().lower()
        if confirm not in ['si', 'sì', 'yes', 'y']:
            return False
        
        # Rimuovi
        try:
            del self.object_templates[object_name]
            if object_name in self.object_stats:
                del self.object_stats[object_name]
            
            # Rimuovi directory
            object_dir = self.objects_photos_path / object_name
            if object_dir.exists():
                shutil.rmtree(object_dir)
            
            self.save_objects_database()
            print(f"✅ Oggetto '{object_name}' rimosso")
            return True
            
        except Exception as e:
            print(f"❌ Errore rimozione: {e}")
            return False
    
    def list_custom_objects(self):
        """Lista oggetti personalizzati"""
        if not self.object_templates:
            print("📭 Nessun oggetto personalizzato")
            return
        
        print(f"\n🎯 OGGETTI PERSONALIZZATI ({len(self.object_templates)} oggetti)")
        print("=" * 50)
        
        for i, (name, templates) in enumerate(self.object_templates.items(), 1):
            stats = self.object_stats.get(name, {})
            print(f"\n{i}. 🎯 {name}")
            print(f"   Template: {len(templates)}")
            print(f"   Riconoscimenti: {stats.get('recognitions', 0)}")
            print(f"   Aggiunto: {stats.get('added_date', 'N/A')[:19]}")
            print(f"   Ultimo visto: {stats.get('last_seen', 'Mai')[:19] if stats.get('last_seen') else 'Mai'}")
    
    def run_complete_custom_system(self):
        """Sistema completo con oggetti personalizzati"""
        print("🚀 SISTEMA CORAL TPU + OGGETTI PERSONALIZZATI")
        print("=" * 60)
        print(f"🔥 Coral TPU: {'Attivo' if self.use_coral else 'Non disponibile'}")
        print(f"🎯 Oggetti custom: {len(self.object_templates)} oggetti")
        print(f"📋 Classi COCO: {len(self.coral_labels)} classi")
        print(f"🎮 Modalità: {self.config['current_mode']}")
        
        # Lista oggetti custom
        if self.object_templates:
            print("\n🎯 Oggetti personalizzati:")
            for name, templates in self.object_templates.items():
                stats = self.object_stats.get(name, {})
                recognitions = stats.get('recognitions', 0)
                print(f"  - {name}: {len(templates)} template, {recognitions} riconoscimenti")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        frame_count = 0
        total_coral_time = 0
        total_custom_time = 0
        
        print("\n🎮 CONTROLLI:")
        print("  Q = Esci")
        print("  S = Salva screenshot")
        print("  M = Cambia modalità (coral/custom/both)")
        print("  A = Aggiungi oggetto custom")
        print("  D = Rimuovi oggetto custom")
        print("  L = Lista oggetti custom")
        print("  + = Aumenta soglia")
        print("  - = Diminuisci soglia")
        print("  I = Info sistema")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                coral_detections = []
                custom_detections = []
                coral_time = 0
                custom_time = 0
                
                # Coral TPU Detection
                if self.config['current_mode'] in ['coral', 'both'] and self.use_coral:
                    coral_detections, coral_time = self.coral_detect_objects(frame)
                    total_coral_time += coral_time
                
                # Custom Objects Detection
                if self.config['current_mode'] in ['custom', 'both'] and self.object_templates:
                    custom_detections, custom_time = self.detect_custom_objects_sliding_window(frame)
                    total_custom_time += custom_time
                
                # Disegna Coral detections
                for detection in coral_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    class_name = detection['class']
                    confidence = detection['confidence']
                    
                    color = self.get_detection_color(class_name, 'coral')
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"[CORAL] {class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Disegna Custom detections
                for detection in custom_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    object_name = detection['class']
                    confidence = detection['confidence']
                    
                    color = self.get_detection_color(object_name, 'custom')
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"[CUSTOM] {object_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Info overlay
                cv2.putText(frame, f"Modalità: {self.config['current_mode']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Coral: {coral_time:.1f}ms | Custom: {custom_time:.1f}ms", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"COCO: {len(coral_detections)} | Custom: {len(custom_detections)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.putText(frame, f"Oggetti custom: {len(self.object_templates)} | Soglia: {self.config['similarity_threshold']:.2f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Controlli
                cv2.putText(frame, "Q=esci | S=salva | M=modalità | A=aggiungi | D=rimuovi | L=lista", 
                           (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
                cv2.imshow('🚀 Sistema Coral TPU + Oggetti Personalizzati', frame)
                
                # Input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"custom_objects_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Salvato: {filename}")
                elif key == ord('m'):
                    modes = ['coral', 'custom', 'both']
                    current_idx = modes.index(self.config['current_mode'])
                    next_idx = (current_idx + 1) % len(modes)
                    self.config['current_mode'] = modes[next_idx]
                    print(f"🔄 Modalità: {self.config['current_mode']}")
                elif key == ord('a'):
                    cv2.destroyAllWindows()
                    name = input("Nome oggetto da aggiungere: ").strip()
                    if name:
                        self.add_custom_object_from_photos(name)
                    cap = cv2.VideoCapture(0)
                elif key == ord('d'):
                    cv2.destroyAllWindows()
                    self.remove_custom_object()
                    cap = cv2.VideoCapture(0)
                elif key == ord('l'):
                    self.list_custom_objects()
                elif key == ord('+'):
                    self.config['similarity_threshold'] = min(0.95, self.config['similarity_threshold'] + 0.05)
                    print(f"🔧 Soglia: {self.config['similarity_threshold']:.2f}")
                elif key == ord('-'):
                    self.config['similarity_threshold'] = max(0.2, self.config['similarity_threshold'] - 0.05)
                    print(f"🔧 Soglia: {self.config['similarity_threshold']:.2f}")
                elif key == ord('i'):
                    self.print_system_info()
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_objects_database()
            
            # Report finale
            if frame_count > 0:
                print(f"\n📊 STATISTICHE FINALI")
                print(f"Frame processati: {frame_count}")
                if total_coral_time > 0:
                    avg_coral = total_coral_time / frame_count
                    print(f"Coral TPU medio: {avg_coral:.1f}ms")
                if total_custom_time > 0:
                    avg_custom = total_custom_time / frame_count
                    print(f"Custom objects medio: {avg_custom:.1f}ms")
    
    def print_system_info(self):
        """Info dettagliate sistema"""
        print(f"\n📊 INFO SISTEMA COMPLETO")
        print("=" * 40)
        print(f"🔥 Coral TPU: {'Attivo' if self.use_coral else 'Non disponibile'}")
        print(f"📍 Modello: {self.models_dir / 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'}")
        print(f"📋 Classi COCO: {len(self.coral_labels)}")
        print(f"🎯 Oggetti custom: {len(self.object_templates)}")
        print(f"⚙️ Template size: {self.config['template_size']}")
        print(f"🎯 Soglia similarità: {self.config['similarity_threshold']}")
        print(f"🔍 NMS threshold: {self.config['nms_threshold']}")
        print(f"🎮 Modalità: {self.config['current_mode']}")

def main():
    parser = argparse.ArgumentParser(description='Sistema Oggetti Personalizzati + Coral TPU')
    parser.add_argument('--run', action='store_true', help='Avvia sistema completo')
    parser.add_argument('--add-object', type=str, help='Aggiungi oggetto da foto')
    parser.add_argument('--remove-object', type=str, help='Rimuovi oggetto')
    parser.add_argument('--list-objects', action='store_true', help='Lista oggetti custom')
    parser.add_argument('--model-info', action='store_true', help='Info modello Coral TPU')
    
    args = parser.parse_args()
    
    system = CustomObjectRecognition()
    
    if args.add_object:
        system.add_custom_object_from_photos(args.add_object)
    elif args.remove_object:
        system.remove_custom_object(args.remove_object)
    elif args.list_objects:
        system.list_custom_objects()
    elif args.model_info:
        system.print_system_info()
    elif args.run:
        system.run_complete_custom_system()
    else:
        print("\n🚀 SISTEMA OGGETTI PERSONALIZZATI + CORAL TPU")
        print("=" * 60)
        print("1. 🔍 Avvia sistema completo")
        print("2. 🎯 Aggiungi oggetto personalizzato")
        print("3. 🗑️ Rimuovi oggetto")
        print("4. 📋 Lista oggetti custom")
        print("5. 📊 Info sistema e modello")
        print("0. ❌ Esci")
        
        while True:
            try:
                choice = input("\nScegli: ").strip()
                if choice == "0":
                    break
                elif choice == "1":
                    system.run_complete_custom_system()
                elif choice == "2":
                    name = input("Nome oggetto: ").strip()
                    if name:
                        system.add_custom_object_from_photos(name)
                elif choice == "3":
                    system.remove_custom_object()
                elif choice == "4":
                    system.list_custom_objects()
                elif choice == "5":
                    system.print_system_info()
                else:
                    print("❌ Opzione non valida")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
