#!/usr/bin/env python3
"""
Sistema COMPLETO: Coral TPU + Riconoscimento Personalizzato + Immagini Statiche
Combina detection oggetti con riconoscimento persone usando foto
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

class CompleteRecognitionSystem:
    """Sistema completo Coral TPU + Riconoscimento personalizzato"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.database_path = Path("people_database")
        self.photos_path = Path("people_photos")
        self.database_path.mkdir(exist_ok=True)
        self.photos_path.mkdir(exist_ok=True)
        
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
            "coral_confidence_threshold": 0.25,
            "nms_threshold": 0.5,
            "current_mode": "both",  # "detection", "personal", "both"
            "adaptive_thresholds": True
        }
        
        # Soglie dinamiche Coral
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
        
        # Statistiche
        self.detection_stats = {}
        self.frame_count = 0
        
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
            return
        
        try:
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
    
    def select_photos_gui(self, person_name):
        """Seleziona foto usando interfaccia grafica"""
        print(f"📁 Seleziona foto per {person_name}...")
        
        # Crea finestra Tkinter nascosta
        root = tk.Tk()
        root.withdraw()
        
        # Seleziona multiple foto
        file_paths = filedialog.askopenfilenames(
            title=f"Seleziona foto per {person_name}",
            filetypes=[
                ("Immagini", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
                ("Tutti i file", "*.*")
            ]
        )
        
        root.destroy()
        return list(file_paths) if file_paths else []
    
    def preprocess_face(self, face_img):
        """Preprocessa volto per template matching"""
        face_resized = cv2.resize(face_img, self.config["template_size"])
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
        face_eq = cv2.equalizeHist(face_gray)
        return cv2.GaussianBlur(face_eq, (3, 3), 0)
    
    def extract_faces_from_image(self, image_path):
        """Estrae volti da immagine statica"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"❌ Impossibile caricare: {image_path}")
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            face_templates = []
            for i, (x, y, w, h) in enumerate(faces):
                face_roi = img[y:y+h, x:x+w]
                template = self.preprocess_face(face_roi)
                face_templates.append(template)
                
                # Salva estratto per verifica
                person_dir = self.photos_path / "extracted_faces"
                person_dir.mkdir(exist_ok=True)
                face_path = person_dir / f"{Path(image_path).stem}_face_{i}.jpg"
                cv2.imwrite(str(face_path), face_roi)
            
            print(f"✅ Estratti {len(face_templates)} volti da {Path(image_path).name}")
            return face_templates
            
        except Exception as e:
            print(f"❌ Errore estrazione volti: {e}")
            return []
    
    def add_person_from_photos(self, person_name):
        """Aggiungi persona usando foto statiche"""
        print(f"\n📷 AGGIUNGI PERSONA DA FOTO: {person_name}")
        
        # Seleziona foto
        photo_paths = self.select_photos_gui(person_name)
        if not photo_paths:
            print("❌ Nessuna foto selezionata")
            return False
        
        print(f"📁 Selezionate {len(photo_paths)} foto")
        
        # Crea directory persona
        person_dir = self.photos_path / person_name
        person_dir.mkdir(exist_ok=True)
        
        all_templates = []
        processed_photos = 0
        
        for photo_path in photo_paths:
            try:
                # Copia foto nella directory persona
                photo_name = f"{person_name}_{processed_photos:03d}{Path(photo_path).suffix}"
                dest_path = person_dir / photo_name
                shutil.copy2(photo_path, dest_path)
                
                # Estrai volti
                face_templates = self.extract_faces_from_image(photo_path)
                if face_templates:
                    all_templates.extend(face_templates)
                    processed_photos += 1
                    print(f"✅ Processata: {Path(photo_path).name} -> {len(face_templates)} volti")
                else:
                    print(f"⚠️ Nessun volto trovato in: {Path(photo_path).name}")
                
            except Exception as e:
                print(f"❌ Errore processando {photo_path}: {e}")
        
        # Salva nel database
        if all_templates:
            if person_name in self.person_templates:
                # Aggiungi ai template esistenti
                existing_count = len(self.person_templates[person_name])
                self.person_templates[person_name].extend(all_templates)
                self.recognition_stats[person_name]['templates_count'] = len(self.person_templates[person_name])
                self.recognition_stats[person_name]['updated_date'] = datetime.now().isoformat()
                
                print(f"✅ Aggiunti {len(all_templates)} template a {person_name}")
                print(f"📊 Template totali: {existing_count} + {len(all_templates)} = {len(self.person_templates[person_name])}")
            else:
                # Nuova persona
                self.person_templates[person_name] = all_templates
                self.recognition_stats[person_name] = {
                    'added_date': datetime.now().isoformat(),
                    'templates_count': len(all_templates),
                    'recognitions': 0,
                    'last_seen': None,
                    'photos_processed': processed_photos
                }
                
                print(f"✅ Nuova persona {person_name} con {len(all_templates)} template")
            
            if self.save_database():
                print(f"💾 Database salvato con successo")
                return True
        
        print(f"❌ Nessun template valido estratto")
        return False
    
    def add_person_camera(self, person_name, num_photos=6):
        """Aggiungi persona tramite camera (modalità esistente)"""
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
        
        person_dir = self.photos_path / person_name
        person_dir.mkdir(exist_ok=True)
        
        while captured_photos < num_photos:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
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
                    
                    photo_path = person_dir / f"{person_name}_cam_{captured_photos:03d}.jpg"
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
            if person_name in self.person_templates:
                # Aggiungi ai template esistenti
                existing_count = len(self.person_templates[person_name])
                self.person_templates[person_name].extend(captured_templates)
                self.recognition_stats[person_name]['templates_count'] = len(self.person_templates[person_name])
                
                print(f"✅ Aggiunti {len(captured_templates)} template da camera a {person_name}")
                print(f"📊 Template totali: {existing_count} + {len(captured_templates)} = {len(self.person_templates[person_name])}")
            else:
                # Nuova persona
                self.person_templates[person_name] = captured_templates
                self.recognition_stats[person_name] = {
                    'added_date': datetime.now().isoformat(),
                    'templates_count': len(captured_templates),
                    'recognitions': 0,
                    'last_seen': None
                }
            
            if self.save_database():
                return True
        
        print(f"❌ Registrazione fallita")
        return False
    
    def remove_person(self, person_name=None):
        """Rimuovi persona dal database"""
        if not self.person_templates:
            print("❌ Database vuoto - nessuna persona da rimuovere")
            return False
        
        # Se non specificato, chiedi quale persona rimuovere
        if not person_name:
            print("\n🗑️ RIMUOVI PERSONA")
            print("Persone disponibili:")
            
            people_list = list(self.person_templates.keys())
            for i, name in enumerate(people_list, 1):
                templates_count = len(self.person_templates[name])
                recognitions = self.recognition_stats.get(name, {}).get('recognitions', 0)
                print(f"  {i}. {name} ({templates_count} template, {recognitions} riconoscimenti)")
            
            print("  0. Annulla")
            
            try:
                choice = input("\nScegli persona da rimuovere (numero): ").strip()
                if choice == "0":
                    print("❌ Operazione annullata")
                    return False
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(people_list):
                    person_name = people_list[choice_idx]
                else:
                    print("❌ Scelta non valida")
                    return False
                    
            except (ValueError, KeyboardInterrupt):
                print("❌ Operazione annullata")
                return False
        
        # Conferma rimozione
        if person_name not in self.person_templates:
            print(f"❌ Persona '{person_name}' non trovata nel database")
            return False
        
        templates_count = len(self.person_templates[person_name])
        recognitions = self.recognition_stats.get(person_name, {}).get('recognitions', 0)
        
        print(f"\n⚠️ CONFERMA RIMOZIONE")
        print(f"Persona: {person_name}")
        print(f"Template: {templates_count}")
        print(f"Riconoscimenti: {recognitions}")
        
        # Directory foto
        person_dir = self.photos_path / person_name
        photos_count = 0
        if person_dir.exists():
            photos = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            photos_count = len(photos)
            print(f"Foto salvate: {photos_count}")
        
        try:
            confirm = input("\n❓ Sei sicuro? Questa azione è IRREVERSIBILE! (si/no): ").strip().lower()
            if confirm not in ['si', 'sì', 'yes', 'y']:
                print("❌ Rimozione annullata")
                return False
        except KeyboardInterrupt:
            print("\n❌ Rimozione annullata")
            return False
        
        # Rimuovi dal database
        try:
            del self.person_templates[person_name]
            if person_name in self.recognition_stats:
                del self.recognition_stats[person_name]
            
            # Rimuovi directory foto (opzionale)
            if person_dir.exists():
                try:
                    remove_photos = input("❓ Rimuovere anche le foto salvate? (si/no): ").strip().lower()
                    if remove_photos in ['si', 'sì', 'yes', 'y']:
                        shutil.rmtree(person_dir)
                        print(f"🗑️ Directory foto rimossa: {person_dir}")
                    else:
                        print(f"📁 Foto conservate in: {person_dir}")
                except Exception as e:
                    print(f"⚠️ Errore rimozione foto: {e}")
            
            # Salva database aggiornato
            if self.save_database():
                print(f"✅ Persona '{person_name}' rimossa con successo")
                print(f"📊 Persone rimanenti: {len(self.person_templates)}")
                return True
            else:
                print(f"❌ Errore salvataggio database")
                return False
                
        except Exception as e:
            print(f"❌ Errore durante rimozione: {e}")
            return False
    
    def list_people(self):
        """Lista tutte le persone nel database"""
        if not self.person_templates:
            print("📭 Database vuoto - nessuna persona registrata")
            return
        
        print(f"\n👥 DATABASE PERSONE ({len(self.person_templates)} persone)")
        print("=" * 50)
        
        for i, (name, templates) in enumerate(self.person_templates.items(), 1):
            stats = self.recognition_stats.get(name, {})
            
            print(f"\n{i}. 👤 {name}")
            print(f"   Template: {len(templates)}")
            print(f"   Riconoscimenti: {stats.get('recognitions', 0)}")
            print(f"   Aggiunto: {stats.get('added_date', 'N/A')[:19]}")
            print(f"   Ultimo visto: {stats.get('last_seen', 'Mai')[:19] if stats.get('last_seen') else 'Mai'}")
            
            # Controlla foto salvate
            person_dir = self.photos_path / name
            if person_dir.exists():
                photos = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
                print(f"   Foto salvate: {len(photos)}")
            else:
                print(f"   Foto salvate: 0")
    
    def cleanup_database(self):
        """Pulizia avanzata database"""
        print(f"\n🧹 PULIZIA DATABASE")
        print("=" * 30)
        
        if not self.person_templates:
            print("📭 Database già vuoto")
            return
        
        print("Opzioni di pulizia:")
        print("1. Rimuovi persone senza riconoscimenti")
        print("2. Rimuovi persone con pochi template (< 3)")
        print("3. Rimuovi tutte le persone (RESET COMPLETO)")
        print("4. Rimuovi solo foto orfane")
        print("0. Annulla")
        
        try:
            choice = input("\nScegli opzione: ").strip()
            
            if choice == "0":
                print("❌ Pulizia annullata")
                return
            elif choice == "1":
                self._cleanup_no_recognitions()
            elif choice == "2":
                self._cleanup_few_templates()
            elif choice == "3":
                self._cleanup_all()
            elif choice == "4":
                self._cleanup_orphan_photos()
            else:
                print("❌ Opzione non valida")
        except KeyboardInterrupt:
            print("\n❌ Pulizia annullata")
    
    def _cleanup_no_recognitions(self):
        """Rimuovi persone senza riconoscimenti"""
        to_remove = []
        
        for name, stats in self.recognition_stats.items():
            if stats.get('recognitions', 0) == 0:
                to_remove.append(name)
        
        if not to_remove:
            print("✅ Nessuna persona senza riconoscimenti")
            return
        
        print(f"🎯 Trovate {len(to_remove)} persone senza riconoscimenti:")
        for name in to_remove:
            templates = len(self.person_templates.get(name, []))
            print(f"  - {name} ({templates} template)")
        
        confirm = input(f"\n❓ Rimuovere {len(to_remove)} persone? (si/no): ").strip().lower()
        if confirm in ['si', 'sì', 'yes', 'y']:
            for name in to_remove:
                if name in self.person_templates:
                    del self.person_templates[name]
                if name in self.recognition_stats:
                    del self.recognition_stats[name]
            
            self.save_database()
            print(f"✅ Rimosse {len(to_remove)} persone senza riconoscimenti")
    
    def _cleanup_few_templates(self):
        """Rimuovi persone con pochi template"""
        to_remove = []
        
        for name, templates in self.person_templates.items():
            if len(templates) < 3:
                to_remove.append(name)
        
        if not to_remove:
            print("✅ Nessuna persona con pochi template")
            return
        
        print(f"🎯 Trovate {len(to_remove)} persone con < 3 template:")
        for name in to_remove:
            templates = len(self.person_templates[name])
            print(f"  - {name} ({templates} template)")
        
        confirm = input(f"\n❓ Rimuovere {len(to_remove)} persone? (si/no): ").strip().lower()
        if confirm in ['si', 'sì', 'yes', 'y']:
            for name in to_remove:
                if name in self.person_templates:
                    del self.person_templates[name]
                if name in self.recognition_stats:
                    del self.recognition_stats[name]
            
            self.save_database()
            print(f"✅ Rimosse {len(to_remove)} persone con pochi template")
    
    def _cleanup_all(self):
        """Reset completo database"""
        count = len(self.person_templates)
        
        confirm = input(f"\n⚠️ RESET COMPLETO - Rimuovere TUTTE le {count} persone? (RESET/no): ").strip()
        if confirm == "RESET":
            self.person_templates.clear()
            self.recognition_stats.clear()
            self.save_database()
            print(f"🧹 Database completamente pulito - {count} persone rimosse")
        else:
            print("❌ Reset annullato")
    
    def _cleanup_orphan_photos(self):
        """Rimuovi foto di persone non più nel database"""
        if not self.photos_path.exists():
            print("📁 Directory foto non esistente")
            return
        
        orphan_dirs = []
        for person_dir in self.photos_path.iterdir():
            if person_dir.is_dir() and person_dir.name not in self.person_templates and person_dir.name != "extracted_faces":
                orphan_dirs.append(person_dir)
        
        if not orphan_dirs:
            print("✅ Nessuna foto orfana trovata")
            return
        
        print(f"🎯 Trovate {len(orphan_dirs)} directory foto orfane:")
        for dir_path in orphan_dirs:
            photos = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
            print(f"  - {dir_path.name} ({len(photos)} foto)")
        
        confirm = input(f"\n❓ Rimuovere {len(orphan_dirs)} directory orfane? (si/no): ").strip().lower()
        if confirm in ['si', 'sì', 'yes', 'y']:
            for dir_path in orphan_dirs:
                try:
                    shutil.rmtree(dir_path)
                    print(f"🗑️ Rimossa: {dir_path.name}")
                except Exception as e:
                    print(f"❌ Errore rimozione {dir_path.name}: {e}")
            
            print(f"✅ Pulizia foto completata")
    
    def get_dynamic_threshold(self, class_name):
        """Ottieni soglia dinamica per classe Coral"""
        if not self.config["adaptive_thresholds"]:
            return self.config["coral_confidence_threshold"]
        
        if class_name in self.class_thresholds:
            return self.class_thresholds[class_name]
        
        return self.config["coral_confidence_threshold"]
    
    def coral_detect_objects(self, frame):
        """Detection oggetti con Coral TPU"""
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
                threshold = self.get_dynamic_threshold(class_name)
                
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
    
    def get_detection_color(self, class_name, confidence, source):
        """Ottieni colore basato su classe, confidenza e sorgente"""
        if source == 'personal':
            if class_name != "Sconosciuto":
                return (0, 255, 0)  # Verde brillante per persone conosciute
            else:
                return (0, 0, 255)  # Rosso per sconosciuti
        else:  # source == 'coral'
            if class_name == "person":
                return (255, 165, 0)  # Arancione per persone generiche
            elif class_name in ["car", "bicycle", "motorcycle", "bus", "truck"]:
                return (0, 255, 255)  # Ciano per veicoli
            elif class_name in ["dog", "cat", "bird", "horse"]:
                return (255, 0, 255)  # Magenta per animali
            elif class_name in ["bottle", "cup", "wine glass", "fork", "knife", "spoon"]:
                return (0, 255, 0)  # Verde per oggetti tavola
            elif class_name in ["phone", "laptop", "mouse", "keyboard", "tv", "remote"]:
                return (255, 255, 0)  # Giallo per elettronica
            else:
                return (128, 255, 128)  # Verde chiaro per altri oggetti
    
    def run_complete_system(self):
        """Sistema completo con tutte le modalità"""
        print("🚀 SISTEMA COMPLETO CORAL TPU + PERSONE")
        print("=" * 50)
        print(f"🔥 Coral TPU: {'Attivo' if self.use_coral else 'Non disponibile'}")
        print(f"👥 Database persone: {len(self.person_templates)} persone")
        print(f"🎯 Modalità: {self.config['current_mode']}")
        
        # Mostra persone nel database
        if self.person_templates:
            print("\n👤 Persone nel database:")
            for name, templates in self.person_templates.items():
                stats = self.recognition_stats.get(name, {})
                recognitions = stats.get('recognitions', 0)
                print(f"  - {name}: {len(templates)} template, {recognitions} riconoscimenti")
        
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
        print("  A = Aggiungi persona (camera)")
        print("  P = Aggiungi persona (foto)")
        print("  D = Rimuovi persona")
        print("  L = Lista persone")
        print("  C = Pulizia database")
        print("  + = Aumenta soglia")
        print("  - = Diminuisci soglia")
        print("  I = Info database")
        
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
                    
                    color = self.get_detection_color(class_name, confidence, 'coral')
                    thickness = 3 if 'person' in class_name.lower() else 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    label = f"[CORAL] {class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Disegna Personal detections
                for detection in personal_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    name = detection['class']
                    confidence = detection['confidence']
                    
                    color = self.get_detection_color(name, confidence, 'personal')
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
                
                cv2.putText(frame, f"Soglia: {self.config['similarity_threshold']:.2f} | Persone: {len(self.person_templates)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Contatori
                coral_persons = len([d for d in coral_detections if 'person' in d['class'].lower()])
                known_persons = len([d for d in personal_detections if d['class'] != "Sconosciuto"])
                
                cv2.putText(frame, f"Coral persone: {coral_persons} | Conosciute: {known_persons}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Controlli
                cv2.putText(frame, "Q=esci | S=salva | M=modalità | A=cam | P=foto | D=rimuovi | L=lista | C=pulizia", 
                           (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                
                cv2.imshow('🚀 Sistema Completo Coral TPU + Persone', frame)
                
                # Input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"complete_system_{timestamp}.jpg"
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
                    name = input("Nome persona da aggiungere (camera): ").strip()
                    if name:
                        self.add_person_camera(name, 6)
                    cap = cv2.VideoCapture(0)
                elif key == ord('p'):
                    cv2.destroyAllWindows()
                    name = input("Nome persona da aggiungere (foto): ").strip()
                    if name:
                        self.add_person_from_photos(name)
                    cap = cv2.VideoCapture(0)
                elif key == ord('d'):
                    cv2.destroyAllWindows()
                    self.remove_person()
                    cap = cv2.VideoCapture(0)
                elif key == ord('l'):
                    self.list_people()
                elif key == ord('c'):
                    cv2.destroyAllWindows()
                    self.cleanup_database()
                    cap = cv2.VideoCapture(0)
                elif key == ord('+') or key == ord('='):
                    self.config['similarity_threshold'] = min(0.95, self.config['similarity_threshold'] + 0.05)
                    print(f"🔧 Soglia: {self.config['similarity_threshold']:.2f}")
                elif key == ord('-'):
                    self.config['similarity_threshold'] = max(0.2, self.config['similarity_threshold'] - 0.05)
                    print(f"🔧 Soglia: {self.config['similarity_threshold']:.2f}")
                elif key == ord('i'):
                    self.print_database_info()
        
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
    
    def print_database_info(self):
        """Stampa info dettagliate del database"""
        print(f"\n📊 DATABASE PERSONE")
        print("=" * 40)
        print(f"Persone totali: {len(self.person_templates)}")
        
        for name, templates in self.person_templates.items():
            stats = self.recognition_stats.get(name, {})
            print(f"\n👤 {name}:")
            print(f"  Template: {len(templates)}")
            print(f"  Riconoscimenti: {stats.get('recognitions', 0)}")
            print(f"  Aggiunto: {stats.get('added_date', 'N/A')}")
            print(f"  Ultimo visto: {stats.get('last_seen', 'Mai')}")
            
            # Controlla directory foto
            person_dir = self.photos_path / name
            if person_dir.exists():
                photos = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
                print(f"  Foto salvate: {len(photos)}")

def main():
    parser = argparse.ArgumentParser(description='Sistema Completo Coral TPU + Persone')
    parser.add_argument('--run', action='store_true', help='Avvia sistema completo')
    parser.add_argument('--add-photos', type=str, help='Aggiungi persona da foto')
    parser.add_argument('--add-camera', type=str, help='Aggiungi persona da camera')
    parser.add_argument('--remove', type=str, help='Rimuovi persona specifica')
    parser.add_argument('--list', action='store_true', help='Lista tutte le persone')
    parser.add_argument('--cleanup', action='store_true', help='Pulizia database')
    
    args = parser.parse_args()
    
    system = CompleteRecognitionSystem()
    
    if args.add_photos:
        system.add_person_from_photos(args.add_photos)
    elif args.add_camera:
        system.add_person_camera(args.add_camera)
    elif args.remove:
        system.remove_person(args.remove)
    elif args.list:
        system.list_people()
    elif args.cleanup:
        system.cleanup_database()
    elif args.run:
        system.run_complete_system()
    else:
        print("\n🚀 SISTEMA COMPLETO CORAL TPU + PERSONE")
        print("=" * 50)
        print("1. 🔍 Avvia sistema completo")
        print("2. 📷 Aggiungi persona da foto")
        print("3. 🎬 Aggiungi persona da camera")
        print("4. �️ Rimuovi persona")
        print("5. 👥 Lista persone")
        print("6. 🧹 Pulizia database")
        print("7. �📊 Info database")
        print("0. ❌ Esci")
        
        while True:
            try:
                choice = input("\nScegli: ").strip()
                if choice == "0":
                    break
                elif choice == "1":
                    system.run_complete_system()
                elif choice == "2":
                    name = input("Nome persona: ").strip()
                    if name:
                        system.add_person_from_photos(name)
                elif choice == "3":
                    name = input("Nome persona: ").strip()
                    if name:
                        system.add_person_camera(name)
                elif choice == "4":
                    system.remove_person()
                elif choice == "5":
                    system.list_people()
                elif choice == "6":
                    system.cleanup_database()
                elif choice == "7":
                    system.print_database_info()
                else:
                    print("❌ Opzione non valida")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
