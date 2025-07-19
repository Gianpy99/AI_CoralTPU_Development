#!/usr/bin/env python3
"""
Sistema di Riconoscimento Facciale Semplificato
Usa OpenCV + modelli base per riconoscimento personalizzato
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import argparse
from datetime import datetime
import pickle
import os

class SimplePersonRecognition:
    """Sistema semplificato per riconoscimento persone usando OpenCV"""
    
    def __init__(self):
        self.database_path = Path("people_database")
        self.database_path.mkdir(exist_ok=True)
        
        self.templates_file = "person_templates.pkl"
        self.config_file = "recognition_config.json"
        
        # Database templates
        self.person_templates = {}
        self.recognition_stats = {}
        
        # Configurazione
        self.config = {
            "template_size": (100, 100),
            "similarity_threshold": 0.6,
            "max_templates_per_person": 10,
            "face_cascade": "haarcascade_frontalface_default.xml"
        }
        
        # Carica face detector
        self.face_cascade = None
        self.load_face_detector()
        
        # Carica database
        self.load_database()
    
    def load_face_detector(self):
        """Carica il rilevatore di volti OpenCV"""
        # Prova diversi path per il cascade
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_default.xml',
            'models/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                self.face_cascade = cv2.CascadeClassifier(path)
                if not self.face_cascade.empty():
                    print(f"✅ Face detector caricato: {path}")
                    return
        
        print("❌ Face detector non trovato")
        print("💡 Scaricando cascade...")
        
        # Download del cascade se non presente
        import urllib.request
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        cascade_file = "haarcascade_frontalface_default.xml"
        
        try:
            urllib.request.urlretrieve(cascade_url, cascade_file)
            self.face_cascade = cv2.CascadeClassifier(cascade_file)
            if not self.face_cascade.empty():
                print(f"✅ Face detector scaricato e caricato")
            else:
                print("❌ Errore caricamento face detector")
        except Exception as e:
            print(f"❌ Errore download: {e}")
    
    def load_database(self):
        """Carica database templates esistente"""
        if Path(self.templates_file).exists():
            try:
                with open(self.templates_file, 'rb') as f:
                    data = pickle.load(f)
                    self.person_templates = data.get('templates', {})
                    self.recognition_stats = data.get('stats', {})
                print(f"✅ Database caricato: {len(self.person_templates)} persone")
                for name, templates in self.person_templates.items():
                    print(f"   - {name}: {len(templates)} templates")
            except Exception as e:
                print(f"⚠️ Errore caricamento database: {e}")
        
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config.update(json.load(f))
                print(f"✅ Configurazione caricata")
            except Exception as e:
                print(f"⚠️ Errore caricamento config: {e}")
    
    def save_database(self):
        """Salva database templates"""
        try:
            data = {
                'templates': self.person_templates,
                'stats': self.recognition_stats,
                'created': datetime.now().isoformat(),
                'total_people': len(self.person_templates)
            }
            
            with open(self.templates_file, 'wb') as f:
                pickle.dump(data, f)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"✅ Database salvato: {len(self.person_templates)} persone")
            return True
        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            return False
    
    def preprocess_face(self, face_img):
        """Preprocessa volto per template matching"""
        # Ridimensiona
        face_resized = cv2.resize(face_img, self.config["template_size"])
        
        # Normalizza illuminazione
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
        
        # Equalizzazione istogramma
        face_eq = cv2.equalizeHist(face_gray)
        
        # Blur per ridurre rumore
        face_blur = cv2.GaussianBlur(face_eq, (3, 3), 0)
        
        return face_blur
    
    def calculate_similarity(self, template1, template2):
        """Calcola similarità tra due template"""
        # Correlazione normalizzata
        result = cv2.matchTemplate(template1, template2, cv2.TM_CCOEFF_NORMED)
        return result[0][0]
    
    def add_person_from_camera(self, person_name, num_photos=8):
        """Aggiungi persona tramite camera"""
        if self.face_cascade is None or self.face_cascade.empty():
            print("❌ Face detector non disponibile")
            return False
        
        print(f"\n🎬 REGISTRAZIONE: {person_name}")
        print(f"📸 Raccoglierò {num_photos} foto del volto")
        print("💡 Posizionati davanti alla camera")
        print("⚠️ Premi SPAZIO per catturare, Q per uscire")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return False
        
        captured_templates = []
        captured_photos = 0
        
        # Crea cartella persona
        person_dir = self.database_path / person_name
        person_dir.mkdir(exist_ok=True)
        
        while captured_photos < num_photos:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Rileva volti
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            display_frame = frame.copy()
            
            # Header info
            cv2.putText(display_frame, f"Registrando: {person_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"Templates: {captured_photos}/{num_photos}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(display_frame, "SPAZIO=cattura, Q=esci", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Disegna volti rilevati
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if len(faces) == 1 else (0, 255, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                if len(faces) == 1:
                    cv2.putText(display_frame, "PRONTO", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Solo 1 volto", 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow(f'Registrazione {person_name}', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and len(faces) == 1:
                # Cattura template
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    # Preprocessa volto
                    template = self.preprocess_face(face_roi)
                    captured_templates.append(template)
                    
                    # Salva foto originale
                    photo_path = person_dir / f"{person_name}_{captured_photos:03d}.jpg"
                    cv2.imwrite(str(photo_path), face_roi)
                    
                    captured_photos += 1
                    print(f"📸 Template {captured_photos}/{num_photos} catturato")
                    
                    # Feedback visivo
                    flash_frame = frame.copy()
                    cv2.rectangle(flash_frame, (x, y), (x+w, y+h), (255, 255, 255), 5)
                    cv2.imshow(f'Registrazione {person_name}', flash_frame)
                    cv2.waitKey(200)
                    
                except Exception as e:
                    print(f"❌ Errore cattura: {e}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Aggiungi al database
        if captured_templates:
            self.person_templates[person_name] = captured_templates
            
            # Inizializza statistiche
            self.recognition_stats[person_name] = {
                'added_date': datetime.now().isoformat(),
                'templates_count': len(captured_templates),
                'recognitions': 0,
                'last_seen': None
            }
            
            if self.save_database():
                print(f"✅ {person_name} aggiunto con {len(captured_templates)} templates")
                return True
        
        print(f"❌ Registrazione fallita per {person_name}")
        return False
    
    def recognize_person(self, face_img):
        """Riconosce persona da immagine volto"""
        if not self.person_templates:
            return "Sconosciuto", 0.0
        
        # Preprocessa volto input
        input_template = self.preprocess_face(face_img)
        
        best_match = "Sconosciuto"
        best_score = 0.0
        
        # Confronta con tutti i template
        for person_name, templates in self.person_templates.items():
            person_scores = []
            
            for template in templates:
                try:
                    similarity = self.calculate_similarity(input_template, template)
                    person_scores.append(similarity)
                except:
                    continue
            
            if person_scores:
                # Prende la migliore somiglianza per questa persona
                person_best_score = max(person_scores)
                
                if person_best_score > best_score:
                    best_score = person_best_score
                    best_match = person_name
        
        # Controlla soglia
        if best_score < self.config["similarity_threshold"]:
            best_match = "Sconosciuto"
        else:
            # Aggiorna statistiche
            if best_match in self.recognition_stats:
                self.recognition_stats[best_match]['recognitions'] += 1
                self.recognition_stats[best_match]['last_seen'] = datetime.now().isoformat()
        
        return best_match, best_score
    
    def recognize_live(self):
        """Riconoscimento dal vivo"""
        if self.face_cascade is None or self.face_cascade.empty():
            print("❌ Face detector non disponibile")
            return
        
        if not self.person_templates:
            print("❌ Database vuoto. Aggiungi persone prima!")
            return
        
        print("🔍 RICONOSCIMENTO PERSONE LIVE")
        print(f"👥 Database: {len(self.person_templates)} persone")
        print("🎮 Controlli: Q=esci, S=salva, A=aggiungi persona, R=reset")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Processa ogni 2 frame per performance
                if frame_count % 2 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Rileva volti
                    start_time = time.time()
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
                    detection_time = (time.time() - start_time) * 1000
                    
                    # Riconosci volti
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Riconoscimento
                        name, confidence = self.recognize_person(face_roi)
                        
                        # Colore in base al riconoscimento
                        color = (0, 255, 0) if name != "Sconosciuto" else (0, 0, 255)
                        
                        # Rettangolo volto
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Nome e confidenza
                        label = f"{name}: {confidence:.2f}" if name != "Sconosciuto" else name
                        
                        # Background label
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x, y-label_height-10), 
                                     (x+label_width, y), color, -1)
                        
                        # Testo label
                        cv2.putText(frame, label, (x, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Info overlay
                cv2.putText(frame, f"Database: {len(self.person_templates)} persone", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Detection: {detection_time:.1f}ms", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Soglia: {self.config['similarity_threshold']:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Controlli
                cv2.putText(frame, "Q=esci | S=salva | A=aggiungi | R=reset | +=soglia+ | -=soglia-", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow('🔍 Riconoscimento Persone Semplificato', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"recognition_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Salvato: {filename}")
                elif key == ord('a'):
                    cv2.destroyAllWindows()
                    name = input("Nome persona da aggiungere: ").strip()
                    if name:
                        self.add_person_from_camera(name, 6)
                    cap = cv2.VideoCapture(0)  # Riapri camera
                elif key == ord('r'):
                    for person in self.recognition_stats:
                        self.recognition_stats[person]['recognitions'] = 0
                    print("🔄 Statistiche resettate")
                elif key == ord('+') or key == ord('='):
                    self.config['similarity_threshold'] = min(0.95, self.config['similarity_threshold'] + 0.05)
                    print(f"🔧 Soglia: {self.config['similarity_threshold']:.2f}")
                elif key == ord('-'):
                    self.config['similarity_threshold'] = max(0.2, self.config['similarity_threshold'] - 0.05)
                    print(f"🔧 Soglia: {self.config['similarity_threshold']:.2f}")
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione utente")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_database()
    
    def show_database_info(self):
        """Mostra info database"""
        print("\n📊 DATABASE RICONOSCIMENTO PERSONE")
        print("=" * 50)
        
        if not self.person_templates:
            print("❌ Database vuoto")
            return
        
        print(f"👥 Totale persone: {len(self.person_templates)}")
        print(f"📸 Soglia similarità: {self.config['similarity_threshold']}")
        print(f"🖼️ Dimensione template: {self.config['template_size']}")
        
        print("\n👤 PERSONE NEL DATABASE:")
        for person_name in self.person_templates:
            templates_count = len(self.person_templates[person_name])
            stats = self.recognition_stats.get(person_name, {})
            recognitions = stats.get('recognitions', 0)
            added_date = stats.get('added_date', 'N/A')
            last_seen = stats.get('last_seen', 'Mai')
            
            print(f"  • {person_name}:")
            print(f"    - Templates: {templates_count}")
            print(f"    - Riconoscimenti: {recognitions}")
            print(f"    - Aggiunto: {added_date[:10] if added_date != 'N/A' else 'N/A'}")
            print(f"    - Ultimo visto: {last_seen[:10] if last_seen != 'Mai' else 'Mai'}")
    
    def delete_person(self, person_name):
        """Rimuovi persona dal database"""
        if person_name not in self.person_templates:
            print(f"❌ {person_name} non trovato nel database")
            return False
        
        # Rimuovi templates
        del self.person_templates[person_name]
        
        # Rimuovi statistiche
        if person_name in self.recognition_stats:
            del self.recognition_stats[person_name]
        
        # Rimuovi cartella
        person_dir = self.database_path / person_name
        if person_dir.exists():
            import shutil
            shutil.rmtree(person_dir)
        
        self.save_database()
        print(f"✅ {person_name} rimosso dal database")
        return True

def main():
    parser = argparse.ArgumentParser(description='Sistema Riconoscimento Persone Semplificato')
    parser.add_argument('--add', type=str, help='Aggiungi persona tramite camera')
    parser.add_argument('--recognize', action='store_true', help='Avvia riconoscimento live')
    parser.add_argument('--info', action='store_true', help='Mostra info database')
    parser.add_argument('--delete', type=str, help='Rimuovi persona dal database')
    
    args = parser.parse_args()
    
    recognition_system = SimplePersonRecognition()
    
    if args.add:
        recognition_system.add_person_from_camera(args.add)
    elif args.recognize:
        recognition_system.recognize_live()
    elif args.info:
        recognition_system.show_database_info()
    elif args.delete:
        recognition_system.delete_person(args.delete)
    else:
        # Menu interattivo
        while True:
            print("\n👤 RICONOSCIMENTO PERSONE SEMPLIFICATO")
            print("=" * 45)
            print("1. 👤 Aggiungi persona")
            print("2. 🔍 Riconoscimento live")
            print("3. 📊 Info database")
            print("4. 🗑️ Rimuovi persona")
            print("5. ⚙️ Configura soglia")
            print("0. ❌ Esci")
            
            try:
                choice = input("\nScegli opzione: ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    name = input("Nome persona: ").strip()
                    if name:
                        recognition_system.add_person_from_camera(name)
                elif choice == "2":
                    recognition_system.recognize_live()
                elif choice == "3":
                    recognition_system.show_database_info()
                elif choice == "4":
                    recognition_system.show_database_info()
                    name = input("Nome da rimuovere: ").strip()
                    if name:
                        recognition_system.delete_person(name)
                elif choice == "5":
                    try:
                        threshold = float(input(f"Soglia attuale: {recognition_system.config['similarity_threshold']}\nNuova soglia (0.2-0.95): "))
                        if 0.2 <= threshold <= 0.95:
                            recognition_system.config['similarity_threshold'] = threshold
                            recognition_system.save_database()
                            print(f"✅ Soglia aggiornata: {threshold}")
                        else:
                            print("❌ Soglia deve essere tra 0.2 e 0.95")
                    except ValueError:
                        print("❌ Valore non valido")
                else:
                    print("❌ Opzione non valida")
            
            except KeyboardInterrupt:
                print("\n👋 Uscita")
                break

if __name__ == "__main__":
    main()
