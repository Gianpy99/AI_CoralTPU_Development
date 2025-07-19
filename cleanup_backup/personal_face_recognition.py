#!/usr/bin/env python3
"""
Sistema di Riconoscimento Facciale Personalizzato
Crea un modello per riconoscere persone specifiche usando Coral TPU
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import argparse
from datetime import datetime
import pickle

# Import per face recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition disponibile")
except ImportError:
    print("❌ face_recognition non installato")
    print("💡 Installa con: pip install face_recognition")
    FACE_RECOGNITION_AVAILABLE = False

# Import Coral TPU
try:
    import tflite_runtime.interpreter as tflite
    from pycoral.utils import edgetpu
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False

class PersonalFaceRecognition:
    """Sistema di riconoscimento facciale personalizzato"""
    
    def __init__(self):
        self.database_path = Path("face_database")
        self.database_path.mkdir(exist_ok=True)
        
        self.encodings_file = "face_encodings.pkl"
        self.config_file = "face_config.json"
        
        # Database persone
        self.known_encodings = []
        self.known_names = []
        self.person_stats = {}
        
        # Configurazione
        self.config = {
            "tolerance": 0.6,  # Soglia riconoscimento (più basso = più rigido)
            "min_face_size": 50,  # Dimensione minima volto
            "max_faces": 10,  # Max volti per frame
            "confidence_threshold": 0.7
        }
        
        # Carica database esistente
        self.load_database()
        
    def load_database(self):
        """Carica database volti esistente"""
        # Carica encodings
        if Path(self.encodings_file).exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_encodings = data.get('encodings', [])
                    self.known_names = data.get('names', [])
                    self.person_stats = data.get('stats', {})
                print(f"✅ Database caricato: {len(self.known_names)} persone")
                for name in set(self.known_names):
                    count = self.known_names.count(name)
                    print(f"   - {name}: {count} foto")
            except Exception as e:
                print(f"⚠️ Errore caricamento database: {e}")
        
        # Carica configurazione
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config.update(json.load(f))
                print(f"✅ Configurazione caricata")
            except Exception as e:
                print(f"⚠️ Errore caricamento config: {e}")
    
    def save_database(self):
        """Salva database volti"""
        try:
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names,
                'stats': self.person_stats,
                'created': datetime.now().isoformat(),
                'total_faces': len(self.known_encodings)
            }
            
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"✅ Database salvato: {len(self.known_encodings)} volti")
            return True
        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            return False
    
    def add_person_from_camera(self, person_name, num_photos=10):
        """Aggiungi persona dal vivo con camera"""
        if not FACE_RECOGNITION_AVAILABLE:
            print("❌ face_recognition non disponibile")
            return False
        
        print(f"\n🎬 REGISTRAZIONE: {person_name}")
        print(f"📸 Raccoglierò {num_photos} foto")
        print("💡 Guarda la camera, muovi leggermente la testa")
        print("⚠️ Premi SPAZIO per catturare, Q per uscire")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return False
        
        # Crea cartella persona
        person_dir = self.database_path / person_name
        person_dir.mkdir(exist_ok=True)
        
        captured_photos = 0
        encodings_collected = []
        
        while captured_photos < num_photos:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Rileva volti
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Disegna overlay
            display_frame = frame.copy()
            
            # Header
            cv2.putText(display_frame, f"Registrando: {person_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"Foto: {captured_photos}/{num_photos}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(display_frame, "SPAZIO=cattura, Q=esci", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Disegna volti rilevati
            for (top, right, bottom, left) in face_locations:
                color = (0, 255, 0) if len(face_locations) == 1 else (0, 255, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                
                # Indica se pronto per cattura
                if len(face_locations) == 1:
                    cv2.putText(display_frame, "PRONTO", 
                               (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Solo 1 volto", 
                               (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            cv2.imshow(f'Registrazione {person_name}', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and len(face_locations) == 1:
                # Cattura foto
                try:
                    # Encoding del volto
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        encoding = face_encodings[0]
                        encodings_collected.append(encoding)
                        
                        # Salva foto
                        photo_path = person_dir / f"{person_name}_{captured_photos:03d}.jpg"
                        cv2.imwrite(str(photo_path), frame)
                        
                        captured_photos += 1
                        print(f"📸 Catturata foto {captured_photos}/{num_photos}")
                        
                        # Feedback visivo
                        flash_frame = frame.copy()
                        cv2.rectangle(flash_frame, (0, 0), (flash_frame.shape[1], flash_frame.shape[0]), (255, 255, 255), 10)
                        cv2.imshow(f'Registrazione {person_name}', flash_frame)
                        cv2.waitKey(200)
                    else:
                        print("⚠️ Encoding fallito, riprova")
                except Exception as e:
                    print(f"❌ Errore cattura: {e}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Aggiungi al database
        if encodings_collected:
            self.known_encodings.extend(encodings_collected)
            self.known_names.extend([person_name] * len(encodings_collected))
            
            # Statistiche
            if person_name not in self.person_stats:
                self.person_stats[person_name] = {
                    'added_date': datetime.now().isoformat(),
                    'photos_count': 0,
                    'recognitions': 0
                }
            
            self.person_stats[person_name]['photos_count'] += len(encodings_collected)
            
            # Salva database
            if self.save_database():
                print(f"✅ {person_name} aggiunto con {len(encodings_collected)} foto")
                return True
        
        print(f"❌ Registrazione fallita per {person_name}")
        return False
    
    def add_person_from_photos(self, person_name, photos_dir):
        """Aggiungi persona da cartella di foto"""
        if not FACE_RECOGNITION_AVAILABLE:
            print("❌ face_recognition non disponibile")
            return False
        
        photos_path = Path(photos_dir)
        if not photos_path.exists():
            print(f"❌ Cartella non trovata: {photos_path}")
            return False
        
        # Trova foto
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        photo_files = []
        for ext in extensions:
            photo_files.extend(photos_path.glob(ext))
        
        if not photo_files:
            print(f"❌ Nessuna foto trovata in {photos_path}")
            return False
        
        print(f"📁 Processando {len(photo_files)} foto per {person_name}")
        
        encodings_collected = []
        processed = 0
        
        for photo_file in photo_files:
            try:
                # Carica immagine
                image = face_recognition.load_image_file(str(photo_file))
                
                # Trova volti
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                if face_encodings:
                    # Prende il primo volto se multipli
                    encodings_collected.append(face_encodings[0])
                    processed += 1
                    print(f"✅ {photo_file.name}: volto estratto")
                else:
                    print(f"⚠️ {photo_file.name}: nessun volto trovato")
                    
            except Exception as e:
                print(f"❌ {photo_file.name}: errore - {e}")
        
        # Aggiungi al database
        if encodings_collected:
            self.known_encodings.extend(encodings_collected)
            self.known_names.extend([person_name] * len(encodings_collected))
            
            # Statistiche
            if person_name not in self.person_stats:
                self.person_stats[person_name] = {
                    'added_date': datetime.now().isoformat(),
                    'photos_count': 0,
                    'recognitions': 0
                }
            
            self.person_stats[person_name]['photos_count'] += len(encodings_collected)
            
            if self.save_database():
                print(f"✅ {person_name}: {len(encodings_collected)} volti aggiunti")
                return True
        
        print(f"❌ Nessun volto utilizzabile per {person_name}")
        return False
    
    def recognize_faces_live(self):
        """Riconoscimento dal vivo"""
        if not FACE_RECOGNITION_AVAILABLE:
            print("❌ face_recognition non disponibile")
            return
        
        if not self.known_encodings:
            print("❌ Database vuoto. Aggiungi persone prima!")
            return
        
        print("🔍 RICONOSCIMENTO FACCIALE LIVE")
        print(f"👥 Database: {len(set(self.known_names))} persone")
        print("🎮 Controlli: Q=esci, S=salva, A=aggiungi persona, R=reset stats")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera non disponibile")
            return
        
        frame_count = 0
        recognition_log = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Processa ogni N frame per performance
                if frame_count % 3 == 0:  # Processa ogni 3 frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Rileva volti
                    start_time = time.time()
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    processing_time = (time.time() - start_time) * 1000
                    
                    # Riconosci volti
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Confronta con database
                        matches = face_recognition.compare_faces(
                            self.known_encodings, face_encoding, tolerance=self.config['tolerance'])
                        
                        name = "Sconosciuto"
                        confidence = 0.0
                        
                        if True in matches:
                            # Trova migliore match
                            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            
                            if matches[best_match_index]:
                                name = self.known_names[best_match_index]
                                confidence = 1 - face_distances[best_match_index]
                                
                                # Aggiorna statistiche
                                if name in self.person_stats:
                                    self.person_stats[name]['recognitions'] += 1
                                
                                # Log riconoscimento
                                recognition_log.append({
                                    'name': name,
                                    'confidence': confidence,
                                    'timestamp': datetime.now().isoformat()
                                })
                        
                        # Disegna rettangolo e nome
                        color = (0, 255, 0) if name != "Sconosciuto" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Nome e confidenza
                        label = f"{name}: {confidence:.2f}" if name != "Sconosciuto" else name
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        cv2.putText(frame, label, (left + 6, bottom - 6), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
                # Info overlay
                cv2.putText(frame, f"Database: {len(set(self.known_names))} persone", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Tempo: {processing_time:.1f}ms", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Controlli
                cv2.putText(frame, "Q=esci | S=salva | A=aggiungi | R=reset", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('🔍 Riconoscimento Facciale Personalizzato', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    cv2.imwrite(f"recognition_{timestamp}.jpg", frame)
                    print(f"📸 Salvato: recognition_{timestamp}.jpg")
                elif key == ord('a'):
                    cv2.destroyAllWindows()
                    name = input("Nome persona da aggiungere: ").strip()
                    if name:
                        self.add_person_from_camera(name, 5)
                    cap = cv2.VideoCapture(0)  # Riapri camera
                elif key == ord('r'):
                    recognition_log.clear()
                    for person in self.person_stats:
                        self.person_stats[person]['recognitions'] = 0
                    print("🔄 Statistiche resettate")
        
        except KeyboardInterrupt:
            print("\n👋 Interruzione utente")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Salva log
            if recognition_log:
                log_file = f"recognition_log_{int(time.time())}.json"
                with open(log_file, 'w') as f:
                    json.dump(recognition_log, f, indent=2)
                print(f"📝 Log salvato: {log_file}")
            
            self.save_database()
    
    def show_database_info(self):
        """Mostra info database"""
        print("\n📊 DATABASE RICONOSCIMENTO FACCIALE")
        print("=" * 50)
        
        if not self.known_names:
            print("❌ Database vuoto")
            return
        
        print(f"👥 Totale persone: {len(set(self.known_names))}")
        print(f"📸 Totale foto: {len(self.known_encodings)}")
        print(f"⚙️ Tolleranza: {self.config['tolerance']}")
        
        print("\n👤 PERSONE NEL DATABASE:")
        for person in set(self.known_names):
            count = self.known_names.count(person)
            stats = self.person_stats.get(person, {})
            recognitions = stats.get('recognitions', 0)
            added_date = stats.get('added_date', 'N/A')
            
            print(f"  • {person}:")
            print(f"    - Foto: {count}")
            print(f"    - Riconoscimenti: {recognitions}")
            print(f"    - Aggiunto: {added_date[:10] if added_date != 'N/A' else 'N/A'}")
    
    def delete_person(self, person_name):
        """Rimuovi persona dal database"""
        if person_name not in self.known_names:
            print(f"❌ {person_name} non trovato nel database")
            return False
        
        # Rimuovi tutti gli encoding della persona
        indices_to_remove = [i for i, name in enumerate(self.known_names) if name == person_name]
        
        for index in sorted(indices_to_remove, reverse=True):
            del self.known_encodings[index]
            del self.known_names[index]
        
        # Rimuovi statistiche
        if person_name in self.person_stats:
            del self.person_stats[person_name]
        
        # Rimuovi cartella foto
        person_dir = self.database_path / person_name
        if person_dir.exists():
            import shutil
            shutil.rmtree(person_dir)
        
        self.save_database()
        print(f"✅ {person_name} rimosso dal database")
        return True

def main():
    parser = argparse.ArgumentParser(description='Sistema Riconoscimento Facciale Personalizzato')
    parser.add_argument('--add-camera', type=str, help='Aggiungi persona tramite camera')
    parser.add_argument('--add-photos', nargs=2, metavar=('NAME', 'FOLDER'), 
                       help='Aggiungi persona da cartella foto')
    parser.add_argument('--recognize', action='store_true', help='Avvia riconoscimento live')
    parser.add_argument('--info', action='store_true', help='Mostra info database')
    parser.add_argument('--delete', type=str, help='Rimuovi persona dal database')
    
    args = parser.parse_args()
    
    # Verifica dipendenze
    if not FACE_RECOGNITION_AVAILABLE:
        print("❌ face_recognition richiesto!")
        print("📦 Installa con:")
        print("   pip install face_recognition")
        print("   conda install -c conda-forge dlib")
        return
    
    face_system = PersonalFaceRecognition()
    
    if args.add_camera:
        face_system.add_person_from_camera(args.add_camera)
    elif args.add_photos:
        name, folder = args.add_photos
        face_system.add_person_from_photos(name, folder)
    elif args.recognize:
        face_system.recognize_faces_live()
    elif args.info:
        face_system.show_database_info()
    elif args.delete:
        face_system.delete_person(args.delete)
    else:
        # Menu interattivo
        while True:
            print("\n🎭 SISTEMA RICONOSCIMENTO FACCIALE")
            print("=" * 40)
            print("1. 👤 Aggiungi persona (camera)")
            print("2. 📁 Aggiungi persona (foto)")
            print("3. 🔍 Riconoscimento live")
            print("4. 📊 Info database")
            print("5. 🗑️ Rimuovi persona")
            print("0. ❌ Esci")
            
            try:
                choice = input("\nScegli opzione: ").strip()
                
                if choice == "0":
                    break
                elif choice == "1":
                    name = input("Nome persona: ").strip()
                    if name:
                        face_system.add_person_from_camera(name)
                elif choice == "2":
                    name = input("Nome persona: ").strip()
                    folder = input("Cartella foto: ").strip()
                    if name and folder:
                        face_system.add_person_from_photos(name, folder)
                elif choice == "3":
                    face_system.recognize_faces_live()
                elif choice == "4":
                    face_system.show_database_info()
                elif choice == "5":
                    face_system.show_database_info()
                    name = input("Nome da rimuovere: ").strip()
                    if name:
                        face_system.delete_person(name)
                else:
                    print("❌ Opzione non valida")
            
            except KeyboardInterrupt:
                print("\n👋 Uscita")
                break

if __name__ == "__main__":
    main()
