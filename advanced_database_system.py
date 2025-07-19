#!/usr/bin/env python3
"""
SISTEMA AVANZATO - Database Riconoscimento Migliorato
Supera le limitazioni di OpenCV con tecnologie moderne
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import pickle
from datetime import datetime
import sqlite3
import hashlib
from collections import defaultdict

# Imports avanzati per AI/ML
try:
    # Face Recognition con deep learning
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    # MediaPipe per detection robusta
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    # Embedding con sentence-transformers
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    # Vector similarity con faiss
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    # Deep features con dlib
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

class AdvancedDatabase:
    """Database avanzato per riconoscimento con multiple tecnologie"""
    
    def __init__(self):
        self.db_path = Path("advanced_recognition_db")
        self.db_path.mkdir(exist_ok=True)
        
        # Database SQL per metadati
        self.sql_db = self.db_path / "metadata.db"
        
        # Database embeddings
        self.embeddings_db = {}
        self.face_encodings_db = {}
        self.visual_features_db = {}
        
        # Configurazione multi-tecnologia
        self.config = {
            "use_face_recognition": FACE_RECOGNITION_AVAILABLE,
            "use_mediapipe": MEDIAPIPE_AVAILABLE,
            "use_dlib": DLIB_AVAILABLE,
            "use_faiss": FAISS_AVAILABLE,
            "similarity_threshold": 0.6,
            "min_confidence": 0.7,
            "max_encodings_per_person": 10,
            "embedding_dimension": 128
        }
        
        # Inizializza componenti
        self.init_sql_database()
        self.init_ai_models()
        
        print(f"🧠 SISTEMA DATABASE AVANZATO")
        print(f"✅ Face Recognition: {'Disponibile' if FACE_RECOGNITION_AVAILABLE else 'Non disponibile'}")
        print(f"✅ MediaPipe: {'Disponibile' if MEDIAPIPE_AVAILABLE else 'Non disponibile'}")
        print(f"✅ Dlib: {'Disponibile' if DLIB_AVAILABLE else 'Non disponibile'}")
        print(f"✅ FAISS: {'Disponibile' if FAISS_AVAILABLE else 'Non disponibile'}")
    
    def init_sql_database(self):
        """Inizializza database SQL per metadati"""
        try:
            conn = sqlite3.connect(self.sql_db)
            cursor = conn.cursor()
            
            # Tabella persone
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_date TEXT NOT NULL,
                    updated_date TEXT,
                    total_encodings INTEGER DEFAULT 0,
                    total_recognitions INTEGER DEFAULT 0,
                    last_seen TEXT,
                    confidence_avg REAL DEFAULT 0.0,
                    source_type TEXT DEFAULT 'mixed',
                    active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Tabella encodings/features
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    encoding_hash TEXT UNIQUE,
                    encoding_type TEXT,
                    confidence REAL,
                    created_date TEXT,
                    source_image TEXT,
                    face_location TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            # Tabella statistiche
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    timestamp TEXT,
                    confidence REAL,
                    method TEXT,
                    processing_time REAL,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Database SQL inizializzato")
            
        except Exception as e:
            print(f"❌ Errore inizializzazione SQL: {e}")
    
    def init_ai_models(self):
        """Inizializza modelli AI avanzati"""
        # MediaPipe Face Detection
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_face_mesh = mp.solutions.face_mesh
                self.mp_drawing = mp.solutions.drawing_utils
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.7
                )
                print("✅ MediaPipe Face Detection inizializzato")
            except Exception as e:
                print(f"⚠️ MediaPipe fallito: {e}")
                self.config["use_mediapipe"] = False
        
        # Dlib face recognition
        if DLIB_AVAILABLE:
            try:
                # Scarica predictor se non esiste
                predictor_path = "shape_predictor_68_face_landmarks.dat"
                if not Path(predictor_path).exists():
                    print("📥 Downloading dlib predictor...")
                    # URL: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
                
                self.dlib_detector = dlib.get_frontal_face_detector()
                print("✅ Dlib detector inizializzato")
            except Exception as e:
                print(f"⚠️ Dlib fallito: {e}")
                self.config["use_dlib"] = False
        
        # FAISS vector index
        if FAISS_AVAILABLE:
            try:
                self.faiss_index = faiss.IndexFlatL2(self.config["embedding_dimension"])
                self.person_id_mapping = {}
                print("✅ FAISS index inizializzato")
            except Exception as e:
                print(f"⚠️ FAISS fallito: {e}")
                self.config["use_faiss"] = False
    
    def add_person_advanced(self, person_name, image_paths):
        """Aggiungi persona con tecnologie multiple"""
        print(f"\n🧠 AGGIUNTA AVANZATA: {person_name}")
        
        try:
            conn = sqlite3.connect(self.sql_db)
            cursor = conn.cursor()
            
            # Inserisci persona
            cursor.execute('''
                INSERT OR IGNORE INTO persons (name, created_date, source_type)
                VALUES (?, ?, ?)
            ''', (person_name, datetime.now().isoformat(), 'advanced'))
            
            # Ottieni ID persona
            cursor.execute('SELECT id FROM persons WHERE name = ?', (person_name,))
            person_id = cursor.fetchone()[0]
            
            all_encodings = []
            total_processed = 0
            
            for img_path in image_paths:
                try:
                    # Carica immagine
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    
                    # Estrai features con multiple tecnologie
                    features = self.extract_multiple_features(image, str(img_path))
                    
                    for feature_data in features:
                        # Hash dell'encoding per unicità
                        encoding_hash = hashlib.md5(
                            feature_data['encoding'].tobytes()
                        ).hexdigest()
                        
                        # Salva nel database SQL
                        cursor.execute('''
                            INSERT OR IGNORE INTO face_encodings
                            (person_id, encoding_hash, encoding_type, confidence, 
                             created_date, source_image, face_location)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            person_id,
                            encoding_hash,
                            feature_data['type'],
                            feature_data['confidence'],
                            datetime.now().isoformat(),
                            Path(img_path).name,
                            json.dumps(feature_data['location'])
                        ))
                        
                        # Salva encoding nel database in memoria
                        if person_name not in self.face_encodings_db:
                            self.face_encodings_db[person_name] = []
                        
                        self.face_encodings_db[person_name].append({
                            'encoding': feature_data['encoding'],
                            'type': feature_data['type'],
                            'confidence': feature_data['confidence'],
                            'hash': encoding_hash
                        })
                        
                        all_encodings.append(feature_data['encoding'])
                    
                    total_processed += 1
                    print(f"  ✅ {Path(img_path).name}: {len(features)} features estratte")
                
                except Exception as e:
                    print(f"  ❌ Errore {Path(img_path).name}: {e}")
            
            # Aggiorna statistiche persona
            cursor.execute('''
                UPDATE persons 
                SET total_encodings = ?, updated_date = ?
                WHERE id = ?
            ''', (len(all_encodings), datetime.now().isoformat(), person_id))
            
            conn.commit()
            conn.close()
            
            # Aggiorna FAISS index se disponibile
            if FAISS_AVAILABLE and all_encodings:
                self.update_faiss_index(person_id, all_encodings)
            
            # Salva database persistente
            self.save_advanced_database()
            
            print(f"✅ {person_name}: {len(all_encodings)} encodings da {total_processed} immagini")
            return True
            
        except Exception as e:
            print(f"❌ Errore aggiunta avanzata: {e}")
            return False
    
    def extract_multiple_features(self, image, source_path):
        """Estrai features con multiple tecnologie"""
        features = []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Face Recognition (dlib-based)
        if self.config["use_face_recognition"] and FACE_RECOGNITION_AVAILABLE:
            try:
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                for encoding, location in zip(face_encodings, face_locations):
                    features.append({
                        'encoding': encoding,
                        'type': 'face_recognition_dlib',
                        'confidence': 0.9,  # Face-recognition è molto accurato
                        'location': location,
                        'source': source_path
                    })
                    
                print(f"    🔹 Face Recognition: {len(face_encodings)} visi")
            except Exception as e:
                print(f"    ⚠️ Face Recognition fallito: {e}")
        
        # 2. MediaPipe Features
        if self.config["use_mediapipe"] and MEDIAPIPE_AVAILABLE:
            try:
                results = self.face_detection.process(rgb_image)
                if results.detections:
                    for detection in results.detections:
                        # Estrai bounding box
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = image.shape
                        
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Estrai face region e crea encoding
                        face_region = rgb_image[y:y+height, x:x+width]
                        if face_region.size > 0:
                            # Resize e normalizza per creare encoding
                            face_resized = cv2.resize(face_region, (128, 128))
                            face_flattened = face_resized.flatten().astype(np.float32)
                            # Normalizza a 128 dimensioni
                            face_normalized = face_flattened[:128] / 255.0
                            
                            features.append({
                                'encoding': face_normalized,
                                'type': 'mediapipe',
                                'confidence': detection.score[0],
                                'location': [y, x+width, y+height, x],  # top, right, bottom, left
                                'source': source_path
                            })
                
                print(f"    🔹 MediaPipe: {len(results.detections) if results.detections else 0} visi")
            except Exception as e:
                print(f"    ⚠️ MediaPipe fallito: {e}")
        
        # 3. OpenCV tradizionale (fallback)
        if not features:  # Solo se altri metodi falliscono
            try:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                for (x, y, w, h) in faces:
                    face_region = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_region, (128, 128))
                    face_normalized = face_resized.flatten().astype(np.float32) / 255.0
                    # Pad o truncate a 128 dimensioni
                    if len(face_normalized) > 128:
                        face_normalized = face_normalized[:128]
                    elif len(face_normalized) < 128:
                        face_normalized = np.pad(face_normalized, (0, 128 - len(face_normalized)))
                    
                    features.append({
                        'encoding': face_normalized,
                        'type': 'opencv_haar',
                        'confidence': 0.7,
                        'location': [y, x+w, y+h, x],
                        'source': source_path
                    })
                
                print(f"    🔹 OpenCV Haar: {len(faces)} visi")
            except Exception as e:
                print(f"    ⚠️ OpenCV fallito: {e}")
        
        return features
    
    def update_faiss_index(self, person_id, encodings):
        """Aggiorna index FAISS per ricerca veloce"""
        if not FAISS_AVAILABLE:
            return
        
        try:
            for encoding in encodings:
                # Assicurati che l'encoding sia 128-dimensionale
                if len(encoding) != 128:
                    # Resize encoding
                    if len(encoding) > 128:
                        encoding = encoding[:128]
                    else:
                        encoding = np.pad(encoding, (0, 128 - len(encoding)))
                
                # Aggiungi al index FAISS
                self.faiss_index.add(encoding.reshape(1, -1).astype(np.float32))
                
                # Mappa ID per retrieval
                current_size = self.faiss_index.ntotal - 1
                self.person_id_mapping[current_size] = person_id
            
            print(f"    🔹 FAISS: {len(encodings)} vettori aggiunti")
        except Exception as e:
            print(f"    ⚠️ FAISS update fallito: {e}")
    
    def recognize_person_advanced(self, face_image):
        """Riconoscimento avanzato con multiple tecnologie"""
        best_match = "Sconosciuto"
        best_confidence = 0.0
        method_used = "none"
        
        try:
            # Estrai features dalla faccia di input
            input_features = self.extract_multiple_features(face_image, "live_capture")
            
            if not input_features:
                return best_match, best_confidence, method_used
            
            # Usa il miglior encoding disponibile
            input_encoding = input_features[0]['encoding']
            
            # 1. Face Recognition comparison (più accurato)
            if FACE_RECOGNITION_AVAILABLE:
                for person_name, person_data in self.face_encodings_db.items():
                    for stored_data in person_data:
                        if stored_data['type'] == 'face_recognition_dlib':
                            try:
                                # Compare con face_recognition
                                rgb_input = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                                input_encodings = face_recognition.face_encodings(rgb_input)
                                
                                if input_encodings:
                                    distances = face_recognition.face_distance(
                                        [stored_data['encoding']], 
                                        input_encodings[0]
                                    )
                                    confidence = 1.0 - distances[0]  # Convert distance to confidence
                                    
                                    if confidence > best_confidence and confidence > self.config["similarity_threshold"]:
                                        best_match = person_name
                                        best_confidence = confidence
                                        method_used = "face_recognition"
                            except:
                                pass
            
            # 2. FAISS similarity search (più veloce)
            if FAISS_AVAILABLE and best_confidence < 0.8:  # Solo se non abbiamo già un match ottimo
                try:
                    # Assicurati dimensione corretta
                    query_vector = input_encoding[:128].reshape(1, -1).astype(np.float32)
                    
                    # Cerca k migliori match
                    distances, indices = self.faiss_index.search(query_vector, k=5)
                    
                    for distance, idx in zip(distances[0], indices[0]):
                        if idx in self.person_id_mapping:
                            # Convert distance to confidence
                            confidence = 1.0 / (1.0 + distance)
                            
                            if confidence > best_confidence and confidence > self.config["similarity_threshold"]:
                                # Ottieni nome persona da ID
                                person_id = self.person_id_mapping[idx]
                                person_name = self.get_person_name_by_id(person_id)
                                
                                if person_name:
                                    best_match = person_name
                                    best_confidence = confidence
                                    method_used = "faiss_similarity"
                except Exception as e:
                    print(f"⚠️ FAISS search fallito: {e}")
            
            # 3. Fallback: confronto manuale encodings
            if best_confidence < 0.7:
                for person_name, person_data in self.face_encodings_db.items():
                    for stored_data in person_data:
                        try:
                            # Calcola similarità coseno
                            similarity = np.dot(input_encoding, stored_data['encoding']) / (
                                np.linalg.norm(input_encoding) * np.linalg.norm(stored_data['encoding'])
                            )
                            
                            if similarity > best_confidence and similarity > self.config["similarity_threshold"]:
                                best_match = person_name
                                best_confidence = similarity
                                method_used = f"manual_{stored_data['type']}"
                        except:
                            pass
            
            # Registra statistiche se riconosciuto
            if best_match != "Sconosciuto":
                self.log_recognition_stats(best_match, best_confidence, method_used)
            
            return best_match, best_confidence, method_used
            
        except Exception as e:
            print(f"❌ Errore riconoscimento avanzato: {e}")
            return "Sconosciuto", 0.0, "error"
    
    def get_person_name_by_id(self, person_id):
        """Ottieni nome persona da ID"""
        try:
            conn = sqlite3.connect(self.sql_db)
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM persons WHERE id = ?', (person_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except:
            return None
    
    def log_recognition_stats(self, person_name, confidence, method):
        """Log statistiche riconoscimento"""
        try:
            conn = sqlite3.connect(self.sql_db)
            cursor = conn.cursor()
            
            # Ottieni person ID
            cursor.execute('SELECT id FROM persons WHERE name = ?', (person_name,))
            person_id = cursor.fetchone()[0]
            
            # Log riconoscimento
            cursor.execute('''
                INSERT INTO recognition_stats
                (person_id, timestamp, confidence, method, processing_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, datetime.now().isoformat(), confidence, method, 0))
            
            # Aggiorna contatori persona
            cursor.execute('''
                UPDATE persons 
                SET total_recognitions = total_recognitions + 1,
                    last_seen = ?,
                    confidence_avg = (
                        SELECT AVG(confidence) 
                        FROM recognition_stats 
                        WHERE person_id = ?
                    )
                WHERE id = ?
            ''', (datetime.now().isoformat(), person_id, person_id))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️ Errore log stats: {e}")
    
    def save_advanced_database(self):
        """Salva database avanzato"""
        try:
            # Salva encodings
            encodings_file = self.db_path / "face_encodings.pkl"
            with open(encodings_file, 'wb') as f:
                pickle.dump(self.face_encodings_db, f)
            
            # Salva FAISS index se disponibile
            if FAISS_AVAILABLE and self.faiss_index.ntotal > 0:
                faiss_file = self.db_path / "faiss_index.bin"
                faiss.write_index(self.faiss_index, str(faiss_file))
                
                mapping_file = self.db_path / "person_mapping.pkl"
                with open(mapping_file, 'wb') as f:
                    pickle.dump(self.person_id_mapping, f)
            
            print("💾 Database avanzato salvato")
            return True
        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            return False
    
    def load_advanced_database(self):
        """Carica database avanzato"""
        try:
            # Carica encodings
            encodings_file = self.db_path / "face_encodings.pkl"
            if encodings_file.exists():
                with open(encodings_file, 'rb') as f:
                    self.face_encodings_db = pickle.load(f)
                print(f"✅ Caricati encodings per {len(self.face_encodings_db)} persone")
            
            # Carica FAISS index
            if FAISS_AVAILABLE:
                faiss_file = self.db_path / "faiss_index.bin"
                mapping_file = self.db_path / "person_mapping.pkl"
                
                if faiss_file.exists() and mapping_file.exists():
                    self.faiss_index = faiss.read_index(str(faiss_file))
                    with open(mapping_file, 'rb') as f:
                        self.person_id_mapping = pickle.load(f)
                    print(f"✅ FAISS index caricato: {self.faiss_index.ntotal} vettori")
            
            return True
        except Exception as e:
            print(f"❌ Errore caricamento: {e}")
            return False
    
    def get_database_stats(self):
        """Ottieni statistiche database"""
        try:
            conn = sqlite3.connect(self.sql_db)
            cursor = conn.cursor()
            
            # Statistiche generali
            cursor.execute('SELECT COUNT(*) FROM persons WHERE active = 1')
            total_persons = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM face_encodings')
            total_encodings = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM recognition_stats')
            total_recognitions = cursor.fetchone()[0]
            
            # Top persone riconosciute
            cursor.execute('''
                SELECT p.name, p.total_recognitions, p.confidence_avg, p.last_seen
                FROM persons p
                WHERE p.active = 1
                ORDER BY p.total_recognitions DESC
                LIMIT 10
            ''')
            top_persons = cursor.fetchall()
            
            # Statistiche per metodo
            cursor.execute('''
                SELECT method, COUNT(*), AVG(confidence)
                FROM recognition_stats
                GROUP BY method
                ORDER BY COUNT(*) DESC
            ''')
            method_stats = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_persons': total_persons,
                'total_encodings': total_encodings,
                'total_recognitions': total_recognitions,
                'top_persons': top_persons,
                'method_stats': method_stats,
                'faiss_vectors': self.faiss_index.ntotal if FAISS_AVAILABLE else 0
            }
            
        except Exception as e:
            print(f"❌ Errore statistiche: {e}")
            return {}

def main():
    """Test sistema database avanzato"""
    print("🧠 TEST SISTEMA DATABASE AVANZATO")
    
    # Crea database
    db = AdvancedDatabase()
    
    # Carica database esistente
    db.load_advanced_database()
    
    # Mostra statistiche
    stats = db.get_database_stats()
    if stats:
        print(f"\n📊 STATISTICHE DATABASE")
        print(f"👥 Persone: {stats['total_persons']}")
        print(f"🔢 Encodings: {stats['total_encodings']}")
        print(f"👁️ Riconoscimenti: {stats['total_recognitions']}")
        print(f"🚀 FAISS vectors: {stats['faiss_vectors']}")
        
        if stats['method_stats']:
            print(f"\n📈 METODI UTILIZZATI:")
            for method, count, avg_conf in stats['method_stats']:
                print(f"  {method}: {count} volte (conf. media: {avg_conf:.3f})")
    
    print(f"\n✅ Sistema database avanzato pronto!")
    print(f"🔧 Metodi disponibili:")
    print(f"  - Face Recognition (dlib): {'✅' if FACE_RECOGNITION_AVAILABLE else '❌'}")
    print(f"  - MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
    print(f"  - FAISS similarity: {'✅' if FAISS_AVAILABLE else '❌'}")
    print(f"  - OpenCV fallback: ✅")

if __name__ == "__main__":
    main()
