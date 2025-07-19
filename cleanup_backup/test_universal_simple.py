#!/usr/bin/env python3
"""
Test semplificato del sistema universale Coral TPU
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "models" / "inference"))
sys.path.insert(0, str(project_root / "src" / "utils"))

print("🚀 Test Sistema Universale Coral TPU")
print("=" * 50)

# Test 1: Import delle librerie
print("📦 Test import librerie...")

try:
    import cv2
    print("✅ OpenCV disponibile")
except ImportError:
    print("❌ OpenCV non disponibile")

try:
    from pycoral.utils import edgetpu
    print("✅ PyCoral disponibile")
    
    # Test Coral TPU
    devices = edgetpu.list_edge_tpus()
    if devices:
        print(f"✅ Coral TPU rilevato: {len(devices)} dispositivi")
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device}")
    else:
        print("⚠️ Coral TPU non rilevato")
        
except ImportError:
    print("❌ PyCoral non disponibile")

try:
    import numpy as np
    import pandas as pd
    from PIL import Image
    print("✅ Librerie scientifiche disponibili")
except ImportError as e:
    print(f"❌ Librerie scientifiche mancanti: {e}")

# Test 2: Camera
print("\n📹 Test camera...")
try:
    import cv2
    
    # Prova ad aprire la camera
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera funzionante: {frame.shape}")
            
            # Salva un frame di test
            cv2.imwrite("camera_test.jpg", frame)
            print("✅ Frame salvato come camera_test.jpg")
        else:
            print("⚠️ Camera aperta ma nessun frame")
        cap.release()
    else:
        print("❌ Impossibile aprire la camera")

except Exception as e:
    print(f"❌ Errore test camera: {e}")

# Test 3: Modelli AI
print("\n🤖 Test modelli AI...")
models_dir = Path("models")
if models_dir.exists():
    tflite_files = list(models_dir.glob("*.tflite"))
    label_files = list(models_dir.glob("*labels.txt"))
    
    print(f"✅ Cartella modelli trovata")
    print(f"📊 Modelli TFLite: {len(tflite_files)}")
    for model in tflite_files:
        print(f"   - {model.name}")
    
    print(f"🏷️ File labels: {len(label_files)}")
    for label in label_files:
        print(f"   - {label.name}")
else:
    print("❌ Cartella modelli non trovata")

# Test 4: Inferenza semplice
print("\n🧠 Test inferenza semplice...")
try:
    import tflite_runtime.interpreter as tflite
    
    # Trova un modello qualsiasi
    models_dir = Path("models")
    tflite_files = list(models_dir.glob("*.tflite"))
    
    if tflite_files:
        model_path = tflite_files[0]
        print(f"🔬 Testando modello: {model_path.name}")
        
        # Prova a caricare il modello
        try:
            interpreter = tflite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✅ Modello caricato")
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            
            # Test con dati casuali
            input_shape = input_details[0]['shape']
            test_input = np.random.random(input_shape).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            print(f"✅ Inferenza test completata")
            print(f"   Output shape: {output.shape}")
            
        except Exception as e:
            print(f"❌ Errore test modello: {e}")
    else:
        print("⚠️ Nessun modello .tflite trovato")

except ImportError:
    print("❌ TensorFlow Lite non disponibile")
except Exception as e:
    print(f"❌ Errore test inferenza: {e}")

# Test 5: Demo classificazione immagine
print("\n🖼️ Test classificazione immagine...")
try:
    if Path("camera_test.jpg").exists():
        # Usa l'immagine appena scattata
        image_path = "camera_test.jpg"
    else:
        # Crea un'immagine test
        import numpy as np
        from PIL import Image
        
        # Crea immagine random 224x224
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(test_image).save("test_image.jpg")
        image_path = "test_image.jpg"
        print("✅ Immagine test creata")
    
    # Carica immagine
    from PIL import Image
    image = Image.open(image_path)
    print(f"✅ Immagine caricata: {image.size}")
    
    # Ridimensiona per classificazione (224x224)
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    print(f"✅ Immagine preprocessata: {image_array.shape}")

except Exception as e:
    print(f"❌ Errore test immagine: {e}")

# Test 6: Demo crypto data
print("\n💰 Test dati crypto...")
try:
    import pandas as pd
    import numpy as np
    
    # Crea dati crypto simulati
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    crypto_data = pd.DataFrame({
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40000, 50000, 100),
        'low': np.random.uniform(40000, 50000, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(100, 1000, 100),
        'sma_20': np.random.uniform(40000, 50000, 100),
        'rsi': np.random.uniform(30, 70, 100),
    }, index=dates)
    
    print(f"✅ Dati crypto generati: {crypto_data.shape}")
    print(f"   Colonne: {list(crypto_data.columns)}")
    
    # Salva per test futuri
    crypto_data.to_csv("crypto_test_data.csv")
    print("✅ Dati salvati in crypto_test_data.csv")

except Exception as e:
    print(f"❌ Errore test crypto: {e}")

# Riepilogo
print("\n" + "=" * 50)
print("🎉 RIEPILOGO TEST")
print("=" * 50)

# Controlla stato componenti
components = {
    "Coral TPU": False,
    "Camera": False,
    "Modelli AI": False,
    "OpenCV": False,
    "TensorFlow Lite": False,
    "Librerie base": False
}

# Aggiorna stato
try:
    import cv2
    components["OpenCV"] = True
except ImportError:
    pass

try:
    from pycoral.utils import edgetpu
    devices = edgetpu.list_edge_tpus()
    if devices:
        components["Coral TPU"] = True
except ImportError:
    pass

try:
    import tflite_runtime.interpreter as tflite
    components["TensorFlow Lite"] = True
except ImportError:
    pass

try:
    import numpy as np
    import pandas as pd
    from PIL import Image
    components["Librerie base"] = True
except ImportError:
    pass

if Path("camera_test.jpg").exists():
    components["Camera"] = True

if any(Path("models").glob("*.tflite")):
    components["Modelli AI"] = True

# Mostra stato
for component, status in components.items():
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {component}")

# Raccomandazioni
ready_count = sum(components.values())
total_count = len(components)

print(f"\n📊 Sistema pronto: {ready_count}/{total_count} componenti")

if ready_count == total_count:
    print("🎉 SISTEMA COMPLETAMENTE FUNZIONANTE!")
    print("\n🚀 Puoi ora usare:")
    print("   python universal_app.py")
    print("   python quick_camera_test.py")
    print("   python quick_crypto_test.py")
elif ready_count >= 4:
    print("⚠️ Sistema quasi pronto - controlla i componenti mancanti")
else:
    print("❌ Sistema non pronto - installa le dipendenze mancanti")
    print("💡 Esegui: python setup_universal.py")

print("\n👋 Test completato!")
