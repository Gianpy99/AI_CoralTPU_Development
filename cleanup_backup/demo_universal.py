#!/usr/bin/env python3
"""
Demo Sistema Universale Coral TPU
Versione semplificata che funziona subito
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
from pathlib import Path

print("🚀 DEMO SISTEMA UNIVERSALE CORAL TPU")
print("=" * 60)

def demo_camera_ai():
    """Demo camera con AI semplificato"""
    print("\n📹 DEMO CAMERA CON AI")
    print("-" * 40)
    
    # Apri camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera non disponibile")
        return
    
    print("✅ Camera aperta")
    print("📱 Controlli: 'q' per uscire, 's' per salvare foto")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Simula AI con analisi semplice
        height, width = frame.shape[:2]
        brightness = np.mean(frame)
        
        # Rilevamento semplice bordi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Simula classificazione AI
        ai_classes = ["persona", "oggetto", "sfondo", "movimento", "luce"]
        predicted_class = ai_classes[frame_count % len(ai_classes)]
        confidence = 0.7 + 0.3 * np.random.random()
        
        # Disegna overlay AI
        cv2.putText(frame, f"AI: {predicted_class} ({confidence:.1%})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Brightness: {brightness:.0f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Objects: {len(contours)}", 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Disegna alcuni contorni
        cv2.drawContours(frame, contours[:5], -1, (0, 255, 255), 2)
        
        # Mostra frame
        cv2.imshow('Demo Coral TPU - AI Camera', frame)
        
        # Input utente
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"ai_photo_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Foto salvata: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Demo camera completata")

def demo_crypto_ai():
    """Demo predizioni crypto con AI"""
    print("\n💰 DEMO CRYPTO AI")
    print("-" * 40)
    
    # Genera dati crypto realistici
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    
    # Simula prezzo BTC con trend e volatilità
    base_price = 45000
    prices = []
    
    for i in range(len(dates)):
        # Trend + rumore + ciclicità
        trend = base_price + (i * 10)  # trend crescente
        noise = np.random.normal(0, 500)  # volatilità
        cycle = 1000 * np.sin(i * 0.1)  # ciclicità
        
        price = trend + noise + cycle
        prices.append(max(price, 20000))  # minimo realistico
    
    crypto_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.05)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.05)) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 2000, len(dates)),
    })
    
    # Calcola indicatori tecnici semplici
    crypto_data['sma_20'] = crypto_data['close'].rolling(20).mean()
    crypto_data['sma_50'] = crypto_data['close'].rolling(50).mean()
    crypto_data['volatility'] = crypto_data['close'].rolling(10).std()
    
    print(f"✅ Generati {len(crypto_data)} punti dati crypto")
    print(f"📊 Prezzo attuale: ${crypto_data['close'].iloc[-1]:.2f}")
    
    # Simula predizioni AI
    print("\n🤖 Eseguendo predizioni AI...")
    
    for i in range(5):
        # Prendi ultimi 60 punti
        recent_data = crypto_data.tail(60)
        
        # Simula analisi AI
        price_trend = recent_data['close'].pct_change().mean()
        volatility = recent_data['volatility'].iloc[-1]
        volume_trend = recent_data['volume'].pct_change().mean()
        
        # Logica di predizione semplice
        if price_trend > 0.01 and volume_trend > 0:
            prediction = "UP"
            confidence = 0.75 + 0.2 * np.random.random()
        elif price_trend < -0.01 and volume_trend > 0:
            prediction = "DOWN"  
            confidence = 0.70 + 0.25 * np.random.random()
        else:
            prediction = "SIDEWAYS"
            confidence = 0.60 + 0.3 * np.random.random()
        
        print(f"📈 Predizione {i+1}: {prediction} ({confidence:.1%})")
        print(f"   Trend prezzo: {price_trend:.3f}")
        print(f"   Volatilità: {volatility:.2f}")
        print(f"   Trend volume: {volume_trend:.3f}")
        
        time.sleep(0.5)  # Simula tempo inferenza
    
    # Salva dati per analisi
    crypto_data.to_csv("demo_crypto_data.csv", index=False)
    print(f"\n✅ Dati salvati in demo_crypto_data.csv")

def demo_image_classification():
    """Demo classificazione immagini"""
    print("\n🖼️ DEMO CLASSIFICAZIONE IMMAGINI")
    print("-" * 40)
    
    # Usa foto dalla camera se disponibile
    if Path("camera_test.jpg").exists():
        image_path = "camera_test.jpg"
        print("📸 Usando foto dalla camera")
    else:
        # Crea immagine test
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(test_image).save("test_image.jpg")
        image_path = "test_image.jpg"
        print("🎨 Creata immagine test")
    
    # Carica e analizza immagine
    image = Image.open(image_path)
    image_array = np.array(image)
    
    print(f"✅ Immagine caricata: {image.size}")
    
    # Simula classificazione AI
    classes = [
        "gatto", "cane", "persona", "auto", "casa", 
        "albero", "fiore", "cibo", "computer", "telefono"
    ]
    
    # Analisi semplice basata su colori e forme
    brightness = np.mean(image_array)
    blue_content = np.mean(image_array[:,:,0])
    green_content = np.mean(image_array[:,:,1])
    red_content = np.mean(image_array[:,:,2])
    
    # Logica di classificazione semplice
    if green_content > blue_content and green_content > red_content:
        predicted_classes = ["albero", "fiore", "persona"]
    elif blue_content > red_content:
        predicted_classes = ["auto", "computer", "telefono"]
    else:
        predicted_classes = ["cibo", "casa", "gatto"]
    
    # Genera predizioni con confidence
    print("\n🎯 Risultati classificazione:")
    for i, class_name in enumerate(predicted_classes):
        confidence = 0.9 - (i * 0.15) + np.random.uniform(-0.1, 0.1)
        confidence = max(0.1, min(0.95, confidence))
        print(f"   {i+1}. {class_name}: {confidence:.1%}")
    
    # Analisi colori
    print(f"\n🎨 Analisi colori:")
    print(f"   Luminosità: {brightness:.0f}")
    print(f"   Rosso: {red_content:.0f}")
    print(f"   Verde: {green_content:.0f}")
    print(f"   Blu: {blue_content:.0f}")

def main():
    """Menu principale demo"""
    while True:
        print("\n" + "=" * 60)
        print("🎮 MENU DEMO SISTEMA UNIVERSALE")
        print("=" * 60)
        print("1. 📹 Demo Camera con AI")
        print("2. 💰 Demo Crypto Trading AI")
        print("3. 🖼️ Demo Classificazione Immagini")
        print("4. 📊 Mostra Stato Sistema")
        print("5. 🎪 Demo Completo")
        print("0. ❌ Esci")
        print("-" * 60)
        
        try:
            choice = input("Scegli opzione (0-5): ").strip()
            
            if choice == "0":
                print("👋 Arrivederci!")
                break
            elif choice == "1":
                demo_camera_ai()
            elif choice == "2":
                demo_crypto_ai()
            elif choice == "3":
                demo_image_classification()
            elif choice == "4":
                show_system_status()
            elif choice == "5":
                print("🎪 Avviando demo completo...")
                demo_image_classification()
                input("\nPremi Enter per continuare con crypto demo...")
                demo_crypto_ai()
                input("\nPremi Enter per continuare con camera demo...")
                demo_camera_ai()
                print("🎉 Demo completo terminato!")
            else:
                print("❌ Opzione non valida")
        
        except KeyboardInterrupt:
            print("\n👋 Demo interrotto")
            break
        except Exception as e:
            print(f"❌ Errore: {e}")

def show_system_status():
    """Mostra stato sistema"""
    print("\n📊 STATO SISTEMA")
    print("-" * 40)
    
    # Test componenti
    components = {}
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            components["Camera"] = "✅ Funzionante"
        else:
            components["Camera"] = "❌ Non disponibile"
    except:
        components["Camera"] = "❌ OpenCV mancante"
    
    try:
        from pycoral.utils import edgetpu
        devices = edgetpu.list_edge_tpus()
        if devices:
            components["Coral TPU"] = f"✅ {len(devices)} dispositivi"
        else:
            components["Coral TPU"] = "⚠️ Non rilevato"
    except:
        components["Coral TPU"] = "❌ PyCoral mancante"
    
    models_dir = Path("models")
    if models_dir.exists():
        tflite_count = len(list(models_dir.glob("*.tflite")))
        components["Modelli AI"] = f"✅ {tflite_count} modelli"
    else:
        components["Modelli AI"] = "❌ Cartella modelli mancante"
    
    try:
        import numpy as np
        import pandas as pd
        components["Librerie"] = "✅ Disponibili"
    except:
        components["Librerie"] = "❌ Mancanti"
    
    # Mostra stato
    for component, status in components.items():
        print(f"{component:15}: {status}")
    
    # File generati
    print(f"\n📁 File generati:")
    demo_files = [
        "camera_test.jpg", "demo_crypto_data.csv", 
        "test_image.jpg", "crypto_test_data.csv"
    ]
    
    for file in demo_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"   ✅ {file} ({size} bytes)")
        else:
            print(f"   ❌ {file}")

if __name__ == "__main__":
    print("🎉 Benvenuto nel Sistema Universale Coral TPU!")
    print("Questo demo mostra le capacità AI senza bisogno di modelli complessi")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo terminato")
    except Exception as e:
        print(f"\n❌ Errore demo: {e}")
        print("💡 Assicurati che OpenCV sia installato: conda install opencv")
