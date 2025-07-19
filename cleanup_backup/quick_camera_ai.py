#!/usr/bin/env python3
"""
Test Rapido Camera AI
Versione minimalista per test immediato
"""

import cv2
import numpy as np
import time

def quick_camera_test():
    """Test camera con AI basic"""
    print("🚀 QUICK CAMERA AI TEST")
    print("=" * 40)
    
    # Apri camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera non disponibile")
        return
    
    print("✅ Camera OK - Premi 'q' per uscire, 's' per salvare")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calcola FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Simula AI detection
        height, width = frame.shape[:2]
        brightness = np.mean(frame)
        
        # Rilevamento semplice movimento/oggetti
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # AI classifications simulate
        ai_objects = ["persona", "oggetto", "movimento", "sfondo", "luce"]
        detected = ai_objects[frame_count % len(ai_objects)]
        confidence = 0.6 + 0.4 * np.random.random()
        
        # Overlay info
        cv2.putText(frame, f"AI: {detected} ({confidence:.1%})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Objects: {len(contours)}", 
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Brightness: {brightness:.0f}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Disegna contorni principali
        cv2.drawContours(frame, contours[:10], -1, (0, 255, 255), 2)
        
        # Aggiungi timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (width - 100, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostra frame
        cv2.imshow('Quick Camera AI Test', frame)
        
        # Input control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"quick_test_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Salvato: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Test completato - {frame_count} frames, FPS medio: {fps:.1f}")

if __name__ == "__main__":
    try:
        quick_camera_test()
    except KeyboardInterrupt:
        print("\n👋 Test interrotto")
    except Exception as e:
        print(f"❌ Errore: {e}")
        print("💡 Assicurati che OpenCV sia installato")
