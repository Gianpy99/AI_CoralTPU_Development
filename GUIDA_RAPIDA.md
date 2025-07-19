# 🚀 Sistema Universale Coral TPU - Guida Rapida

## ✨ Cosa Hai Ottenuto

Il tuo sistema originale per crypto trading è stato **trasformato** in una **piattaforma AI universale** che supporta:

- 💰 **Crypto Trading AI** (originale + migliorato)
- 📹 **Computer Vision in tempo reale** con camera
- 🖼️ **Classificazione immagini**
- 🎯 **Rilevamento oggetti**
- 👤 **Riconoscimento facciale**
- ⚡ **Accelerazione hardware Coral TPU**

## 🎮 Come Usare il Sistema

### 🚀 Lancio Rapido

```bash
# Demo semplificato (funziona subito)
python demo_universal.py

# Sistema completo (richiede setup)
python universal_app.py
```

### 📹 Modalità Camera AI

```bash
# Avvia camera con AI in tempo reale
python universal_app.py --mode vision

# Controlli durante l'uso:
# q = Esci
# s = Salva foto con AI
# c = Cambia modello AI
# p = Pausa/riprendi
```

### 🖼️ Analizza Singola Foto

```bash
# Analizza una foto specifica
python universal_app.py --mode photo --image "percorso/foto.jpg"
```

### 💰 Crypto Trading (Originale Migliorato)

```bash
# Predizioni crypto con AI
python universal_app.py --mode crypto
```

### 🎪 Menu Interattivo

```bash
# Menu completo con tutte le opzioni
python universal_app.py --mode menu
```

## 🔧 Configurazione Modelli

Modifica `inference_config.json` per personalizzare:

```json
{
  "models": {
    "bird_classifier": {
      "model_path": "models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
      "labels_path": "models/inat_bird_labels.txt",
      "task_type": "classification"
    }
  }
}
```

## 📊 Modelli AI Inclusi

1. **🐦 Bird Classifier** - Riconosce 965 specie di uccelli
2. **🎯 Object Detection** - Rileva 80+ oggetti COCO
3. **👤 Face Detection** - Rilevamento volti in tempo reale

## ⚙️ Risoluzione Problemi

### 🔹 Camera Non Funziona
```bash
# Verifica camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'ERRORE')"
```

### 🔹 Coral TPU Non Rilevato
```bash
# Verifica TPU
python -c "from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())"
```

### 🔹 Reinstalla Dipendenze
```bash
# OpenCV
conda install opencv

# Coral TPU
pip install pycoral

# Altri componenti
pip install pillow scikit-learn matplotlib
```

## 🎯 Esempi d'Uso Creativi

### 📹 Sorveglianza AI
- Avvia modalità camera
- Rileva automaticamente persone/oggetti
- Salva foto quando rileva attività

### 🖼️ Catalogazione Foto
- Analizza cartelle di foto
- Classifica automaticamente contenuti
- Organizza per categoria AI

### 💰 Trading Automatico
- Monitora prezzi crypto
- Predizioni AI in tempo reale
- Alert automatici

### 🎨 Analisi Artistica
- Carica opere d'arte
- Analisi stile e contenuto
- Riconoscimento artisti

## 📁 Struttura File

```
AI_CoralTPU_Development/
├── universal_app.py         # 🚀 App principale
├── universal_inference.py   # 🧠 Motore AI universale
├── camera_vision.py         # 📹 Gestione camera
├── demo_universal.py        # 🎮 Demo semplificato
├── inference_config.json    # ⚙️ Configurazione
├── setup_universal.py       # 📦 Installazione
├── models/                  # 🤖 Modelli AI
│   ├── *.tflite            # Modelli ottimizzati
│   └── *.txt               # Etichette
└── GUIDA_RAPIDA.md          # 📖 Questa guida
```

## 🎊 Caratteristiche Avanzate

### ⚡ Accelerazione Hardware
- Rileva automaticamente Coral TPU
- Fallback CPU se TPU non disponibile
- Ottimizzazioni per prestazioni massime

### 🔄 Multi-Modello
- Carica multipli modelli simultaneamente
- Cambio modello in tempo reale
- Preprocessing automatico per tipo

### 📊 Analisi Performance
- Metriche inferenza in tempo reale
- FPS camera live
- Utilizzo risorse

### 💾 Persistenza Dati
- Salvataggio automatico risultati
- Export CSV per analisi
- History predizioni

## 🚀 Prossimi Passi

1. **Testa il demo**: `python demo_universal.py`
2. **Prova la camera**: Modalità vision per vedere l'AI live
3. **Personalizza**: Modifica `inference_config.json`
4. **Aggiungi modelli**: Scarica nuovi modelli EdgeTPU
5. **Estendi**: Crea nuove funzionalità basate su questo framework

## 🎉 Congratulazioni!

Hai trasformato un sistema crypto in una **piattaforma AI universale**! 

🔥 **Cosa puoi fare ora:**
- Riconoscimento oggetti con la tua camera
- Analisi foto con AI professionale  
- Trading crypto migliorato con AI
- Sviluppo di nuove applicazioni AI

Il sistema è **pronto per l'uso** e **facilmente estendibile**! 🚀
