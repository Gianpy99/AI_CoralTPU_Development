# Coral TPU Crypto AI Trading System - Quick Start Guide

## 🚀 Sistema Completo Creato!

## 📊 STATO ATTUALE DEL SISTEMA

### ✅ **COMPLETAMENTE FUNZIONANTE:**
- Sistema base di trading ✅
- Configurazione Python ✅  
- Strutture dati e logica trading ✅
- Portfolio e risk management ✅
- Raccolta dati crypto ✅
- Test suite funzionante ✅
- **🎉 CORAL TPU COMPLETAMENTE OPERATIVO** ✅
- **Edge TPU Runtime installato** ✅
- **PyCoral libraries funzionanti** ✅
- **TensorFlow Lite pronto** ✅

### 🔧 **Opzionale da Completare:**
- Configurazione API keys exchange (per trading live)
- Modelli AI addestrati personalizzati

### 🧪 **Test Disponibili:**
```bash
# Test sistema base (FUNZIONA!)
python simple_test.py

# Test Coral TPU detection (🎉 CORAL TPU RILEVATO!)
python test_coral_tpu.py

# Test sistema completo
python demo.py --test
```

### 🎯 **RISULTATI CORAL TPU:**
```
🎉 Coral TPU is fully functional!
   ✅ Hardware detected: Device 0 (PCI ApexDevice0)
   ✅ Windows Device Manager: Coral PCIe Accelerator
   ✅ Runtime installed and operational
   ✅ Python libraries available
   ✅ Ready for AI inference
```

### 🖥️ **CONFERMA HARDWARE:**
- **Gestione Dispositivi**: ✅ Coral Accelerator devices → Coral PCIe Accelerator
- **Driver Status**: ✅ Funzionante correttamente
- **Edge TPU Runtime**: ✅ Installato e operativo
- **PyCoral Detection**: ✅ Device 0 rilevato

Hai ora a disposizione un sistema completo per il trading di criptovalute utilizzando il Coral TPU di Google. Ecco cosa è stato creato:

## 📁 Struttura del Progetto

```
AI_CoralTPU_Development/
├── src/                          # Codice sorgente principale
│   ├── config/                   # Configurazione sistema
│   ├── data/                     # Raccolta dati crypto
│   ├── models/                   # AI e inferenza TPU
│   ├── trading/                  # Motore di trading
│   └── utils/                    # Utilità e dashboard
├── scripts/                      # Script di setup e utilità
├── notebooks/                    # Jupyter notebooks per analisi
├── models/                       # Modelli .tflite per TPU
├── data/                         # Cache dati di mercato
├── logs/                         # Log del sistema
├── requirements.txt              # Dipendenze Python
├── .env.template                 # Template configurazione
├── main.py                       # Entry point principale
├── demo.py                       # Demo e test sistema
├── Dockerfile                    # Container Docker
└── README.md                     # Documentazione completa
```

## 🛠️ Setup Rapido

### 1. ✅ Test Sistema Base (COMPLETATO)
```bash
# Il sistema base è già funzionante!
python simple_test.py
```

### 2. 🔧 Installa Coral TPU (NECESSARIO)
```bash
# Test stato attuale Coral TPU
python test_coral_tpu.py

# Segui la guida completa
# Vedi: CORAL_TPU_SETUP.md
```

### 3. Configura l'Ambiente
```bash
# Copia il template di configurazione
copy .env.template .env

# Modifica .env con le tue API keys
notepad .env
```

### 4. Test del Sistema Completo
```bash
# Test rapido sistema
python simple_test.py

# Test Coral TPU
python test_coral_tpu.py

# Demo completa (con dati simulati)
python demo.py
```

### 4. Avvia il Sistema
```bash
# Sistema completo
python main.py
```

## � STATO ATTUALE DEL SISTEMA

### ✅ **Completato e Funzionante:**
- Sistema base di trading ✅
- Configurazione Python ✅  
- Strutture dati e logica trading ✅
- Portfolio e risk management ✅
- Raccolta dati crypto ✅
- Test suite funzionante ✅

### 🔧 **Da Completare:**
- Installazione Coral TPU hardware
- Edge TPU Runtime per Windows
- Librerie PyCoral e TensorFlow Lite
- Configurazione API keys exchange

### 🧪 **Test Disponibili:**
```bash
# Test sistema base (FUNZIONA!)
python simple_test.py

# Test Coral TPU detection
python test_coral_tpu.py

# Test sistema completo
python demo.py --test
```

## �🔧 Componenti Principali

### 📊 Raccolta Dati
- **Exchanges supportati**: Binance, Coinbase Pro
- **Dati in tempo reale**: OHLCV, order book, trades
- **Indicatori tecnici**: SMA, EMA, MACD, RSI, Bollinger Bands
- **File**: `src/data/collectors/crypto_collector.py`

### 🧠 AI e Coral TPU
- **Inferenza ultra-veloce** con Google Coral TPU
- **Fallback automatico** su CPU se TPU non disponibile
- **Modelli ottimizzati** per Edge TPU
- **File**: `src/models/inference/tpu_inference.py`

### 💰 Motore di Trading
- **Modalità simulazione** e live
- **Gestione del rischio** integrata
- **Portfolio management**
- **Stop loss/Take profit automatici**
- **File**: `src/trading/engine/trading_engine.py`

### 📱 Dashboard Web
- **Monitoraggio in tempo reale** su http://localhost:8000
- **Grafici interattivi** delle performance
- **Log e metriche** live
- **File**: `src/utils/monitoring/dashboard.py`

## 🎯 Training di Modelli

### Raccolta Dati Storici
```bash
python src/data/collectors/crypto_collector.py --symbol BTCUSDT --hours 720 --output data/BTCUSDT_30d.csv
```

### Training del Modello
```bash
python src/models/training/train_model.py --data data/BTCUSDT_30d.csv --model-type dense_classifier --epochs 50
```

### Test del Modello
```bash
python src/utils/test_tpu.py
```

## 🐳 Docker Deployment

### Build dell'Immagine
```bash
docker build -t coral-tpu-trading .
```

### Run con Docker Compose
```bash
docker-compose up -d
```

## 📝 Configurazione API Keys

Modifica il file `.env`:

```env
# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Trading Configuration
TRADING_MODE=simulation  # o 'live' per trading reale
DEFAULT_SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT
MAX_POSITION_SIZE=0.02  # 2% max per posizione
CONFIDENCE_THRESHOLD=0.7  # Soglia confidenza AI

# Sistema
ENABLE_DASHBOARD=true
DASHBOARD_PORT=8000
```

## 🎛️ Strategies di Trading

Il sistema include diverse strategie:

1. **AI Direction Prediction**: Usa il Coral TPU per predire direzione prezzi
2. **Mean Reversion**: Sfrutta ritorni alla media
3. **Momentum**: Segue i trend di mercato
4. **Technical Analysis**: Combina indicatori multipli

## 📈 Monitoraggio

### Dashboard Web
- Accedi a http://localhost:8000
- Visualizza portfolio in tempo reale
- Monitora performance e trades

### Log Files
- `logs/trading_system.log`: Log completo sistema
- Log rotazione automatica ogni 100MB

## ⚠️ Importante: Sicurezza

1. **Non usare mai soldi reali** senza test approfonditi
2. **Inizia sempre in modalità simulazione**
3. **Testa le strategie** con backtesting
4. **Mantieni API keys sicure**
5. **Monitora sempre il sistema** quando attivo

## 🔗 Comandi Utili

```bash
# Setup completo
python scripts/setup.py

# Test sistema
python src/utils/test_tpu.py

# Demo con dati live (serve API key)
python demo.py

# Sistema completo
python main.py

# Jupyter notebooks
jupyter notebook notebooks/
```

## 📚 Prossimi Passi

1. **Installa Coral TPU runtime** dal sito di Google
2. **Configura le API keys** degli exchange
3. **Testa il sistema** in modalità simulazione
4. **Analizza i risultati** con i notebook Jupyter
5. **Ottimizza le strategie** in base ai tuoi obiettivi

## 🆘 Support

- Controlla i **log files** per errori
- Usa `python demo.py --test` per diagnostica
- Consulta la documentazione in `README.md`
- I notebook Jupyter contengono esempi dettagliati

## 🚀 SISTEMA UNIVERSALE CORAL TPU

Il tuo sistema è ora **UNIVERSALE**! Oltre al trading crypto, puoi usare il Coral TPU per:

### 📹 **Computer Vision Live**
```bash
# Classificazione immagini in tempo reale
python universal_app.py --mode vision --vision-type classification

# Rilevamento oggetti live
python universal_app.py --mode vision --vision-type detection
```

### 📸 **Foto con AI**
```bash
# Scatta foto e analizza con AI
python universal_app.py --mode photo
```

### 💰 **Trading Crypto** (originale)
```bash
# Predizioni crypto con AI
python universal_app.py --mode crypto
```

### 🎮 **Menu Interattivo**
```bash
# Avvia il menu principale
python universal_app.py
```

### 🎪 **Demo Completo**
```bash
# Mostra tutte le funzionalità
python universal_app.py --mode demo
```

## ⚙️ **Setup Rapido Nuovo Sistema**

```bash
# 1. Installa dipendenze aggiuntive (OpenCV, PIL, etc.)
python setup_universal.py

# 2. Testa tutto il sistema
python universal_app.py --mode demo

# 3. Test rapidi
python quick_camera_test.py    # 📸 Test camera
python quick_crypto_test.py    # 💰 Test crypto
```

## 🎯 **Controlli Camera Live**
Durante lo stream video:
- **`q`** - Esci
- **`s`** - Salva frame
- **`c`** - Cambia modello AI
- **`p`** - Pausa/Riprendi

## 📊 **Modelli AI Disponibili**
- **crypto_trading**: Predizioni trading
- **image_classification**: Riconoscimento immagini
- **object_detection**: Rilevamento oggetti
- **face_detection**: Rilevamento volti

## 🔧 **Configurazione**
Modifica `inference_config.json` per personalizzare:
- Soglie di confidenza
- Risoluzione camera
- Modelli da caricare

## 🎉 Buon Divertimento!

Hai ora un sistema **UNIVERSALE** di AI con Coral TPU che supporta:
- **💰 Trading crypto** con AI in tempo reale
- **👁️ Computer vision** con camera live
- **📸 Analisi foto** con AI avanzata
- **🔍 Riconoscimento oggetti** e classificazione

**📖 Documentazione completa**: `README_UNIVERSAL.md`

**DISCLAIMER**: Questo sistema è per scopi educativi. Il trading comporta rischi. Non investire mai più di quanto puoi permetterti di perdere!
