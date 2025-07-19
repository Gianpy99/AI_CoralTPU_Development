# Coral TPU Crypto AI Trading System - Quick Start Guide

## 🚀 Sistema Completo Creato!

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

### 1. Installa le Dipendenze
```bash
pip install -r requirements.txt
```

### 2. Configura l'Ambiente
```bash
# Copia il template di configurazione
copy .env.template .env

# Modifica .env con le tue API keys
notepad .env
```

### 3. Test del Sistema
```bash
# Mostra informazioni sistema
python demo.py --info

# Test rapido
python demo.py --test

# Demo completa (con dati simulati)
python demo.py
```

### 4. Avvia il Sistema
```bash
# Sistema completo
python main.py
```

## 🔧 Componenti Principali

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

## 🎉 Buon Trading!

Hai ora un sistema completo di trading crypto con AI e Coral TPU. Ricorda:
- **Testa sempre** prima di usare denaro reale
- **Monitora costantemente** le performance
- **Aggiorna regolarmente** i modelli AI
- **Gestisci sempre il rischio**

**DISCLAIMER**: Questo sistema è per scopi educativi. Il trading comporta rischi. Non investire mai più di quanto puoi permetterti di perdere!
