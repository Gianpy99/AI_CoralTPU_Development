# Coral TPU Crypto AI Trading System

Un sistema completo che utilizza Google Coral TPU per prendere decisioni di trading in tempo reale su criptovalute utilizzando modelli AI ottimizzati per edge computing.

## Caratteristiche

- 🚀 **Coral TPU Integration**: Sfrutta la potenza del Google Coral TPU per inferenza AI ultra-veloce
- 📊 **Real-time Data**: Raccolta dati in tempo reale da multiple exchange di criptovalute
- 🤖 **AI Models**: Modelli di machine learning ottimizzati per Edge TPU
- 📈 **Technical Analysis**: Analisi tecnica avanzata con indicatori personalizzati
- ⚡ **Fast Decision Making**: Sistema di decisioni rapide per trading ad alta frequenza
- 📱 **Monitoring**: Dashboard di monitoraggio in tempo reale
- 🔒 **Risk Management**: Sistema integrato di gestione del rischio

## Prerequisiti

- Google Coral TPU (USB Accelerator o Dev Board)
- Python 3.8+
- Windows/Linux/macOS

## Installazione

### 1. Setup Coral TPU

```bash
# Installa il runtime Edge TPU
# Per Windows: Scarica da https://coral.ai/software/#edgetpu-runtime
# Per Linux:
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
```

### 2. Installa le dipendenze Python

```bash
pip install -r requirements.txt
```

### 3. Configura le API keys

Crea un file `.env` nella directory root:

```env
# Exchange API Keys
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
COINBASE_API_KEY=your_coinbase_api_key
COINBASE_SECRET_KEY=your_coinbase_secret_key

# Data Sources
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Quick Start

1. **Test Coral TPU Connection**:
```bash
python src/utils/test_tpu.py
```

2. **Collect Sample Data**:
```bash
python src/data/collectors/crypto_collector.py --symbol BTCUSDT --hours 24
```

3. **Train a Model**:
```bash
python src/models/training/train_model.py --data data/BTCUSDT.csv
```

4. **Run Trading System**:
```bash
python main.py
```

## Struttura del Progetto

```
AI_CoralTPU_Development/
├── src/
│   ├── data/
│   │   ├── collectors/          # Data collection modules
│   │   ├── preprocessors/       # Data preprocessing
│   │   └── storage/            # Data storage handlers
│   ├── models/
│   │   ├── architectures/      # Model architectures
│   │   ├── training/           # Training scripts
│   │   └── inference/          # TPU inference engine
│   ├── trading/
│   │   ├── strategies/         # Trading strategies
│   │   ├── portfolio/          # Portfolio management
│   │   └── risk/              # Risk management
│   ├── utils/
│   │   ├── technical_analysis/ # Technical indicators
│   │   ├── visualization/      # Plotting and charts
│   │   └── notifications/      # Alert system
│   └── config/                 # Configuration files
├── models/                     # Trained .tflite models
├── data/                       # Data cache and storage
├── logs/                       # Application logs
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter analysis notebooks
├── docker/                     # Docker configurations
└── scripts/                    # Utility scripts
```

## Disclaimer

⚠️ **ATTENZIONE**: Questo software è solo per scopi educativi e di ricerca. Il trading di criptovalute comporta rischi significativi. Non utilizzare con denaro reale senza una completa comprensione dei rischi coinvolti.
