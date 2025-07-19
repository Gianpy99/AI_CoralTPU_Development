# 🔧 Guida Installazione Coral TPU per Windows

## ⚠️ STATO ATTUALE
Il test del sistema ha rivelato che il Coral TPU non è ancora configurato. Ecco cosa serve:

### ✅ Già Installato:
- Windows 10 (compatibile)
- Python 3.13.3
- NumPy e Pillow

### ❌ Da Installare:
- Coral TPU hardware (non rilevato)
- Edge TPU Runtime
- Librerie Python (tflite-runtime, pycoral)

## 🛠️ PASSAGGI DI INSTALLAZIONE

### 1. 📦 Hardware Coral TPU
**Se non hai ancora un Coral TPU:**
- Coral USB Accelerator: https://coral.ai/products/accelerator/
- Coral Dev Board: https://coral.ai/products/dev-board/

**Se hai già il dispositivo:**
- Collega il Coral TPU via USB
- Verifica che appaia in Gestione Dispositivi di Windows

### 2. 💿 Edge TPU Runtime per Windows

**Opzione A: Download Manuale (Consigliato)**
1. Vai su: https://coral.ai/software/#edgetpu-runtime
2. Scarica "Edge TPU runtime library (Windows)"
3. Estrai il file ZIP
4. Esegui `install.bat` come amministratore

**Opzione B: Installazione Automatica**
```powershell
# Esegui PowerShell come amministratore
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Scarica e installa
$url = "https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20220711.zip"
$output = "$env:TEMP\edgetpu_runtime.zip"
Invoke-WebRequest -Uri $url -OutFile $output
Expand-Archive -Path $output -DestinationPath "$env:TEMP\edgetpu_runtime"
Start-Process -FilePath "$env:TEMP\edgetpu_runtime\install.bat" -Verb RunAs
```

### 3. 🐍 Librerie Python

**Per il tuo ambiente Python attuale:**

```bash
# Installa TensorFlow Lite (versione compatibile)
pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-win_amd64.whl

# Installa PyCoral
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

**Se questo non funziona, usa l'ambiente conda:**
```bash
# Attiva ambiente conda
conda activate base

# Installa le dipendenze
conda install -c conda-forge numpy pillow
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

### 4. 🔧 Configurazione Sistema

**Aggiungi al PATH (opzionale):**
```
C:\Program Files\EdgeTPU\
```

**Riavvia il computer** dopo l'installazione del runtime.

### 5. ✅ Verifica Installazione

Dopo aver completato i passaggi sopra, esegui nuovamente il test:

```bash
python test_coral_tpu.py
```

Dovresti vedere:
- ✅ TensorFlow Lite Runtime: Available
- ✅ PyCoral Edge TPU Utils: Available  
- ✅ Edge TPU devices found: 1
- 🎉 Coral TPU is ready for use!

## 🚨 TROUBLESHOOTING

### Problema: "No Edge TPU devices detected"
**Soluzioni:**
1. Verifica connessione USB
2. Controlla Gestione Dispositivi Windows
3. Reinstalla driver USB
4. Riavvia il computer

### Problema: "Cannot import PyCoral"
**Soluzioni:**
1. Verifica ambiente Python corretto
2. Reinstalla PyCoral con `--force-reinstall`
3. Controlla conflitti di versione

### Problema: Driver non riconosciuto
**Soluzioni:**
1. Installa driver manualmente da Gestione Dispositivi
2. Usa driver generico "libusb-win32"
3. Controlla compatibilità Windows

## 🔗 Link Utili

- **Documentazione Ufficiale**: https://coral.ai/docs/accelerator/get-started/
- **Download Runtime**: https://coral.ai/software/#edgetpu-runtime
- **GitHub PyCoral**: https://github.com/google-coral/pycoral
- **Forum Support**: https://github.com/google-coral/edgetpu/issues

## 🎯 PROSSIMI PASSI

Dopo aver installato tutto:

1. **Testa il sistema**: `python test_coral_tpu.py`
2. **Scarica un modello**: Modelli pre-addestrati da coral.ai
3. **Testa inferenza**: `python demo.py --test`
4. **Avvia sistema completo**: `python main.py`

## ⚡ MODELLI DI TEST

Una volta che il TPU funziona, puoi testare con questi modelli:

```bash
# Classificazione immagini
wget https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite

# Rilevamento oggetti  
wget https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
```

---

**📝 Nota**: L'installazione può richiedere 15-30 minuti e un riavvio del sistema. Una volta completata, avrai un sistema di trading AI completamente funzionale con accelerazione Coral TPU!
