@echo off
echo ============================================
echo        SISTEMA UNIVERSALE CORAL TPU
echo ============================================
echo.

echo Scegli modalita:
echo 1. Demo Universale (Raccomandato)
echo 2. Camera AI Live
echo 3. Sistema Completo
echo 4. Test Quick Camera
echo 5. Analisi Crypto
echo 0. Esci
echo.

set /p choice="Inserisci numero (0-5): "

if "%choice%"=="0" goto :eof
if "%choice%"=="1" python demo_universal.py
if "%choice%"=="2" python universal_app.py --mode vision
if "%choice%"=="3" python universal_app.py
if "%choice%"=="4" python quick_camera_ai.py
if "%choice%"=="5" python universal_app.py --mode crypto

echo.
echo Premi un tasto per continuare...
pause >nul
goto :eof
