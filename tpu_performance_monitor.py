#!/usr/bin/env python3
"""
TPU PERFORMANCE MONITOR - Monitoraggio Real-time Performance e Temperatura
Monitora performance Coral TPU, temperatura sistema e statistiche live
"""

import time
import psutil
import threading
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import deque

try:
    import wmi
    WMI_AVAILABLE = True
except ImportError:
    WMI_AVAILABLE = False

try:
    from pycoral.utils import edgetpu
    CORAL_AVAILABLE = True
except ImportError:
    CORAL_AVAILABLE = False

class TPUPerformanceMonitor:
    """Monitor completo per performance TPU e sistema"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        
        # Metriche TPU
        self.tpu_metrics = {
            'inference_times': deque(maxlen=100),
            'fps_history': deque(maxlen=50),
            'temperature': deque(maxlen=30),
            'power_usage': deque(maxlen=30),
            'utilization': deque(maxlen=50)
        }
        
        # Metriche sistema
        self.system_metrics = {
            'cpu_usage': deque(maxlen=50),
            'memory_usage': deque(maxlen=50),
            'gpu_temperature': deque(maxlen=30),
            'cpu_temperature': deque(maxlen=30),
            'disk_usage': deque(maxlen=20)
        }
        
        # Configurazione
        self.config = {
            'update_interval': 1.0,  # secondi
            'temperature_warning': 80,  # °C
            'cpu_warning': 90,  # %
            'memory_warning': 85,  # %
            'save_logs': True,
            'log_file': 'tpu_performance_log.json'
        }
        
        # Inizializza sensori
        self.init_sensors()
        
        print("🌡️ TPU Performance Monitor inizializzato")
    
    def init_sensors(self):
        """Inizializza sensori di sistema"""
        self.wmi_interface = None
        
        if WMI_AVAILABLE:
            try:
                self.wmi_interface = wmi.WMI(namespace="root\\wmi")
                print("✅ Sensori WMI inizializzati")
            except Exception as e:
                print(f"⚠️ WMI non disponibile: {e}")
        
        # Verifica TPU
        if CORAL_AVAILABLE:
            try:
                devices = edgetpu.list_edge_tpus()
                if devices:
                    print(f"✅ Coral TPU rilevato: {len(devices)} dispositivi")
                    self.tpu_device = devices[0]
                else:
                    print("⚠️ Nessun Coral TPU trovato")
                    self.tpu_device = None
            except Exception as e:
                print(f"⚠️ Errore TPU detection: {e}")
                self.tpu_device = None
    
    def get_cpu_temperature(self):
        """Ottieni temperatura CPU"""
        try:
            if WMI_AVAILABLE and self.wmi_interface:
                # Prova sensori WMI
                temperature_info = self.wmi_interface.query("SELECT * FROM MSAcpi_ThermalZoneTemperature")
                if temperature_info:
                    # Converti da decimi di Kelvin a Celsius
                    temp_kelvin = temperature_info[0].CurrentTemperature / 10
                    temp_celsius = temp_kelvin - 273.15
                    return temp_celsius
            
            # Fallback: usa psutil se disponibile
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current
            
            return None
            
        except Exception as e:
            return None
    
    def get_gpu_temperature(self):
        """Ottieni temperatura GPU"""
        try:
            # Per NVIDIA GPU
            import subprocess
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=temperature.gpu', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            
        except Exception:
            pass
        
        return None
    
    def get_tpu_metrics(self, inference_time=None):
        """Aggiorna metriche TPU"""
        metrics = {}
        
        if inference_time is not None:
            self.tpu_metrics['inference_times'].append(inference_time)
            
            # Calcola FPS
            if len(self.tpu_metrics['inference_times']) > 1:
                avg_inference = np.mean(list(self.tpu_metrics['inference_times'])[-10:])
                fps = 1000.0 / avg_inference if avg_inference > 0 else 0
                self.tpu_metrics['fps_history'].append(fps)
                metrics['current_fps'] = fps
                metrics['avg_fps'] = np.mean(list(self.tpu_metrics['fps_history']))
        
        # Simulazione temperatura TPU (non sempre disponibile via API)
        if self.tpu_device:
            # Stima temperatura basata su utilizzo
            if self.tpu_metrics['inference_times']:
                recent_times = list(self.tpu_metrics['inference_times'])[-5:]
                avg_time = np.mean(recent_times)
                # Stima temperatura: più veloce = più caldo
                estimated_temp = 35 + (10 / avg_time) if avg_time > 0 else 35
                estimated_temp = min(85, max(25, estimated_temp))
                self.tpu_metrics['temperature'].append(estimated_temp)
                metrics['tpu_temperature'] = estimated_temp
        
        return metrics
    
    def get_system_metrics(self):
        """Ottieni metriche sistema"""
        metrics = {}
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.system_metrics['cpu_usage'].append(cpu_percent)
        metrics['cpu_usage'] = cpu_percent
        
        # Memoria
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.system_metrics['memory_usage'].append(memory_percent)
        metrics['memory_usage'] = memory_percent
        metrics['memory_available'] = memory.available / (1024**3)  # GB
        
        # Temperatura CPU
        cpu_temp = self.get_cpu_temperature()
        if cpu_temp:
            self.system_metrics['cpu_temperature'].append(cpu_temp)
            metrics['cpu_temperature'] = cpu_temp
        
        # Temperatura GPU
        gpu_temp = self.get_gpu_temperature()
        if gpu_temp:
            self.system_metrics['gpu_temperature'].append(gpu_temp)
            metrics['gpu_temperature'] = gpu_temp
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.system_metrics['disk_usage'].append(disk_percent)
        metrics['disk_usage'] = disk_percent
        
        return metrics
    
    def check_warnings(self, metrics):
        """Controlla warning temperature/performance"""
        warnings = []
        
        # Temperature warnings
        if 'cpu_temperature' in metrics:
            if metrics['cpu_temperature'] > self.config['temperature_warning']:
                warnings.append(f"🔥 CPU temperatura alta: {metrics['cpu_temperature']:.1f}°C")
        
        if 'gpu_temperature' in metrics:
            if metrics['gpu_temperature'] > self.config['temperature_warning']:
                warnings.append(f"🔥 GPU temperatura alta: {metrics['gpu_temperature']:.1f}°C")
        
        if 'tpu_temperature' in metrics:
            if metrics['tpu_temperature'] > self.config['temperature_warning']:
                warnings.append(f"🔥 TPU temperatura alta: {metrics['tpu_temperature']:.1f}°C")
        
        # Performance warnings
        if metrics['cpu_usage'] > self.config['cpu_warning']:
            warnings.append(f"⚠️ CPU usage alto: {metrics['cpu_usage']:.1f}%")
        
        if metrics['memory_usage'] > self.config['memory_warning']:
            warnings.append(f"⚠️ Memoria usage alta: {metrics['memory_usage']:.1f}%")
        
        return warnings
    
    def start_monitoring(self):
        """Avvia monitoraggio in background"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🚀 Monitoraggio performance avviato")
    
    def stop_monitoring(self):
        """Ferma monitoraggio"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("⏹️ Monitoraggio performance fermato")
    
    def _monitor_loop(self):
        """Loop principale monitoraggio"""
        while self.monitoring:
            try:
                # Raccogli metriche
                system_metrics = self.get_system_metrics()
                tpu_metrics = self.get_tpu_metrics()
                
                all_metrics = {**system_metrics, **tpu_metrics}
                
                # Controlla warnings
                warnings = self.check_warnings(all_metrics)
                if warnings:
                    for warning in warnings:
                        print(warning)
                
                # Salva log se configurato
                if self.config['save_logs']:
                    self._save_metrics_log(all_metrics)
                
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                print(f"⚠️ Errore monitoring: {e}")
                time.sleep(1)
    
    def _save_metrics_log(self, metrics):
        """Salva metriche su file"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            log_file = Path(self.config['log_file'])
            
            # Carica log esistenti
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Aggiungi nuovo entry
            logs.append(log_entry)
            
            # Mantieni solo ultimi 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Salva
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Errore salvataggio log: {e}")
    
    def get_current_stats(self):
        """Ottieni statistiche correnti"""
        system_metrics = self.get_system_metrics()
        tpu_metrics = self.get_tpu_metrics()
        
        stats = {
            'system': system_metrics,
            'tpu': tpu_metrics,
            'averages': {}
        }
        
        # Calcola medie
        if self.system_metrics['cpu_usage']:
            stats['averages']['avg_cpu'] = np.mean(list(self.system_metrics['cpu_usage']))
        
        if self.system_metrics['memory_usage']:
            stats['averages']['avg_memory'] = np.mean(list(self.system_metrics['memory_usage']))
        
        if self.tpu_metrics['inference_times']:
            stats['averages']['avg_inference_time'] = np.mean(list(self.tpu_metrics['inference_times']))
        
        if self.tpu_metrics['fps_history']:
            stats['averages']['avg_fps'] = np.mean(list(self.tpu_metrics['fps_history']))
        
        return stats
    
    def record_inference(self, inference_time_ms):
        """Registra tempo inferenza per statistiche TPU"""
        self.get_tpu_metrics(inference_time_ms)
    
    def get_performance_overlay_text(self):
        """Ottieni testo per overlay performance"""
        stats = self.get_current_stats()
        
        lines = []
        
        # TPU Stats
        if 'current_fps' in stats['tpu']:
            lines.append(f"TPU FPS: {stats['tpu']['current_fps']:.1f}")
        
        if self.tpu_metrics['inference_times']:
            avg_time = np.mean(list(self.tpu_metrics['inference_times'])[-10:])
            lines.append(f"Inference: {avg_time:.1f}ms")
        
        if 'tpu_temperature' in stats['tpu']:
            temp = stats['tpu']['tpu_temperature']
            temp_color = "🔥" if temp > 70 else "🌡️"
            lines.append(f"{temp_color} TPU: {temp:.1f}°C")
        
        # Sistema
        if 'cpu_usage' in stats['system']:
            cpu = stats['system']['cpu_usage']
            cpu_icon = "⚠️" if cpu > 80 else "💻"
            lines.append(f"{cpu_icon} CPU: {cpu:.1f}%")
        
        if 'memory_usage' in stats['system']:
            mem = stats['system']['memory_usage']
            mem_icon = "⚠️" if mem > 80 else "🧠"
            lines.append(f"{mem_icon} RAM: {mem:.1f}%")
        
        if 'cpu_temperature' in stats['system']:
            temp = stats['system']['cpu_temperature']
            temp_icon = "🔥" if temp > 70 else "🌡️"
            lines.append(f"{temp_icon} CPU: {temp:.1f}°C")
        
        return lines
    
    def print_detailed_report(self):
        """Stampa report dettagliato"""
        print("\n🌡️ REPORT PERFORMANCE DETTAGLIATO")
        print("=" * 50)
        
        stats = self.get_current_stats()
        
        # TPU Performance
        print("🔥 TPU PERFORMANCE:")
        if self.tpu_metrics['inference_times']:
            times = list(self.tpu_metrics['inference_times'])
            print(f"  Inference time: {np.mean(times):.1f}ms (avg)")
            print(f"  Min/Max: {np.min(times):.1f}/{np.max(times):.1f}ms")
        
        if self.tpu_metrics['fps_history']:
            fps = list(self.tpu_metrics['fps_history'])
            print(f"  FPS: {np.mean(fps):.1f} (avg)")
        
        if self.tpu_metrics['temperature']:
            temps = list(self.tpu_metrics['temperature'])
            print(f"  Temperatura: {np.mean(temps):.1f}°C (avg)")
        
        # Sistema
        print("\n💻 SISTEMA:")
        if self.system_metrics['cpu_usage']:
            cpu = list(self.system_metrics['cpu_usage'])
            print(f"  CPU: {np.mean(cpu):.1f}% (avg)")
        
        if self.system_metrics['memory_usage']:
            mem = list(self.system_metrics['memory_usage'])
            print(f"  Memoria: {np.mean(mem):.1f}% (avg)")
        
        if self.system_metrics['cpu_temperature']:
            temps = list(self.system_metrics['cpu_temperature'])
            print(f"  CPU Temp: {np.mean(temps):.1f}°C (avg)")
        
        # Raccomandazioni
        print("\n💡 RACCOMANDAZIONI:")
        if stats['system'].get('cpu_usage', 0) > 85:
            print("  ⚠️ CPU usage alto - considera ottimizzazioni")
        if stats['system'].get('memory_usage', 0) > 85:
            print("  ⚠️ Memoria alta - controlla memory leaks")
        if stats['system'].get('cpu_temperature', 0) > 75:
            print("  🔥 Temperatura alta - controlla ventilazione")

def main():
    """Test TPU Performance Monitor"""
    print("🌡️ TEST TPU PERFORMANCE MONITOR")
    print("=" * 50)
    
    monitor = TPUPerformanceMonitor()
    
    # Avvia monitoraggio
    monitor.start_monitoring()
    
    try:
        print("Monitoraggio attivo... (Ctrl+C per fermare)")
        
        # Simula alcune inferenze
        for i in range(20):
            # Simula tempo inferenza variabile
            simulated_time = 5.0 + np.random.normal(0, 1.0)
            simulated_time = max(3.0, simulated_time)
            
            monitor.record_inference(simulated_time)
            
            # Stampa overlay ogni 5 iterazioni
            if i % 5 == 0:
                overlay_lines = monitor.get_performance_overlay_text()
                print(f"\n📊 Performance Overlay {i//5 + 1}:")
                for line in overlay_lines:
                    print(f"  {line}")
            
            time.sleep(1)
        
        # Report finale
        monitor.print_detailed_report()
        
    except KeyboardInterrupt:
        print("\n⏹️ Fermando monitoraggio...")
    
    finally:
        monitor.stop_monitoring()
        print("✅ Monitor fermato")

if __name__ == "__main__":
    main()
