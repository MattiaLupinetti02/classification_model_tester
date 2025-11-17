import psutil
import time
import threading
from IPython.display import clear_output, display, HTML
import ipywidgets as widgets
import os
import signal

# Calcola automaticamente il limite RAM = totale - 2 GB di margine
TOTAL_RAM = psutil.virtual_memory().total / 1024**3
LIMIT_GB = TOTAL_RAM - 0.5  # esempio: su 16 GB → 14 GB
WARN_THRESHOLD = 0.80
CRIT_THRESHOLD = 0.98

def color_for_usage(usage_ratio):
    """Ritorna il colore in base alla soglia."""
    if usage_ratio < WARN_THRESHOLD:
        return "green"
    elif usage_ratio < CRIT_THRESHOLD:
        return "orange"
    else:
        return "red"

def stop_execution():
    """Ferma l'esecuzione corrente nel notebook."""
    os.kill(os.getpid(), signal.SIGINT)

def monitor_resources(interval=1, auto_stop=False):
    out = widgets.Output()
    display(out)

    def update():
        while True:
            with out:
                clear_output(wait=True)
                mem = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=None)

                usage_gb = mem.used / 1024**3
                usage_ratio = usage_gb / LIMIT_GB
                color = color_for_usage(usage_ratio)

                html_content = f"""
                <div style="font-family: monospace; padding: 10px; border: 2px solid {color}; border-radius: 10px;">
                    <h3 style="color:{color}; margin:0;">📊 Monitor Risorse</h3>
                    <p>Uso CPU: <b>{cpu:.2f}%</b></p>
                    <p>RAM Usata: <b>{usage_gb:.2f} GB</b> / {LIMIT_GB:.2f} GB (limite)</p>
                    <p>Disponibile: <b>{mem.available / 1024**3:.2f} GB</b> (Totale fisico: {TOTAL_RAM:.2f} GB)</p>
                </div>
                """

                display(HTML(html_content))

                if auto_stop and usage_ratio >= CRIT_THRESHOLD:
                    print("Uso RAM critico! Interrompo l'esecuzione...")
                    stop_execution()

            time.sleep(interval)

    thread = threading.Thread(target=update, daemon=True)
    thread.start()

# Avvio monitor con limite massimo possibile e auto_stop disattivato
monitor_resources(interval=2, auto_stop=True)
