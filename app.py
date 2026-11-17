import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, stft
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Monitor Sísmico IoT - Análisis DSP",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 Estación de Análisis Sísmico IoT (ESP32 + MPU-6050)")
st.markdown(
    "Plataforma de procesamiento digital de señales (DSP) para caracterización estructural, "
    "filtrado pasa-altos de gravedad ($1g$), análisis espectral **rFFT triaxial** y **espectrogramas STFT**."
)

# --- FUNCIONES DE PROCESAMIENTO DSP ---
def filtro_pasa_altos(datos: np.ndarray, cutoff_hz: float, fs: float, orden: int = 4) -> np.ndarray:
    if cutoff_hz <= 0 or len(datos) <= orden * 3:
        return datos - np.mean(datos)
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = butter(orden, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, datos)

def calcular_rfft_triaxial(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float):
    n = len(ax)
    ventana = np.hamming(n)
    factor_escala = 2.0 / np.sum(ventana)

    fft_x = np.abs(np.fft.rfft(ax * ventana)) * factor_escala
    fft_y = np.abs(np.fft.rfft(ay * ventana)) * factor_escala
    fft_z = np.abs(np.fft.rfft(az * ventana)) * factor_escala

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # Combinación energética válida en frecuencia (invariante a la inclinación del sensor)
    amp_total = np.sqrt(fft_x**2 + fft_y**2 + fft_z**2)
    return freqs, fft_x, fft_y, fft_z, amp_total

# --- BARRA LATERAL: CARGA Y PARÁMETROS ---
with st.sidebar:
    st.header("1. Cargar Registro CSV")
    archivo = st.file_uploader(
        "Sube el CSV generado por el receptor UDP",
        type=["csv"],
        help="Columnas esperadas: seq, fecha_hora, timestamp_ms, ax, ay, az"
    )

    st.divider()
    st.header("2. Filtros y Escala")
    usar_filtro = st.checkbox("Activar Filtro Pasa-Altos (Remover 1g)", value=True)
    freq_corte = st.slider("Frecuencia de corte HPF (Hz)", 0.1, 2.0, 0.5, 0.1, disabled=not usar_filtro)

    unidad = st.radio(
        "Unidad / Escala de Amplitud",
        options=["m/s² (Lineal)", "Gal - cm/s² (Sismología)", "dB (Escala Logarítmica)"],
        index=0
    )

    st.divider()
    st.header("3. Parámetros Espectrales")
    banda_sismica = st.slider("Banda de interés sísmico (Hz)", 0.5, 25.0, (1.0, 10.0), 0.5)
    nperseg_stft = st.select_slider("Ventana STFT (muestras)", options=[64, 128, 256, 512], value=128)

# --- CUERPO PRINCIPAL ---
if archivo is None:
    st.info("👈 Sube un archivo `.csv` en el panel izquierdo para comenzar el análisis (ej. `datos_arduino_20261116_190348.csv`).")
    st.stop()

try:
    df = pd.read_csv(archivo)
    columnas_req = {"seq", "timestamp_ms", "ax", "ay", "az"}
    if not columnas_req.issubset(set(df.columns)):
        st.error(f"El CSV no tiene las columnas requeridas: {columnas_req}")
        st.stop()
except Exception as e:
    st.error(f"Error al leer el archivo CSV: {e}")
    st.stop()

# Reconstrucción temporal y auditoría de paquetes
df = df.sort_values("seq").reset_index(drop=True)
t_seg_total = (df["timestamp_ms"].values - df["timestamp_ms"].values[0]) / 1000.0
dt_medio = np
