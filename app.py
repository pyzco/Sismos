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

# --- MODO OSCURO / CLARO ---
if "modo_oscuro" not in st.session_state:
    st.session_state.modo_oscuro = True  # oscuro por defecto

with st.sidebar:
    st.toggle("🌙 Modo oscuro", key="modo_oscuro")
    st.divider()

MODO_OSCURO = st.session_state.modo_oscuro

if MODO_OSCURO:
    COLOR_FONDO = "#0e1117"
    COLOR_TEXTO = "#fafafa"
    COLOR_TARJETA = "#1c1f26"
    COLOR_BORDE = "#30333d"
    COLOR_RESULTANTE = "#00e5ff"  # cian brillante -> visible sobre fondo oscuro
    PLOTLY_TEMPLATE = "plotly_dark"
else:
    COLOR_FONDO = "#ffffff"
    COLOR_TEXTO = "#111111"
    COLOR_TARJETA = "#f0f2f6"
    COLOR_BORDE = "#d8dce3"
    COLOR_RESULTANTE = "#111111"  # negro -> visible sobre fondo claro
    PLOTLY_TEMPLATE = "plotly_white"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {COLOR_FONDO};
        color: {COLOR_TEXTO};
    }}
    [data-testid="stSidebar"] {{
        background-color: {COLOR_TARJETA};
        border-right: 1px solid {COLOR_BORDE};
    }}
    [data-testid="stMetric"] {{
        background-color: {COLOR_TARJETA};
        border: 1px solid {COLOR_BORDE};
        padding: 12px;
        border-radius: 10px;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
        color: {COLOR_TEXTO} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌋 Estación de Análisis Sísmico IoT (ESP32 + MPU-6050)")
st.markdown(
    "Plataforma de procesamiento digital de señales (DSP) para caracterización estructural, "
    "filtrado pasa-altos de gravedad ($1g$), análisis espectral **rFFT triaxial** y **espectrogramas STFT**."
)

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

df = df.sort_values("seq").reset_index(drop=True)
t_seg_total = (df["timestamp_ms"].values - df["timestamp_ms"].values[0]) / 1000.0
dt_medio = np.median(np.diff(t_seg_total)) if len(t_seg_total) > 1 else 0.01
fs_real = round(1.0 / dt_medio, 2) if dt_medio > 0 else 100.0

saltos = np.diff(df["seq"].values)
perdidas = int(np.sum(saltos[saltos > 1] - 1))
integridad = (len(df) / (len(df) + perdidas)) * 100.0

st.subheader("⏱️ Selección de Intervalo Temporal")
duracion_total = float(t_seg_total[-1])
rango_t = st.slider(
    "Selecciona el rango de tiempo a analizar (segundos desde el inicio):",
    min_value=0.0,
    max_value=max(duracion_total, 0.1),
    value=(0.0, max(duracion_total, 0.1)),
    step=0.1
)

mascara = (t_seg_total >= rango_t[0]) & (t_seg_total <= rango_t[1])
df_sub = df.loc[mascara].copy()
t_sub = t_seg_total[mascara]

if len(df_sub) < 32:
    st.warning("Selecciona un intervalo de al menos 0.5 segundos (32 muestras) para calcular la FFT.")
    st.stop()

# --- PROCESAMIENTO DE SEÑALES ---
ax_raw = df_sub["ax"].values
ay_raw = df_sub["ay"].values
az_raw = df_sub["az"].values

if usar_filtro:
    ax_proc = filtro_pasa_altos(ax_raw, freq_corte, fs_real)
    ay_proc = filtro_pasa_altos(ay_raw, freq_corte, fs_real)
    az_proc = filtro_pasa_altos(az_raw, freq_corte, fs_real)
else:
    ax_proc, ay_proc, az_proc = ax_raw, ay_raw, az_raw

# Conversión de escala visual
factor_unidad = 100.0 if "Gal" in unidad else 1.0
etiqueta_y = "cm/s² (Gal)" if "Gal" in unidad else "m/s²"

ax_vis = ax_proc * factor_unidad
ay_vis = ay_proc * factor_unidad
az_vis = az_proc * factor_unidad

# Cálculo FFT sobre datos filtrados en m/s² (para métricas físicas consistentes)
ax_din = filtro_pasa_altos(ax_raw, freq_corte, fs_real)
ay_din = filtro_pasa_altos(ay_raw, freq_corte, fs_real)
az_din = filtro_pasa_altos(az_raw, freq_corte, fs_real)
freqs, fx, fy, fz, f_total = calcular_rfft_triaxial(ax_din, ay_din, az_din, fs_real)

# Métricas dentro de la banda sísmica (ej. 1 - 10 Hz)
mask_banda = (freqs >= banda_sismica[0]) & (freqs <= banda_sismica[1])
if np.any(mask_banda):
    idx_pico = np.argmax(f_total[mask_banda])
    freq_dominante = freqs[mask_banda][idx_pico]
    amp_pico = f_total[mask_banda][idx_pico]
    energia_total = np.sum(f_total[freqs >= 0.5]**2) + 1e-12
    energia_banda = np.sum(f_total[mask_banda]**2)
    fraccion_energia = (energia_banda / energia_total) * 100.0
else:
    freq_dominante, amp_pico, fraccion_energia = 0.0, 0.0, 0.0

# --- TARJETAS DE MÉTRICAS ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Muestras en Tramo", f"{len(df_sub):,}", f"{t_sub[-1] - t_sub[0]:.2f} s")
c2.metric("Frecuencia Real (Fs)", f"{fs_real:.1f} Hz", f"Integridad: {integridad:.1f}%")
c3.metric("Frecuencia Dominante", f"{freq_dominante:.2f} Hz", f"Banda {banda_sismica[0]}-{banda_sismica[1]} Hz")
c4.metric("Amplitud Pico Espectral", f"{amp_pico:.4f} m/s²", f"{amp_pico*100:.2f} Gal")
c5.metric("Energía en Banda Sísmica", f"{fraccion_energia:.1f} %", "Ratio E_banda / E_total")

# --- PESTAÑAS DE VISUALIZACIÓN ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Ondas en el Tiempo",
    "📊 Espectro rFFT Triaxial",
    "🔥 Espectrograma (STFT)",
    "📋 Datos y Resumen"
])

with tab1:
    fig_tiempo = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Eje X (Horizontal)", "Eje Y (Horizontal)", "Eje Z (Vertical / Inclinado)"),
        vertical_spacing=0.07
    )
    fig_tiempo.add_trace(go.Scatter(x=t_sub, y=ax_vis, name="Ax", line=dict(color="#1f77b4", width=1.2)), row=1, col=1)
    fig_tiempo.add_trace(go.Scatter(x=t_sub, y=ay_vis, name="Ay", line=dict(color="#2ca02c", width=1.2)), row=2, col=1)
    fig_tiempo.add_trace(go.Scatter(x=t_sub, y=az_vis, name="Az", line=dict(color="#d62728", width=1.2)), row=3, col=1)

    tipo_eje_y = "log" if "dB" in unidad else "linear"
    if "dB" in unidad:
        fig_tiempo.update_traces(y=np.abs(ax_vis) + 1e-5, row=1, col=1)
        fig_tiempo.update_traces(y=np.abs(ay_vis) + 1e-5, row=2, col=1)
        fig_tiempo.update_traces(y=np.abs(az_vis) + 1e-5, row=3, col=1)

    fig_tiempo.update_yaxes(title_text=etiqueta_y, type=tipo_eje_y)
    fig_tiempo.update_xaxes(title_text="Tiempo (s)", row=3, col=1)
    fig_tiempo.update_layout(
        height=620, showlegend=False, margin=dict(t=40, b=30),
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO
    )
    st.plotly_chart(fig_tiempo, use_container_width=True)

with tab2:
    fig_fft = go.Figure()
    escala_fft = 20 * np.log10(f_total + 1e-6) if "dB" in unidad else f_total * factor_unidad
    escala_fx = 20 * np.log10(fx + 1e-6) if "dB" in unidad else fx * factor_unidad
    escala_fy = 20 * np.log10(fy + 1e-6) if "dB" in unidad else fy * factor_unidad
    escala_fz = 20 * np.log10(fz + 1e-6) if "dB" in unidad else fz * factor_unidad
    y_fft_label = "Magnitud (dB)" if "dB" in unidad else f"Amplitud ({etiqueta_y})"

    fig_fft.add_trace(go.Scatter(x=freqs, y=escala_fft, name="Resultante Espectral √(|X|²+|Y|²+|Z|²)", line=dict(color=COLOR_RESULTANTE, width=2.5)))
    fig_fft.add_trace(go.Scatter(x=freqs, y=escala_fx, name="Eje X", line=dict(color="#1f77b4", width=1, dash="dot")))
    fig_fft.add_trace(go.Scatter(x=freqs, y=escala_fy, name="Eje Y", line=dict(color="#2ca02c", width=1, dash="dot")))
    fig_fft.add_trace(go.Scatter(x=freqs, y=escala_fz, name="Eje Z", line=dict(color="#d62728", width=1, dash="dot")))

    fig_fft.add_vrect(
        x0=banda_sismica[0], x1=banda_sismica[1],
        fillcolor="orange", opacity=0.15, line_width=0,
        annotation_text="Banda Sísmica", annotation_position="top left"
    )
    fig_fft.update_layout(
        title="Espectro de Frecuencias (rFFT con Ventana Hamming)",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title=y_fft_label,
        xaxis=dict(range=[0, min(25, fs_real / 2)]),
        height=480,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO
    )
    st.plotly_chart(fig_fft, use_container_width=True)

with tab3:
    nperseg = min(nperseg_stft, len(ax_din))
    f_stft, t_stft, Zxx_x = stft(ax_din, fs=fs_real, window="hamming", nperseg=nperseg, noverlap=nperseg // 2)
    _, _, Zxx_y = stft(ay_din, fs=fs_real, window="hamming", nperseg=nperseg, noverlap=nperseg // 2)
    _, _, Zxx_z = stft(az_din, fs=fs_real, window="hamming", nperseg=nperseg, noverlap=nperseg // 2)

    potencia_triaxial = np.abs(Zxx_x)**2 + np.abs(Zxx_y)**2 + np.abs(Zxx_z)**2
    potencia_db = 10 * np.log10(potencia_triaxial + 1e-10)
    t_stft_real = t_stft + t_sub[0]

    fig_spec = go.Figure(data=go.Heatmap(
        z=potencia_db,
        x=t_stft_real,
        y=f_stft,
        colorscale="Inferno",
        colorbar=dict(title="Potencia (dB)")
    ))
    fig_spec.add_hline(y=banda_sismica[0], line_dash="dash", line_color="cyan")
    fig_spec.add_hline(y=banda_sismica[1], line_dash="dash", line_color="cyan")
    fig_spec.update_layout(
        title="Espectrograma Tiempo-Frecuencia Triaxial (STFT)",
        xaxis_title="Tiempo (s)",
        yaxis_title="Frecuencia (Hz)",
        yaxis=dict(range=[0, min(25, fs_real / 2)]),
        height=500,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO
    )
    st.plotly_chart(fig_spec, use_container_width=True)

with tab4:
    st.markdown("### Diagnóstico para Calibración de Umbrales del ESP32")
    st.write(f"- **Archivo analizado:** `{archivo.name}`")
    st.write(f"- **Muestras totales:** `{len(df)}` | **Muestras perdidas:** `{perdidas}` ({integridad:.2f}% integridad)")
    st.write(f"- **Pico espectral detectado en {banda_sismica[0]}–{banda_sismica[1]} Hz:** `{amp_pico:.5f} m/s²` a `{freq_dominante:.2f} Hz`")
    st.dataframe(df_sub.head(200), use_container_width=True)