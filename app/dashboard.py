# app/dashboard.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import joblib
import os

# -------------------- CONFIGURAÇÃO DA PÁGINA --------------------
st.set_page_config(
    page_title="Manutenção Preditiva - Motores",
    page_icon="🛠️",
    layout="wide"
)

# -------------------- FUNÇÃO DE EXTRAÇÃO DE CARACTERÍSTICAS --------------------


def extrair_caracteristicas(sinal, fs=12000):
    n = len(sinal)
    rms = np.sqrt(np.mean(sinal**2))
    kurtosis = stats.kurtosis(sinal)
    skewness = stats.skew(sinal)
    pico = np.max(np.abs(sinal))
    fator_crista = pico / (rms + 1e-10)
    magnitude = np.abs(np.fft.rfft(sinal))
    frequencias = np.fft.rfftfreq(n, d=1/fs)
    min_len = min(len(magnitude), len(frequencias))
    magnitude = magnitude[:min_len]
    frequencias = frequencias[:min_len]
    idx_pico = np.argmax(magnitude)
    freq_dominante = frequencias[idx_pico]
    energia_baixa = np.sum(
        magnitude[(frequencias >= 0) & (frequencias < 1000)])
    energia_media = np.sum(
        magnitude[(frequencias >= 1000) & (frequencias < 3000)])
    energia_alta = np.sum(
        magnitude[(frequencias >= 3000) & (frequencias < 6000)])
    return {
        'rms': rms, 'kurtosis': kurtosis, 'skewness': skewness,
        'pico': pico, 'fator_crista': fator_crista,
        'freq_dominante': freq_dominante,
        'energia_baixa': energia_baixa,
        'energia_media': energia_media,
        'energia_alta': energia_alta
    }


# -------------------- CARREGAR MODELOS (caminhos absolutos) --------------------
MODELS_DIR = Path(__file__).parent.parent / 'models'
DATA_DIR = Path(__file__).parent.parent / 'data' / '1797 RPM'

modelo = joblib.load(MODELS_DIR / 'xgboost_cwru.pkl')
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
encoder = joblib.load(MODELS_DIR / 'label_encoder.pkl')

# -------------------- TÍTULO E DESCRIÇÃO --------------------
st.title("🛠️ Manutenção Preditiva de Motores Elétricos")
st.markdown("""
Faça upload de um arquivo de vibração (`.npz`) de um motor elétrico para diagnosticar 
automaticamente se há falha, o tipo e a severidade.
""")

# -------------------- SIDEBAR: OPÇÕES DO USUÁRIO --------------------
st.sidebar.header("⚙️ Parâmetros de Análise")
arquivo_exemplo = st.sidebar.selectbox(
    "Ou use um arquivo de exemplo:",
    ["Nenhum", "Normal", "IR007", "B014", "OR021"]
)
uploaded_file = st.sidebar.file_uploader(
    "Upload de arquivo .npz", type=["npz"])

janela_inicio = st.sidebar.slider(
    "Início da janela (amostras)", 0, 120000, 0, step=1024)
tamanho_janela = st.sidebar.number_input(
    "Tamanho da janela", 512, 4096, 1024, step=512)

# -------------------- CARREGAR SINAL --------------------
sinal = None
nome_arquivo = ""

if uploaded_file is not None:
    dados = np.load(uploaded_file)
    if 'DE' in dados:
        sinal = dados['DE'].flatten()
        nome_arquivo = uploaded_file.name
    else:
        st.error("Arquivo .npz não contém a chave 'DE'. Verifique o formato.")
elif arquivo_exemplo != "Nenhum":
    mapeamento = {
        "Normal": "1797_Normal.npz",
        "IR007": "1797_IR_7_DE12.npz",
        "B014": "1797_B_14_DE12.npz",
        "OR021": "1797_OR@6_21_DE12.npz"
    }
    caminho = DATA_DIR / mapeamento[arquivo_exemplo]
    if caminho.exists():
        dados = np.load(caminho)
        sinal = dados['DE'].flatten()
        nome_arquivo = caminho.name
    else:
        st.error(f"Arquivo de exemplo não encontrado: {caminho}")

# -------------------- ANÁLISE E DIAGNÓSTICO --------------------
if sinal is not None:
    if janela_inicio + tamanho_janela > len(sinal):
        st.error(
            "Janela ultrapassa o tamanho do sinal. Ajuste o início ou o tamanho.")
    else:
        janela = sinal[janela_inicio:janela_inicio+tamanho_janela]
        features = extrair_caracteristicas(janela)
        X = np.array([list(features.values())]).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred_id = modelo.predict(X_scaled)[0]
        pred_classe = encoder.inverse_transform([pred_id])[0]
        proba = modelo.predict_proba(X_scaled)[0][pred_id]

        st.header("🔍 Diagnóstico")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Classe Predita", pred_classe)
        with col2:
            st.metric("Confiança", f"{proba:.2%}")
        with col3:
            if pred_classe == "Normal":
                st.success("Motor saudável ✅")
            else:
                st.warning("Falha detectada ⚠️")

        st.subheader("📊 Características Extraídas")
        import pandas as pd
        df_feat = pd.DataFrame([features]).T.rename(columns={0: "Valor"})
        st.dataframe(df_feat.style.format("{:.4f}"))

        st.subheader("📈 Análise do Sinal")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        tempo = np.arange(0, len(janela)) / 12000
        ax1.plot(tempo, janela, linewidth=0.8)
        ax1.set_title("Sinal de Vibração - Domínio do Tempo")
        ax1.set_xlabel("Tempo (s)")
        ax1.set_ylabel("Aceleração (g)")
        ax1.grid(True, alpha=0.3)

        magnitude = np.abs(np.fft.rfft(janela))
        freq = np.fft.rfftfreq(len(janela), d=1/12000)
        ax2.plot(freq, magnitude, linewidth=0.8)
        ax2.set_title("Espectro de Frequência")
        ax2.set_xlabel("Frequência (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("👆 Faça upload de um arquivo ou selecione um exemplo para começar.")
