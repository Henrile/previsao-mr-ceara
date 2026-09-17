import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predição de MR - Ceará", layout="wide")

# Carregar o modelo treinado com sistema de cache do Streamlit
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgb_mr_model.pkl')
    except FileNotFoundError:
        return None

pipeline = load_model()

st.title("Predição do Módulo de Resiliência (MR) - Solos do Ceará")
st.markdown("Estimativa rápida a partir das propriedades físicas e do estado de tensão.")

# Inserção da imagem do fluxograma
st.image("Fluxo de Previsão de Resiliência dos Solos.png", caption="Fluxo de processamento e previsão do Módulo de Resiliência", width=800)

st.divider()

st.header("Propriedades do Material e Ensaio")
col1, col2, col3 = st.columns(3)

with col1:
    ot = st.number_input("Umidade Ótima - OT (%)", value=8.0, step=0.1)
    den = st.number_input("Massa Específica Seca Máx - DEN (g/cm³)", value=2.16, step=0.01)
    cbr = st.number_input("Índice de Suporte Califórnia - CBR (%)", value=16.0, step=0.1)
    ll = st.number_input("Limite de Liquidez - LL (%)", value=0.0, step=1.0)

with col2:
    ip = st.number_input("Índice de Plasticidade - IP (%)", value=0.0, step=1.0)
    p2_0 = st.number_input("Passante 2,0 mm (%)", value=49.0, step=1.0)
    p0_42 = st.number_input("Passante 0,42 mm (%)", value=26.0, step=1.0)
    p0_074 = st.number_input("Passante 0,074 mm (%)", value=8.0, step=1.0)

with col3:
    aashto = st.selectbox("Classificação AASHTO", ["A-1-a", "A-1-b", "A-2-4", "A-2-5", "A-2-6", "A-2-7", "A-3", "A-4", "A-5", "A-6", "A-7-5", "A-7-6"])
    sigma3 = st.number_input("Tensão Confinante - σ3 (MPa)", value=0.021, format="%.3f")
    sigmad = st.number_input("Tensão Desviadora - σd (MPa)", value=0.041, format="%.3f")

if st.button("Calcular Módulo de Resiliência", type="primary"):
    if pipeline is None:
        st.error("Erro: O arquivo 'xgb_mr_model.pkl' não foi encontrado. Execute o train.py primeiro.")
    else:
        # 1. Montagem do dataframe de entrada
        input_data = pd.DataFrame({
            'OT': [ot], 'DEN': [den], 'CBR': [cbr], 'LL': [ll], 'IP': [ip],
            'P2_0': [p2_0], 'P0_42': [p0_42], 'P0_074': [p0_074],
            'Class': [aashto], 'sigma3': [sigma3], 'sigmad': [sigmad]
        })
        
        # 2. Predição (o valor retornado está em escala logarítmica)
        pred_log = pipeline.predict(input_data)[0]
        
        # 3. Reconversão para a unidade original de MPa
        pred_mr = np.expm1(pred_log)
        
        st.success(f"### Módulo de Resiliência (MR): {pred_mr:,.2f} MPa")
