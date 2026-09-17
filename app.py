import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuração da página em modo "wide" para aproveitar a tela toda
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
st.divider()

# -----------------------------------------------------------------
# CRIAÇÃO DO LAYOUT LADO A LADO
# col_img fica com proporção 1 (esquerda) e col_form com 1.2 (direita, um pouco maior)
col_img, col_form = st.columns([1, 1.2], gap="large") 

# LADO ESQUERDO: Imagem
with col_img:
    st.image("Fluxo de Previsão de Resiliência dos Solos.png", caption="Fluxo de processamento e previsão do Módulo de Resiliência", use_container_width=True)

# LADO DIREITO: Campos de entrada (o seu "quadrado vermelho")
with col_form:
    st.header("Propriedades do Material e Ensaio")
    
    # Criando sub-colunas dentro do lado direito para organizar os campos
    c1, c2 = st.columns(2)
    
    with c1:
        ot = st.number_input("Umidade Ótima - OT (%)", value=8.0, step=0.1)
        den = st.number_input("Massa Específica Seca Máx - DEN (g/cm³)", value=2.16, step=0.01)
        cbr = st.number_input("Índice de Suporte Califórnia - CBR (%)", value=16.0, step=0.1)
        ll = st.number_input("Limite de Liquidez - LL (%)", value=0.0, step=1.0)
        ip = st.number_input("Índice de Plasticidade - IP (%)", value=0.0, step=1.0)
        
        st.markdown("---") # Divisória para as tensões
        sigma3 = st.number_input("Tensão Confinante - σ3 (MPa)", value=0.021, format="%.3f")

    with c2:
        p2_0 = st.number_input("Passante 2,0 mm (%)", value=49.0, step=1.0)
        p0_42 = st.number_input("Passante 0,42 mm (%)", value=26.0, step=1.0)
        p0_074 = st.number_input("Passante 0,074 mm (%)", value=8.0, step=1.0)
        aashto = st.selectbox("Classificação AASHTO", ["A-1-a", "A-1-b", "A-2-4", "A-2-5", "A-2-6", "A-2-7", "A-3", "A-4", "A-5", "A-6", "A-7-5", "A-7-6"])
        
        # O markdown("<br>") adiciona um espaço em branco para compensar a linha extra que a esquerda tem (IP), mantendo as linhas divisórias perfeitamente alinhadas!
        st.markdown("<br>", unsafe_allow_html=True) 
        st.markdown("---") 
        sigmad = st.number_input("Tensão Desviadora - σd (MPa)", value=0.041, format="%.3f")

    st.markdown("<br>", unsafe_allow_html=True) # Espaçamento
    
    # Botão e lógica de previsão (também dentro do lado direito)
    if st.button("Calcular Módulo de Resiliência", type="primary", use_container_width=True):
        if pipeline is None:
            st.error("Erro: O arquivo 'xgb_mr_model.pkl' não foi encontrado. Execute o train.py primeiro.")
        else:
            # 1. Montagem do dataframe de entrada
            input_data = pd.DataFrame({
                'OT': [ot], 'DEN': [den], 'CBR': [cbr], 'LL': [ll], 'IP': [ip],
                'P2_0': [p2_0], 'P0_42': [p0_42], 'P0_074': [p0_074],
                'Class': [aashto], 'sigma3': [sigma3], 'sigmad': [sigmad]
            })
            
            # 2. Predição
            pred_log = pipeline.predict(input_data)[0]
            
            # 3. Reconversão para a unidade original de MPa
            pred_mr = np.expm1(pred_log)
            
            st.success(f"### Módulo de Resiliência Previsto: {pred_mr:,.2f} MPa")
