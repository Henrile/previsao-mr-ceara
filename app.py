import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Configuração da página em modo "wide"
st.set_page_config(
    page_title="Predição de MR - Ceará",
    page_icon="🛣️",
    layout="wide"
)

# 2. Carregar o modelo treinado com cache do Streamlit
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgb_mr_model.pkl')
    except FileNotFoundError:
        return None

pipeline = load_model()

# 3. Controle de Navegação das Etapas
if "etapa" not in st.session_state:
    st.session_state.etapa = 1

def ir_para_parametros():
    st.session_state.etapa = 2

def voltar_para_fluxo():
    st.session_state.etapa = 1

# 4. Cabeçalho Geral
st.title("Predição do Módulo de Resiliência (MR) - Solos do Ceará")
st.markdown("Estimativa rápida a partir das propriedades físicas e do estado de tensão.")
st.divider()

# =========================================================
# ETAPA 1: FLUXOGRAMA
# =========================================================
if st.session_state.etapa == 1:
    st.subheader("1. Fluxograma Metodológico")
    st.caption("Conheça o processo de tratamento de dados e modelagem preditiva antes de inserir os parâmetros.")

    caminho_imagem = "Fluxo de Previsão de Resiliência dos Solos.png"
    if os.path.exists(caminho_imagem):
        # Exibe a imagem centralizada ou em destaque
        col_esq, col_centro, col_dir = st.columns([1, 8, 1])
        with col_centro:
            st.image(
                caminho_imagem, 
                caption="Fluxo de processamento e previsão do Módulo de Resiliência", 
                use_container_width=True
            )
    else:
        st.warning(f"Imagem '{caminho_imagem}' não encontrada no diretório atual.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Linha com o botão/seta para avançar
    col_vazia, col_avanco = st.columns([4, 1])
    with col_avanco:
        st.button("Inserir Parâmetros ➡️", on_click=ir_para_parametros, type="primary", use_container_width=True)

# =========================================================
# ETAPA 2: PARÂMETROS E PREVISÃO
# =========================================================
elif st.session_state.etapa == 2:
    # Botão de retorno ao fluxograma
    col_voltar, col_espaco = st.columns([1, 4])
    with col_voltar:
        st.button("⬅️ Voltar ao Fluxograma", on_click=voltar_para_fluxo, use_container_width=True)

    st.subheader("2. Propriedades do Material e Ensaio")

    # Primeira dupla de colunas (Propriedades físicas do solo)
    c1, c2 = st.columns(2)
    
    with c1:
        ot = st.number_input("Umidade Ótima - OT (%)", value=8.0, step=0.1)
        den = st.number_input("Massa Específica Seca Máx - DEN (g/cm³)", value=2.16, step=0.01)
        cbr = st.number_input("Índice de Suporte Califórnia - CBR (%)", value=16.0, step=0.1)
        ll = st.number_input("Limite de Liquidez - LL (%)", value=0.0, step=1.0)
        ip = st.number_input("Índice de Plasticidade - IP (%)", value=0.0, step=1.0)

    with c2:
        p2_0 = st.number_input("Passante 2,0 mm (%)", value=49.0, step=1.0)
        p0_42 = st.number_input("Passante 0,42 mm (%)", value=26.0, step=1.0)
        p0_074 = st.number_input("Passante 0,074 mm (%)", value=8.0, step=1.0)
        aashto = st.selectbox("Classificação AASHTO", [
            "A-1-a", "A-1-b", "A-2-4", "A-2-5", "A-2-6", 
            "A-2-7", "A-3", "A-4", "A-5", "A-6", "A-7-5", "A-7-6"
        ])
        
    st.markdown("---")

    # Segunda dupla de colunas (Tensões)
    c3, c4 = st.columns(2)
    
    with c3:
        sigma3 = st.number_input("Tensão Confinante - σ3 (MPa)", value=0.021, format="%.3f")
        
    with c4:
        sigmad = st.number_input("Tensão Desviadora - σd (MPa)", value=0.041, format="%.3f")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botão e lógica de previsão
    if st.button("Calcular Módulo de Resiliência 🚀", type="primary", use_container_width=True):
        if pipeline is None:
            st.error("Erro: O arquivo 'xgb_mr_model.pkl' não foi encontrado. Certifique-se de que o modelo está na raiz do projeto.")
        else:
            with st.spinner("Processando dados e aplicando modelo XGBoost..."):
                # Montagem do dataframe de entrada
                input_data = pd.DataFrame({
                    'OT': [ot], 'DEN': [den], 'CBR': [cbr], 'LL': [ll], 'IP': [ip],
                    'P2_0': [p2_0], 'P0_42': [p0_42], 'P0_074': [p0_074],
                    'Class': [aashto], 'sigma3': [sigma3], 'sigmad': [sigmad]
                })
                
                # Predição (retorno na escala logarítmica)
                pred_log = pipeline.predict(input_data)[0]
                
                # Reconversão para a unidade original de MPa
                pred_mr = np.expm1(pred_log)
                
                st.success("✅ Previsão concluída com sucesso!")
                st.metric(label="Módulo de Resiliência (MR) Previsto", value=f"{pred_mr:,.2f} MPa")
                
                st.info("""
                *Nota técnica: Esta previsão atua como apoio exploratório e não substitui o ensaio laboratorial definitivo para projetos de pavimentação.*
                """)
