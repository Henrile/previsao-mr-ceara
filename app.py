import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import base64

# 1. Configuração da página
st.set_page_config(
    page_title="Predição de MR - Ceará",
    page_icon="🛣️",
    layout="wide"
)

# CSS Global
st.markdown("""
    <style>
    div.stButton > button {
        font-size: 1.13rem !important; 
        padding: 0.6rem 1.2rem !important;
        height: auto !important;
        transition: 0.3s;
    }
    img {
        max-width: 100%;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Carregar o modelo
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgb_mr_model.pkl')
    except FileNotFoundError:
        return None

pipeline = load_model()

# 3. Navegação
if "etapa" not in st.session_state:
    st.session_state.etapa = 1

def ir_para_parametros():
    st.session_state.etapa = 2

def voltar_para_fluxo():
    st.session_state.etapa = 1

# Função para converter a imagem em Base64 (necessário para o iframe HTML)
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 4. Interface Geral
st.title("Predição do Módulo de Resiliência (MR) - Solos do Ceará")
st.markdown("Estimativa rápida a partir das propriedades físicas e do estado de tensão.")
st.divider()

# =========================================================
# ETAPA 1: FLUXOGRAMA (COM ZOOM E ANIMAÇÃO)
# =========================================================
if st.session_state.etapa == 1:
    st.subheader("1. Fluxograma Metodológico")
    st.caption("🔍 Role o mouse ou faça o **movimento de pinça no celular** para dar zoom. Clique, arraste e solte para ver a animação elástica.")

    caminho_imagem = "Fluxo de Previsão de Resiliência dos Solos.png"
    
    if os.path.exists(caminho_imagem):
        img_base64 = get_base64_image(caminho_imagem)
        
        # HTML/CSS/JS para Pan & Zoom com efeito de mola (puxar e soltar)
        custom_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
        <style>
            #container {{
                width: 70%; /* Reduz a área da imagem em 30% em relação à tela (100-30) */
                height: 500px;
                margin: 0 auto;
                overflow: hidden;
                border: 2px dashed #ccc;
                border-radius: 10px;
                position: relative;
                background-color: #f9f9f9;
                cursor: grab;
            }}
            #container:active {{
                cursor: grabbing;
            }}
            #zoom-img {{
                width: 100%;
                height: 100%;
                object-fit: contain;
                transform-origin: center center;
                transition: transform 0.1s ease-out; /* Suavidade no pan/zoom */
            }}
            /* Classe adicionada quando solta a imagem (efeito mola/despuxar) */
            .spring-back {{
                transition: transform 0.6s cubic-bezier(0.25, 1.5, 0.5, 1) !important;
            }}
        </style>
        </head>
        <body>
            <div id="container">
                <img id="zoom-img" src="data:image/png;base64,{img_base64}" alt="Fluxograma" draggable="false" />
            </div>

            <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>
            <script>
                // Inicializa a biblioteca Panzoom para gerenciar pinça (mobile) e mouse (desktop)
                const elem = document.getElementById('zoom-img');
                const pz = panzoom(elem, {{
                    maxZoom: 5,
                    minZoom: 0.5,
                    bounds: true,
                    boundsPadding: 0.1
                }});

                // Adiciona o efeito elástico (puxar e soltar)
                let isDragging = false;
                
                elem.addEventListener('panzoomstart', () => {{
                    isDragging = true;
                    elem.classList.remove('spring-back');
                }});

                elem.addEventListener('panzoomend', () => {{
                    isDragging = false;
                    // Ao soltar, se estiver fora do centro (pan), nós damos um pequeno efeito de volta
                    elem.classList.add('spring-back');
                    
                    // Opcional: Se quiser que volte sempre pro centro ao soltar, descomente a linha abaixo:
                    // pz.moveTo(0, 0); 
                }});
            </script>
        </body>
        </html>
        """
        
        # Renderiza o componente HTML no Streamlit (70% de largura fica controlado no CSS acima, o iframe pega tudo)
        components.html(custom_html, height=520)
        
    else:
        st.warning(f"Imagem '{caminho_imagem}' não encontrada no diretório atual.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botão de avanço para a etapa 2
    col_vazia, col_avanco = st.columns([4, 1])
    with col_avanco:
        st.button("Inserir Parâmetros ➡️", on_click=ir_para_parametros, type="primary", use_container_width=True)

# =========================================================
# ETAPA 2: PARÂMETROS E PREVISÃO
# =========================================================
elif st.session_state.etapa == 2:
    # Botão de retorno
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
    
    # Botão de previsão centralizado
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        btn_calcular = st.button("Calcular Módulo de Resiliência 🚀", type="primary", use_container_width=True)

    if btn_calcular:
        if pipeline is None:
            st.error("Erro: O arquivo 'xgb_mr_model.pkl' não foi encontrado. Certifique-se de que o modelo está na raiz do projeto.")
        else:
            with st.spinner("Processando dados e aplicando modelo XGBoost..."):
                input_data = pd.DataFrame({
                    'OT': [ot], 'DEN': [den], 'CBR': [cbr], 'LL': [ll], 'IP': [ip],
                    'P2_0': [p2_0], 'P0_42': [p0_42], 'P0_074': [p0_074],
                    'Class': [aashto], 'sigma3': [sigma3], 'sigmad': [sigmad]
                })
                
                pred_log = pipeline.predict(input_data)[0]
                pred_mr = np.expm1(pred_log)
                
                st.success("✅ Previsão concluída com sucesso!")
                st.metric(label="Módulo de Resiliência (MR) Previsto", value=f"{pred_mr:,.2f} MPa")
                
                st.info("""
                *Nota técnica: Esta previsão atua como apoio exploratório e não substitui o ensaio laboratorial definitivo para projetos de pavimentação.*
                """)
