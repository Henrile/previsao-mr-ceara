import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import base64

# ==========================================
# 1. Configuração da Página
# ==========================================
st.set_page_config(
    page_title="Predição de MR - Ceará",
    page_icon="🛣️",
    layout="wide"
)

# ==========================================
# 2. CSS Global (Textos Responsivos e Botões Base)
# ==========================================
st.markdown("""
    <style>
    /* Transição suave padrão para todos os botões */
    div.stButton > button {
        height: auto !important;
        transition: 0.3s;
    }
    
    /* Controle de visibilidade dos textos de instrução (PC vs Mobile) */
    .texto-desktop { display: block; color: #555; font-size: 0.9rem; margin-bottom: 10px; }
    .texto-mobile { display: none; color: #555; font-size: 0.9rem; margin-bottom: 10px; }
    
    /* Quando a tela for menor que 768px (Celulares) */
    @media (max-width: 768px) {
        .texto-desktop { display: none; }
        .texto-mobile { display: block; }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Funções de Suporte e Carregamento
# ==========================================
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgb_mr_model.pkl')
    except FileNotFoundError:
        return None

pipeline = load_model()

# Lógica de navegação entre as telas
if "etapa" not in st.session_state:
    st.session_state.etapa = 1

def ir_para_parametros():
    st.session_state.etapa = 2

def voltar_para_fluxo():
    st.session_state.etapa = 1

# Função para converter imagem em Base64 para usar no HTML/JS
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ==========================================
# 4. Cabeçalho Geral
# ==========================================
st.title("Predição do Módulo de Resiliência (MR) - Solos do Ceará")
st.markdown("Estimativa rápida a partir das propriedades físicas e do estado de tensão.")
st.divider()

# =========================================================
# ETAPA 1: FLUXOGRAMA (COM ZOOM E ANIMAÇÃO)
# =========================================================
if st.session_state.etapa == 1:
    
    # CSS Específico da Etapa 1: Botão "Inserir Parâmetros" ~40% maior
    st.markdown("""
        <style>
        button[data-testid="baseButton-primary"] {
            font-size: 1.4rem !important;  
            padding: 0.9rem 1.8rem !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("1. Fluxograma Metodológico")
    
    # Instruções dinâmicas para PC e Celular
    st.markdown('<div class="texto-desktop">🔍 <b>No Computador:</b> Role o scroll do mouse para dar zoom. Clique, arraste e solte para ver a animação elástica.</div>', unsafe_allow_html=True)
    st.markdown('<div class="texto-mobile">🔍 <b>No Celular:</b> Faça o movimento de pinça na tela para dar zoom. Toque, arraste e solte para ver a animação.</div>', unsafe_allow_html=True)

    caminho_imagem = "Fluxo de Previsão de Resiliência dos Solos.png"
    
    if os.path.exists(caminho_imagem):
        img_base64 = get_base64_image(caminho_imagem)
        
        # HTML/CSS ajustado para não ter sobras brancas ao redor da imagem
        custom_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
        <style>
            body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
            
            #container {{
                margin: 0 auto;
                overflow: hidden;
                border: 2px dashed #ccc;
                border-radius: 10px;
                position: relative;
                background-color: #f9f9f9;
                cursor: grab;
                box-sizing: border-box;
                width: 100%;
                height: 250px; /* Altura reduzida no Celular */
            }}
            
            @media (min-width: 768px) {{
                #container {{
                    width: 70%;
                    height: 380px; /* Altura ajustada no PC para abraçar bem a imagem */
                }}
            }}

            #container:active {{ cursor: grabbing; }}
            
            #zoom-img {{
                width: 100%;
                height: 100%;
                object-fit: contain; /* Ajusta a imagem dentro da div sem distorcer */
                transform-origin: center center;
                transition: transform 0.1s ease-out;
            }}
            
            .spring-back {{
                transition: transform 0.6s cubic-bezier(0.25, 1.5, 0.5, 1) !important;
            }}
        </style>
        </head>
        <body>
            <div id="container">
                <img id="zoom-img" src="data:image/png;base64,{img_base64}" alt="Fluxograma" draggable="false" />
            </div>

            <!-- Biblioteca JavaScript para lidar com o gesto de pinça e pan -->
            <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>
            <script>
                const elem = document.getElementById('zoom-img');
                const pz = panzoom(elem, {{
                    maxZoom: 5,
                    minZoom: 0.5,
                    bounds: true,
                    boundsPadding: 0.1
                }});

                elem.addEventListener('panzoomstart', () => {{
                    elem.classList.remove('spring-back');
                }});

                elem.addEventListener('panzoomend', () => {{
                    elem.classList.add('spring-back');
                }});
            </script>
        </body>
        </html>
        """
        
        # Componente com altura exata para não gerar barras de rolagem desnecessárias
        components.html(custom_html, height=400)
        
    else:
        st.warning(f"Imagem '{caminho_imagem}' não encontrada no diretório atual.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botão Inserir Parâmetros Centralizado (Aumentado)
    _, col_avanco, _ = st.columns([1, 2, 1])
    with col_avanco:
        st.button("Inserir Parâmetros ➡️", on_click=ir_para_parametros, type="primary", use_container_width=True)


# =========================================================
# ETAPA 2: PARÂMETROS E PREVISÃO
# =========================================================
elif st.session_state.etapa == 2:
    
    # CSS Específico da Etapa 2: Botão Calcular ~26% maior | Botão Voltar normal
    st.markdown("""
        <style>
        /* Botão Calcular (Primary) */
        button[data-testid="baseButton-primary"] {
            font-size: 1.26rem !important; 
            padding: 0.75rem 1.5rem !important;
            font-weight: bold;
        }
        
        /* Botão Voltar (Secondary/Padrão) */
        button[data-testid="baseButton-secondary"] {
            font-size: 1.0rem !important;
            padding: 0.5rem 1.0rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Botão de retorno
    st.button("⬅️ Voltar ao Fluxograma", on_click=voltar_para_fluxo)
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("2. Propriedades do Material e Ensaio")

    # Primeira dupla de colunas (Propriedades físicas)
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
    
    # Botão de previsão centralizado (Aumentado em 26%)
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        btn_calcular = st.button("Calcular Módulo de Resiliência 🚀", type="primary", use_container_width=True)

    # Lógica de predição
    if btn_calcular:
        if pipeline is None:
            st.error("Erro: O arquivo 'xgb_mr_model.pkl' não foi encontrado. Certifique-se de que ele está na mesma pasta do script.")
        else:
            with st.spinner("Processando dados e aplicando modelo XGBoost..."):
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
