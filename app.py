import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import logging
from pathlib import Path

from dashboard import render_dashboard
from soil_data import AASHTO, FEATURES, LABELS, SIEVES


BASE_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)

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
    return joblib.load(BASE_DIR / "xgb_mr_model.pkl")


def render_prediction(values):
    if any(not np.isfinite(values[name]) or values[name] < 0 for name in FEATURES if name != "Class"):
        st.error("Informe valores numéricos finitos e não negativos.")
        return
    if values["DEN"] == 0:
        st.error("A massa específica seca máxima deve ser maior que zero.")
        return
    passing = [values[name] for name in SIEVES if name in values]
    if not 0 <= passing[2] <= passing[1] <= passing[0] <= 100:
        st.error("Confira a granulometria. Use percentuais de 0 a 100 com passante de 2,0 mm ≥ 0,42 mm ≥ 0,074 mm.")
        return

    try:
        with st.spinner("Calculando o MR com XGBoost…"):
            pipeline = load_model()
            categories = pipeline.named_steps["preprocessor"].named_transformers_["cat"].categories_[0]
            if values["Class"] not in categories:
                st.warning(f"A classe {values['Class']} não aparece no treinamento. Interprete a estimativa com cautela.")
            inputs = pd.DataFrame([values], columns=FEATURES)
            predicted = float(np.expm1(pipeline.predict(inputs)[0]))
    except FileNotFoundError:
        st.error("O arquivo xgb_mr_model.pkl deve estar na pasta do aplicativo.")
        return
    except Exception:
        LOGGER.exception("Falha ao calcular o módulo de resiliência")
        st.error("Não foi possível calcular o MR. Confira as versões em requirements.txt e o arquivo do modelo.")
        return

    if not np.isfinite(predicted) or predicted <= 0:
        st.error("O modelo produziu um MR fora do intervalo físico.")
        return
    st.metric(label="Módulo de Resiliência (MR) previsto", value=f"{predicted:,.2f} MPa")
    st.info("Use a previsão como estimativa preliminar. O dimensionamento definitivo requer ensaio de laboratório.")


# Lógica de navegação entre as telas
if "etapa" not in st.session_state:
    st.session_state.etapa = 1


def ir_para_parametros():
    st.session_state.etapa = 2


def voltar_para_fluxo():
    st.session_state.etapa = 1


# Função para converter imagem em Base64 para usar no HTML/JS
@st.cache_data(show_spinner=False)
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# ==========================================
# 4. Cabeçalho Geral
# ==========================================
st.title("Predição do Módulo de Resiliência (MR) - Solos do Ceará")
st.markdown("Estime o MR com XGBoost e explore os resultados de ensaios de laboratório.")
prediction_view, dashboard_view = st.tabs(["Previsão de MR", "Painel de ensaios"])

with dashboard_view:
    render_dashboard()

# =========================================================
# ETAPA 1: FLUXOGRAMA (COM ZOOM E ANIMAÇÃO)
# =========================================================
if st.session_state.etapa == 1:

    # CSS Específico da Etapa 1: Botão "Inserir Parâmetros" ~40% maior
    prediction_view.markdown("""
        <style>
        button[data-testid="baseButton-primary"] {
            font-size: 1.4rem !important;
            padding: 0.9rem 1.8rem !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    prediction_view.subheader("1. Fluxograma Metodológico")

    # Instruções dinâmicas para PC e Celular
    prediction_view.markdown(
        '<div class="texto-desktop">Use a roda do mouse para ampliar. Clique e arraste para mover a imagem.</div>',
        unsafe_allow_html=True,
    )
    prediction_view.markdown(
        '<div class="texto-mobile">Faça o gesto de pinça para ampliar. Toque e arraste para mover a imagem.</div>',
        unsafe_allow_html=True,
    )

    caminho_imagem = BASE_DIR / "Fluxo de Previsão de Resiliência dos Solos.png"

    if caminho_imagem.is_file():
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
            <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"
                integrity="sha384-eexGhKur8uM8+ZcFzz42RmxqFkn5dlKkACVXVptLAlKhEkBGbGpyMfVxcRNjK6ka"
                crossorigin="anonymous"></script>
            <script>
                if (typeof panzoom === 'function') {{
                    const elem = document.getElementById('zoom-img');
                    panzoom(elem, {{
                        maxZoom: 5,
                        minZoom: 0.5,
                        bounds: true,
                        boundsPadding: 0.1
                    }});
                    elem.addEventListener('panzoomstart', () => elem.classList.remove('spring-back'));
                    elem.addEventListener('panzoomend', () => elem.classList.add('spring-back'));
                }}
            </script>
        </body>
        </html>
        """

        # Componente com altura exata para não gerar barras de rolagem desnecessárias
        prediction_view.iframe(custom_html, height=400, tab_index=0, alt="Fluxograma metodológico interativo")

    else:
        prediction_view.warning(f"A imagem '{caminho_imagem.name}' não está na pasta do aplicativo.")

    prediction_view.markdown("<br>", unsafe_allow_html=True)

    # Botão Inserir Parâmetros Centralizado (Aumentado)
    _, col_avanco, _ = prediction_view.columns([1, 2, 1])
    with col_avanco:
        st.button("Inserir Parâmetros ➡️", on_click=ir_para_parametros, type="primary", width="stretch")


# =========================================================
# ETAPA 2: PARÂMETROS E PREVISÃO
# =========================================================
elif st.session_state.etapa == 2:

    # CSS Específico da Etapa 2: Botão Calcular ~26% maior | Botão Voltar normal
    prediction_view.markdown("""
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
    prediction_view.button("⬅️ Voltar ao Fluxograma", on_click=voltar_para_fluxo)
    prediction_view.markdown("<br>", unsafe_allow_html=True)

    prediction_view.subheader("2. Propriedades do Material e Ensaio")

    with prediction_view.form("previsao_mr", border=False):
        physical, grain = st.columns(2)
        values = {
            "OT": physical.number_input("Umidade Ótima - OT (%)", value=8.0, step=0.1),
            "DEN": physical.number_input("Massa Específica Seca Máx - DEN (g/cm³)", value=2.16, step=0.01),
            "CBR": physical.number_input("Índice de Suporte Califórnia - CBR (%)", value=16.0, step=0.1),
            "LL": physical.number_input("Limite de Liquidez - LL (%)", value=0.0, step=1.0),
            "IP": physical.number_input("Índice de Plasticidade - IP (%)", value=0.0, step=1.0),
            "P2_0": grain.number_input(LABELS["P2_0"], value=49.0, step=1.0),
            "P0_42": grain.number_input(LABELS["P0_42"], value=26.0, step=1.0),
            "P0_074": grain.number_input(LABELS["P0_074"], value=8.0, step=1.0),
            "Class": grain.selectbox("Classificação AASHTO", AASHTO),
        }

        st.divider()
        confining, deviator = st.columns(2)
        values["sigma3"] = confining.number_input("Tensão Confinante - σ₃ (MPa)", value=0.021, step=0.001, format="%.3f")
        values["sigmad"] = deviator.number_input("Tensão Desviadora - σd (MPa)", value=0.041, step=0.001, format="%.3f")

        _, submit_col, _ = st.columns([1, 2, 1])
        submitted = submit_col.form_submit_button("Calcular Módulo de Resiliência", type="primary", width="stretch")

    if submitted:
        with prediction_view:
            render_prediction(values)
