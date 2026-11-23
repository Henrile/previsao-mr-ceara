import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

def get_theme_css(theme_name):
    dark = theme_name == "Escuro"
    palette = {
        "bg": "#0a1118" if dark else "#f5f7fb",
        "surface": "rgba(15, 23, 32, 0.82)" if dark else "rgba(255, 255, 255, 0.82)",
        "surface-strong": "rgba(16, 26, 36, 0.9)" if dark else "rgba(255,255,255,0.55)",
        "border": "rgba(148, 163, 184, 0.2)" if dark else "rgba(22, 131, 117, 0.15)",
        "primary": "#7ad9c5" if dark else "#168375",
        "primary-dark": "#b9f1e6" if dark else "#0f685d",
        "accent": "#f4b98c" if dark else "#a45d27",
        "text": "#edf4ff" if dark else "#1f2a37",
        "muted": "#c2d0df" if dark else "#5f6b7a",
        "shadow": "rgba(15, 23, 42, 0.35)" if dark else "rgba(15, 23, 42, 0.05)",
        "grad-a": "#0a1118" if dark else "#f5f7fb",
        "grad-b": "#101a24" if dark else "#eef4f2",
    }

    return f"""
    <style>
    :root {{
        --bg-soft: {palette['bg']};
        --card-bg: {palette['surface']};
        --surface-strong: {palette['surface-strong']};
        --border-soft: {palette['border']};
        --primary: {palette['primary']};
        --primary-dark: {palette['primary-dark']};
        --accent: {palette['accent']};
        --text: {palette['text']};
        --muted: {palette['muted']};
        --shadow: {palette['shadow']};
    }}

    div[data-testid="stAppViewContainer"] {{
        background: linear-gradient(180deg, {palette['grad-a']} 0%, {palette['grad-b']} 100%);
    }}

    div[data-testid="stMain"] {{
        padding-top: 1rem;
    }}

    div.stButton > button {{
        height: auto !important;
        transition: 0.3s ease-in-out;
        border-radius: 0.8rem !important;
        box-shadow: 0 8px 22px rgba(22, 131, 117, 0.12);
    }}

    div.stButton > button:hover {{
        transform: translateY(-1px);
        filter: brightness(1.03);
    }}

    .app-shell {{
        background: var(--surface-strong);
        border: 1px solid var(--border-soft);
        border-radius: 1.1rem;
        padding: 1.1rem 1.2rem 0.8rem;
        box-shadow: 0 10px 30px var(--shadow);
    }}

    .header-panel {{
        background: linear-gradient(135deg, rgba(22, 131, 117, 0.08), rgba(164, 93, 39, 0.06));
        border: 1px solid var(--border-soft);
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }}

    .header-panel h1 {{
        margin-bottom: 0.15rem;
        color: var(--text);
    }}

    .header-panel p {{
        margin: 0;
        color: var(--muted);
        font-size: 1rem;
    }}

    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.75rem;
        background: rgba(22, 131, 117, 0.08);
        border: 1px solid rgba(22, 131, 117, 0.15);
        border-radius: 999px;
        color: var(--primary-dark);
        font-weight: 600;
        font-size: 0.82rem;
    }}

    .info-card {{
        background: var(--card-bg);
        border: 1px solid var(--border-soft);
        border-radius: 0.9rem;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 18px rgba(20, 28, 34, 0.04);
    }}

    .info-card h4 {{
        margin: 0 0 0.3rem 0;
        color: var(--text);
    }}

    .info-card p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.4;
    }}

    .flow-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.8rem;
        list-style: none;
        margin: 1rem 0 0;
        padding: 0;
    }}

    .flow-card {{
        min-width: 0;
        min-height: 180px;
        padding: 1rem;
        background: var(--card-bg);
        border: 1px solid var(--border-soft);
        border-radius: 0.9rem;
        box-shadow: 0 8px 18px rgba(20, 28, 34, 0.04);
    }}

    .flow-card--result {{
        border-color: var(--primary);
        box-shadow: 0 8px 22px rgba(22, 131, 117, 0.12);
    }}

    .flow-card__heading {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }}

    .flow-card__icon {{
        font-size: 1.35rem;
        line-height: 1;
    }}

    .flow-card__step {{
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}

    .flow-card h3 {{
        margin: 0 0 0.45rem;
        color: var(--text);
        font-size: 1rem;
    }}

    .flow-card p {{
        margin: 0;
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.45;
    }}

    .flow-card code {{
        color: var(--primary-dark);
        font-size: 0.82rem;
        font-weight: 600;
    }}

    @media (max-width: 768px) {{
        .app-shell {{ padding: 0.85rem 0.8rem 0.5rem; }}
        .flow-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        .flow-card {{ min-height: 0; padding: 0.85rem; }}
    }}

    @media (max-width: 520px) {{
        .flow-grid {{ grid-template-columns: minmax(0, 1fr); }}
    }}
    </style>
    """


if "theme" not in st.session_state:
    st.session_state.theme = "Claro"

st.markdown(get_theme_css(st.session_state.get("theme", "Claro")), unsafe_allow_html=True)

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


# ==========================================
# 4. Cabeçalho Geral
# ==========================================

def render_app_header():
    current_theme = st.session_state.get("theme", "Claro")
    st.radio(
        "Tema",
        ["Claro", "Escuro"],
        index=0 if current_theme == "Claro" else 1,
        horizontal=True,
        key="theme",
    )

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='header-panel'>
            <div class='pill'>🚧 Estimativa geotécnica assistida por IA</div>
            <h1>Predição do Módulo de Resiliência (MR) - Solos do Ceará</h1>
            <p>Estime o MR com XGBoost e explore os resultados de ensaios de laboratório com apoio visual e análise comparativa.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro_cols = st.columns(3)
    with intro_cols[0]:
        st.markdown(
            """
            <div class='info-card'>
                <h4>📘 Entrada</h4>
                <p>Propriedades físicas, granulométricas e estados de tensão do material.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_cols[1]:
        st.markdown(
            """
            <div class='info-card'>
                <h4>🧠 Modelo</h4>
                <p>Regressor XGBoost treinado para estimar o MR em MPa.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with intro_cols[2]:
        st.markdown(
            """
            <div class='info-card'>
                <h4>📊 Validação</h4>
                <p>Comparação com ensaios, filtros por solo, classe e tensão.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


render_app_header()
prediction_view, dashboard_view = st.tabs(["Previsão de MR", "Painel de ensaios"])

with dashboard_view:
    render_dashboard()

# =========================================================
# ETAPA 1: FLUXO DE CÁLCULO
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

    prediction_view.subheader("1. Como a estimativa é calculada")
    prediction_view.markdown(
        """
        <ol class="flow-grid" aria-label="Etapas do cálculo do módulo de resiliência">
            <li class="flow-card">
                <div class="flow-card__heading">
                    <span class="flow-card__icon" aria-hidden="true">🧪</span>
                    <span class="flow-card__step">Etapa 1</span>
                </div>
                <h3>Dados de entrada</h3>
                <p>Propriedades do solo, classe AASHTO e tensões do ensaio.</p>
            </li>
            <li class="flow-card">
                <div class="flow-card__heading">
                    <span class="flow-card__icon" aria-hidden="true">✓</span>
                    <span class="flow-card__step">Etapa 2</span>
                </div>
                <h3>Validação</h3>
                <p>Confere valores, densidade positiva e ordem dos percentuais passantes.</p>
            </li>
            <li class="flow-card">
                <div class="flow-card__heading">
                    <span class="flow-card__icon" aria-hidden="true">⚙️</span>
                    <span class="flow-card__step">Etapa 3</span>
                </div>
                <h3>Preparação</h3>
                <p>Organiza as variáveis e codifica a classe AASHTO para o modelo.</p>
            </li>
            <li class="flow-card">
                <div class="flow-card__heading">
                    <span class="flow-card__icon" aria-hidden="true">🤖</span>
                    <span class="flow-card__step">Etapa 4</span>
                </div>
                <h3>Modelo XGBoost</h3>
                <p>Estima o valor transformado <code>ln(1 + MR)</code>.</p>
            </li>
            <li class="flow-card flow-card--result">
                <div class="flow-card__heading">
                    <span class="flow-card__icon" aria-hidden="true">📈</span>
                    <span class="flow-card__step">Etapa 5</span>
                </div>
                <h3>Resultado</h3>
                <p>Retorna à escala original: <code>MR = exp(y) − 1</code>, em MPa.</p>
            </li>
        </ol>
        """,
        unsafe_allow_html=True,
    )

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
