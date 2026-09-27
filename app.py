import streamlit as st
import os

# Configuração da página
st.set_page_config(
    page_title="Previsão MR Ceará",
    page_icon="📈",
    layout="centered"
)

# Inicializa o estado da etapa na sessão
if "etapa" not in st.session_state:
    st.session_state.etapa = 1

def ir_para_parametros():
    st.session_state.etapa = 2

def voltar_para_fluxograma():
    st.session_state.etapa = 1

# ==========================================
# ETAPA 1: FLUXOGRAMA DO PROCESSO
# ==========================================
if st.session_state.etapa == 1:
    st.title("Fluxograma da Metodologia")
    st.caption("Visão geral do pipeline de análise e modelo preditivo.")

    # Caminho da imagem do fluxograma
    caminho_imagem = "fluxograma.png"

    if os.path.exists(caminho_imagem):
        st.image(caminho_imagem, caption="Fluxograma do Modelo de Previsão", use_container_width=True)
    else:
        st.info("Coloque o arquivo 'fluxograma.png' no mesmo diretório deste script para exibi-lo.")

    st.markdown("---")

    col_vazia, col_botao = st.columns([3, 1])
    with col_botao:
        st.button("Avançar ➡️", on_click=ir_para_parametros, use_container_width=True)

# ==========================================
# ETAPA 2: PARÂMETROS E PREVISÃO
# ==========================================
elif st.session_state.etapa == 2:
    st.button("⬅️ Voltar ao Fluxograma", on_click=voltar_para_fluxograma)

    st.title("Inserção de Parâmetros")
    st.markdown("Preencha os valores abaixo para calcular a estimativa.")

    with st.form("form_parametros"):
        col1, col2 = st.columns(2)
        
        with col1:
            municipio = st.selectbox(
                "Região / Município:",
                ["Fortaleza", "Juazeiro do Norte", "Sobral", "Crato", "Outro"]
            )
            parametro_a = st.number_input("Parâmetro A (ex: Índice / Taxa):", min_value=0.0, value=10.0, step=0.1)

        with col2:
            ano = st.number_input("Ano de Referência:", min_value=2000, max_value=2030, value=2026, step=1)
            parametro_b = st.number_input("Parâmetro B (ex: Variável Explicativa):", min_value=0.0, value=5.0, step=0.5)

        botao_calcular = st.form_submit_button("Gerar Previsão 🚀", use_container_width=True)

    if botao_calcular:
        # Exemplo simples de execução/inferência
        resultado_estimado = (parametro_a * 1.5) + (parametro_b * 0.8)
        
        st.success("Cálculo realizado com sucesso!")
        st.metric(label="Resultado Previsto", value=f"{resultado_estimado:.2f}")
