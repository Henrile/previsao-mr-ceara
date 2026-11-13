from hashlib import sha256
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl.utils.exceptions import InvalidFileException

from soil_data import AASHTO, LABELS, SIEVES, normalize_aashto, normalize_data, read_csv


TEAL = "#168375"
OCHRE = "#A45D27"
UNCLASSIFIED = "Sem classificação"


@st.cache_data(scope="session", max_entries=4, show_spinner=False)
def workbook_sheets(contents):
    with pd.ExcelFile(BytesIO(contents)) as workbook:
        return workbook.sheet_names


@st.cache_data(scope="session", max_entries=4, show_spinner=False)
def read_ensaios(contents, sheet):
    source = BytesIO(contents)
    raw = read_csv(source) if sheet is None else pd.read_excel(source, sheet_name=sheet, dtype=object)
    data = normalize_data(raw)
    if "Class" in data:
        data["Class"] = normalize_aashto(data["Class"])
    return data


def format_number(value, decimals=0):
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def filter_ensaios(data, source_key):
    mask = pd.Series(True, index=data.index)
    filters = [name for name in ("Class", "Amostra", "sigma3", "sigmad") if name in data]
    if not filters:
        return data

    for name, column in zip(filters, st.columns(len(filters))):
        values = data[name]
        key = f"{name}_{source_key}"
        if name == "Class":
            classes = values.fillna(UNCLASSIFIED)
            chosen = column.multiselect("Classes AASHTO", sorted(classes.unique()), key=key, placeholder="Todas as classes")
            if chosen:
                mask &= classes.isin(chosen)
        else:
            stress = name in {"sigma3", "sigmad"}
            selected = column.selectbox(
                LABELS[name], sorted(values.dropna().unique()), index=None, key=key,
                placeholder="Todas as tensões" if stress else "Todos os solos",
                format_func=(lambda x: f"{x:.6g}".replace(".", ",")) if stress else str,
            )
            if selected is not None:
                mask &= values.eq(selected).fillna(False)

    return data.loc[mask]


def render_histogram(mr, source_key):
    st.markdown("**Distribuição do MR**")
    if mr.empty:
        st.caption("Nenhum valor de MR positivo no recorte.")
        return

    lower, upper = mr.min(), mr.max()
    if lower == upper:
        lower, upper = lower * 0.95, upper * 1.05
    scale = st.segmented_control(
        "Escala do MR", ["Linear", "Logarítmica"], key=f"mr_scale_{source_key}",
        default="Logarítmica" if upper / lower > 100 else "Linear", selection_mode="single",
    )
    logarithmic = scale == "Logarítmica"
    bins = min(30, max(5, int(np.sqrt(len(mr)))))
    edges = np.geomspace(lower, upper, bins + 1) if logarithmic else np.linspace(lower, upper, bins + 1)
    counts, edges = np.histogram(mr, bins=edges)
    distribution = pd.DataFrame({"inicio": edges[:-1], "fim": edges[1:], "registros": counts})
    chart = alt.Chart(distribution).mark_bar(color=TEAL, binSpacing=1).encode(
        x=alt.X("inicio:Q", bin="binned", title=LABELS["MR"],
                scale=alt.Scale(type="log" if logarithmic else "linear", zero=False), axis=alt.Axis(tickCount=6)),
        x2="fim:Q",
        y=alt.Y("registros:Q", title="Padrões de ensaio", axis=alt.Axis(tickMinStep=1)),
        tooltip=[
            alt.Tooltip("inicio:Q", title="MR inicial (MPa)", format=".2f"),
            alt.Tooltip("fim:Q", title="MR final (MPa)", format=".2f"),
            alt.Tooltip("registros:Q", title="Padrões de ensaio"),
        ],
    ).properties(height=265)
    st.altair_chart(chart, alt="Histograma dos valores positivos de MR medido")
    st.caption(f"MR de {format_number(mr.min(), 2)} a {format_number(mr.max(), 2)} MPa no recorte.")
    return logarithmic


def render_classes(data):
    st.markdown("**Ensaios por classe AASHTO**")
    if "Class" not in data:
        st.caption("Inclua uma coluna AASHTO para ver esta distribuição.")
        return

    counts = data["Class"].fillna(UNCLASSIFIED).value_counts().rename_axis("classe").reset_index(name="registros")
    chart = alt.Chart(counts).mark_bar(color=OCHRE, cornerRadiusEnd=3).encode(
        x=alt.X("registros:Q", title="Padrões de ensaio", axis=alt.Axis(tickCount=6, tickMinStep=1)),
        y=alt.Y("classe:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("classe:N", title=LABELS["Class"]), alt.Tooltip("registros:Q", title="Padrões de ensaio")],
    ).properties(height=300)
    st.altair_chart(chart, alt="Número de padrões de ensaio por classe AASHTO")


def render_scatter(data, fields, source_key, logarithmic):
    options = [name for name in fields if name != "MR"]
    if not options:
        st.info("Inclua uma coluna numérica além do MR para comparar as variáveis.")
        return

    preferred = "sigmad" if "sigmad" in options else options[0]
    variable = st.selectbox("Variável no eixo horizontal", options, index=options.index(preferred),
                            format_func=LABELS.get, key=f"scatter_{source_key}")
    columns = [variable, "MR"] + [name for name in ("Amostra", "Class") if name in data]
    points = data[columns].dropna(subset=[variable, "MR"])
    if points.empty:
        st.info("Não há pares válidos no recorte atual.")
        return
    if len(points) > 5000:
        st.caption(f"O gráfico exibe uma amostra de 5.000 dos {format_number(len(points))} pares válidos.")
        points = points.sample(5000, random_state=42)

    tooltip = [alt.Tooltip(f"{name}:Q", title=LABELS[name], format=".4~g") for name in (variable, "MR")]
    tooltip.extend(alt.Tooltip(f"{name}:N", title=LABELS[name]) for name in columns[2:])
    chart = alt.Chart(points).mark_circle(color=TEAL, opacity=0.58, size=48).encode(
        x=alt.X(f"{variable}:Q", title=LABELS[variable], scale=alt.Scale(zero=False), axis=alt.Axis(tickCount=6)),
        y=alt.Y("MR:Q", title=LABELS["MR"], scale=alt.Scale(type="log" if logarithmic else "linear", zero=False),
                axis=alt.Axis(tickCount=6)),
        tooltip=tooltip,
    ).properties(height=320).interactive()
    st.altair_chart(chart, alt=f"Dispersão do MR medido em relação a {LABELS[variable]}")


def render_correlations(data, fields):
    fields = [name for name in fields if data[name].nunique() > 1]
    if len(fields) < 2:
        st.info("A matriz requer pelo menos duas variáveis numéricas com variação no recorte.")
        return

    corr = data[fields].corr(method="pearson", min_periods=3)
    indicators = corr.rename(index=LABELS, columns=LABELS).rename_axis(index="linha", columns="coluna")
    cells = indicators.stack(future_stack=True).rename("r").reset_index()
    order = [LABELS[name] for name in fields]
    chart = alt.Chart(cells).mark_rect().encode(
        x=alt.X("coluna:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45, labelLimit=155)),
        y=alt.Y("linha:N", sort=order, title=None),
        color=alt.Color("r:Q", title="Pearson (r)", scale=alt.Scale(domain=[-1, 1], scheme="brownbluegreen")),
        tooltip=[alt.Tooltip("linha:N", title="Variável 1"), alt.Tooltip("coluna:N", title="Variável 2"),
                 alt.Tooltip("r:Q", title="Pearson (r)", format=".3f")],
    ).properties(height=max(180, len(fields) * 24))
    st.altair_chart(chart, alt="Matriz de correlação de Pearson das variáveis numéricas")
    st.caption("Cada célula usa pelo menos três pares válidos. Um mesmo solo pode aparecer em vários estados de tensão.")


def render_quality(data):
    missing = data.isna().sum()
    completeness = pd.DataFrame({
        "Variável": [LABELS[name] for name in data],
        "Ausentes ou ilegíveis": missing.to_numpy(),
        "% do recorte": (100 * missing / len(data)).round(1).to_numpy(),
    })
    flags = [f"**{format_number(data['MR'].le(0).sum())}** registros com MR menor ou igual a zero"]
    sieves = [name for name in SIEVES if name in data]
    if sieves:
        passing = data[sieves]
        outside = ((passing < 0) | (passing > 100)).any(axis=1)
        inverted = passing.ffill(axis=1).diff(axis=1).gt(0).any(axis=1)
        count = format_number((outside | inverted).sum())
        flags.append(f"**{count}** registros com passantes fora de 0–100% ou em ordem incoerente")
    if "Class" in data:
        unexpected = data["Class"].notna() & ~data["Class"].isin(AASHTO)
        flags.append(f"**{format_number(unexpected.sum())}** registros com classe AASHTO fora da lista do app")
    st.markdown("\n".join(f"- {flag}" for flag in flags))
    st.dataframe(completeness, hide_index=True)


@st.fragment
def render_dashboard():
    st.header("Painel de ensaios")
    st.caption("MR medido em laboratório, com recortes por solo, classe AASHTO e estado de tensão.")
    upload = st.file_uploader(
        "Base de ensaios", type=["xlsx", "csv"], key="mr_data_file",
        help="Use a planilha original ou a consolidada da pesquisa. As tensões e o MR devem estar em MPa.",
    )
    if upload is None:
        st.info("Envie a planilha de ensaios (.xlsx) ou um CSV para abrir o painel.")
        st.caption("Inclua a coluna MR ou MR (MPa). Os filtros aparecem quando o arquivo contém Nome, AASHTO, σ₃ ou σd.")
        return

    contents = upload.getvalue()
    source_key = sha256(contents).hexdigest()[:16]
    sheet = None
    try:
        if Path(upload.name).suffix.lower() == ".xlsx":
            sheets = workbook_sheets(contents)
            if not sheets:
                st.error("A planilha não contém abas de dados.")
                return
            sheet = st.selectbox("Aba da planilha", sheets, key=f"sheet_{source_key}") if len(sheets) > 1 else sheets[0]
            source_key = f"{source_key}_{sheet}"
        data = read_ensaios(contents, sheet)
    except (BadZipFile, InvalidFileException, KeyError, OSError, ParseError, ValueError):
        st.error("Não foi possível ler o arquivo. Confira se ele abre no Excel ou exporte a aba de ensaios como CSV.")
        return

    if "MR" not in data or data.empty:
        st.error("Selecione uma aba ou um arquivo com registros e uma coluna MR ou MR (MPa).")
        return

    selected = filter_ensaios(data, source_key)
    if selected.empty:
        st.warning("Nenhum ensaio corresponde aos filtros selecionados.")
        return

    measured = selected["MR"].where(selected["MR"].gt(0))
    valid = measured.dropna()
    sample_count = format_number(selected["Amostra"].nunique()) if "Amostra" in selected else "—"
    median = f"{format_number(valid.median(), 1)} MPa" if not valid.empty else "—"
    cards = st.columns(4)
    cards[0].metric("Padrões no recorte", format_number(len(selected)), border=True)
    cards[1].metric("Solos identificados", sample_count, border=True)
    cards[2].metric("MR mediano", median, border=True)
    cards[3].metric("MR positivo", format_number(len(valid)), border=True, help="Valores de MR numéricos e positivos.")

    distribution, classes = st.columns([3, 2])
    with distribution.container(border=True):
        logarithmic = render_histogram(valid, source_key)
    with classes.container(border=True):
        render_classes(selected)

    chart_data = selected.assign(MR=measured)
    numeric = chart_data.select_dtypes(include="number").columns.tolist()
    scatter, correlation, quality, records = st.tabs(["Dispersão", "Correlações", "Qualidade", "Registros"])
    with scatter:
        render_scatter(chart_data, numeric, source_key, logarithmic)
    with correlation:
        render_correlations(chart_data, numeric)
    with quality:
        render_quality(selected)
    with records:
        st.download_button(
            "Baixar recorte em CSV", selected.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name="ensaios_mr_filtrados.csv", mime="text/csv", on_click="ignore",
        )
        st.dataframe(selected.rename(columns=LABELS), hide_index=True)
