import csv
import unicodedata
from io import StringIO

import numpy as np
import pandas as pd


FEATURES = ["OT", "DEN", "CBR", "LL", "IP", "P2_0", "P0_42", "P0_074", "Class", "sigma3", "sigmad"]
SIEVES = ["P25_4", "P9_5", "P4_76", "P2_0", "P0_42", "P0_074"]
AASHTO = ["A-1-a", "A-1-b", "A-2-4", "A-2-5", "A-2-6", "A-2-7", "A-3", "A-4", "A-5", "A-6", "A-7-5", "A-7-6"]

LABELS = {
    "Amostra": "Solo / amostra",
    "Class": "Classe AASHTO",
    "MR": "MR medido (MPa)",
    "sigma3": "σ₃ (MPa)",
    "sigmad": "σd (MPa)",
    "OT": "OT (%)",
    "DEN": "DEN (g/cm³)",
    "CBR": "CBR (%)",
    "EXP": "Expansão (%)",
    "LL": "LL (%)",
    "IP": "IP (%)",
    "P25_4": "Passante 25,4 mm (%)",
    "P9_5": "Passante 9,5 mm (%)",
    "P4_76": "Passante 4,76 mm (%)",
    "P2_0": "Passante 2,0 mm (%)",
    "P0_42": "Passante 0,42 mm (%)",
    "P0_074": "Passante 0,074 mm (%)",
}

ALIASES = {
    "Amostra": ("Nome", "ID da amostra", "Solo"),
    "Class": ("AASHTO", "Classificação AASHTO"),
    "MR": ("MR (MPa)",),
    "sigma3": ("σ3",),
    "sigmad": ("σd",),
    "OT": (),
    "DEN": (),
    "CBR": (),
    "EXP": ("EXP (%)",),
    "LL": (),
    "IP": (),
    "P25_4": ("25.4", "25,4 mm (%)"),
    "P9_5": ("9.5", "9,5 mm (%)"),
    "P4_76": ("4.76", "4,76 mm (%)"),
    "P2_0": ("2", "2.0", "2,0 mm (%)"),
    "P0_42": ("0.42", "0,42 mm (%)"),
    "P0_074": ("0.074", "0,074 mm (%)"),
}


def read_csv(source):
    """Lê o CSV com separador de campo padrão ou do Excel em português."""
    payload = source.read()
    try:
        content = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = payload.decode("cp1252")

    header = next((line for line in content.splitlines() if line.strip()), "")
    delimiter = ";" if len(next(csv.reader([header], delimiter=";"), [])) > 1 else ","
    return pd.read_csv(StringIO(content), sep=delimiter, dtype=str, keep_default_na=False)


def header_key(value):
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def normalize_data(raw):
    """Padroniza os cabeçalhos da dissertação e mantém os registros da fonte."""
    headers = {header_key(column): column for column in raw.columns}
    # A versão original tem 'Class ' textual e 'Class' com o código numérico.
    if "Class " in raw.columns:
        headers["class"] = "Class "

    normalized = pd.DataFrame(index=raw.index)
    for name, aliases in ALIASES.items():
        keys = map(header_key, (*aliases, LABELS[name], name))
        column = next((headers[key] for key in keys if key in headers), None)
        if column is None:
            continue
        values = raw[column].astype("string").str.strip()
        if name in {"Class", "Amostra"}:
            normalized[name] = values.replace("", pd.NA)
        else:
            numbers = pd.to_numeric(values.str.replace(",", ".", regex=False), errors="coerce")
            normalized[name] = numbers.astype(float).replace([np.inf, -np.inf], np.nan)
    return normalized


def normalize_aashto(classes):
    classes = classes.str.upper().str.replace(r"\s+", "", regex=True)
    return classes.replace({"A-1-A": "A-1-a", "A-1-B": "A-1-b", "UNKNOWN": pd.NA})
