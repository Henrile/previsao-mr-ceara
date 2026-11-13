import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from soil_data import FEATURES, normalize_data, read_csv


def load_training_data(path, sheet):
    if path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(path, sheet_name=sheet, dtype=object)
    elif path.suffix.lower() == ".csv":
        with path.open("rb") as source:
            raw = read_csv(source)
    else:
        raise ValueError("Use um arquivo .xlsx ou .csv.")

    data = normalize_data(raw)
    missing = [name for name in [*FEATURES, "MR"] if name not in data]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(missing)}.")
    if data.empty or not data["MR"].gt(0).all():
        raise ValueError("A base deve conter registros com MR numérico e positivo em todas as linhas.")

    data["Class"] = data["Class"].fillna("Unknown").astype(str)
    return data


def build_pipeline():
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["Class"])],
        remainder="passthrough",
    )
    model = XGBRegressor(
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        n_estimators=200,
        subsample=1.0,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_grouped(data):
    if "Amostra" not in data or data["Amostra"].isna().any() or data["Amostra"].nunique() < 5:
        raise ValueError("A validação requer ao menos cinco solos e identificação preenchida em cada registro.")

    inputs, mr, soils = data[FEATURES], data["MR"], data["Amostra"]
    results = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(inputs, mr, soils), start=1):
        pipeline = build_pipeline()
        pipeline.fit(inputs.iloc[train_idx], np.log1p(mr.iloc[train_idx]))
        predicted = np.expm1(pipeline.predict(inputs.iloc[test_idx]))
        observed = mr.iloc[test_idx]
        results.append({
            "Dobra": fold,
            "Solos no teste": soils.iloc[test_idx].nunique(),
            "R²": r2_score(observed, predicted) if observed.nunique() > 1 else np.nan,
            "MAE (MPa)": mean_absolute_error(observed, predicted),
            "RMSE (MPa)": root_mean_squared_error(observed, predicted),
        })
    return pd.DataFrame(results).set_index("Dobra")


def main():
    parser = argparse.ArgumentParser(description="Treina ou avalia o preditor de MR em MPa.")
    parser.add_argument("base", type=Path, help="Planilha XLSX ou arquivo CSV de ensaios.")
    parser.add_argument("--sheet", default=0, help="Nome da aba do Excel. Por padrão, lê a primeira aba.")
    parser.add_argument("--output", type=Path, default=Path("xgb_mr_model.pkl"), help="Caminho do modelo salvo.")
    parser.add_argument("--validate", action="store_true", help="Executa validação agrupada por solo e mostra as métricas.")
    args = parser.parse_args()

    try:
        data = load_training_data(args.base, args.sheet)
        if args.validate:
            results = evaluate_grouped(data)
            print(results.to_string(float_format="{:.3f}".format))
            print("\nMédia e desvio-padrão entre dobras")
            print(results.drop(columns="Solos no teste").agg(["mean", "std"]).to_string(float_format="{:.3f}".format))
            return
    except (OSError, ValueError) as error:
        parser.error(str(error))

    pipeline = build_pipeline()
    pipeline.fit(data[FEATURES], np.log1p(data["MR"]))
    joblib.dump(pipeline, args.output)
    print(f"Modelo treinado com {len(data)} padrões de ensaio e salvo em {args.output}.")


if __name__ == "__main__":
    main()
