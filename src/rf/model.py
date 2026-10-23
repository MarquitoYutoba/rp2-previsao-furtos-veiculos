from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preparar_dataset_rf_enriquecido(
    grid_cells: pd.DataFrame,
    train_events_path: Path,
) -> pd.DataFrame:
    """
    Enriquece a malha espacial com atributos temporais agregados extraídos 
    dos eventos de treino (ex: mês médio, hora média, proporção de ocorrências à noite).
    """
    df_events = pd.read_parquet(train_events_path)
    
    # Extração de atributos temporais auxiliares a partir do datetime_ocorrencia
    df_events["datetime_ocorrencia"] = pd.to_datetime(df_events["datetime_ocorrencia"])
    df_events["mes"] = df_events["datetime_ocorrencia"].dt.month
    df_events["hora"] = df_events["datetime_ocorrencia"].dt.hour
    df_events["eh_noite"] = df_events["hora"].apply(lambda h: 1 if 18 <= h or h < 6 else 0)

    # Agregação por célula da malha para criar o perfil temporal de cada local
    perfil_celula = df_events.groupby("cell_id").agg(
        mes_medio=("mes", "mean"),
        hora_media=("hora", "mean"),
        prop_noite=("eh_noite", "mean"),
    ).reset_index()

    # União com a tabela da malha (grid_cells)
    dataset = pd.merge(grid_cells, perfil_celula, on="cell_id", how="left")
    
    # Preencher valores omissos para células sem eventos no treino
    dataset["mes_medio"] = dataset["mes_medio"].fillna(df_events["mes"].mean())
    dataset["hora_media"] = dataset["hora_media"].fillna(df_events["hora"].mean())
    dataset["prop_noite"] = dataset["prop_noite"].fillna(0.0)

    return dataset


def treinar_e_avaliar_rf_enriquecido(
    dataset: pd.DataFrame,
    features: list[str],
    params: dict,
    alphas: list[float] | tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> tuple[RandomForestRegressor, pd.DataFrame, dict, dict]:
    """
    Treina o Random Forest com atributos espaciais e temporais agregados.
    """
    X_train = dataset[features]
    y_train = dataset["n_treino"]
    y_test = dataset["n_teste"]

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    preds_test = model.predict(dataset[features])
    
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds_test))),
        "mae": float(mean_absolute_error(y_test, preds_test)),
        "r2": float(r2_score(y_test, preds_test)),
    }

    results_test = dataset[["cell_id", "x_center", "y_center", "n_teste"]].copy()
    results_test["pred_rf"] = preds_test

    total_crimes_teste = int(y_test.sum())
    ranked = results_test.sort_values("pred_rf", ascending=False).reset_index(drop=True)
    n_cells = len(ranked)
    
    pai_results = {}
    for alpha in alphas:
        k = max(1, int(np.ceil(alpha * n_cells)))
        top_cells = set(ranked.head(k)["cell_id"].tolist())
        crimes_hits = int(dataset.loc[dataset["cell_id"].isin(top_cells), "n_teste"].sum())
        pai = (crimes_hits / total_crimes_teste) / alpha if total_crimes_teste > 0 else 0.0
        pai_results[f"pai_{int(round(alpha * 100))}"] = float(pai)

    return model, results_test, metrics, pai_results


def pai_curve_rf(
    results_test: pd.DataFrame,
    *,
    col_pred: str = "pred_rf",
    col_target: str = "n_teste",
    col_cell: str = "cell_id",
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Calcula a curva PAI completa para o Random Forest.
    """
    ranked = results_test.sort_values(col_pred, ascending=False).reset_index(drop=True)
    n_cells = len(ranked)
    total_crimes = int(ranked[col_target].sum())
    
    if n_cells == 0 or total_crimes == 0:
        return pd.DataFrame(columns=["alpha", "pai", "crimes_pct", "n_cells_top"])

    top_cells_ordered = ranked[col_cell].tolist()

    rows = []
    for k in range(1, n_cells + 1):
        alpha = k / n_cells
        crimes_hits = int(ranked.head(k)[col_target].sum())
        crimes_pct = crimes_hits / total_crimes
        pai = (crimes_pct / alpha) if alpha > 0 else 0.0
        rows.append({
            "alpha": alpha,
            "pai": pai,
            "crimes_pct": crimes_pct,
            "n_cells_top": k,
        })

    curve = pd.DataFrame(rows)
    if n_points > 0 and len(curve) > n_points:
        idx = np.linspace(0, len(curve) - 1, n_points, dtype=int)
        curve = curve.iloc[idx].reset_index(drop=True)
    return curve