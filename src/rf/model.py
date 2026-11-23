"""Treino, métricas e PAI do Random Forest (painel semanal)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def treinar_rf_semanal(
    painel: pd.DataFrame,
    features: list[str],
    params: dict,
    *,
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> tuple[RandomForestRegressor, pd.DataFrame, dict, dict, pd.DataFrame]:
    """
    Treina em semanas 2024–2025; avalia semanas 2026 no painel e agrega PAI por célula.
    """
    treino = painel.loc[painel["split"] == "treino"].copy()
    teste = painel.loc[painel["split"] == "teste"].copy()
    if treino.empty or teste.empty:
        raise ValueError("Painel sem linhas de treino ou teste.")

    X_tr = treino[features]
    y_tr = treino["n_furtos"]
    model = RandomForestRegressor(**params)
    model.fit(X_tr, y_tr)

    teste = teste.copy()
    teste["pred_rf"] = model.predict(teste[features])

    y_true = teste["n_furtos"]
    y_pred = teste["pred_rf"]
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_linhas_treino": int(len(treino)),
        "n_linhas_teste": int(len(teste)),
    }

    por_celula = (
        teste.groupby("cell_id", as_index=False)
        .agg(
            pred_total=("pred_rf", "sum"),
            n_obs_semanas=("pred_rf", "size"),
        )
    )
    n_teste = painel.groupby("cell_id", as_index=False)["n_teste"].max()
    ranking = por_celula.merge(n_teste[["cell_id", "n_teste"]], on="cell_id", how="left")
    ranking["n_teste"] = ranking["n_teste"].fillna(0).astype(int)

    total_crimes = int(ranking["n_teste"].sum())
    ranked = ranking.sort_values("pred_total", ascending=False).reset_index(drop=True)
    n_cells = len(ranked)

    pai_results: dict[str, float] = {}
    for alpha in alphas:
        k = max(1, int(np.ceil(alpha * n_cells)))
        top = set(ranked.head(k)["cell_id"].tolist())
        hits = int(ranking.loc[ranking["cell_id"].isin(top), "n_teste"].sum())
        pai = (hits / total_crimes) / alpha if total_crimes > 0 else float("nan")
        pai_results[f"pai_{int(round(alpha * 100))}"] = float(pai)

    return model, teste, metrics, pai_results, ranking


def pai_curve_rf_celulas(
    ranking: pd.DataFrame,
    *,
    col_pred: str = "pred_total",
    col_target: str = "n_teste",
    col_cell: str = "cell_id",
    n_points: int = 100,
) -> pd.DataFrame:
    """Curva PAI agregada por célula (mesmo universo que STKDE)."""
    ranked = ranking.sort_values(col_pred, ascending=False).reset_index(drop=True)
    n_cells = len(ranked)
    total = int(ranked[col_target].sum())
    if n_cells == 0 or total == 0:
        return pd.DataFrame(columns=["alpha", "pai", "crimes_pct", "n_cells_top"])

    rows: list[dict[str, float | int]] = []
    for k in range(1, n_cells + 1):
        alpha = k / n_cells
        hits = int(ranked.head(k)[col_target].sum())
        crimes_pct = hits / total
        pai = crimes_pct / alpha if alpha > 0 else float("nan")
        rows.append({"alpha": alpha, "pai": pai, "crimes_pct": crimes_pct, "n_cells_top": k})

    curve = pd.DataFrame(rows)
    if n_points > 0 and len(curve) > n_points:
        idx = np.linspace(0, len(curve) - 1, n_points, dtype=int)
        curve = curve.iloc[idx].reset_index(drop=True)
    return curve