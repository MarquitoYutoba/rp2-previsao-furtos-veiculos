"""PAI sobre células ranqueadas por densidade prevista."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pai_por_alpha(
    cells: pd.DataFrame,
    test_cell_ids: np.ndarray,
    *,
    col_lambda: str = "lambda",
    col_cell: str = "cell_id",
    alphas: list[float] | tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> dict[str, float]:
    """
    cells: uma linha por célula, com col_lambda já calculada em t_eval.
    test_cell_ids: cell_id de cada evento de teste (dentro_malha).
    """
    if len(cells) == 0 or len(test_cell_ids) == 0:
        return {f"pai_{int(a*100)}": float("nan") for a in alphas}

    ranked = cells.sort_values(col_lambda, ascending=False).reset_index(drop=True)
    n_cells = len(ranked)
    n_crimes = len(test_cell_ids)
    out: dict[str, float] = {}

    for alpha in alphas:
        k = max(1, int(np.ceil(alpha * n_cells)))
        top = set(ranked.head(k)[col_cell].tolist())
        hits = sum(1 for c in test_cell_ids if c in top)
        pai = (hits / n_crimes) / alpha
        key = f"pai_{int(round(alpha * 100))}"
        out[key] = float(pai)

    return out


def pai_curve(
    cells: pd.DataFrame,
    test_cell_ids: np.ndarray,
    *,
    col_lambda: str = "lambda",
    col_cell: str = "cell_id",
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Curva PAI(α) para α uniforme em (0, 1].
    α = k / n_celulas (k células de maior λ).
    """
    if len(cells) == 0 or len(test_cell_ids) == 0:
        return pd.DataFrame(columns=["alpha", "pai", "crimes_pct", "n_cells_top"])

    ranked = cells.sort_values(col_lambda, ascending=False).reset_index(drop=True)
    n_cells = len(ranked)
    n_crimes = len(test_cell_ids)
    top_cells_ordered = ranked[col_cell].tolist()

    rows: list[dict[str, float | int]] = []
    for k in range(1, n_cells + 1):
        alpha = k / n_cells
        top = set(top_cells_ordered[:k])
        hits = sum(1 for c in test_cell_ids if c in top)
        crimes_pct = hits / n_crimes
        pai = crimes_pct / alpha if alpha > 0 else float("nan")
        rows.append(
            {
                "alpha": alpha,
                "pai": pai,
                "crimes_pct": crimes_pct,
                "n_cells_top": k,
            }
        )

    curve = pd.DataFrame(rows)
    if n_points > 0 and len(curve) > n_points:
        idx = np.linspace(0, len(curve) - 1, n_points, dtype=int)
        curve = curve.iloc[idx].reset_index(drop=True)
    return curve