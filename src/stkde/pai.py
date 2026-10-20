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