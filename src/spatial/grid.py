"""
Passo 3 — malha regular UTM e atribuição de células (STKDE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GridSpec:
    cell_size_m: float
    x_origin: float
    y_origin: float
    nx: int
    ny: int

    @property
    def x_max(self) -> float:
        return self.x_origin + self.nx * self.cell_size_m

    @property
    def y_max(self) -> float:
        return self.y_origin + self.ny * self.cell_size_m


@dataclass
class RelatorioGrid:
    cell_size_m: float = 500.0
    nx: int = 0
    ny: int = 0
    n_celulas_com_evento: int = 0
    n_treino: int = 0
    n_teste: int = 0
    treino_dentro: int = 0
    treino_fora: int = 0
    teste_dentro: int = 0
    teste_fora: int = 0
    max_n_treino_por_celula: int = 0
    max_n_teste_por_celula: int = 0
    celulas_so_treino: int = 0
    celulas_so_teste: int = 0
    celulas_treino_e_teste: int = 0


def grid_spec_para_dict(spec: GridSpec) -> dict[str, Any]:
    return {
        "cell_size_m": spec.cell_size_m,
        "x_origin": spec.x_origin,
        "y_origin": spec.y_origin,
        "nx": spec.nx,
        "ny": spec.ny,
        "x_max": spec.x_max,
        "y_max": spec.y_max,
    }


def relatorio_grid_para_dict(rel: RelatorioGrid) -> dict[str, Any]:
    return rel.__dict__.copy()


def construir_grid_spec_treino(
    df_treino: pd.DataFrame,
    *,
    cell_size_m: float = 500.0,
    padding_cells: int = 0,
    col_x: str = "x_utm",
    col_y: str = "y_utm",
) -> GridSpec:
    if df_treino.empty:
        raise ValueError("Treino vazio: não dá para definir a malha.")
    for col in (col_x, col_y):
        if col not in df_treino.columns:
            raise ValueError(f"Coluna ausente: {col}")

    cs = float(cell_size_m)
    pad = max(0, int(padding_cells))

    x = df_treino[col_x].to_numpy(dtype=float)
    y = df_treino[col_y].to_numpy(dtype=float)

    x_origin = np.floor(x.min() / cs) * cs
    y_origin = np.floor(y.min() / cs) * cs

    nx = int(np.floor((x.max() - x_origin) / cs)) + 1 + pad
    ny = int(np.floor((y.max() - y_origin) / cs)) + 1 + pad

    return GridSpec(
        cell_size_m=cs,
        x_origin=float(x_origin),
        y_origin=float(y_origin),
        nx=max(nx, 1),
        ny=max(ny, 1),
    )


def atribuir_celulas(
    df: pd.DataFrame,
    spec: GridSpec,
    *,
    col_x: str = "x_utm",
    col_y: str = "y_utm",
) -> pd.DataFrame:
    out = df.copy()
    cs = spec.cell_size_m

    ix = np.floor((out[col_x].to_numpy(dtype=float) - spec.x_origin) / cs).astype(np.int64)
    iy = np.floor((out[col_y].to_numpy(dtype=float) - spec.y_origin) / cs).astype(np.int64)

    dentro = (ix >= 0) & (ix < spec.nx) & (iy >= 0) & (iy < spec.ny)

    out["ix"] = ix
    out["iy"] = iy
    out["dentro_malha"] = dentro
    out["cell_id"] = np.where(dentro, iy * spec.nx + ix, -1).astype(np.int64)
    return out


def contagem_por_celula(
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    spec: GridSpec,
) -> pd.DataFrame:
    """Uma linha por célula com evento; colunas n_treino, n_teste e centro UTM."""
    tr_in = treino.loc[treino["dentro_malha"]]
    te_in = teste.loc[teste["dentro_malha"]]

    c_tr = tr_in.groupby("cell_id", as_index=False).size().rename(columns={"size": "n_treino"})
    c_te = te_in.groupby("cell_id", as_index=False).size().rename(columns={"size": "n_teste"})

    cells = pd.merge(c_tr, c_te, on="cell_id", how="outer").fillna(0)
    cells["n_treino"] = cells["n_treino"].astype(int)
    cells["n_teste"] = cells["n_teste"].astype(int)

    cells["ix"] = (cells["cell_id"] % spec.nx).astype(int)
    cells["iy"] = (cells["cell_id"] // spec.nx).astype(int)
    cs = spec.cell_size_m
    cells["x_center"] = spec.x_origin + (cells["ix"] + 0.5) * cs
    cells["y_center"] = spec.y_origin + (cells["iy"] + 0.5) * cs
    return cells.sort_values("cell_id").reset_index(drop=True)


def registrar_relatorio(
    rel: RelatorioGrid,
    spec: GridSpec,
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    cells: pd.DataFrame,
) -> None:
    rel.cell_size_m = spec.cell_size_m
    rel.nx = spec.nx
    rel.ny = spec.ny
    rel.n_treino = int(len(treino))
    rel.n_teste = int(len(teste))
    rel.treino_dentro = int(treino["dentro_malha"].sum())
    rel.treino_fora = int((~treino["dentro_malha"]).sum())
    rel.teste_dentro = int(teste["dentro_malha"].sum())
    rel.teste_fora = int((~teste["dentro_malha"]).sum())
    rel.n_celulas_com_evento = int(len(cells))
    if len(cells):
        rel.max_n_treino_por_celula = int(cells["n_treino"].max())
        rel.max_n_teste_por_celula = int(cells["n_teste"].max())
        so_tr = (cells["n_treino"] > 0) & (cells["n_teste"] == 0)
        so_te = (cells["n_teste"] > 0) & (cells["n_treino"] == 0)
        ambos = (cells["n_treino"] > 0) & (cells["n_teste"] > 0)
        rel.celulas_so_treino = int(so_tr.sum())
        rel.celulas_so_teste = int(so_te.sum())
        rel.celulas_treino_e_teste = int(ambos.sum())