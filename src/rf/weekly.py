"""
Random Forest — painel célula × semana (opção B: alinhado ao relatório).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VEIC_COLS = ["DESCR_TIPO_VEICULO", "DESCR_TIPOLOCAL", "ANO_FABRICACAO", "MES_OCORRENCIA"]
MERGE_KEYS = ["NUM_BO", "ANO_ARQUIVO"]


def _ciclo_semana(serie: pd.Series) -> tuple[pd.Series, pd.Series]:
    woy = serie.dt.isocalendar().week.astype(int).clip(1, 52)
    ang = 2.0 * np.pi * woy.to_numpy(dtype=float) / 52.0
    return pd.Series(np.sin(ang), index=serie.index), pd.Series(np.cos(ang), index=serie.index)


def carregar_eventos_com_veiculos(
    train_path: Path,
    test_path: Path,
    csv_path: Path,
) -> pd.DataFrame:
    tr = pd.read_parquet(train_path)
    te = pd.read_parquet(test_path)
    ev = pd.concat([tr, te], ignore_index=True)
    ev = ev.loc[ev["dentro_malha"]].copy()
    ev["datetime_ocorrencia"] = pd.to_datetime(ev["datetime_ocorrencia"])

    cols_veic = [c for c in VEIC_COLS if c not in ev.columns]
    if cols_veic and csv_path.is_file():
        raw = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            usecols=lambda c: c in set(MERGE_KEYS + VEIC_COLS),
        )
        ev = ev.merge(raw, on=MERGE_KEYS, how="left", suffixes=("", "_csv"))

    return ev


def _inicio_semana_w_sun(serie: pd.Series) -> pd.Series:
    """Alinha com to_period('W-SUN').start_time (âncora na segunda-feira)."""
    return serie.dt.to_period("W-SUN").dt.start_time


def _grade_semanas(min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> pd.DatetimeIndex:
    pr = pd.period_range(min_ts.to_period("W-SUN"), max_ts.to_period("W-SUN"), freq="W-SUN")
    return pr.start_time


def _perfil_veiculo_historico(ev: pd.DataFrame, anos_treino: tuple[int, ...]) -> pd.DataFrame:
    """Perfil veicular por célula — apenas anos de treino (sem vazamento do alvo)."""
    sub = ev.loc[ev["ANO_ARQUIVO"].isin(anos_treino)].copy()
    sub["is_automovel"] = (
        sub["DESCR_TIPO_VEICULO"]
        .astype("string")
        .str.contains("Automovel", case=False, na=False)
        .astype(float)
    )
    sub["is_moto"] = (
        sub["DESCR_TIPO_VEICULO"]
        .astype("string")
        .str.contains("Motociclo", case=False, na=False)
        .astype(float)
    )
    sub["is_via_publica"] = (
        sub["DESCR_TIPOLOCAL"]
        .astype("string")
        .str.contains("Via P", case=False, na=False)
        .astype(float)
    )
    sub["ano_fab"] = pd.to_numeric(sub["ANO_FABRICACAO"], errors="coerce")
    sub["mes_oc"] = pd.to_numeric(sub["MES_OCORRENCIA"], errors="coerce")
    perfil = sub.groupby("cell_id", as_index=False).agg(
        pct_automovel=("is_automovel", "mean"),
        pct_moto=("is_moto", "mean"),
        pct_via_publica=("is_via_publica", "mean"),
        ano_fab_medio=("ano_fab", "mean"),
        mes_medio_hist=("mes_oc", "mean"),
    )
    for col in ("pct_automovel", "pct_moto", "pct_via_publica", "ano_fab_medio", "mes_medio_hist"):
        perfil[col] = perfil[col].fillna(0.0)
    return perfil


def _agregar_cell_semana(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["semana_inicio"] = _inicio_semana_w_sun(ev["datetime_ocorrencia"])
    g = ev.groupby(["cell_id", "semana_inicio"], as_index=False).agg(
        n_furtos=("NUM_BO", "count"),
        ano_arquivo=("ANO_ARQUIVO", "max"),
    )
    g["sin_semana"], g["cos_semana"] = _ciclo_semana(pd.to_datetime(g["semana_inicio"]))
    return g


def _painel_completo(
    cell_semana: pd.DataFrame,
    grid_cells: pd.DataFrame,
    perfil_veic: pd.DataFrame,
    anos_treino: tuple[int, ...],
    anos_teste: tuple[int, ...],
) -> pd.DataFrame:
    cells = grid_cells["cell_id"].unique()
    anos_painel = tuple(sorted(set(anos_treino) | set(anos_teste)))
    cs = cell_semana.loc[cell_semana["semana_inicio"].dt.year.isin(anos_painel)].copy()
    week_min = pd.Timestamp(f"{min(anos_painel)}-01-01")
    week_max = cs["semana_inicio"].max()
    if pd.isna(week_max):
        raise ValueError("Sem ocorrências no intervalo de anos do painel.")
    semanas = _grade_semanas(week_min, pd.Timestamp(week_max))
    idx = pd.MultiIndex.from_product([cells, semanas], names=["cell_id", "semana_inicio"])
    base = pd.DataFrame(index=idx).reset_index()
    df = base.merge(cs, on=["cell_id", "semana_inicio"], how="left")
    df["n_furtos"] = df["n_furtos"].fillna(0).astype(int)

    df = df.merge(perfil_veic, on="cell_id", how="left")
    for col in ("pct_automovel", "pct_moto", "pct_via_publica", "ano_fab_medio", "mes_medio_hist"):
        df[col] = df[col].fillna(0.0)
    df["mes_medio"] = df["mes_medio_hist"]
    df["sin_semana"] = df["sin_semana"].fillna(
        np.sin(2 * np.pi * df["semana_inicio"].dt.isocalendar().week.astype(float) / 52)
    )
    df["cos_semana"] = df["cos_semana"].fillna(
        np.cos(2 * np.pi * df["semana_inicio"].dt.isocalendar().week.astype(float) / 52)
    )

    df = df.merge(
        grid_cells[["cell_id", "x_center", "y_center", "ix", "iy", "n_teste"]],
        on="cell_id",
        how="left",
    )
    df = df.sort_values(["cell_id", "semana_inicio"]).reset_index(drop=True)

    df["trimestre"] = df["semana_inicio"].dt.quarter
    df["ano_calendario"] = df["semana_inicio"].dt.year

    for lag in (1, 2, 4):
        df[f"lag_{lag}"] = df.groupby("cell_id")["n_furtos"].shift(lag).fillna(0)

    treino_mask = df["ano_calendario"].isin(anos_treino)
    hist = df.loc[treino_mask].groupby(["cell_id", "trimestre"])["n_furtos"].mean().reset_index()
    hist = hist.rename(columns={"n_furtos": "media_trimestre_hist"})
    df = df.merge(hist, on=["cell_id", "trimestre"], how="left")
    df["media_trimestre_hist"] = df["media_trimestre_hist"].fillna(0.0)

    df["split"] = np.where(
        df["ano_calendario"].isin(anos_teste),
        "teste",
        np.where(df["ano_calendario"].isin(anos_treino), "treino", "fora"),
    )
    return df


def construir_painel_semanal(
    *,
    train_grid: Path,
    test_grid: Path,
    csv_amostra: Path,
    grid_cells: pd.DataFrame,
    anos_treino: tuple[int, ...] = (2024, 2025),
    anos_teste: tuple[int, ...] = (2026,),
) -> pd.DataFrame:
    ev = carregar_eventos_com_veiculos(train_grid, test_grid, csv_amostra)
    cell_semana = _agregar_cell_semana(ev)
    perfil = _perfil_veiculo_historico(ev, anos_treino)
    return _painel_completo(cell_semana, grid_cells, perfil, anos_treino, anos_teste)