"""
Passo 2 — split temporal treino/teste por ANO_ARQUIVO para STKDE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RelatorioSplit:
    total_entrada: int = 0
    n_treino: int = 0
    n_teste: int = 0
    descartados_ano: int = 0
    anos_treino: list[int] = field(default_factory=list)
    anos_teste: list[int] = field(default_factory=list)
    por_ano_treino: dict[int, int] = field(default_factory=dict)
    por_ano_teste: dict[int, int] = field(default_factory=dict)
    t0_treino: str | None = None
    t_horas_recalculado: bool = False

    def registrar(self, treino: pd.DataFrame, teste: pd.DataFrame) -> None:
        self.n_treino = int(len(treino))
        self.n_teste = int(len(teste))
        if "ANO_ARQUIVO" in treino.columns and len(treino):
            self.por_ano_treino = (
                treino["ANO_ARQUIVO"].value_counts().sort_index().astype(int).to_dict()
            )
        if "ANO_ARQUIVO" in teste.columns and len(teste):
            self.por_ano_teste = (
                teste["ANO_ARQUIVO"].value_counts().sort_index().astype(int).to_dict()
            )


def relatorio_split_para_dict(rel: RelatorioSplit) -> dict[str, Any]:
    return {
        "total_entrada": rel.total_entrada,
        "n_treino": rel.n_treino,
        "n_teste": rel.n_teste,
        "descartados_ano": rel.descartados_ano,
        "anos_treino": rel.anos_treino,
        "anos_teste": rel.anos_teste,
        "por_ano_treino": rel.por_ano_treino,
        "por_ano_teste": rel.por_ano_teste,
        "t0_treino": rel.t0_treino,
        "t_horas_recalculado": rel.t_horas_recalculado,
    }


def _recalcular_t_horas(
    treino: pd.DataFrame, teste: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if treino.empty:
        raise ValueError("Treino vazio: não é possível definir t0 para t_horas.")
    t0 = treino["datetime_ocorrencia"].min()
    treino = treino.copy()
    teste = teste.copy()
    for df in (treino, teste):
        df["t_horas"] = (df["datetime_ocorrencia"] - t0).dt.total_seconds() / 3600.0
    return treino, teste, t0


def split_treino_teste(
    df: pd.DataFrame,
    *,
    coluna_ano: str = "ANO_ARQUIVO",
    anos_treino: list[int] | tuple[int, ...] = (2024, 2025),
    anos_teste: list[int] | tuple[int, ...] = (2026,),
    recalcular_t_horas: bool = True,
    descartar_fora_dos_anos: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, RelatorioSplit]:
    """
    Divide eventos limpos em treino e teste por ano do arquivo SSP.

    Se descartar_fora_dos_anos=True, linhas cujo ano não está em treino∪teste
    são removidas (com contagem no relatório).
    """
    rel = RelatorioSplit(
        anos_treino=[int(a) for a in anos_treino],
        anos_teste=[int(a) for a in anos_teste],
    )
    rel.total_entrada = int(len(df))

    if coluna_ano not in df.columns:
        raise ValueError(f"Coluna ausente: {coluna_ano}")
    if "datetime_ocorrencia" not in df.columns:
        raise ValueError("Esperado datetime_ocorrencia (rode o passo 1 antes).")

    anos_treino_set = set(rel.anos_treino)
    anos_teste_set = set(rel.anos_teste)
    if anos_treino_set & anos_teste_set:
        raise ValueError("anos_treino e anos_teste não podem se sobrepor.")

    ano = df[coluna_ano].astype(int)
    mask_treino = ano.isin(rel.anos_treino)
    mask_teste = ano.isin(rel.anos_teste)

    if descartar_fora_dos_anos:
        mask_uso = mask_treino | mask_teste
        rel.descartados_ano = int((~mask_uso).sum())
        base = df.loc[mask_uso].copy()
        ano = base[coluna_ano].astype(int)
        mask_treino = ano.isin(rel.anos_treino)
        mask_teste = ano.isin(rel.anos_teste)
    else:
        base = df.copy()

    treino = base.loc[mask_treino].reset_index(drop=True)
    teste = base.loc[mask_teste].reset_index(drop=True)

    if recalcular_t_horas:
        treino, teste, t0 = _recalcular_t_horas(treino, teste)
        rel.t0_treino = t0.isoformat()
        rel.t_horas_recalculado = True

    rel.registrar(treino, teste)
    return treino, teste, rel


def carregar_eventos_limpos(caminho: Path) -> pd.DataFrame:
    """Lê Parquet se existir; senão CSV."""
    if caminho.suffix.lower() == ".parquet":
        return pd.read_parquet(caminho)
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    if "datetime_ocorrencia" in df.columns:
        df["datetime_ocorrencia"] = pd.to_datetime(df["datetime_ocorrencia"])
    if "hora_informada" in df.columns:
        df["hora_informada"] = df["hora_informada"].astype(bool)
    return df