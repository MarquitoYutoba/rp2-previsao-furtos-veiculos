"""
Passo 1 — carregar amostra CSV e limpar georreferência + tempo para STKDE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyproj

COLUNAS_STKDE = [
    "ANO_ARQUIVO",
    "ANO_BO",
    "NUM_BO",
    "DATA_OCORRENCIA_BO",
    "HORA_OCORRENCIA",
    "LATITUDE",
    "LONGITUDE",
]

COLUNAS_SAIDA = COLUNAS_STKDE + [
    "datetime_ocorrencia",
    "hora_informada",
    "t_horas",
    "x_utm",
    "y_utm",
]


@dataclass
class RelatorioLimpeza:
    total_entrada: int = 0
    removidos_coord_zero: int = 0
    removidos_fora_envelope: int = 0
    removidos_data_invalida: int = 0
    removidos_alinhamento_ano: int = 0
    total_saida: int = 0
    hora_informada: int = 0
    hora_ausente: int = 0
    por_ano_arquivo: dict[int, int] = field(default_factory=dict)

    def registrar_contagens(self, df: pd.DataFrame) -> None:
        self.total_saida = int(len(df))
        self.hora_informada = int(df["hora_informada"].sum())
        self.hora_ausente = int((~df["hora_informada"]).sum())
        if "ANO_ARQUIVO" in df.columns:
            self.por_ano_arquivo = (
                df["ANO_ARQUIVO"].value_counts().sort_index().astype(int).to_dict()
            )


def relatorio_para_dict(rel: RelatorioLimpeza) -> dict[str, Any]:
    return {
        "total_entrada": rel.total_entrada,
        "removidos_coord_zero": rel.removidos_coord_zero,
        "removidos_fora_envelope": rel.removidos_fora_envelope,
        "removidos_data_invalida": rel.removidos_data_invalida,
        "removidos_alinhamento_ano": rel.removidos_alinhamento_ano,
        "total_saida": rel.total_saida,
        "hora_informada": rel.hora_informada,
        "hora_ausente": rel.hora_ausente,
        "por_ano_arquivo": rel.por_ano_arquivo,
    }


def _parse_hora(serie: pd.Series) -> pd.Series:
    """Converte HORA_OCORRENCIA para timedelta; aceita HH:MM:SS e variantes."""
    s = serie.astype("string").str.strip()
    s = s.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return pd.to_timedelta(s, errors="coerce")


def _adicionar_utm(df: pd.DataFrame, epsg: int) -> pd.DataFrame:
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    x, y = transformer.transform(
        df["LONGITUDE"].to_numpy(dtype=float),
        df["LATITUDE"].to_numpy(dtype=float),
    )
    out = df.copy()
    out["x_utm"] = x
    out["y_utm"] = y
    return out


def carregar_e_limpar(
    caminho_csv: Path,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    alinhamento_ano_arquivo: bool = False,
    utm_epsg: int = 31983,
) -> tuple[pd.DataFrame, RelatorioLimpeza]:
    """
    Carrega CSV da amostra e retorna DataFrame pronto para STKDE + relatório.
    """
    rel = RelatorioLimpeza()
    df = pd.read_csv(caminho_csv, encoding="utf-8-sig")
    rel.total_entrada = int(len(df))

    faltando = [c for c in COLUNAS_STKDE if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {faltando}")

    df = df[COLUNAS_STKDE].copy()

    # Coordenadas (0, 0)
    zero = (df["LATITUDE"] == 0) & (df["LONGITUDE"] == 0)
    rel.removidos_coord_zero = int(zero.sum())
    df = df.loc[~zero]

    # Envelope municipal
    lat, lon = df["LATITUDE"], df["LONGITUDE"]
    dentro = lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
    rel.removidos_fora_envelope = int((~dentro).sum())
    df = df.loc[dentro]

    # Data e datetime
    data = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")
    invalida = data.isna()
    rel.removidos_data_invalida = int(invalida.sum())
    df = df.loc[~invalida].copy()
    data = data.loc[~invalida]

    hora_delta = _parse_hora(df["HORA_OCORRENCIA"])
    hora_ok = hora_delta.notna()
    datetime_oc = data + hora_delta.fillna(pd.Timedelta(0))
    df["datetime_ocorrencia"] = datetime_oc
    df["hora_informada"] = hora_ok

    if alinhamento_ano_arquivo:
        ano_data = df["datetime_ocorrencia"].dt.year
        alinhado = ano_data == df["ANO_ARQUIVO"]
        rel.removidos_alinhamento_ano = int((~alinhado).sum())
        df = df.loc[alinhado]

    # Tempo numérico (horas desde a primeira ocorrência — útil para kernel temporal)
    t0 = df["datetime_ocorrencia"].min()
    df["t_horas"] = (df["datetime_ocorrencia"] - t0).dt.total_seconds() / 3600.0

    df = _adicionar_utm(df, utm_epsg)
    df = df[COLUNAS_SAIDA].reset_index(drop=True)
    rel.registrar_contagens(df)
    return df, rel
