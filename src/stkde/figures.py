"""Figuras e tabelas LaTeX — resultados STKDE (passo 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from matplotlib.colors import LogNorm


def _utm_to_wgs84(x: np.ndarray, y: np.ndarray, epsg: int = 31983) -> tuple[np.ndarray, np.ndarray]:
    transformer = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def plot_density_map(
    density: pd.DataFrame,
    test: pd.DataFrame | None,
    *,
    best: dict[str, Any],
    out_path: Path,
    map_test_max: int = 3000,
    cell_size_m: float = 500.0,
) -> None:
    lon, lat = _utm_to_wgs84(
        density["x_center"].to_numpy(dtype=float),
        density["y_center"].to_numpy(dtype=float),
    )
    lam = density["lambda"].to_numpy(dtype=float)
    lam_pos = np.maximum(lam, np.nanmin(lam[lam > 0]) if np.any(lam > 0) else 1e-12)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    sc = ax.scatter(
        lon,
        lat,
        c=lam_pos,
        s=18,
        cmap="inferno",
        norm=LogNorm(),
        alpha=0.85,
        linewidths=0,
    )
    if test is not None and len(test):
        te = test
        if map_test_max > 0 and len(te) > map_test_max:
            te = te.sample(map_test_max, random_state=42)
        lon_t, lat_t = _utm_to_wgs84(
            te["x_utm"].to_numpy(dtype=float),
            te["y_utm"].to_numpy(dtype=float),
        )
        ax.scatter(lon_t, lat_t, s=4, c="#4fc3f7", alpha=0.35, linewidths=0, label="Teste (2026)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"STKDE — λ em t_eval (h_s={best['h_s']:.0f} m, h_t={best['h_t']:.0f} h)"
    )
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(sc, ax=ax, shrink=0.72)
    cb.set_label("Densidade λ (escala log)")
    if test is not None and len(test):
        ax.legend(loc="upper right", framealpha=0.9)
    fig.text(
        0.01,
        0.01,
        f"Malha {cell_size_m:.0f} m UTM · {len(density)} células com λ",
        fontsize=8,
        color="gray",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pai_curve(
    curve: pd.DataFrame,
    *,
    best: dict[str, Any],
    out_path: Path,
    mark_alphas: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(curve["alpha"] * 100, curve["pai"], color="#1565c0", lw=2, label="STKDE")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="Referência (uniforme)")

    for alpha in mark_alphas:
        key = f"pai_{int(round(alpha * 100))}"
        if key in best:
            ax.scatter([alpha * 100], [best[key]], s=50, zorder=5, color="#c62828")
            ax.annotate(
                f"{alpha*100:.0f}%: {best[key]:.2f}",
                (alpha * 100, best[key]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

    ax.set_xlabel("Área priorizada (% das células)")
    ax.set_ylabel("PAI")
    ax.set_title("Curva PAI — teste 2026")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tuning_heatmap(
    results: pd.DataFrame,
    *,
    metric: str,
    out_path: Path,
) -> None:
    pivot = results.pivot(index="h_t", columns="h_s", values=metric)
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    ax.set_xlabel("h_s (m)")
    ax.set_ylabel("h_t (h)")
    ax.set_title(f"Tuning STKDE — {metric}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_tuning_latex(results: pd.DataFrame, out_path: Path) -> None:
    """Tabela booktabs para o artigo (grid completo)."""
    df = results.copy()
    df["loglik_test"] = df["loglik_test"].map(lambda x: f"{float(x):.1f}")
    for col in ("pai_1", "pai_5", "pai_10", "pai_20"):
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2f}")

    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\caption{Resultados do tuning STKDE (treino 2024--2025, teste 2026).}",
        r"\label{tab:stkde_tuning}",
        r"\small",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$h_s$ (m) & $h_t$ (h) & log-lik & PAI$_{1\%}$ & PAI$_{5\%}$ & PAI$_{10\%}$ & PAI$_{20\%}$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{int(row['h_s'])} & {int(row['h_t'])} & {row['loglik_test']} & "
            f"{row.get('pai_1', '-')} & {row.get('pai_5', '-')} & "
            f"{row.get('pai_10', '-')} & {row.get('pai_20', '-')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def plot_comparacao_pai_curve(
    curve_stkde: pd.DataFrame,
    curve_rf: pd.DataFrame,
    *,
    out_path: Path,
) -> None:
    """Plota as curvas PAI completas de ambos os modelos para comparação direta."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    
    # Curva STKDE
    ax.plot(curve_stkde["alpha"] * 100, curve_stkde["pai"], color="#1565c0", lw=2, label="STKDE (Ótimo)")
    
    # Curva Random Forest Puro
    ax.plot(curve_rf["alpha"] * 100, curve_rf["pai"], color="#d32f2f", lw=2, ls="--", label="Random Forest (Puro)")
    
    # Linha de referência uniforme
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="Referência (Uniforme)")

    ax.set_xlabel("Área priorizada (% das células)")
    ax.set_ylabel("Índice de Precisão Preditiva (PAI)")
    ax.set_title("Comparativo PAI — STKDE vs. Random Forest (Teste 2026)")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)