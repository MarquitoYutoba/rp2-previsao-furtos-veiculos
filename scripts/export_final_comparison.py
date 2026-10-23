"""
Script final: Gera gráfico comparativo de Curva PAI e Tabela LaTeX consolidada (STKDE vs. RF).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stkde.figures import plot_comparacao_pai_curve


def write_comparison_latex(stkde_best: dict, rf_report: dict, out_path: Path) -> None:
    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        r"\caption{Comparativo de Desempenho Preditivo — STKDE vs. Random Forest (Teste 2026).}",
        r"\label{tab:comparativo_modelos}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Modelo & PAI$_{1\%}$ & PAI$_{5\%}$ & PAI$_{10\%}$ & PAI$_{20\%}$ \\",
        r"\midrule",
    ]
    
    # Linha STKDE
    lines.append(
        f"STKDE (Otimizado) & {stkde_best.get('pai_1', 0):.2f} & "
        f"{stkde_best.get('pai_5', 0):.2f} & "
        f"{stkde_best.get('pai_10', 0):.2f} & "
        f"{stkde_best.get('pai_20', 0):.2f} \\\\"
    )
    
    # Linha Random Forest
    rf_pais = rf_report.get("pai_scores", {})
    lines.append(
        f"Random Forest + STKDE & {rf_pais.get('pai_1', 0):.2f} & "
        f"{rf_pais.get('pai_5', 0):.2f} & "
        f"{rf_pais.get('pai_10', 0):.2f} & "
        f"{rf_pais.get('pai_20', 0):.2f} \\\\"
    )
    
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    reports_dir = ROOT / "reports" / "stkde"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Carregar dados do STKDE e do RF
    stkde_dir = ROOT / "data" / "processed" / "stkde"
    rf_dir = ROOT / "data" / "processed" / "rf"

    if not (stkde_dir / "best_params.json").is_file() or not (rf_dir / "rf_eval_report.json").is_file():
        print("[ERRO] Rode os scripts de avaliação do STKDE e do Random Forest antes.")
        return 1

    stkde_best = json.loads((stkde_dir / "best_params.json").read_text(encoding="utf-8"))
    rf_report = json.loads((rf_dir / "rf_eval_report.json").read_text(encoding="utf-8"))
    
    # Carregar as curvas PAI completas de ambos os modelos
    curve_stkde = pd.read_csv(reports_dir / "pai_curve.csv")
    curve_rf = pd.read_csv(rf_dir / "pai_curve_rf.csv")

    # 1. Gerar Gráfico Comparativo com as duas curvas contínuas
    out_fig = reports_dir / "fig_comparacao_stkde_rf.png"
    plot_comparacao_pai_curve(curve_stkde, curve_rf, out_path=out_fig)

    # 2. Gerar Tabela LaTeX Comparativa
    out_tex = reports_dir / "tab_comparativo_modelos.tex"
    write_comparison_latex(stkde_best, rf_report, out_tex)

    print(f"[OK] Artefatos finais gerados com sucesso em: {reports_dir.relative_to(ROOT)}")
    print(f"     - Gráfico: {out_fig.relative_to(ROOT)}")
    print(f"     - Tabela LaTeX: {out_tex.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())