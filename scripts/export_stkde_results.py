"""
Passo 5: mapas, curva PAI, heatmaps e tabela LaTeX a partir do passo 4.
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

from src.stkde.figures import (  # noqa: E402
    plot_density_map,
    plot_pai_curve,
    plot_tuning_heatmap,
    write_tuning_latex,
)
from src.stkde.pai import pai_curve  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "stkde.yaml").read_text(encoding="utf-8"))
    stkde_dir = ROOT / cfg["output_dir"]
    reports_dir = ROOT / cfg.get("reports_dir", "reports/stkde")
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_path = stkde_dir / "best_params.json"
    if not best_path.is_file():
        print("[ERRO] Rode scripts/run_stkde_eval.py antes.")
        return 1

    best = json.loads(best_path.read_text(encoding="utf-8"))
    tuning = pd.read_csv(stkde_dir / "tuning_results.csv")
    density = pd.read_parquet(stkde_dir / "density_best.parquet")
    test = pd.read_parquet(ROOT / cfg["test_parquet"])
    test_in = test.loc[test["dentro_malha"]].copy()

    n_curve = int(cfg.get("pai_curve_points", 100))
    curve = pai_curve(
        density,
        test_in["cell_id"].to_numpy(),
        n_points=n_curve,
    )
    curve.to_csv(reports_dir / "pai_curve.csv", index=False, encoding="utf-8-sig")

    map_test_max = int(cfg.get("map_test_max", 3000))
    cell_size = float(cfg.get("cell_size_m", 500))

    plot_density_map(
        density,
        test_in,
        best=best,
        out_path=reports_dir / "fig_stkde_densidade.png",
        map_test_max=map_test_max,
        cell_size_m=cell_size,
    )
    plot_pai_curve(
        curve,
        best=best,
        out_path=reports_dir / "fig_stkde_pai_curva.png",
        mark_alphas=tuple(cfg.get("pai_alphas", [0.01, 0.05, 0.10, 0.20])),
    )
    plot_tuning_heatmap(
        tuning,
        metric="pai_10",
        out_path=reports_dir / "fig_stkde_heatmap_pai10.png",
    )
    plot_tuning_heatmap(
        tuning,
        metric="loglik_test",
        out_path=reports_dir / "fig_stkde_heatmap_loglik.png",
    )
    write_tuning_latex(tuning, reports_dir / "tab_stkde_tuning.tex")

    summary = {
        "best_params": best,
        "figuras": {
            "densidade": str((reports_dir / "fig_stkde_densidade.png").relative_to(ROOT)),
            "pai_curva": str((reports_dir / "fig_stkde_pai_curva.png").relative_to(ROOT)),
            "heatmap_pai10": str((reports_dir / "fig_stkde_heatmap_pai10.png").relative_to(ROOT)),
            "heatmap_loglik": str(
                (reports_dir / "fig_stkde_heatmap_loglik.png").relative_to(ROOT)
            ),
        },
        "tabela_latex": str((reports_dir / "tab_stkde_tuning.tex").relative_to(ROOT)),
        "pai_curve_csv": str((reports_dir / "pai_curve.csv").relative_to(ROOT)),
    }
    (reports_dir / "export_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Figuras e tabelas em: {reports_dir.relative_to(ROOT)}")
    for name, rel in summary["figuras"].items():
        print(f"     - {name}: {rel}")
    print(f"     - latex: {summary['tabela_latex']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
