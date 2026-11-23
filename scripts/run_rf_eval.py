"""
Avaliação do Random Forest — painel célula × semana (lags T-1, T-2, T-4, perfil veicular).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rf.model import pai_curve_rf_celulas, treinar_rf_semanal
from src.rf.weekly import construir_painel_semanal


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "rf.yaml").read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    grid_cells = pd.read_parquet(ROOT / cfg["grid_cells"])
    painel = construir_painel_semanal(
        train_grid=ROOT / cfg["train_grid"],
        test_grid=ROOT / cfg["test_grid"],
        csv_amostra=ROOT / cfg["amostra_csv"],
        grid_cells=grid_cells,
        anos_treino=tuple(cfg["anos_treino"]),
        anos_teste=tuple(cfg["anos_teste"]),
    )
    painel.to_parquet(out_dir / "rf_panel_weekly.parquet", index=False)

    alphas = tuple(cfg.get("pai_alphas", [0.01, 0.05, 0.10, 0.20]))
    model, teste_linhas, metrics, pai_results, ranking = treinar_rf_semanal(
        painel,
        features=cfg["features"],
        params=cfg["model_params"],
        alphas=alphas,
    )

    curve = pai_curve_rf_celulas(ranking)
    curve.to_csv(out_dir / "pai_curve_rf.csv", index=False, encoding="utf-8-sig")
    ranking.to_parquet(out_dir / "rf_ranking_cells.parquet", index=False)

    report = {
        "modelo": "RandomForestRegressor",
        "painel": "celula_x_semana",
        "features": cfg["features"],
        "metrics_semanais_teste_2026": metrics,
        "pai_scores_por_celula": pai_results,
        "nota_pai": (
            "PAI agregado por célula: soma das predições semanais 2026 ranqueada "
            "vs n_teste (malha STKDE)."
        ),
    }
    (out_dir / "rf_eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Painel: {len(painel):,} linhas (treino+teste)")
    print(f"[OK] Teste semanal 2026 -> RMSE: {metrics['rmse']:.3f} | MAE: {metrics['mae']:.3f} | R2: {metrics['r2']:.3f}")
    print("[OK] PAI (por célula, teste 2026):")
    for k, v in pai_results.items():
        print(f"     {k}: {v:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())