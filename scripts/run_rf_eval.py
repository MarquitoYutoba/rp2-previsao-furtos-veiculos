"""
Avaliação do Random Forest com atributos espaciais e temporais agregados.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rf.model import preparar_dataset_rf_enriquecido, treinar_e_avaliar_rf_enriquecido, pai_curve_rf


def main() -> int:
    cfg_rf = yaml.safe_load((ROOT / "configs" / "rf.yaml").read_text(encoding="utf-8"))
    out_dir = ROOT / cfg_rf["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_cells = pd.read_parquet(ROOT / "data/processed/grid_cells.parquet")
    # Corrigido para apontar para o parquet que possui a coluna cell_id atribuída
    train_events_path = ROOT / "data/processed/stkde_train_grid.parquet"

    dataset = preparar_dataset_rf_enriquecido(grid_cells, train_events_path)

    features = cfg_rf["features"]
    params = cfg_rf["model_params"]
    alphas = [0.01, 0.05, 0.10, 0.20]

    print("[INFO] Treinando Random Forest multivariado (Espacial + Temporal)...")
    model, preds_test, metrics, pai_results = treinar_e_avaliar_rf_enriquecido(
        dataset,
        features=features,
        params=params,
        alphas=alphas,
    )

    curve_rf = pai_curve_rf(preds_test)
    curve_rf.to_csv(out_dir / "pai_curve_rf.csv", index=False, encoding="utf-8-sig")

    output_report = {
        "metrics": metrics,
        "pai_scores": pai_results,
    }

    preds_test.to_parquet(out_dir / "rf_predictions_test.parquet", index=False)
    (out_dir / "rf_eval_report.json").write_text(
        json.dumps(output_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Random Forest treinado com sucesso!")
    print(f"     Métricas de Teste -> RMSE: {metrics['rmse']:.3f} | MAE: {metrics['mae']:.3f} | R2: {metrics['r2']:.3f}")
    print(f"     Desempenho PAI do RF:")
    for k, v in pai_results.items():
        print(f"       - {k.upper()}: {v:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())