"""
Passo 4: grid search STKDE, PAI e densidade na melhor banda.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stkde.kde import (  # noqa: E402
    _t_eval_from_config,
    loglik_test_events,
    stkde_density_at_points,
)
from src.stkde.pai import pai_por_alpha  # noqa: E402


def main() -> int:
    cfg_path = ROOT / "configs" / "stkde.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = pd.read_parquet(ROOT / cfg["train_parquet"])
    te = pd.read_parquet(ROOT / cfg["test_parquet"])
    cells = pd.read_parquet(ROOT / cfg["grid_cells_parquet"])

    tr = tr.loc[tr["dentro_malha"]].copy()
    te_in = te.loc[te["dentro_malha"]].copy()

    train_xyt = tr[["x_utm", "y_utm", "t_horas"]].to_numpy(dtype=float)
    test_xyt = te_in[["x_utm", "y_utm", "t_horas"]].to_numpy(dtype=float)
    t_eval = _t_eval_from_config(
        train_xyt[:, 2],
        test_xyt[:, 2],
        str(cfg.get("t_eval_mode", "median_test")),
        float(cfg.get("t_eval_fixed_hours", 0.0)),
    )

    eval_xy = cells[["x_center", "y_center"]].to_numpy(dtype=float)
    chunk_eval = int(cfg.get("chunk_size_eval", 256))
    chunk_ll = int(cfg.get("chunk_size_test_ll", 512))
    alphas = cfg.get("pai_alphas", [0.01, 0.05, 0.10, 0.20])

    rows = []
    for h_s in cfg["h_s_list"]:
        for h_t in cfg["h_t_list"]:
            lam = stkde_density_at_points(
                train_xyt, eval_xy, t_eval, float(h_s), float(h_t), chunk_size=chunk_eval
            )
            cells_tmp = cells.copy()
            cells_tmp["lambda"] = lam
            pai = pai_por_alpha(
                cells_tmp,
                te_in["cell_id"].to_numpy(),
                alphas=alphas,
            )
            ll = loglik_test_events(
                train_xyt, test_xyt, float(h_s), float(h_t), chunk_size=chunk_ll
            )
            row = {"h_s": h_s, "h_t": h_t, "t_eval": t_eval, "loglik_test": ll, **pai}
            rows.append(row)
            print(f"  h_s={h_s} h_t={h_t} loglik={ll:.1f} {pai}")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "tuning_results.csv", index=False, encoding="utf-8-sig")

    select_by = str(cfg.get("select_by", "loglik_test"))
    if select_by == "pai_at_10":
        best_row = results.loc[results["pai_10"].idxmax()]
    else:
        best_row = results.loc[results["loglik_test"].idxmax()]

    h_s_best = float(best_row["h_s"])
    h_t_best = float(best_row["h_t"])
    lam_best = stkde_density_at_points(
        train_xyt, eval_xy, t_eval, h_s_best, h_t_best, chunk_size=chunk_eval
    )
    density = cells.copy()
    density["lambda"] = lam_best
    density.to_parquet(out_dir / "density_best.parquet", index=False)

    best = {
        "h_s": h_s_best,
        "h_t": h_t_best,
        "t_eval": t_eval,
        "t_eval_mode": cfg.get("t_eval_mode"),
        "select_by": select_by,
        "loglik_test": float(best_row["loglik_test"]),
        **{k: float(best_row[k]) for k in best_row.index if str(k).startswith("pai_")},
    }
    (out_dir / "best_params.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "n_treino": int(len(tr)),
        "n_teste_dentro": int(len(te_in)),
        "n_celulas": int(len(cells)),
        "t_eval": t_eval,
        "best": best,
        "n_combinacoes": int(len(results)),
    }
    (out_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Melhor: h_s={h_s_best}, h_t={h_t_best} ({select_by})")
    print(f"[OK] Gravado: {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())