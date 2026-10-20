"""
Passo 3: malha 500 m UTM, atribui células e grava artefatos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split import carregar_eventos_limpos  # noqa: E402
from src.spatial.grid import (  # noqa: E402
    RelatorioGrid,
    atribuir_celulas,
    construir_grid_spec_treino,
    contagem_por_celula,
    grid_spec_para_dict,
    registrar_relatorio,
    relatorio_grid_para_dict,
)


def _resolver_entrada(root: Path, parquet_rel: str, csv_rel: str) -> Path:
    pq = root / parquet_rel
    csv = root / csv_rel
    if pq.is_file():
        return pq
    if csv.is_file():
        return csv
    raise FileNotFoundError(f"Não encontrado: {pq} nem {csv}")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    gcfg = cfg["grid"]
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_path = _resolver_entrada(
            ROOT,
            gcfg["arquivos_entrada"]["train_parquet"],
            gcfg["arquivos_entrada"]["train_csv"],
        )
        test_path = _resolver_entrada(
            ROOT,
            gcfg["arquivos_entrada"]["test_parquet"],
            gcfg["arquivos_entrada"]["test_csv"],
        )
    except FileNotFoundError as e:
        print(f"[ERRO] {e}. Rode o passo 2 antes.")
        return 1

    treino = carregar_eventos_limpos(train_path)
    teste = carregar_eventos_limpos(test_path)

    col_x = str(gcfg.get("col_x", "x_utm"))
    col_y = str(gcfg.get("col_y", "y_utm"))

    spec = construir_grid_spec_treino(
        treino,
        cell_size_m=float(gcfg.get("cell_size_m", 500)),
        padding_cells=int(gcfg.get("padding_cells", 0)),
        col_x=col_x,
        col_y=col_y,
    )

    treino_g = atribuir_celulas(treino, spec, col_x=col_x, col_y=col_y)
    teste_g = atribuir_celulas(teste, spec, col_x=col_x, col_y=col_y)
    cells = contagem_por_celula(treino_g, teste_g, spec)

    rel = RelatorioGrid()
    registrar_relatorio(rel, spec, treino_g, teste_g, cells)

    paths = {
        "grid_spec": out_dir / "grid_spec.json",
        "train_grid_parquet": out_dir / "stkde_train_grid.parquet",
        "train_grid_csv": out_dir / "stkde_train_grid.csv",
        "test_grid_parquet": out_dir / "stkde_test_grid.parquet",
        "test_grid_csv": out_dir / "stkde_test_grid.csv",
        "grid_cells_parquet": out_dir / "grid_cells.parquet",
        "grid_report": out_dir / "grid_report.json",
    }

    paths["grid_spec"].write_text(
        json.dumps(grid_spec_para_dict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    treino_g.to_parquet(paths["train_grid_parquet"], index=False)
    teste_g.to_parquet(paths["test_grid_parquet"], index=False)
    treino_g.to_csv(paths["train_grid_csv"], index=False, encoding="utf-8-sig")
    teste_g.to_csv(paths["test_grid_csv"], index=False, encoding="utf-8-sig")
    cells.to_parquet(paths["grid_cells_parquet"], index=False)

    report = relatorio_grid_para_dict(rel)
    report["arquivos"] = {
        "train_entrada": str(train_path.relative_to(ROOT)),
        "test_entrada": str(test_path.relative_to(ROOT)),
        **{k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }
    paths["grid_report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Malha: {spec.nx} x {spec.ny} células ({spec.cell_size_m} m)")
    print(f"     origem UTM: ({spec.x_origin:.1f}, {spec.y_origin:.1f})")
    print(f"[OK] Treino dentro/fora: {rel.treino_dentro:,} / {rel.treino_fora:,}")
    print(f"[OK] Teste  dentro/fora: {rel.teste_dentro:,} / {rel.teste_fora:,}")
    print(f"[OK] Células com evento: {rel.n_celulas_com_evento:,}")
    print(f"     max eventos/célula (treino/teste): {rel.max_n_treino_por_celula} / {rel.max_n_teste_por_celula}")
    for k, p in paths.items():
        print(f"[OK] Gravado: {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())