"""
Passo 2: lê eventos limpos, faz split treino/teste e grava em data/processed/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.split import (  # noqa: E402
    carregar_eventos_limpos,
    relatorio_split_para_dict,
    split_treino_teste,
)


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    split_cfg = cfg["split"]
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_in = ROOT / split_cfg["arquivos_entrada"]["parquet"]
    csv_in = ROOT / split_cfg["arquivos_entrada"]["csv"]
    entrada = parquet_in if parquet_in.is_file() else csv_in
    if not entrada.is_file():
        print(f"[ERRO] Rode o passo 1 antes. Arquivo não encontrado: {entrada}")
        return 1

    df = carregar_eventos_limpos(entrada)
    treino, teste, rel = split_treino_teste(
        df,
        coluna_ano=str(split_cfg.get("coluna_ano", "ANO_ARQUIVO")),
        anos_treino=split_cfg["anos_treino"],
        anos_teste=split_cfg["anos_teste"],
        recalcular_t_horas=bool(split_cfg.get("recalcular_t_horas", True)),
    )

    paths = {
        "train_csv": out_dir / "stkde_train.csv",
        "train_parquet": out_dir / "stkde_train.parquet",
        "test_csv": out_dir / "stkde_test.csv",
        "test_parquet": out_dir / "stkde_test.parquet",
        "report": out_dir / "split_report.json",
    }

    treino.to_csv(paths["train_csv"], index=False, encoding="utf-8-sig")
    teste.to_csv(paths["test_csv"], index=False, encoding="utf-8-sig")
    treino.to_parquet(paths["train_parquet"], index=False)
    teste.to_parquet(paths["test_parquet"], index=False)

    report = relatorio_split_para_dict(rel)
    report["arquivos"] = {
        "entrada": str(entrada.relative_to(ROOT)),
        **{k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }
    paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Entrada: {rel.total_entrada:,} linhas")
    print(f"[OK] Treino:  {rel.n_treino:,}  {rel.por_ano_treino}")
    print(f"[OK] Teste:   {rel.n_teste:,}  {rel.por_ano_teste}")
    if rel.descartados_ano:
        print(f"     - descartados (ano fora): {rel.descartados_ano:,}")
    if rel.t_horas_recalculado:
        print(f"     - t0 treino (t_horas): {rel.t0_treino}")
    for label, p in paths.items():
        if label != "report":
            print(f"[OK] Gravado: {p.relative_to(ROOT)}")
    print(f"[OK] Gravado: {paths['report'].relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())