"""
Passo 1: carrega CSV, limpa e grava artefatos em data/processed/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean import carregar_e_limpar, relatorio_para_dict  # noqa: E402


def main() -> int:
    config_path = ROOT / "configs" / "data.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    input_csv = ROOT / cfg["input_csv"]
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df, rel = carregar_e_limpar(
        input_csv,
        lat_min=float(cfg["lat_min"]),
        lat_max=float(cfg["lat_max"]),
        lon_min=float(cfg["lon_min"]),
        lon_max=float(cfg["lon_max"]),
        alinhamento_ano_arquivo=bool(cfg.get("alinhamento_ano_arquivo", False)),
        utm_epsg=int(cfg.get("utm_epsg", 31983)),
    )

    csv_out = out_dir / "stkde_events_clean.csv"
    parquet_out = out_dir / "stkde_events_clean.parquet"
    report_out = out_dir / "cleaning_report.json"

    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_out, index=False)

    report = relatorio_para_dict(rel)
    report["arquivos"] = {
        "entrada": str(input_csv.relative_to(ROOT)),
        "csv": str(csv_out.relative_to(ROOT)),
        "parquet": str(parquet_out.relative_to(ROOT)),
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Entrada: {rel.total_entrada:,} linhas")
    print(f"[OK] Saida:   {rel.total_saida:,} linhas")
    print(f"     - coord (0,0):     -{rel.removidos_coord_zero:,}")
    print(f"     - fora envelope:   -{rel.removidos_fora_envelope:,}")
    print(f"     - data invalida:   -{rel.removidos_data_invalida:,}")
    print(f"     - ano desalinhado: -{rel.removidos_alinhamento_ano:,}")
    print(f"     - hora informada:  {rel.hora_informada:,} | ausente: {rel.hora_ausente:,}")
    print(f"     - por ANO_ARQUIVO: {rel.por_ano_arquivo}")
    print(f"[OK] Gravado: {csv_out.relative_to(ROOT)}")
    print(f"[OK] Gravado: {parquet_out.relative_to(ROOT)}")
    print(f"[OK] Gravado: {report_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
