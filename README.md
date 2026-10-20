# rp2-previsao-furtos-veiculos

Pipeline de **previsão espaço-temporal de furtos de veículos** (SSP-SP, São Paulo) com **STKDE** (*Spatiotemporal Kernel Density Estimation*). O Random Forest será integrado depois; neste repositório está implementado o fluxo STKDE completo (limpeza → split temporal → malha 500 m → tuning → figuras).

**Amostra:** 30.000 registros (`Furto (art. 155)`, `S.PAULO`), ~10.000 por ano-arquivo (2024, 2025, 2026).

---

## Estrutura do repositório

```
configs/
  data.yaml          # caminhos, envelope SP, split, malha
  stkde.yaml         # bandas h_s/h_t, tuning e exportação
data/
  amostra_completa_30000.csv   # entrada (versionada)
  processed/                   # artefatos gerados (ignorado no Git)
scripts/
  prepare_stkde_data.py        # passo 1
  split_stkde_data.py          # passo 2
  build_stkde_grid.py          # passo 3
  run_stkde_eval.py            # passo 4
  export_stkde_results.py      # passo 5
src/
  data/      # limpeza e split
  spatial/   # malha UTM
  stkde/     # KDE, PAI, figuras
reports/
  stkde/     # figuras PNG, tabela LaTeX, CSV (versionável)
```

---

## Pré-requisitos

- **Python 3.10+** (recomendado)
- **Git**

Verifique:

```powershell
python --version
git --version
```

---

## 1. Ambiente virtual e dependências

Na **raiz do repositório**:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> A pasta `.venv` não vai para o Git. Sempre ative o ambiente antes de rodar os scripts.

Para sair do ambiente: `deactivate`.

---

## 2. Pipeline STKDE — visão geral

| Passo | Script | O que faz |
|-------|--------|-----------|
| **1** | `prepare_stkde_data.py` | Carrega o CSV, remove `(0,0)` e datas inválidas, trata hora ausente, projeta **UTM 31983**, cria `datetime_ocorrencia` e `t_horas`. |
| **2** | `split_stkde_data.py` | **Treino:** `ANO_ARQUIVO` 2024–2025 · **Teste:** 2026. Recalcula `t_horas` com origem no treino. |
| **3** | `build_stkde_grid.py` | Malha **500 m** (UTM) a partir do treino; atribui `cell_id` a cada evento; contagens por célula. |
| **4** | `run_stkde_eval.py` | STKDE gaussiano, *grid search* em `h_s` e `h_t`, log-verossimilhança no teste, **PAI**, densidade no melhor par. |
| **5** | `export_stkde_results.py` | Mapas PNG, curva PAI, heatmaps e `tab_stkde_tuning.tex` em `reports/stkde/`. |

Execute **na ordem**, com o venv ativo:

```powershell
python scripts/prepare_stkde_data.py
python scripts/split_stkde_data.py
python scripts/build_stkde_grid.py
python scripts/run_stkde_eval.py
python scripts/export_stkde_results.py
```

O passo **4** pode levar alguns minutos (16 combinações × ~18k eventos de treino).

---

## 3. Saídas principais

### `data/processed/` (local, `.gitignore`)

| Arquivo | Descrição |
|---------|-----------|
| `stkde_events_clean.parquet` | Eventos após limpeza |
| `stkde_train.parquet` / `stkde_test.parquet` | Split temporal |
| `stkde_train_grid.parquet` / `stkde_test_grid.parquet` | Eventos + célula da malha |
| `grid_spec.json` / `grid_cells.parquet` | Definição e resumo da malha |
| `stkde/tuning_results.csv` | Resultados do *tuning* |
| `stkde/best_params.json` | Melhor `(h_s, h_t)` |
| `stkde/density_best.parquet` | λ por célula (melhor modelo) |

Relatórios JSON: `cleaning_report.json`, `split_report.json`, `grid_report.json`, `stkde/eval_report.json`.

### `reports/stkde/` (pode ir para o Git)

- `fig_stkde_densidade.png`, `fig_stkde_pai_curva.png`
- `fig_stkde_heatmap_pai10.png`, `fig_stkde_heatmap_loglik.png`
- `tab_stkde_tuning.tex`, `pai_curve.csv`, `export_report.json`

---

## 4. Configuração

### `configs/data.yaml`

- **`input_csv`:** amostra de entrada.
- **Envelope lat/lon:** filtro aproximado de São Paulo.
- **`alinhamento_ano_arquivo`:** se `true`, exige ano da data = `ANO_ARQUIVO` (padrão: `false`).
- **`split`:** anos de treino/teste e caminhos dos artefatos limpos.
- **`grid`:** `cell_size_m: 500`, colunas UTM, arquivos de treino/teste.

### `configs/stkde.yaml`

- **`h_s_list` / `h_t_list`:** bandas espacial (m) e temporal (h).
- **`t_eval_mode`:** `median_test` (PAI/mapa no tempo mediano do teste), `max_train` ou `fixed`.
- **`select_by`:** `loglik_test` ou `pai_at_10` para escolher o melhor par.
- **`reports_dir`:** pasta das figuras exportadas (passo 5).

Edite os YAMLs e rode de novo apenas os passos afetados (ex.: mudou só `stkde.yaml` → passos 4 e 5).

---

## 5. Resultados esperados (referência)

Com a amostra atual, após limpeza restam **28.161** eventos; treino **18.838**, teste **9.323** (1 evento de teste pode ficar fora do *bbox* da malha).

Melhor par típico do grid padrão: **`h_s = 500` m**, **`h_t = 2160` h** (~90 dias), com **PAI@10% ≈ 2,8** (sobre células com atividade — ver limitações no artigo).

---

## 6. Boas práticas

- Não commitar `.venv/` nem `data/processed/`.
- Commitar código, `configs/`, `reports/stkde/` (figuras/tabela) quando fizer sentido para o grupo.
- Novas dependências: instale no venv e atualize `requirements.txt` com cuidado (`pip freeze` traz pacotes demais; prefira pinar só o que o projeto usa).

---

## 7. Solução de problemas

| Problema | Ação |
|----------|------|
| `ImportError: pyarrow` | `pip install pyarrow` ou reinstale `requirements.txt`. |
| Passo 2/3/4: arquivo não encontrado | Rode os passos anteriores. |
| PowerShell bloqueia `Activate.ps1` | Ajuste a política de execução do usuário ou use CMD com `.venv\Scripts\activate.bat`. |
| STKDE muito lento | Reduza `h_s_list`/`h_t_list` em `stkde.yaml` para testes. |

---

## Licença / autores

Projeto acadêmico — EACH-USP, RP II (G12). Consulte o orientador sobre uso e citação dos dados SSP-SP.
