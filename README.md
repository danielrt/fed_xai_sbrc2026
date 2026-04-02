# Do Aprendizado Centralizado ao Federado: O Que Acontece com a Explicabilidade dos Modelos?

Este repositório reúne o código referente ao artigo *Do Aprendizado Centralizado ao Federado: 
O Que Acontece com a Explicabilidade dos Modelos?*. O objetivo do trabalho foi propor uma 
metodologia para analisar o impacto do **aprendizado federado (FL)** sobre explicações **SHAP** 
em diferentes cenários de treino e particionamento de dados. O dataset EHMS 
([Enhanced Healthcare Monitoring System](https://www.cse.wustl.edu/~jain/ehms/index.html)).
foi usado para aplicar a metodologia proposta.

## Visão geral do fluxo

A aplicação da metodologia consiste nos seguintes passos:

1. criação os folds para treinamento usando a estratégia de validação cruzada;
2. divisão do dataset para geração dos clientes para o treinamento federado, com particionamento seguindo a mesma distribuição do dataset original (Uni) e distribuição desbalanceada em relação ao dataset original (NonUni); 
3. treinamentos dos modelos central, federado Uni e federado NonUni, com validação cruzada;
3. calculo dos valores SHAP para os modelos e cenários de comparação;
4. comparação entre as explicações para os cenários;
5. agregação de resultados globais;
6. geração de gráficos para visualização das métricas de comparação entre os diferentes cenários.

## Pré-requisitos

### Sistema

- Ubuntu versões 20.04, 22.04 ou 24.04
- Python 3
- MininetFed 2.0 instalado e funcional

**Para instalar o MininetFed 2.0, siga as instruções em:**

https://github.com/lprm-ufes/MininetFed-2.0-SBRC-2026

### Configuração do Ambiente Python


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn torch shap matplotlib scipy joblib
```

## Sequência de comandos

### 1) Clonar o repositório

```bash
git clone https://github.com/danielrt/fed_xai_sbrc2026.git
cd fed_xai_sbrc2026
```

---

### 2) Criar os folds centrais, os datasets federados e os diretórios montados dos clientes

```bash
python3 run_cv_folds_pipeline.py \
  --csv dataset/wustl-ehms-2020_with_attacks_categories.csv \
  --cv-root cv \
  --k 5 \
  --seed 42 \
  --n-clients 4 \
  --alpha-size 0.8 \
  --alpha-class 0.8 \
  --variants iid,noniid \
  --client-code-dir fed/client_code \
  --mount-root mounted_clients \
  --clean
```

### 3) Treinar os modelos central, federado IID e federado non-IID

```bash
python3 run_train_all_folds.py \
  --k 5 \
  --cv-root cv \
  --mount-root mounted_clients \
  --results-root training_results \
  --n-clients 4 \
  --central-script ehms_central.py \
  --fed-script fed/ehms_fed.py \
  --fed-server-dir fed/server \
  --seed 42 \
  --epochs 150 \
  --patience 10
```

### 4) Calcular os valores SHAP para todos os folds e cenários

```bash
python3 run_shap_all_folds.py \
  --k 5 \
  --cv-root cv \
  --results-root training_results \
  --out-root shap_values \
  --compute-script compute_shap_values.py \
  --background-size 500 \
  --label-col Label \
  --id-col instance_id \
  --n-clients 4
```

### 5) Comparar explicações SHAP entre cenários

```bash
python3 run_compare_shap_all_folds.py \
  --k 5 \
  --shap-root shap_values \
  --out-root comparisons \
  --compare-script compare_shap_instancewise.py \
  --rbo-p 0.9 \
  --topk 5,10,15
```

### 6) Agregar as médias globais dos SHAP entre folds

```bash
python3 summarize_shap_global_means.py \
  --shap-root shap_values \
  --out-root global_means \
  --ci-level 0.95
```

### 7) Gerar gráficos para visualização das métricas de comparação entre os diferentes cenários.

```bash
python3 plot_top15_shap_global_bars.py \
  --means-root global_means \
  --out-dir fig_top_features_ci \
  --top-n 15
```

**O que gera:**

Arquivos `.pdf` e `.png` em `fig_top_features_ci/`, por exemplo:

- `top15_MC_DC_global_means.pdf`
- `top15_MA_DC_iid_global_means.pdf`
- `top15_MA_DC_noniid_global_means.pdf`
- e equivalentes para os cenários `MA_DL_*` disponíveis

```bash
python3 plot_ecdf_cosine_distance.py \
  --compare-root comparisons \
  --k 5 \
  --out-dir fig_ecdf_cosine
```

**O que gera:**

Arquivos `.pdf` e `.png` em `fig_ecdf_cosine/`.

```bash
python3 plot_pmf_jaccard_topk_informative_bins.py \
  --compare-root comparisons \
  --kfolds 5 \
  --out-dir fig_pmf_jaccard_informative \
  --threshold 0.6
```

**O que gera:**

Arquivos `.pdf` e `.png` em `fig_pmf_jaccard_informative/`.

```bash
python3 plot_pmf_topk_sign_consistency_informative_bins.py \
  --compare-root comparisons \
  --kfolds 5 \
  --out-dir fig_pmf_sign_informative \
  --threshold 0.6
```

**O que gera:**

Arquivos `.pdf` e `.png` em `fig_pmf_sign_informative/`.