#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_shap_instancewise.py

Compara dois arquivos data_test_shap_values*.csv (mesmas instâncias e mesmas features),
calculando métricas por instância com base apenas nas colunas shap_*.

Saídas (na pasta --out-dir):
- cosine_distance.csv
- kendall_tau_b.csv
- spearman_rho.csv
- l1_distance.csv                     (|shap| L1-normalizado por instância)
- js_divergence.csv                   (Jensen–Shannon, |shap| L1-normalizado, log2)
- rbo_p0.9.csv                        (RBO com p configurável, default 0.9)
- jaccard_topk_5.csv / _10.csv / _15.csv
- topk_sign_consistency_5.csv / _10.csv / _15.csv

E também um arquivo de métricas globais (uma linha):
- global_topk_metrics.csv
  colunas:
    overlap_topk_global_5/10/15
    topk_sign_consistency_global_5/10/15

Regras:
- Alinha por instance_id (ordena e faz inner-join)
- Exige mesmo conjunto de colunas shap_*
- Para L1 e JSD: normaliza por instância em abs(shap): p = |shap| / sum(|shap|).
  Se sum(|shap|) ~ 0 => usa distribuição uniforme.

Dependências:
- pandas, numpy
- scipy (para kendalltau e spearmanr). Se não tiver scipy, o script falha com erro claro.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# util
# -------------------------

def _require_scipy():
    try:
        from scipy.stats import kendalltau, spearmanr  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Este script precisa de scipy para Kendall/Spearman.\n"
            "Instale com: pip install scipy\n"
            f"Erro original: {e}"
        )


def find_shap_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("shap_")]
    if not cols:
        raise RuntimeError("Não encontrei colunas shap_* no CSV.")
    return cols


def load_and_prepare(path: Path, id_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise RuntimeError(f"CSV {path} não tem coluna '{id_col}'.")
    # ordenar por instance_id para garantir alinhamento
    df = df.sort_values(id_col).reset_index(drop=True)
    return df


def align_two(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    id_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # interseção de instâncias
    a_ids = df_a[[id_col]].copy()
    b_ids = df_b[[id_col]].copy()

    # inner merge para garantir mesmas instâncias e mesma ordem
    merged = a_ids.merge(b_ids, on=id_col, how="inner")
    if merged.empty:
        raise RuntimeError("Nenhuma instância em comum entre os dois arquivos (inner join vazio).")

    # filtrar e reordenar por merged ids
    df_a2 = df_a.merge(merged, on=id_col, how="inner").sort_values(id_col).reset_index(drop=True)
    df_b2 = df_b.merge(merged, on=id_col, how="inner").sort_values(id_col).reset_index(drop=True)

    shap_a = find_shap_cols(df_a2)
    shap_b = find_shap_cols(df_b2)

    if set(shap_a) != set(shap_b):
        missing_in_b = sorted(list(set(shap_a) - set(shap_b)))
        missing_in_a = sorted(list(set(shap_b) - set(shap_a)))
        raise RuntimeError(
            "Conjunto de colunas shap_* diferente entre os arquivos.\n"
            f"  faltando no B: {missing_in_b[:20]}{'...' if len(missing_in_b)>20 else ''}\n"
            f"  faltando no A: {missing_in_a[:20]}{'...' if len(missing_in_a)>20 else ''}\n"
        )

    # ordem canônica das colunas shap_* (alfabética, estável)
    shap_cols = sorted(list(set(shap_a)))
    return df_a2, df_b2, shap_cols


def cosine_distance_row(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < eps and ny < eps:
        return 0.0
    if nx < eps or ny < eps:
        return 1.0
    cos = float(np.dot(x, y) / (nx * ny))
    # estabilidade numérica
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos


def l1_normalized_abs(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.abs(v).astype(np.float64, copy=False)
    s = float(np.sum(a))
    if s < eps:
        # distribuição uniforme
        return np.full_like(a, 1.0 / max(1, a.size), dtype=np.float64)
    return a / s


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence com log base 2 (retorna em [0,1] quando p,q são distribuições).
    p,q devem ser >=0 e somar 1.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        # KL(a||b) com 0*log(0/.) = 0
        a = np.clip(a, 0.0, None)
        b = np.clip(b, eps, None)
        mask = a > 0
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask]))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def rbo_score(rank_a: List[int], rank_b: List[int], p: float = 0.9) -> float:
    """
    Rank-Biased Overlap (RBO) para rankings finitos (sem extrapolação infinita).
    Implementação padrão: sum_{d=1..k} (1-p)*p^{d-1} * A_d
    onde A_d é o overlap proporcional (|S_a(d) ∩ S_b(d)| / d).
    rank_* são listas de índices (mesmo universo de itens).
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p do RBO precisa estar em (0,1).")

    k = max(len(rank_a), len(rank_b))
    seen_a = set()
    seen_b = set()

    score = 0.0
    for d in range(1, k + 1):
        if d <= len(rank_a):
            seen_a.add(rank_a[d - 1])
        if d <= len(rank_b):
            seen_b.add(rank_b[d - 1])
        overlap = len(seen_a.intersection(seen_b))
        A_d = overlap / d
        score += (1.0 - p) * (p ** (d - 1)) * A_d
    return float(score)


def topk_set_from_abs(v: np.ndarray, k: int) -> List[int]:
    k = min(int(k), int(v.size))
    if k <= 0:
        return []
    idx = np.argsort(-np.abs(v))  # desc por abs
    return idx[:k].tolist()


def topk_rank_from_abs(v: np.ndarray) -> List[int]:
    return np.argsort(-np.abs(v)).tolist()


def jaccard_topk(v1: np.ndarray, v2: np.ndarray, k: int) -> float:
    s1 = set(topk_set_from_abs(v1, k))
    s2 = set(topk_set_from_abs(v2, k))
    if not s1 and not s2:
        return 1.0
    inter = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(inter / union) if union else 1.0


def topk_sign_consistency(v1: np.ndarray, v2: np.ndarray, k: int) -> float:
    """
    Considera TOP-k por abs em cada vetor e mede consistência de sinal APENAS no overlap.
    Retorna:
      (#features no overlap com mesmo sinal) / (#features no overlap)
    Se overlap vazio => NaN (ou 0.0 se você preferir; aqui uso NaN pra ficar explícito).
    """
    s1 = set(topk_set_from_abs(v1, k))
    s2 = set(topk_set_from_abs(v2, k))
    ov = sorted(list(s1.intersection(s2)))
    if len(ov) == 0:
        return float("nan")
    same = 0
    for j in ov:
        sign1 = 1 if v1[j] > 0 else (-1 if v1[j] < 0 else 0)
        sign2 = 1 if v2[j] > 0 else (-1 if v2[j] < 0 else 0)
        if sign1 == sign2:
            same += 1
    return float(same / len(ov))


def global_rank_and_sign(df: pd.DataFrame, shap_cols: List[str]) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Produz:
    - ranking global por mean(|shap|)
    - mean_abs por feature
    - mean_signed por feature (para consistência de sinal global)
    """
    X = df[shap_cols].to_numpy(dtype=np.float64, copy=False)
    mean_abs = np.mean(np.abs(X), axis=0)
    mean_signed = np.mean(X, axis=0)
    rank = np.argsort(-mean_abs).tolist()
    return rank, mean_abs, mean_signed


def save_single_metric(out_path: Path, ids: np.ndarray, values: np.ndarray, colname: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"instance_id": ids, colname: values})
    df.to_csv(out_path, index=False)


# -------------------------
# main
# -------------------------

def main():
    _require_scipy()
    from scipy.stats import kendalltau, spearmanr  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="CSV A: data_test_shap_values.csv")
    ap.add_argument("--b", required=True, help="CSV B: data_test_shap_values.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--id-col", default="instance_id")
    ap.add_argument("--rbo-p", type=float, default=0.9)
    ap.add_argument("--topk", default="5,10,15", help="Lista de k separada por vírgula (ex: 5,10,15)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    if not ks:
        raise RuntimeError("--topk vazio/ inválido.")

    df_a = load_and_prepare(Path(args.a), id_col=args.id_col)
    df_b = load_and_prepare(Path(args.b), id_col=args.id_col)
    df_a, df_b, shap_cols = align_two(df_a, df_b, id_col=args.id_col)

    ids = df_a[args.id_col].to_numpy()

    Xa = df_a[shap_cols].to_numpy(dtype=np.float64, copy=False)
    Xb = df_b[shap_cols].to_numpy(dtype=np.float64, copy=False)

    n = Xa.shape[0]
    d = Xa.shape[1]

    cosine = np.zeros((n,), dtype=np.float64)
    ktau = np.zeros((n,), dtype=np.float64)
    srho = np.zeros((n,), dtype=np.float64)
    l1 = np.zeros((n,), dtype=np.float64)
    jsd = np.zeros((n,), dtype=np.float64)
    rbo = np.zeros((n,), dtype=np.float64)

    jaccard_by_k: Dict[int, np.ndarray] = {k: np.zeros((n,), dtype=np.float64) for k in ks}
    sign_by_k: Dict[int, np.ndarray] = {k: np.full((n,), np.nan, dtype=np.float64) for k in ks}

    for i in range(n):
        va = Xa[i]
        vb = Xb[i]

        cosine[i] = cosine_distance_row(va, vb)

        # Kendall tau-b / Spearman rho (por ranking dos valores SHAP, com sinal)
        kt = kendalltau(va, vb, variant="b", nan_policy="propagate")
        sp = spearmanr(va, vb, nan_policy="propagate")

        ktau[i] = float(kt.statistic) if kt.statistic is not None else float("nan")
        srho[i] = float(sp.statistic) if sp.statistic is not None else float("nan")

        # L1 + JSD em distribuições |shap| normalizadas por instância
        pa = l1_normalized_abs(va)
        pb = l1_normalized_abs(vb)

        l1[i] = float(np.sum(np.abs(pa - pb)))
        jsd[i] = js_divergence(pa, pb)

        # RBO (ranking por abs)
        ra = topk_rank_from_abs(va)
        rb = topk_rank_from_abs(vb)
        rbo[i] = rbo_score(ra, rb, p=float(args.rbo_p))

        # top-k metrics
        for k in ks:
            jaccard_by_k[k][i] = jaccard_topk(va, vb, k=k)
            sign_by_k[k][i] = topk_sign_consistency(va, vb, k=k)

    # salvar métricas por instância
    save_single_metric(out_dir / "cosine_distance.csv", ids, cosine, "cosine_distance")
    save_single_metric(out_dir / "kendall_tau_b.csv", ids, ktau, "kendall_tau_b")
    save_single_metric(out_dir / "spearman_rho.csv", ids, srho, "spearman_rho")
    save_single_metric(out_dir / "l1_distance.csv", ids, l1, "l1_distance")
    save_single_metric(out_dir / "js_divergence.csv", ids, jsd, "js_divergence")
    save_single_metric(out_dir / f"rbo_p{args.rbo_p}.csv", ids, rbo, f"rbo_p{args.rbo_p}")

    for k in ks:
        save_single_metric(out_dir / f"jaccard_topk_{k}.csv", ids, jaccard_by_k[k], f"jaccard_topk_{k}")
        save_single_metric(out_dir / f"topk_sign_consistency_{k}.csv", ids, sign_by_k[k], f"topk_sign_consistency_{k}")

    # -------------------------
    # métricas globais top-k (por fold)
    # -------------------------
    rank_a, mean_abs_a, mean_signed_a = global_rank_and_sign(df_a, shap_cols)
    rank_b, mean_abs_b, mean_signed_b = global_rank_and_sign(df_b, shap_cols)

    # overlap global top-k por ranking mean(|shap|)
    # sign consistency global: no overlap do top-k, compara sinal do mean(shap)
    glob: Dict[str, float] = {}
    for k in ks:
        top_a = set(rank_a[:k])
        top_b = set(rank_b[:k])
        ov = sorted(list(top_a.intersection(top_b)))

        # overlap jaccard global top-k
        denom = len(top_a.union(top_b))
        glob[f"overlap_topk_global_{k}"] = float(len(ov) / denom) if denom else 1.0

        # sign consistency global top-k (no overlap)
        if len(ov) == 0:
            glob[f"topk_sign_consistency_global_{k}"] = float("nan")
        else:
            same = 0
            for j in ov:
                s1 = 1 if mean_signed_a[j] > 0 else (-1 if mean_signed_a[j] < 0 else 0)
                s2 = 1 if mean_signed_b[j] > 0 else (-1 if mean_signed_b[j] < 0 else 0)
                if s1 == s2:
                    same += 1
            glob[f"topk_sign_consistency_global_{k}"] = float(same / len(ov))

    pd.DataFrame([glob]).to_csv(out_dir / "global_topk_metrics.csv", index=False)

    print(f"[OK] Comparação concluída: n={n}, d={d}")
    print(f"[OUT] {out_dir.resolve()}")


if __name__ == "__main__":
    main()
