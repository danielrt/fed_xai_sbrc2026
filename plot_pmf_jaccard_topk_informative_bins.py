#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_pmf_jaccard_topk_informative_bins_fixedcolors.py

PMF do #matches (overlap inteiro m) derivado do Jaccard top-k (k=5,10,15),
mas plota SOMENTE bins "informativos":

- Um bin agregado "< m_min" (tudo abaixo do limiar)
- Bins individuais: m_min, m_min+1, ..., k

Onde:
  m_min = ceil(threshold * k)   (default threshold=0.6)

Agregação:
- Para cada cenário e fold:
    lê jaccard_topk_k.csv (um valor por instância)
    converte Jaccard -> m (inteiro 0..k)
    computa PMF nos bins informativos
- Depois:
    média sobre folds + IC 0.95 (t-interval; amostras = folds)

IMPORTANTE: cores são FIXAS por cenário (mesmas cores para qualquer figura e
para o script de sign-consistency).

Entrada esperada (compare_root = out_root do driver):
  compare_root/<CENARIO>/fold<i>/jaccard_topk_<k>.csv

Saídas:
  out_dir/pmf_matches_informative_top5.pdf (+ png)
  out_dir/pmf_matches_informative_top10.pdf (+ png)
  out_dir/pmf_matches_informative_top15.pdf (+ png)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy opcional
try:
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ============================================================
# Estilo SBC-friendly
# ============================================================

def set_sbc_style(font_size: int = 8) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": font_size,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })


# ============================================================
# CORES FIXAS (cenários) — IGUAL EM TODOS OS SCRIPTS
# ============================================================

SCENARIO_LABELS = {
    "MA_DC_iid_vs_MC_DC": "MF_DT (Uni) vs MC_DT",
    "MA_DC_noniid_vs_MC_DC": "MF_DT (NonUni) vs MC_DT",
    "MA_DL_iid_vs_MA_DC_iid": "MF_DL (Uni) vs MF_DT (Uni)",
    "MA_DL_noniid_vs_MA_DC_noniid": "MF_DL (NonUni) vs MF_DT (NonUni)",
}

# Paleta fixa: 4 cores do tab10 (índices estáveis)
_CMAP = plt.get_cmap("tab10")
SCENARIO_COLORS = {
    "MA_DC_iid_vs_MC_DC": _CMAP(0),
    "MA_DC_noniid_vs_MC_DC": _CMAP(1),
    "MA_DL_iid_vs_MA_DC_iid": _CMAP(2),
    "MA_DL_noniid_vs_MA_DC_noniid": _CMAP(3),
}

SCENARIOS_ORDER = [
    "MA_DC_iid_vs_MC_DC",
    "MA_DC_noniid_vs_MC_DC",
    "MA_DL_iid_vs_MA_DC_iid",
    "MA_DL_noniid_vs_MA_DC_noniid",
]

def scenario_color(key: str):
    if key not in SCENARIO_COLORS:
        raise KeyError(f"Cenário sem cor definida: {key}")
    return SCENARIO_COLORS[key]


# ============================================================
# Estatística: mean + IC
# ============================================================

def t_crit(ci_level: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    alpha = 1.0 - float(ci_level)
    if _HAS_SCIPY:
        return float(student_t.ppf(1.0 - alpha / 2.0, df=df))
    if abs(ci_level - 0.95) < 1e-9:
        return 1.96
    return 1.96


def mean_ci(values: np.ndarray, ci_level: float = 0.95) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(v))
    if n == 1:
        return m, m, m
    s = float(np.std(v, ddof=1))
    sem = s / math.sqrt(n)
    tc = t_crit(ci_level=ci_level, df=n - 1)
    half = tc * sem if np.isfinite(tc) else float("nan")
    return m, (m - half), (m + half)


# ============================================================
# Jaccard -> overlap inteiro m
# ============================================================

def jaccard_to_overlap_int(j: np.ndarray, k: int) -> np.ndarray:
    """
    J = m/(2k-m) => m = (2kJ)/(1+J)
    """
    j = np.asarray(j, dtype=np.float64)
    j = np.clip(j, 0.0, 1.0)
    m = (2.0 * k * j) / (1.0 + j + 1e-12)
    m_int = np.rint(m).astype(int)
    return np.clip(m_int, 0, k)


# ============================================================
# PMF com bins informativos
# ============================================================

def pmf_informative_bins(m_int: np.ndarray, k: int, m_min: int) -> Tuple[np.ndarray, List[str]]:
    """
    probs:
      idx 0 = P(m < m_min)
      idx i>0 = P(m == (m_min + i - 1))
    labels: ["<m_min","m_min",...,"k"]
    """
    m_int = np.asarray(m_int, dtype=int)
    m_int = m_int[(m_int >= 0) & (m_int <= k)]
    labels = [f"<{m_min}"] + [str(x) for x in range(m_min, k + 1)]

    if m_int.size == 0:
        probs = np.full((len(labels),), np.nan, dtype=np.float64)
        return probs, labels

    total = float(m_int.size)
    probs = [float(np.sum(m_int < m_min) / total)]
    for m in range(m_min, k + 1):
        probs.append(float(np.sum(m_int == m) / total))

    return np.array(probs, dtype=np.float64), labels


# ============================================================
# IO
# ============================================================

def read_jaccard_values(csv_path: Path, col: str) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"Não encontrei: {csv_path}")
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise RuntimeError(f"CSV sem coluna '{col}': {csv_path}")
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return v[np.isfinite(v)]


# ============================================================
# Plot
# ============================================================

def plot_pmf_grouped(
    x_labels: List[str],
    scenario_to_stats: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_pdf: Path,
    out_png: Path,
    fig_w_in: float,
    fig_h_in: float,
    show_legend: bool,
) -> None:
    bins = np.arange(len(x_labels), dtype=np.float64)

    keys_in_plot = [k for k in SCENARIOS_ORDER if k in scenario_to_stats]
    n_scen = len(keys_in_plot)

    group_width = 0.86
    bar_w = group_width / max(1, n_scen)
    offsets = (np.arange(n_scen) - (n_scen - 1) / 2.0) * bar_w

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_subplot(111)
    ax.set_title("")

    for si, skey in enumerate(keys_in_plot):
        mean_p, lo, hi = scenario_to_stats[skey]

        yerr = np.vstack([
            np.maximum(0.0, mean_p - lo),
            np.maximum(0.0, hi - mean_p),
        ])

        x = bins + offsets[si]
        ax.bar(
            x,
            mean_p,
            width=bar_w * 0.95,
            yerr=yerr,
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            color=scenario_color(skey),
            linewidth=0.0,
            label=SCENARIO_LABELS[skey],
        )

    ax.set_xticks(bins)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    if show_legend:
        ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.6, labelspacing=0.2)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-root", required=True)
    ap.add_argument("--kfolds", type=int, required=True)
    ap.add_argument("--out-dir", default="fig_pmf_jaccard_informative")
    ap.add_argument("--ci-level", type=float, default=0.95)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--font-size", type=int, default=8)

    # SBC 1 coluna
    ap.add_argument("--fig-w", type=float, default=3.35)
    ap.add_argument("--fig-h", type=float, default=2.45)

    ap.add_argument("--no-legend", action="store_true")
    args = ap.parse_args()

    compare_root = Path(args.compare_root)
    if not compare_root.exists():
        raise FileNotFoundError(f"compare-root não existe: {compare_root}")

    kfolds = int(args.kfolds)
    if kfolds <= 0:
        raise RuntimeError("--kfolds precisa ser > 0")

    thr = float(args.threshold)
    if not (0.0 < thr <= 1.0):
        raise RuntimeError("--threshold deve estar em (0,1]. Ex: 0.6")

    set_sbc_style(font_size=int(args.font_size))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for topk in [5, 10, 15]:
        col = f"jaccard_topk_{topk}"
        m_min = int(math.ceil(thr * topk))

        x_labels = [f"<{m_min}"] + [str(x) for x in range(m_min, topk + 1)]
        nbins = len(x_labels)

        scenario_to_stats: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for skey in SCENARIOS_ORDER:
            scen_dir = compare_root / skey
            if not scen_dir.exists():
                print(f"[SKIP] Cenário não encontrado: {scen_dir}")
                continue

            pmfs: List[np.ndarray] = []
            for fi in range(kfolds):
                csv_path = scen_dir / f"fold{fi}" / f"jaccard_topk_{topk}.csv"
                if not csv_path.exists():
                    print(f"[WARN] faltando: {csv_path}")
                    continue

                jvals = read_jaccard_values(csv_path, col=col)
                m_int = jaccard_to_overlap_int(jvals, k=topk)
                pmf, _ = pmf_informative_bins(m_int, k=topk, m_min=m_min)

                if np.any(~np.isfinite(pmf)) or pmf.size != nbins:
                    print(f"[WARN] PMF inválida em {csv_path}. Pulando fold.")
                    continue

                pmfs.append(pmf)

            if not pmfs:
                print(f"[SKIP] Sem folds válidos para {skey} em topk={topk}")
                continue

            M = np.stack(pmfs, axis=0)  # (n_folds, nbins)

            mean_p = np.zeros((nbins,), dtype=np.float64)
            lo = np.zeros((nbins,), dtype=np.float64)
            hi = np.zeros((nbins,), dtype=np.float64)

            for b in range(nbins):
                m, l, h = mean_ci(M[:, b], ci_level=float(args.ci_level))
                mean_p[b], lo[b], hi[b] = m, l, h

            scenario_to_stats[skey] = (mean_p, lo, hi)

        if not scenario_to_stats:
            print(f"[SKIP] Nenhum cenário válido para topk={topk}")
            continue

        out_pdf = out_dir / f"pmf_matches_informative_top{topk}.pdf"
        out_png = out_dir / f"pmf_matches_informative_top{topk}.png"

        plot_pmf_grouped(
            x_labels=x_labels,
            scenario_to_stats=scenario_to_stats,
            out_pdf=out_pdf,
            out_png=out_png,
            fig_w_in=float(args.fig_w),
            fig_h_in=float(args.fig_h),
            show_legend=not bool(args.no_legend),
        )

        print(f"[OK] {out_pdf.name} (+ PNG) | topk={topk} m_min={m_min}")

    print("\n[DONE] Jaccard PMF informativa gerada em:", out_dir.resolve())


if __name__ == "__main__":
    main()
