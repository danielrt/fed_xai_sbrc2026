#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_shap_global_means.py (FIX: features fora de ordem + NORMALIZA L1 opcional)

Mudanças:
- NÃO exige mesma ordem de features entre folds/clients.
- Exige mesmo CONJUNTO de features, e alinha por 'feature'.
- (NOVO) Opção --normalize-l1:
    Normaliza (L1) o vetor mean_abs_shap por fold/cliente (divide pela soma),
    antes de agregar entre folds. Isso reduz “explosão” do IC quando a escala
    varia entre folds.

Regras:
- MC_DC e MA_DC_*:
  - shap_global_means.csv: média + IC por feature ao longo dos folds (amostras = folds)
  - base_value_means.txt: média simples do base_value por fold (base_value é escalar por arquivo)

- MA_DL_*:
  Passo 1) Por cliente: igual aos casos acima (amostras = folds)
    - client<i>/shap_global_means.csv
    - client<i>/base_value_means.txt
  Passo 2) All clients: baseado nos "means" por cliente (amostras = clientes)
    - shap_global_means_all_clients.csv: mean+IC entre clientes usando a coluna "mean"
      de cada client<i>/shap_global_means.csv
    - base_value_means_all_clients.txt: média simples dos base_value_means.txt por cliente
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# SciPy só para t-quantil (IC). Se não tiver, cai em z~1.96.
try:
    from scipy.stats import t as student_t
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------
# Descoberta de folds
# ----------------------------

def find_fold_dirs(shap_root: Path) -> List[Path]:
    if not shap_root.exists():
        raise FileNotFoundError(f"shap_root não existe: {shap_root}")
    folds = [p for p in shap_root.iterdir() if p.is_dir() and p.name.startswith("k_fold")]

    def _knum(p: Path) -> int:
        s = p.name.replace("k_fold", "")
        return int(s) if s.isdigit() else 10**9

    return sorted(folds, key=_knum)


# ----------------------------
# Leitura dos arquivos
# ----------------------------

def read_shap_global(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"shap_global.csv não encontrado: {path}")
    df = pd.read_csv(path)
    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        raise RuntimeError(f"shap_global.csv inválido em {path} (esperado: feature, mean_abs_shap)")
    out = df[["feature", "mean_abs_shap"]].copy()
    out["feature"] = out["feature"].astype(str)
    return out


def read_base_value_scalar(path: Path) -> float:
    """
    Lê base_value como escalar do arquivo data_test_shap_values.csv.
    Regra: base_value é o mesmo para todas as linhas do arquivo.
    """
    if not path.exists():
        raise FileNotFoundError(f"data_test_shap_values.csv não encontrado: {path}")

    df = pd.read_csv(path, usecols=["base_value"])
    if df.empty:
        return float("nan")

    v0 = float(df["base_value"].iloc[0])

    vmax = float(df["base_value"].max())
    vmin = float(df["base_value"].min())
    if not np.isfinite(v0) or not np.isfinite(vmax) or not np.isfinite(vmin):
        return float("nan")

    if abs(vmax - vmin) > 1e-8:
        print(f"[WARN] base_value não parece constante em {path} (min={vmin}, max={vmax}). Usando o primeiro valor: {v0}")

    return v0


# ----------------------------
# Estatística (IC)
# ----------------------------

def t_crit(alpha: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    if _HAS_SCIPY:
        return float(student_t.ppf(1.0 - alpha / 2.0, df=df))
    # fallback normal approx
    zmap = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}
    return float(zmap.get(alpha, 1.96))


def mean_ci(values: np.ndarray, ci_level: float = 0.95) -> Tuple[float, float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    n = int(v.size)
    m = float(np.mean(v)) if n else float("nan")
    if n <= 1:
        return m, float("nan"), m, m

    s = float(np.std(v, ddof=1))
    sem = s / math.sqrt(n)
    alpha = 1.0 - float(ci_level)
    tc = t_crit(alpha=alpha, df=n - 1)
    half = tc * sem if np.isfinite(tc) else float("nan")
    return m, s, (m - half), (m + half)


def build_feature_table(per_sample_feature_values: Dict[str, List[float]], ci_level: float) -> pd.DataFrame:
    rows = []
    ci_low_name = f"ci{int(ci_level*100)}_low"
    ci_high_name = f"ci{int(ci_level*100)}_high"

    for feat, vals in per_sample_feature_values.items():
        v = np.array(vals, dtype=np.float64)
        m, s, lo, hi = mean_ci(v, ci_level=ci_level)
        rows.append({
            "feature": feat,
            "n": int(v.size),
            "mean": m,
            "std": s,
            ci_low_name: lo,
            ci_high_name: hi,
        })

    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return df


# ----------------------------
# Alinhamento por feature + normalização
# ----------------------------

def align_values_by_ref_features(
    df_sg: pd.DataFrame,
    ref_features: List[str],
    src_path: Path,
) -> np.ndarray:
    """
    Retorna um vetor (len=ref_features) com mean_abs_shap alinhado por feature.
    Exige que o conjunto de features seja idêntico.
    """
    cur_feats = df_sg["feature"].astype(str).tolist()
    if set(cur_feats) != set(ref_features):
        missing = sorted(list(set(ref_features) - set(cur_feats)))
        extra = sorted(list(set(cur_feats) - set(ref_features)))
        raise RuntimeError(
            f"[ERRO] Conjunto de features diferente em {src_path}.\n"
            f"  missing={missing[:20]}{'...' if len(missing)>20 else ''}\n"
            f"  extra={extra[:20]}{'...' if len(extra)>20 else ''}"
        )

    s = df_sg.set_index("feature")["mean_abs_shap"]
    v = s.reindex(ref_features).to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(v)):
        raise RuntimeError(f"[ERRO] Valores NaN/inf ao alinhar features em {src_path}")
    return v


def l1_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    s = float(np.sum(v))
    if not np.isfinite(s) or s <= eps:
        return v
    return v / s


# ----------------------------
# Agregação: casos MC_DC / MA_DC_*
# ----------------------------

def aggregate_case_over_folds(
    fold_dirs: List[Path],
    rel_case_dir: Path,
    out_dir: Path,
    ci_level: float,
    normalize_l1: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    base_vals: List[float] = []

    ref_features: Optional[List[str]] = None
    per_feat: Dict[str, List[float]] = {}

    for fdir in fold_dirs:
        case_dir = fdir / rel_case_dir

        sg_path = case_dir / "shap_global.csv"
        dt_path = case_dir / "data_test_shap_values.csv"

        sg = read_shap_global(sg_path)
        bv = read_base_value_scalar(dt_path)

        if ref_features is None:
            ref_features = sg["feature"].astype(str).tolist()
            for feat in ref_features:
                per_feat.setdefault(feat, [])

        vals = align_values_by_ref_features(sg, ref_features, src_path=sg_path)

        if normalize_l1:
            vals = l1_normalize(vals)

        for feat, v in zip(ref_features, vals):
            per_feat[feat].append(float(v))

        base_vals.append(float(bv))

    df_out = build_feature_table(per_feat, ci_level=ci_level)
    df_out.to_csv(out_dir / "shap_global_means.csv", index=False)

    base_mean = float(np.mean(np.asarray(base_vals, dtype=np.float64))) if base_vals else float("nan")
    (out_dir / "base_value_means.txt").write_text(f"{base_mean:.12f}\n", encoding="utf-8")

    print(f"[OK] {out_dir} -> shap_global_means.csv + base_value_means.txt")


# ----------------------------
# Agregação: MA_DL_* (2 passos)
# ----------------------------

def aggregate_ma_dl_per_client(
    fold_dirs: List[Path],
    rel_case_dir: Path,   # MA_DL_iid or MA_DL_noniid
    out_dir: Path,
    ci_level: float,
    n_clients: Optional[int],
    normalize_l1: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_clients is None:
        first = fold_dirs[0] / rel_case_dir
        if not first.exists():
            raise FileNotFoundError(f"Não encontrei {first} para detectar clientes.")
        client_dirs = sorted([p for p in first.iterdir() if p.is_dir() and p.name.startswith("client")])
        if not client_dirs:
            raise RuntimeError(f"Não encontrei client*/ em {first}")
        n_clients = len(client_dirs)

    for ci in range(n_clients):
        base_vals: List[float] = []

        ref_features: Optional[List[str]] = None
        per_feat: Dict[str, List[float]] = {}

        for fdir in fold_dirs:
            cdir = fdir / rel_case_dir / f"client{ci}"

            sg_path = cdir / "shap_global.csv"
            dt_path = cdir / "data_test_shap_values.csv"

            sg = read_shap_global(sg_path)
            bv = read_base_value_scalar(dt_path)

            if ref_features is None:
                ref_features = sg["feature"].astype(str).tolist()
                for feat in ref_features:
                    per_feat.setdefault(feat, [])

            vals = align_values_by_ref_features(sg, ref_features, src_path=sg_path)

            if normalize_l1:
                vals = l1_normalize(vals)

            for feat, v in zip(ref_features, vals):
                per_feat[feat].append(float(v))

            base_vals.append(float(bv))

        c_out = out_dir / f"client{ci}"
        c_out.mkdir(parents=True, exist_ok=True)

        df_out = build_feature_table(per_feat, ci_level=ci_level)
        df_out.to_csv(c_out / "shap_global_means.csv", index=False)

        base_mean = float(np.mean(np.asarray(base_vals, dtype=np.float64))) if base_vals else float("nan")
        (c_out / "base_value_means.txt").write_text(f"{base_mean:.12f}\n", encoding="utf-8")

    print(f"[OK] {out_dir} -> per-client means gerados")
    return int(n_clients)


def aggregate_ma_dl_all_clients_from_client_means(
    out_dir: Path,
    ci_level: float,
    n_clients: int,
) -> None:
    per_feat_across_clients: Dict[str, List[float]] = {}
    base_client_means: List[float] = []

    ref_features: Optional[List[str]] = None

    for ci in range(n_clients):
        cdir = out_dir / f"client{ci}"

        df = pd.read_csv(cdir / "shap_global_means.csv")
        if "feature" not in df.columns or "mean" not in df.columns:
            raise RuntimeError(f"[ERRO] shap_global_means.csv inválido em {cdir} (esperado: feature, mean)")

        df["feature"] = df["feature"].astype(str)

        if ref_features is None:
            ref_features = df["feature"].tolist()
            for feat in ref_features:
                per_feat_across_clients.setdefault(feat, [])
        else:
            if set(df["feature"].tolist()) != set(ref_features):
                raise RuntimeError("[ERRO] Conjunto de features diferente entre clientes (não deveria).")

        s = df.set_index("feature")["mean"].reindex(ref_features)
        vals = s.to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(vals)):
            raise RuntimeError(f"[ERRO] NaN/inf ao alinhar client means em {cdir}")

        for feat, v in zip(ref_features, vals):
            per_feat_across_clients[feat].append(float(v))

        bv_txt = (cdir / "base_value_means.txt").read_text(encoding="utf-8").strip()
        base_client_means.append(float(bv_txt))

    df_all = build_feature_table(per_feat_across_clients, ci_level=ci_level)
    df_all.to_csv(out_dir / "shap_global_means_all_clients.csv", index=False)

    base_all = float(np.mean(np.asarray(base_client_means, dtype=np.float64))) if base_client_means else float("nan")
    (out_dir / "base_value_means_all_clients.txt").write_text(f"{base_all:.12f}\n", encoding="utf-8")

    print(f"[OK] {out_dir} -> all_clients (a partir de client means) gerado")


def aggregate_ma_dl(
    fold_dirs: List[Path],
    rel_case_dir: Path,
    out_dir: Path,
    ci_level: float,
    n_clients: Optional[int],
    normalize_l1: bool,
) -> None:
    n = aggregate_ma_dl_per_client(
        fold_dirs=fold_dirs,
        rel_case_dir=rel_case_dir,
        out_dir=out_dir,
        ci_level=ci_level,
        n_clients=n_clients,
        normalize_l1=normalize_l1,
    )
    aggregate_ma_dl_all_clients_from_client_means(
        out_dir=out_dir,
        ci_level=ci_level,
        n_clients=n,
    )


# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--shap-root", default="shap_values", help="Raiz dos SHAP por fold (shap_values/)")
    ap.add_argument("--out-root", default=".", help="Onde criar as pastas *_global_means/ (default: diretório atual)")
    ap.add_argument("--ci-level", type=float, default=0.95, help="Nível do IC (ex: 0.95)")
    ap.add_argument("--n-clients", type=int, default=None, help="Opcional: número de clientes (auto-detect se omitido)")
    ap.add_argument("--skip-noniid", action="store_true", help="Pula MA_DC_noniid e MA_DL_noniid")

    ap.add_argument(
        "--normalize-l1",
        action="store_true",
        help="Normaliza L1 (divide mean_abs_shap pela soma) por fold/cliente antes de agregar.",
    )

    args = ap.parse_args()

    shap_root = Path(args.shap_root)
    out_root = Path(args.out_root)

    fold_dirs = find_fold_dirs(shap_root)
    if not fold_dirs:
        raise RuntimeError(f"Nenhum k_fold* encontrado em {shap_root}")

    # Casos DC
    cases = [
        (Path("MC_DC"), out_root / "MC_DC_global_means"),
        (Path("MA_DC_iid"), out_root / "MA_DC_iid_global_means"),
    ]
    if not args.skip_noniid:
        cases.append((Path("MA_DC_noniid"), out_root / "MA_DC_noniid_global_means"))

    for rel_case_dir, out_dir in cases:
        aggregate_case_over_folds(
            fold_dirs=fold_dirs,
            rel_case_dir=rel_case_dir,
            out_dir=out_dir,
            ci_level=float(args.ci_level),
            normalize_l1=bool(args.normalize_l1),
        )

    # Casos DL (2 passos)
    aggregate_ma_dl(
        fold_dirs=fold_dirs,
        rel_case_dir=Path("MA_DL_iid"),
        out_dir=out_root / "MA_DL_iid_global_means",
        ci_level=float(args.ci_level),
        n_clients=args.n_clients,
        normalize_l1=bool(args.normalize_l1),
    )

    if not args.skip_noniid:
        aggregate_ma_dl(
            fold_dirs=fold_dirs,
            rel_case_dir=Path("MA_DL_noniid"),
            out_dir=out_root / "MA_DL_noniid_global_means",
            ci_level=float(args.ci_level),
            n_clients=args.n_clients,
            normalize_l1=bool(args.normalize_l1),
        )

    print("\n[DONE] Global means + CI gerados com sucesso.")


if __name__ == "__main__":
    main()
