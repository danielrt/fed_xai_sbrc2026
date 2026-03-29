#!/usr/bin/env python3
"""
compute_shap_values.py (ATUALIZADO)

Computa SHAP (logit) para:
- data_test.csv (inteiro)

e salva em --out-dir:
- data_test_shap_values.csv   (instance_id, base_value, f_x, prob, x_*, shap_*)
- shap_global.csv             (feature, mean_abs_shap)  [a partir do data_test]

Carrega modelo a partir de:
A) NPZ posicional (p0..pN): ex: training_results/.../iid/best.model.npz
B) NPZ state_dict-like: ex: training_results/.../central/best_model.npz
C) MininetFed TrainingData JSON legado (best.model/.json): {"weights":[base64...]} aplicado em model.parameters()
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
from torch import nn


# ============================================================
# Modelo (TEM que ser igual ao do treino/cliente)
# ============================================================

class EHMSANN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(40, 20),
            nn.ReLU(),

            nn.Linear(20, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Utils gerais
# ============================================================

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_features(df: pd.DataFrame, label_col="Label", id_col="instance_id") -> List[str]:
    feats = [c for c in df.columns if c not in {label_col, id_col}]
    if not feats:
        raise RuntimeError("Nenhuma feature encontrada (removendo Label/instance_id sobrou vazio).")
    return feats


def df_to_tensor(df: pd.DataFrame, feats: List[str], device: torch.device) -> torch.Tensor:
    X = df[feats].to_numpy(dtype=np.float32, copy=True)
    return torch.tensor(X, dtype=torch.float32, device=device)


def ensure_2d_shap(arr: Any, n: int, d: int) -> np.ndarray:
    arr = arr.detach().cpu().numpy() if torch.is_tensor(arr) else np.asarray(arr)
    if arr.ndim == 2 and arr.shape == (n, d):
        return arr
    if arr.ndim == 3 and arr.shape == (n, d, 1):
        return arr[:, :, 0]
    if arr.ndim == 3 and arr.shape == (n, 1, d):
        return arr[:, 0, :]
    raise RuntimeError(f"SHAP com shape não suportado: {arr.shape} (esperado {(n,d)} ou {(n,d,1)} ou {(n,1,d)})")


# ============================================================
# Base64 <-> ndarray (compatível com mininetfed legado)
# ============================================================

def base64_to_ndarray(s: str) -> np.ndarray:
    raw = base64.b64decode(s.encode("utf-8"))
    import io
    bio = io.BytesIO(raw)
    arr = np.load(bio, allow_pickle=False)
    return arr


# ============================================================
# Carregamento do modelo (NPZ híbrido + JSON legado)
# ============================================================

def _load_weights_from_trainingdata_json(path: Path) -> List[np.ndarray]:
    txt = path.read_text(encoding="utf-8").strip()
    d = json.loads(txt)
    if not isinstance(d, dict):
        raise RuntimeError("best.model JSON não é um dict no topo.")
    if "weights" not in d:
        raise RuntimeError(f"JSON não tem chave 'weights'. Chaves: {list(d.keys())[:20]}")
    if not isinstance(d["weights"], list):
        raise RuntimeError("Campo 'weights' não é lista.")
    weights = [base64_to_ndarray(w) for w in d["weights"]]
    return weights


def _npz_is_positional(data: np.lib.npyio.NpzFile) -> bool:
    files = list(data.files)
    if not files:
        return False
    if "p0" in files:
        return True

    ok = True
    for k in files:
        if not k.startswith("p"):
            ok = False
            break
        try:
            int(k[1:])
        except Exception:
            ok = False
            break
    return ok


def _load_weights_positional_from_npz(data: np.lib.npyio.NpzFile) -> List[np.ndarray]:
    def _kint(k: str) -> int:
        if not k.startswith("p"):
            raise RuntimeError(f"NPZ não é posicional (esperado p0..pN). Achou chave: {k}")
        return int(k[1:])
    keys = sorted(data.files, key=_kint)
    return [np.asarray(data[k]) for k in keys]


def _load_state_dict_from_npz(data: np.lib.npyio.NpzFile) -> Dict[str, torch.Tensor]:
    sd: Dict[str, torch.Tensor] = {}
    for k in data.files:
        sd[k] = torch.tensor(np.asarray(data[k]))
    if not sd:
        raise RuntimeError("NPZ state_dict-like vazio.")
    return sd


def load_model_any(model_path: Path, input_dim: int, device: torch.device) -> nn.Module:
    model = EHMSANN(input_dim=input_dim).to(device)
    model.eval()

    suf = model_path.suffix.lower()

    if suf == ".npz":
        data = np.load(model_path)

        if _npz_is_positional(data):
            weights_list = _load_weights_positional_from_npz(data)
            params = list(model.parameters())

            if len(weights_list) != len(params):
                raise RuntimeError(
                    f"Número de pesos no NPZ ({len(weights_list)}) != número de parâmetros do modelo ({len(params)})."
                )

            with torch.no_grad():
                for i, (p, w) in enumerate(zip(params, weights_list)):
                    w = np.asarray(w)
                    if tuple(p.data.shape) != tuple(w.shape):
                        raise RuntimeError(
                            f"Shape incompatível em p{i}: param={tuple(p.data.shape)} vs npz={tuple(w.shape)}"
                        )
                    p.data.copy_(torch.from_numpy(w).to(device=device, dtype=p.data.dtype))
            return model

        # state_dict-like
        sd = _load_state_dict_from_npz(data)
        ref_sd = model.state_dict()

        # mover para device + dtype esperado
        fixed_sd: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k not in ref_sd:
                raise RuntimeError(f"Chave '{k}' não existe no state_dict do modelo. Arquitetura diferente?")
            fixed_sd[k] = v.to(device=device, dtype=ref_sd[k].dtype)

        model.load_state_dict(fixed_sd, strict=True)
        return model

    # JSON legado (best.model/.json) começando com '{'
    first = model_path.open("rb").read(1)
    if first == b"{":
        weights_list = _load_weights_from_trainingdata_json(model_path)
        params = list(model.parameters())

        if len(weights_list) != len(params):
            raise RuntimeError(
                f"Número de pesos no arquivo ({len(weights_list)}) != número de parâmetros do modelo ({len(params)})."
            )

        with torch.no_grad():
            for i, (p, w) in enumerate(zip(params, weights_list)):
                w = np.asarray(w)
                if tuple(p.data.shape) != tuple(w.shape):
                    raise RuntimeError(
                        f"Shape incompatível ao carregar pesos em idx={i}: param={tuple(p.data.shape)} vs weight={tuple(w.shape)}"
                    )
                p.data.copy_(torch.from_numpy(w).to(device=device, dtype=p.data.dtype))
        return model

    raise RuntimeError(
        f"Formato de modelo não suportado: {model_path} "
        "(esperado: .npz posicional p0..pN, .npz state_dict-like, ou JSON TrainingData começando com '{')."
    )


# ============================================================
# SHAP GradientExplainer (logit)
# ============================================================

def compute_shap_gradient(
    model: nn.Module,
    X_background: torch.Tensor,
    X_explain: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
      shap_values (N,D)
      base_values (N,) (E[f(X)] no espaço logit)
    """
    import shap

    explainer = shap.GradientExplainer(model, X_background)
    sv = explainer.shap_values(X_explain)
    ev = getattr(explainer, "expected_value", None)

    if isinstance(sv, list):
        sv = sv[0]

    sv = ensure_2d_shap(sv, n=int(X_explain.shape[0]), d=int(X_explain.shape[1]))

    n = sv.shape[0]
    if ev is None:
        with torch.no_grad():
            base_scalar = float(model(X_background).detach().mean().cpu().item())
        base = np.full((n,), base_scalar, dtype=np.float64)
    else:
        ev = np.asarray(ev).reshape(-1)
        base_scalar = float(ev[0]) if ev.size else 0.0
        base = np.full((n,), base_scalar, dtype=np.float64)

    return sv.astype(np.float64), base


# ============================================================
# Salvamento (com x_* e shap_*)
# ============================================================

def save_local_csv_with_x(
    out_path: Path,
    df_src: pd.DataFrame,
    feats: List[str],
    sv: np.ndarray,
    base: np.ndarray,
    id_col: str = "instance_id",
):
    X = df_src[feats].to_numpy(dtype=np.float64, copy=True)
    fx = base + sv.sum(axis=1)   # logit
    prob = sigmoid_np(fx)        # sigmoid(logit)

    out = pd.DataFrame()
    if id_col in df_src.columns:
        out[id_col] = df_src[id_col].to_numpy()

    out["base_value"] = base
    out["f_x"] = fx
    out["prob"] = prob

    for j, c in enumerate(feats):
        out[f"x_{c}"] = X[:, j]
    for j, c in enumerate(feats):
        out[f"shap_{c}"] = sv[:, j]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] {out_path} | shape={out.shape}")


def save_global_csv(out_path: Path, feats: List[str], sv: np.ndarray):
    mean_abs = np.mean(np.abs(sv), axis=0)
    df = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] {out_path} | shape={df.shape}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-npz", required=True, help="Modelo: .npz posicional p0..pN OU .npz state_dict-like OU JSON TrainingData (best.model)")
    parser.add_argument("--data-test", required=True, help="CSV de teste do fold (data_test.csv)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--background-size", type=int, default=500)
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--id-col", default="instance_id")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # garantir shap instalado
    import shap  # noqa: F401

    df_test = pd.read_csv(args.data_test)
    feats = load_features(df_test, label_col=args.label_col, id_col=args.id_col)

    model_path = Path(args.model_npz)
    model = load_model_any(model_path, input_dim=len(feats), device=device)

    # background do próprio test (consistente)
    bg_n = min(args.background_size, len(df_test))
    bg_df = df_test.sample(n=bg_n, random_state=42) if len(df_test) > bg_n else df_test
    Xb = df_to_tensor(bg_df, feats, device=device)

    Xt = df_to_tensor(df_test, feats, device=device)

    print("[INFO] Calculando SHAP (GradientExplainer, logit) no data_test...")
    sv_test, base_test = compute_shap_gradient(model, Xb, Xt)

    save_local_csv_with_x(
        out_dir / "data_test_shap_values.csv",
        df_test, feats, sv_test, base_test, id_col=args.id_col
    )
    save_global_csv(out_dir / "shap_global.csv", feats, sv_test)

    print("\n[OK] Concluído.")


if __name__ == "__main__":
    main()
