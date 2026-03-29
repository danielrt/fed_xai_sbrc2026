#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def load_ehms_df(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)

    network_features = [
        "Sport", "Dport",
        "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
        "SrcGap", "DstGap", "SIntPkt", "DIntPkt",
        "SIntPktAct", "DIntPktAct", "SrcJitter", "DstJitter",
        "sMaxPktSz", "dMaxPktSz", "sMinPktSz", "dMinPktSz",
        "Dur", "Trans", "TotPkts", "TotBytes",
        "Load", "Loss", "pLoss", "pSrcLoss", "pDstLoss", "Rate"
    ]
    bio_features = [
        "Temp", "SpO2", "Pulse_Rate",
        "SYS", "DIA", "Heart_rate",
        "Resp_Rate", "ST",
    ]

    features = [c for c in (network_features + bio_features) if c in df.columns]

    # manter só numéricas
    numeric_features = []
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)
        else:
            print(f"[WARN] Removendo coluna não numérica: {c}")
    features = numeric_features

    if "Label" not in df.columns:
        raise RuntimeError("A coluna 'Label' não existe no CSV.")

    df_model = df[features + ["Label"]].dropna().copy()

    # instance_id estável
    df_model["instance_id"] = df_model.index.astype(np.int64)

    cols = ["instance_id"] + features + ["Label"]
    df_model = df_model[cols]

    return df_model, features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-root", default="cv", help="Diretório raiz de saída (ex: cv/)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true", default=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df_raw, features = load_ehms_df(args.csv)
    y = df_raw["Label"].to_numpy()

    skf = StratifiedKFold(n_splits=args.k, shuffle=args.shuffle, random_state=args.seed)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df_raw, y)):
        fold_dir = out_root / f"fold_{fold_idx}"
        central_dir = fold_dir / "dataset_central"
        central_dir.mkdir(parents=True, exist_ok=True)

        df_train = df_raw.iloc[train_idx].copy()
        df_test = df_raw.iloc[test_idx].copy()

        # scaler fit somente no treino do fold
        scaler = StandardScaler()
        scaler.fit(df_train[features].to_numpy(dtype=np.float32))

        df_train[features] = scaler.transform(df_train[features].to_numpy(dtype=np.float32))
        df_test[features] = scaler.transform(df_test[features].to_numpy(dtype=np.float32))

        df_train.to_csv(central_dir / "data_train.csv", index=False)
        df_test.to_csv(central_dir / "data_test.csv", index=False)
        joblib.dump(scaler, central_dir / "scaler.joblib")

        # sanity checks (no leakage by instance_id)
        train_ids = set(df_train["instance_id"].tolist())
        test_ids = set(df_test["instance_id"].tolist())
        inter = train_ids.intersection(test_ids)
        if inter:
            raise RuntimeError(f"[LEAK] Fold {fold_idx}: instance_id em treino e teste! Ex: {list(inter)[:10]}")

        print(f"[OK] fold_{fold_idx}: central train={len(df_train)} test={len(df_test)} -> {central_dir}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
