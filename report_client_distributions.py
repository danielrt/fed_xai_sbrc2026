#!/usr/bin/env python3
"""
report_client_distributions.py

Exemplos:
  python3 report_client_distributions.py --clients-dir fed/clients
  python3 report_client_distributions.py --clients-dir fed/clients --show-test
  python3 report_client_distributions.py --clients-dir fed/clients --label-col Label --warn-min-size 200

Saída:
- Para cada cliente:
    TRAIN: total, class_0, class_1, ... (contagem e %)
    (opcional) TEST idem
- Agregado ALL_CLIENTS:
    TRAIN_SUM e (opcional) TEST_SUM
- Alertas:
    * cliente single-class
    * cliente muito pequeno
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def find_client_dirs(clients_root: Path) -> List[Path]:
    if not clients_root.exists():
        raise FileNotFoundError(f"--clients-dir não existe: {clients_root}")
    dirs = [p for p in clients_root.iterdir() if p.is_dir() and p.name.startswith("client")]
    dirs = sorted(dirs, key=lambda p: int(p.name.replace("client", "")) if p.name.replace("client", "").isdigit() else p.name)
    return dirs


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


def label_counts(df: pd.DataFrame, label_col: str) -> Tuple[Dict[int, int], int]:
    if label_col not in df.columns:
        raise RuntimeError(f"CSV não tem coluna '{label_col}'. Colunas: {list(df.columns)[:30]} ...")
    y = df[label_col].to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    d = {int(c): int(n) for c, n in zip(classes, counts)}
    total = int(len(y))
    return d, total


def fmt_counts(counts: Dict[int, int], total: int, all_classes: List[int]) -> List[str]:
    lines = []
    for c in all_classes:
        n = counts.get(int(c), 0)
        pct = (100.0 * n / total) if total > 0 else 0.0
        lines.append(f"  - class_{c}: {n:6d}  ({pct:6.2f}%)")
    return lines


def is_single_class(counts: Dict[int, int]) -> bool:
    present = [c for c, n in counts.items() if n > 0]
    return len(present) <= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients-dir", required=True, help="Ex.: fed/clients")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--train-file", default="data_train.csv")
    ap.add_argument("--test-file", default="data_test.csv")
    ap.add_argument("--show-test", action="store_true")
    ap.add_argument("--warn-min-size", type=int, default=200, help="Avisa se TRAIN do cliente tiver menos que isso")
    args = ap.parse_args()

    clients_root = Path(args.clients_dir)
    client_dirs = find_client_dirs(clients_root)

    print(f"[INFO] Encontrados {len(client_dirs)} clientes em: {clients_root.resolve()}")

    # Descobrir conjunto de classes olhando todos os TRAINs
    all_classes_set = set()
    for cdir in client_dirs:
        df_tr = load_csv(cdir / args.train_file)
        counts, _ = label_counts(df_tr, args.label_col)
        all_classes_set.update(counts.keys())
    all_classes = sorted(list(all_classes_set))

    if not all_classes:
        raise RuntimeError("Não consegui detectar classes (all_classes vazio).")

    # Acumuladores
    train_sum_counts: Dict[int, int] = {c: 0 for c in all_classes}
    test_sum_counts: Dict[int, int] = {c: 0 for c in all_classes}
    train_sum_total = 0
    test_sum_total = 0

    # Alertas
    alerts = []

    for cdir in client_dirs:
        client_name = cdir.name

        # ---------- TRAIN ----------
        df_tr = load_csv(cdir / args.train_file)
        tr_counts, tr_total = label_counts(df_tr, args.label_col)

        print(f"\n==================== {client_name} | TRAIN ====================")
        print(f"Total: {tr_total}")
        for line in fmt_counts(tr_counts, tr_total, all_classes):
            print(line)

        # acumuladores
        train_sum_total += tr_total
        for c in all_classes:
            train_sum_counts[c] += tr_counts.get(c, 0)

        # alertas
        if tr_total < args.warn_min_size:
            alerts.append(f"[WARN] {client_name}: TRAIN muito pequeno (total={tr_total} < {args.warn_min_size})")
        if is_single_class(tr_counts):
            alerts.append(f"[WARN] {client_name}: TRAIN single-class (classes={[(k,v) for k,v in tr_counts.items() if v>0]})")

        # ---------- TEST ----------
        if args.show_test:
            df_te = load_csv(cdir / args.test_file)
            te_counts, te_total = label_counts(df_te, args.label_col)

            print(f"\n==================== {client_name} | TEST ====================")
            print(f"Total: {te_total}")
            for line in fmt_counts(te_counts, te_total, all_classes):
                print(line)

            test_sum_total += te_total
            for c in all_classes:
                test_sum_counts[c] += te_counts.get(c, 0)

            if is_single_class(te_counts):
                alerts.append(f"[WARN] {client_name}: TEST single-class (classes={[(k,v) for k,v in te_counts.items() if v>0]})")

    # ---------- ALL CLIENTS ----------
    print(f"\n==================== ALL_CLIENTS | TRAIN_SUM ====================")
    print(f"Total: {train_sum_total}")
    for line in fmt_counts(train_sum_counts, train_sum_total, all_classes):
        print(line)

    if args.show_test:
        print(f"\n==================== ALL_CLIENTS | TEST_SUM ====================")
        print(f"Total: {test_sum_total}")
        for line in fmt_counts(test_sum_counts, test_sum_total, all_classes):
            print(line)

    # ---------- ALERTS ----------
    if alerts:
        print("\n==================== ALERTS ====================")
        for a in alerts:
            print(a)
    else:
        print("\n[OK] Nenhum alerta: sem clientes single-class e tamanhos OK.")


if __name__ == "__main__":
    main()
