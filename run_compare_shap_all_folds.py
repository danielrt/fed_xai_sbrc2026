#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_cmd(cmd: list[str], timeout: Optional[int] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--k", type=int, required=True, help="Número de folds")
    ap.add_argument("--shap-root", default="shap_values", help="Raiz shap_values/")
    ap.add_argument("--out-root", required=True, help="Pasta raiz de saída (ex: output/)")
    ap.add_argument("--compare-script", default="compare_shap_instancewise.py")
    ap.add_argument("--timeout", type=int, default=0)

    ap.add_argument("--rbo-p", type=float, default=0.9)
    ap.add_argument("--topk", default="5,10,15")

    args = ap.parse_args()

    k = int(args.k)
    shap_root = Path(args.shap_root)
    out_root = Path(args.out_root)
    compare_script = Path(args.compare_script)

    if not compare_script.exists():
        raise FileNotFoundError(f"compare-script não encontrado: {compare_script}")
    if not shap_root.exists():
        raise FileNotFoundError(f"shap-root não existe: {shap_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    timeout = None if int(args.timeout) <= 0 else int(args.timeout)

    # helpers
    def fold_dir(i: int) -> Path:
        return shap_root / f"k_fold{i}"

    def file_in(case_dir: Path) -> Path:
        # sempre data_test_shap_values.csv
        return case_dir / "data_test_shap_values.csv"

    def file_in_dl(case_dir: Path) -> Path:
        # MA_DL_* deve usar o merge all_clients
        # (você disse que seu run_shap_all_folds.py já garante esse arquivo)
        return case_dir / "data_test_shap_values_all_clients.csv"

    comparisons = [
        ("MA_DC_iid_vs_MC_DC",
         lambda fd: file_in(fd / "MA_DC_iid"),
         lambda fd: file_in(fd / "MC_DC")),

        ("MA_DC_noniid_vs_MC_DC",
         lambda fd: file_in(fd / "MA_DC_noniid"),
         lambda fd: file_in(fd / "MC_DC")),

        ("MA_DL_iid_vs_MA_DC_iid",
         lambda fd: file_in_dl(fd / "MA_DL_iid"),
         lambda fd: file_in(fd / "MA_DC_iid")),

        ("MA_DL_noniid_vs_MA_DC_noniid",
         lambda fd: file_in_dl(fd / "MA_DL_noniid"),
         lambda fd: file_in(fd / "MA_DC_noniid")),
    ]

    for i in range(k):
        fd = fold_dir(i)
        if not fd.exists():
            raise FileNotFoundError(f"Fold não encontrado: {fd}")

        for comp_name, a_path_fn, b_path_fn in comparisons:
            a_path = a_path_fn(fd)
            b_path = b_path_fn(fd)

            if not a_path.exists():
                raise FileNotFoundError(f"[{comp_name}] arquivo A não encontrado: {a_path}")
            if not b_path.exists():
                raise FileNotFoundError(f"[{comp_name}] arquivo B não encontrado: {b_path}")

            out_dir = out_root / comp_name / f"fold{i}"
            out_dir.mkdir(parents=True, exist_ok=True)

            run_cmd([
                sys.executable, str(compare_script),
                "--a", str(a_path),
                "--b", str(b_path),
                "--out-dir", str(out_dir),
                "--rbo-p", str(args.rbo_p),
                "--topk", str(args.topk),
            ], timeout=timeout)

    print("\n[DONE] Comparações concluídas.")
    print(f"[OUT] {out_root.resolve()}")


if __name__ == "__main__":
    main()
