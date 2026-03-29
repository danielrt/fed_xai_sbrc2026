#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def run_capture(cmd: list[str], out_file: Path) -> None:
    """
    Executa cmd e salva stdout+stderr em out_file.
    """
    print("\n$ " + " ".join(cmd) + f"  > {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cv-root", default="cv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n-clients", type=int, default=4)

    # noniid params
    ap.add_argument("--alpha-size", type=float, default=0.8)
    ap.add_argument("--alpha-class", type=float, default=0.8)

    ap.add_argument("--variants", default="iid,noniid",
                    help="Ex: iid,noniid (default) ou iid")
    ap.add_argument("--client-code-dir", default="fed/client_code")

    ap.add_argument("--mount-root", default="mounted_clients",
                    help="Raiz onde montar por fold (ex: mounted_clients/fold_0/clients/<variant>/client<i>)")
    ap.add_argument("--clean", action="store_true",
                    help="Limpa montagem de cada variante por fold antes de montar")

    # report options
    ap.add_argument("--report-script", default="report_client_distributions.py")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--warn-min-size", type=int, default=200)
    args = ap.parse_args()

    cv_root = Path(args.cv_root)
    cv_root.mkdir(parents=True, exist_ok=True)

    mount_root = Path(args.mount_root)
    mount_root.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ("iid", "noniid"):
            raise ValueError(f"Variant inválida: {v}. Use iid e/ou noniid.")

    # 1) folds central
    run([
        "python3", "make_cv_folds_central.py",
        "--csv", args.csv,
        "--out-root", args.cv_root,
        "--k", str(args.k),
        "--seed", str(args.seed),
    ])

    # 2) dataset_fed (iid e noniid) por fold
    for fold in range(args.k):
        run([
            "python3", "make_cv_folds_client_datasets.py",
            "--cv-root", args.cv_root,
            "--fold", str(fold),
            "--n-clients", str(args.n_clients),
            "--seed", str(args.seed),
            "--alpha-size", str(args.alpha_size),
            "--alpha-class", str(args.alpha_class),
            "--variants", ",".join(variants),
        ])

        # 3) mount por fold e por variante
        dataset_fed = str(Path(args.cv_root) / f"fold_{fold}" / "dataset_fed")
        out_clients_root = Path(args.mount_root) / f"fold_{fold}" / "clients"

        for variant in variants:
            cmd = [
                "python3", "mount_clients.py",
                "--dataset-fed", dataset_fed,
                "--variant", variant,
                "--client-code-dir", args.client_code_dir,
                "--out-clients-root", str(out_clients_root),
            ]
            if args.clean:
                cmd.append("--clean-variant")
            run(cmd)

            # 4) report distributions -> mounted_clients/fold_i/clients/<variant>/distribution_report.txt
            clients_dir = out_clients_root / variant  # contém client0..clientN
            report_out = clients_dir / "distribution_report.txt"

            run_capture([
                "python3", args.report_script,
                "--clients-dir", str(clients_dir),
                "--show-test",
                "--label-col", args.label_col,
                "--warn-min-size", str(args.warn-min-size) if False else str(args.warn_min_size),
            ], report_out)

    print("\nConcluído: pipeline CV completo (IID + nonIID no mesmo k-fold, test=partitioned).")


if __name__ == "__main__":
    main()
