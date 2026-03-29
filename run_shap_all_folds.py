#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def run_cmd(cmd: list[str], timeout: Optional[int] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout)


def _find_client_dirs(ma_dl_dir: Path) -> list[Path]:
    if not ma_dl_dir.exists():
        return []
    clients = [p for p in ma_dl_dir.iterdir() if p.is_dir() and p.name.startswith("client")]
    def key(p: Path):
        s = p.name.replace("client", "")
        return int(s) if s.isdigit() else s
    return sorted(clients, key=key)


def merge_ma_dl_all_clients_for_fold(
    ma_dl_dir: Path,
    out_name: str = "data_test_shap_values_all_clients.csv",
    add_client_id: bool = True,
    sort_by: str | None = "instance_id",
) -> None:
    """
    Lê:
      ma_dl_dir/client*/data_test_shap_values.csv
    Gera:
      ma_dl_dir/data_test_shap_values_all_clients.csv
    """
    client_dirs = _find_client_dirs(ma_dl_dir)
    if not client_dirs:
        print(f"[SKIP] Nenhum client*/ encontrado em: {ma_dl_dir}")
        return

    dfs = []
    ref_cols = None

    for cdir in client_dirs:
        f = cdir / "data_test_shap_values.csv"
        if not f.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {f}")

        df = pd.read_csv(f)

        if add_client_id:
            df.insert(0, "client_id", cdir.name)

        cols = list(df.columns)
        if ref_cols is None:
            ref_cols = cols
        else:
            # Se só mudou ordem, realinha; se mudou conjunto, erro.
            if cols != ref_cols:
                if set(cols) != set(ref_cols):
                    missing = sorted(list(set(ref_cols) - set(cols)))
                    extra = sorted(list(set(cols) - set(ref_cols)))
                    raise RuntimeError(
                        f"[ERRO] Colunas diferentes em {f}\n"
                        f"missing={missing}\n"
                        f"extra={extra}\n"
                    )
                df = df.reindex(columns=ref_cols)

        dfs.append(df)

    out = pd.concat(dfs, axis=0, ignore_index=True)

    if sort_by is not None and sort_by in out.columns:
        out = out.sort_values(by=sort_by, kind="mergesort").reset_index(drop=True)

    out_path = ma_dl_dir / out_name
    out.to_csv(out_path, index=False)
    print(f"[OK] {out_path} | rows={len(out)} | cols={len(out.columns)} | clients={len(client_dirs)}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--k", type=int, required=True, help="Número de folds (ex: 5)")
    ap.add_argument("--cv-root", default="cv", help="Raiz dos folds (cv/)")
    ap.add_argument("--results-root", default="training_results", help="Raiz dos resultados de treino (training_results/)")
    ap.add_argument("--out-root", default="shap_values", help="Raiz onde salvar os SHAP (shap_values/)")
    ap.add_argument("--compute-script", default="compute_shap_values.py", help="Script compute_shap_values.py atualizado")

    ap.add_argument("--background-size", type=int, default=500)
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--id-col", default="instance_id")

    ap.add_argument("--timeout", type=int, default=0, help="Timeout por execução de SHAP em segundos (0=sem timeout)")
    ap.add_argument("--skip-noniid", action="store_true", help="Pula MA_DC_noniid e MA_DL_noniid")
    ap.add_argument("--skip-ma-dl", action="store_true", help="Pula MA_DL_* (por cliente) e merges all_clients")

    # onde ficam os testes locais por cliente (dataset_fed dentro do cv)
    ap.add_argument(
        "--dataset-fed-subdir",
        default="dataset_fed",
        help="Subdir dentro de cv/fold_i/ que contém iid/client*/data_test.csv, noniid/client*/data_test.csv (default: dataset_fed)",
    )
    ap.add_argument("--n-clients", type=int, default=4)

    # merge options
    ap.add_argument("--merge-add-client-id", action="store_true", help="Adiciona coluna client_id no merged all_clients")
    ap.add_argument("--merge-sort-by", default="instance_id", help="Coluna para ordenar no merged (default instance_id). Use '' para não ordenar.")
    ap.add_argument("--merged-name", default="data_test_shap_values_all_clients.csv")

    args = ap.parse_args()

    k = int(args.k)
    cv_root = Path(args.cv_root)
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)
    compute_script = Path(args.compute_script)

    if not compute_script.exists():
        raise FileNotFoundError(f"compute-script não encontrado: {compute_script}")

    timeout = None if int(args.timeout) <= 0 else int(args.timeout)

    out_root.mkdir(parents=True, exist_ok=True)

    sort_by = args.merge_sort_by.strip() or None

    for i in range(k):
        central_ds_dir = cv_root / f"fold_{i}" / "dataset_central"
        data_test_global = central_ds_dir / "data_test.csv"
        if not data_test_global.exists():
            raise FileNotFoundError(f"data_test.csv não encontrado: {data_test_global}")

        # modelos
        central_model = results_root / f"k_fold{i}" / "central" / "best_model.npz"
        iid_model = results_root / f"k_fold{i}" / "iid" / "best.model.npz"
        noniid_model = results_root / f"k_fold{i}" / "noniid" / "best.model.npz"

        # output fold
        fold_out = out_root / f"k_fold{i}"
        (fold_out / "MC_DC").mkdir(parents=True, exist_ok=True)
        (fold_out / "MA_DC_iid").mkdir(parents=True, exist_ok=True)
        (fold_out / "MA_DC_noniid").mkdir(parents=True, exist_ok=True)

        (fold_out / "MA_DL_iid").mkdir(parents=True, exist_ok=True)
        (fold_out / "MA_DL_noniid").mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 100)
        print(f"[FOLD {i}] global_test={data_test_global}")
        print("#" * 100)

        # ---------------- MC_DC (central em DC) ----------------
        if not central_model.exists():
            raise FileNotFoundError(f"Modelo central não encontrado: {central_model}")

        run_cmd([
            sys.executable, str(compute_script),
            "--model-npz", str(central_model),
            "--data-test", str(data_test_global),
            "--out-dir", str(fold_out / "MC_DC"),
            "--background-size", str(args.background_size),
            "--label-col", args.label_col,
            "--id-col", args.id_col,
        ], timeout=timeout)

        # ---------------- MA_DC_iid (agregado iid em DC) ----------------
        if not iid_model.exists():
            raise FileNotFoundError(f"Modelo iid não encontrado: {iid_model}")

        run_cmd([
            sys.executable, str(compute_script),
            "--model-npz", str(iid_model),
            "--data-test", str(data_test_global),
            "--out-dir", str(fold_out / "MA_DC_iid"),
            "--background-size", str(args.background_size),
            "--label-col", args.label_col,
            "--id-col", args.id_col,
        ], timeout=timeout)

        # ---------------- MA_DC_noniid (agregado noniid em DC) ----------------
        if not args.skip_noniid:
            if not noniid_model.exists():
                raise FileNotFoundError(f"Modelo noniid não encontrado: {noniid_model}")

            run_cmd([
                sys.executable, str(compute_script),
                "--model-npz", str(noniid_model),
                "--data-test", str(data_test_global),
                "--out-dir", str(fold_out / "MA_DC_noniid"),
                "--background-size", str(args.background_size),
                "--label-col", args.label_col,
                "--id-col", args.id_col,
            ], timeout=timeout)
        else:
            print("[SKIP] MA_DC_noniid")

        # ---------------- MA_DL_* (agregado em cada DL de cada client) ----------------
        if not args.skip_ma_dl:
            fed_dir = cv_root / f"fold_{i}" / args.dataset_fed_subdir

            # IID clients
            iid_clients_root = fed_dir / "iid"
            for cj in range(args.n_clients):
                client_test = iid_clients_root / f"client{cj}" / "data_test.csv"
                if not client_test.exists():
                    raise FileNotFoundError(f"[ERRO] Não achei test do client{iid_clients_root.name}/client{cj}: {client_test}")

                out_client = fold_out / "MA_DL_iid" / f"client{cj}"
                out_client.mkdir(parents=True, exist_ok=True)

                run_cmd([
                    sys.executable, str(compute_script),
                    "--model-npz", str(iid_model),              # modelo agregado iid
                    "--data-test", str(client_test),            # test local do cliente
                    "--out-dir", str(out_client),
                    "--background-size", str(args.background_size),
                    "--label-col", args.label_col,
                    "--id-col", args.id_col,
                ], timeout=timeout)

            # NONIID clients
            if not args.skip_noniid:
                noniid_clients_root = fed_dir / "noniid"
                for cj in range(args.n_clients):
                    client_test = noniid_clients_root / f"client{cj}" / "data_test.csv"
                    if not client_test.exists():
                        raise FileNotFoundError(f"[ERRO] Não achei test do client{noniid_clients_root.name}/client{cj}: {client_test}")

                    out_client = fold_out / "MA_DL_noniid" / f"client{cj}"
                    out_client.mkdir(parents=True, exist_ok=True)

                    run_cmd([
                        sys.executable, str(compute_script),
                        "--model-npz", str(noniid_model),         # modelo agregado noniid
                        "--data-test", str(client_test),          # test local do cliente
                        "--out-dir", str(out_client),
                        "--background-size", str(args.background_size),
                        "--label-col", args.label_col,
                        "--id-col", args.id_col,
                    ], timeout=timeout)
            else:
                print("[SKIP] MA_DL_noniid")

            # -------- MERGE all_clients ao final do fold --------
            print("\n" + "-" * 80)
            print(f"[MERGE] FOLD {i}: gerando data_test_shap_values_all_clients.csv em MA_DL_*")
            print("-" * 80)

            merge_ma_dl_all_clients_for_fold(
                ma_dl_dir=(fold_out / "MA_DL_iid"),
                out_name=args.merged_name,
                add_client_id=bool(args.merge_add_client_id),
                sort_by=sort_by,
            )

            if not args.skip_noniid:
                merge_ma_dl_all_clients_for_fold(
                    ma_dl_dir=(fold_out / "MA_DL_noniid"),
                    out_name=args.merged_name,
                    add_client_id=bool(args.merge_add_client_id),
                    sort_by=sort_by,
                )

        else:
            print("[SKIP] MA_DL_* (por cliente) + merge all_clients")

    print("\n[DONE] SHAP concluído para todos os folds.")
    print(f"[OUT] {out_root.resolve()}")


if __name__ == "__main__":
    main()
