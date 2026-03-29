#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_client_code(client_code_dir: Path, dst_client_dir: Path) -> None:
    if not client_code_dir.exists():
        raise FileNotFoundError(f"client_code dir não existe: {client_code_dir}")

    for item in client_code_dir.iterdir():
        if item.is_file() and (item.suffix == ".py" or item.name == "client_requirements.txt"):
            shutil.copy2(item, dst_client_dir / item.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-fed", required=True,
                    help="Caminho para dataset_fed (ex: cv/fold_0/dataset_fed)")
    ap.add_argument("--variant", choices=["iid", "noniid"], required=True,
                    help="Qual variante montar (iid ou noniid)")
    ap.add_argument("--client-code-dir", default="fed/client_code")
    ap.add_argument("--out-clients-root", default="fed/clients",
                    help="Raiz onde montar (ex: fed/clients). O script cria /<variant>/client<i>")
    ap.add_argument("--clean-variant", action="store_true",
                    help="Remove apenas out-clients-root/<variant> antes de montar")
    args = ap.parse_args()

    dataset_fed = Path(args.dataset_fed)
    variant = args.variant
    client_code_dir = Path(args.client_code_dir)
    out_root = Path(args.out_clients_root)

    src_variant_dir = dataset_fed / variant
    if not src_variant_dir.exists():
        raise FileNotFoundError(f"Variant dir não existe: {src_variant_dir}")

    clients = sorted([p for p in src_variant_dir.glob("client*") if p.is_dir()])
    if not clients:
        raise RuntimeError(f"Nenhum client* encontrado em: {src_variant_dir}")

    out_variant_dir = out_root / variant
    if args.clean_variant and out_variant_dir.exists():
        shutil.rmtree(out_variant_dir)

    out_variant_dir.mkdir(parents=True, exist_ok=True)

    for csrc in clients:
        cdst = out_variant_dir / csrc.name
        cdst.mkdir(parents=True, exist_ok=True)

        shutil.copy2(csrc / "data_train.csv", cdst / "data_train.csv")
        shutil.copy2(csrc / "data_test.csv", cdst / "data_test.csv")

        copy_client_code(client_code_dir, cdst)

        print(f"[OK] Montado {cdst}")

    print(f"\nConcluído. Clientes montados em: {out_variant_dir}")


if __name__ == "__main__":
    main()
