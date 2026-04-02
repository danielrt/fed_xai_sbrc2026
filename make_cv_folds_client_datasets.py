#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# Helpers de validação
# ============================================================

def ensure_disjoint_by_id(df_parts: List[pd.DataFrame], id_col: str = "instance_id") -> None:
    seen = set()
    for i, dfi in enumerate(df_parts):
        ids = dfi[id_col].tolist()
        dup = [x for x in ids if x in seen]
        if dup:
            raise RuntimeError(f"[LEAK] instance_id duplicado entre clientes. Ex: client{i}: {dup[:10]}")
        seen.update(ids)


def ensure_union_equals(
    df_parts: List[pd.DataFrame],
    df_ref: pd.DataFrame,
    id_col: str = "instance_id",
    name: str = "split"
) -> None:
    ref_ids = set(df_ref[id_col].tolist())
    parts_ids = set()
    for p in df_parts:
        parts_ids.update(p[id_col].tolist())

    if parts_ids != ref_ids:
        missing = list(ref_ids - parts_ids)[:10]
        extra = list(parts_ids - ref_ids)[:10]
        raise RuntimeError(
            f"[ERROR] União dos ids ({name}) != referência.\n"
            f"  - missing (ex): {missing}\n"
            f"  - extra   (ex): {extra}\n"
            f"  - ref={len(ref_ids)} parts={len(parts_ids)}"
        )


def class_counts(df: pd.DataFrame, label_col: str) -> Dict[int, int]:
    y = df[label_col].to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): int(k) for c, k in zip(classes, counts)}


# ============================================================
# Split IID (treino e teste separadamente)
# ============================================================

def split_iid_stratified(df, n_clients, label_col, seed, verbose=False):
    import numpy as np
    import pandas as pd

    parts = [[] for _ in range(n_clients)]

    for label_value, group in df.groupby(label_col):
        group = group.sample(frac=1.0, random_state=seed)
        index_chunks = np.array_split(group.index.to_numpy(), n_clients)

        for client_idx, idx_chunk in enumerate(index_chunks):
            if len(idx_chunk) == 0:
                continue

            chunk_df = df.loc[idx_chunk].copy()
            if not chunk_df.empty:
                parts[client_idx].append(chunk_df)

        if verbose:
            sizes = [len(idx_chunk) for idx_chunk in index_chunks]
            print(f"[IID] classe={label_value} -> distribuição por cliente: {sizes}")

    client_dfs = []
    for client_idx in range(n_clients):
        valid_parts = [
            part for part in parts[client_idx]
            if isinstance(part, pd.DataFrame) and not part.empty
        ]

        if not valid_parts:
            raise RuntimeError(
                f"Cliente {client_idx} ficou sem amostras no split IID. "
                f"Tente reduzir n_clients ou revisar a distribuição das classes."
            )

        client_df = pd.concat(valid_parts, axis=0)
        client_df = client_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        if verbose:
            print(f"[IID] cliente {client_idx}: {len(client_df)} amostras")

        client_dfs.append(client_df)

    return client_dfs


# ============================================================
# Split NONIID Dirichlet + "pós-correção" (sem single-class e sem muito pequeno)
# ============================================================

def _dirichlet_initial_split_indices(
    df: pd.DataFrame,
    n_clients: int,
    label_col: str,
    seed: int,
    alpha_size: float,
    alpha_class: float,
) -> Tuple[np.ndarray, List[List[int]], np.ndarray]:
    """
    Retorna:
    - classes (np.ndarray)
    - client_idx (lista de lista de indices int)
    - sizes (np.ndarray) tamanhos alvo por cliente
    """
    rng = np.random.default_rng(seed)

    y = df[label_col].to_numpy()
    classes, _ = np.unique(y, return_counts=True)
    n_classes = len(classes)

    if n_classes < 2:
        # tudo uma classe só: devolve divisão qualquer (ainda disjunta)
        idx = np.array(df.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        splits = np.array_split(idx, n_clients)
        client_idx = [list(map(int, s)) for s in splits]
        sizes = np.array([len(s) for s in splits], dtype=int)
        return classes, client_idx, sizes

    # tamanhos por cliente (Dirichlet)
    size_props = rng.dirichlet(alpha=[alpha_size] * n_clients)
    sizes = np.floor(size_props * len(df)).astype(int)
    while sizes.sum() < len(df):
        sizes[np.argmax(size_props)] += 1

    # distribuição de classes por cliente (Dirichlet)
    class_props = rng.dirichlet(alpha=[alpha_class] * n_classes, size=n_clients)

    # IMPORTANTE: copy=True para evitar "array is read-only"
    idx_by_class = {
        int(c): np.array(df[df[label_col] == c].index.to_numpy(), copy=True)
        for c in classes
    }
    for c in idx_by_class:
        rng.shuffle(idx_by_class[c])

    # alocação desejada (cliente x classe)
    alloc = np.zeros((n_clients, n_classes), dtype=int)
    for i in range(n_clients):
        raw = class_props[i] * sizes[i]
        alloc[i] = np.floor(raw).astype(int)
        while alloc[i].sum() < sizes[i]:
            alloc[i, np.argmax(class_props[i])] += 1

    # alocar respeitando disponibilidade
    parts_idx = [[] for _ in range(n_clients)]
    cursor = {int(c): 0 for c in classes}

    for i in range(n_clients):
        for j, c in enumerate(classes):
            c = int(c)
            need = int(alloc[i, j])
            avail = len(idx_by_class[c]) - cursor[c]
            take = min(need, avail)
            if take > 0:
                sel = idx_by_class[c][cursor[c]: cursor[c] + take]
                cursor[c] += take
                parts_idx[i].append(sel)

    used = (
        np.concatenate([np.concatenate(p) for p in parts_idx if p], axis=0)
        if any(parts_idx)
        else np.array([], dtype=int)
    )

    remaining = np.array(df.index.difference(used).to_numpy(), copy=True)
    rng.shuffle(remaining)

    # completar tamanhos com remanescentes
    for i in range(n_clients):
        cur = np.concatenate(parts_idx[i], axis=0) if parts_idx[i] else np.array([], dtype=int)
        missing = int(sizes[i] - len(cur))
        if missing > 0 and len(remaining) > 0:
            take = min(missing, len(remaining))
            parts_idx[i].append(remaining[:take])
            remaining = remaining[take:]

    client_idx: List[List[int]] = []
    for i in range(n_clients):
        idx = np.concatenate(parts_idx[i], axis=0) if parts_idx[i] else np.array([], dtype=int)
        client_idx.append(list(map(int, idx)))

    return classes, client_idx, sizes


def _counts_for_client(
    df: pd.DataFrame,
    idxs: List[int],
    label_col: str,
    classes: np.ndarray
) -> Dict[int, int]:
    out = {int(c): 0 for c in classes}
    if not idxs:
        return out
    y = df.loc[idxs, label_col].to_numpy()
    for c in classes:
        out[int(c)] = int((y == c).sum())
    return out


def _move_samples(
    df: pd.DataFrame,
    donor_idxs: List[int],
    recv_idxs: List[int],
    label_col: str,
    want_class: Optional[int],
    n_move: int,
    rng: np.random.Generator
) -> int:
    """
    Move até n_move amostras do donor -> receiver.
    Se want_class != None, tenta mover apenas dessa classe.
    Retorna quantas foram movidas.
    """
    if not donor_idxs or n_move <= 0:
        return 0

    if want_class is None:
        candidates = donor_idxs.copy()
    else:
        y = df.loc[donor_idxs, label_col].to_numpy()
        candidates = [donor_idxs[i] for i in range(len(donor_idxs)) if int(y[i]) == int(want_class)]

    if not candidates:
        return 0

    rng.shuffle(candidates)
    take = min(n_move, len(candidates))
    moved = candidates[:take]
    moved_set = set(moved)

    # remove do donor
    donor_idxs[:] = [x for x in donor_idxs if x not in moved_set]
    # adiciona no receiver
    recv_idxs.extend(moved)
    rng.shuffle(recv_idxs)
    return take


def _enforce_constraints(
    df: pd.DataFrame,
    client_idx: List[List[int]],
    label_col: str,
    seed: int,
    min_total: int,
    min_per_class: int,
) -> None:
    """
    Pós-correção IN-PLACE em client_idx garantindo:
    - cada cliente com pelo menos min_total amostras (quando possível)
    - cada cliente com pelo menos min_per_class de cada classe (quando possível e se global tem >=2 classes)

    Importante: mantém disjunção e preserva união (apenas move amostras entre clientes).
    """
    rng = np.random.default_rng(seed)

    y = df[label_col].to_numpy()
    classes, global_counts = np.unique(y, return_counts=True)
    classes = classes.astype(int)
    n_classes = len(classes)

    # Se o dataset global não tem 2 classes, não há como impor min_per_class em 2 classes.
    if n_classes < 2:
        # ainda pode tentar min_total
        pass

    # limites para não travar
    max_iters = 50 * len(client_idx)

    for _ in range(max_iters):
        changed = False

        counts = [_counts_for_client(df, client_idx[i], label_col, classes) for i in range(len(client_idx))]
        sizes = [len(client_idx[i]) for i in range(len(client_idx))]

        # 1) Corrigir clientes muito pequenos
        for i in range(len(client_idx)):
            if sizes[i] >= min_total:
                continue

            need = min_total - sizes[i]
            if need <= 0:
                continue

            # escolhe doador: maior tamanho que ainda possa doar sem cair abaixo min_total
            donors = sorted(range(len(client_idx)), key=lambda d: sizes[d], reverse=True)
            for d in donors:
                if d == i:
                    continue
                if sizes[d] <= min_total:
                    continue

                can_give = sizes[d] - min_total
                take = min(need, can_give)
                if take <= 0:
                    continue

                # move aleatoriamente (qualquer classe)
                moved = _move_samples(df, client_idx[d], client_idx[i], label_col, want_class=None, n_move=take, rng=rng)
                if moved > 0:
                    changed = True
                    sizes[i] += moved
                    sizes[d] -= moved
                    need -= moved
                if need <= 0:
                    break

        # recomputa depois de mover tamanhos
        counts = [_counts_for_client(df, client_idx[i], label_col, classes) for i in range(len(client_idx))]
        sizes = [len(client_idx[i]) for i in range(len(client_idx))]

        # 2) Corrigir single-class / falta de classe (quando global tem >=2 classes)
        if n_classes >= 2 and min_per_class > 0:
            for i in range(len(client_idx)):
                missing = [c for c in classes if counts[i].get(int(c), 0) < min_per_class]
                if not missing:
                    continue

                for mc in missing:
                    # encontrar doador com sobra dessa classe
                    donors = []
                    for d in range(len(client_idx)):
                        if d == i:
                            continue
                        # doador precisa ter sobra da classe e não pode ficar "quebrado" (>=min_per_class)
                        if counts[d].get(int(mc), 0) > (min_per_class):
                            donors.append(d)

                    # prioriza doadores maiores
                    donors = sorted(donors, key=lambda d: sizes[d], reverse=True)

                    for d in donors:
                        # tentar mover o mínimo necessário dessa classe
                        need = min_per_class - counts[i].get(int(mc), 0)
                        if need <= 0:
                            break

                        # mas o doador não pode cair abaixo min_total
                        # e não pode cair abaixo min_per_class para essa classe
                        can_give_by_total = max(0, sizes[d] - min_total)
                        can_give_by_class = max(0, counts[d].get(int(mc), 0) - min_per_class)
                        can_give = min(can_give_by_total, can_give_by_class)
                        if can_give <= 0:
                            continue

                        take = min(need, can_give)
                        moved = _move_samples(df, client_idx[d], client_idx[i], label_col, want_class=int(mc), n_move=take, rng=rng)
                        if moved > 0:
                            changed = True
                            sizes[i] += moved
                            sizes[d] -= moved
                            counts[i][int(mc)] = counts[i].get(int(mc), 0) + moved
                            counts[d][int(mc)] = counts[d].get(int(mc), 0) - moved

        if not changed:
            break

    # Não levanta erro automaticamente aqui porque pode ser impossível satisfazer (ex: classe rara demais)
    # A validação final decide se deve falhar ou apenas warn.


def split_noniid_dirichlet_with_constraints(
    df: pd.DataFrame,
    n_clients: int,
    label_col: str,
    seed: int,
    alpha_size: float,
    alpha_class: float,
    min_total: int,
    min_per_class: int,
    strict: bool = True,
) -> List[pd.DataFrame]:
    """
    NONIID via Dirichlet + pós-correção para:
    - evitar clientes muito pequenos
    - evitar clientes single-class (quando possível)

    strict=True: se não conseguir satisfazer, falha com erro.
    strict=False: apenas segue e imprime warn no caller (via checagens externas).
    """
    classes, client_idx, _sizes = _dirichlet_initial_split_indices(
        df, n_clients, label_col, seed, alpha_size, alpha_class
    )

    # pós-correção
    _enforce_constraints(
        df=df,
        client_idx=client_idx,
        label_col=label_col,
        seed=seed + 999,  # seed diferente pra correção
        min_total=min_total,
        min_per_class=min_per_class,
    )

    # construir dfs
    out = []
    for i in range(n_clients):
        out.append(df.loc[client_idx[i]].sample(frac=1.0, random_state=seed))

    # validações (se strict)
    if strict:
        # disjunção e união
        ensure_disjoint_by_id(out, id_col="instance_id")
        ensure_union_equals(out, df, id_col="instance_id", name="noniid")

        # constraints básicas
        y = df[label_col].to_numpy()
        classes_glob = np.unique(y).astype(int).tolist()

        for i, dfi in enumerate(out):
            # min_total
            if len(dfi) < min_total:
                raise RuntimeError(f"[STRICT] client{i} ficou muito pequeno: {len(dfi)} < {min_total}")

            # min_per_class se dataset global tem >=2 classes
            if len(classes_glob) >= 2 and min_per_class > 0:
                cc = class_counts(dfi, label_col)
                for c in classes_glob:
                    if cc.get(int(c), 0) < min_per_class:
                        raise RuntimeError(
                            f"[STRICT] client{i} ficou com classe_{c} insuficiente: "
                            f"{cc.get(int(c), 0)} < {min_per_class}"
                        )

    return out


# ============================================================
# Escrita: agora com TEST particionado também
# ============================================================

def write_variant(
    fold_dir: Path,
    variant: str,
    train_parts: List[pd.DataFrame],
    test_parts: List[pd.DataFrame],
    n_clients: int,
) -> None:
    """
    Cria:
      fold_dir/dataset_fed/<variant>/client<i>/data_train.csv
      fold_dir/dataset_fed/<variant>/client<i>/data_test.csv
    """
    fed_variant_dir = fold_dir / "dataset_fed" / variant
    if fed_variant_dir.exists():
        shutil.rmtree(fed_variant_dir)
    fed_variant_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_clients):
        cdir = fed_variant_dir / f"client{i}"
        cdir.mkdir(parents=True, exist_ok=True)
        train_parts[i].to_csv(cdir / "data_train.csv", index=False)
        test_parts[i].to_csv(cdir / "data_test.csv", index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", default="cv", help="Raiz com fold_*/dataset_central/")
    ap.add_argument("--fold", type=int, default=None, help="Se definido, processa apenas este fold")
    ap.add_argument("--n-clients", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    # noniid params
    ap.add_argument("--alpha-size", type=float, default=0.8)
    ap.add_argument("--alpha-class", type=float, default=0.8)

    # constraints noniid (para TREINO e TESTE)
    ap.add_argument("--min-train-total", type=int, default=200, help="Min amostras no TRAIN por cliente (noniid).")
    ap.add_argument("--min-test-total", type=int, default=40, help="Min amostras no TEST por cliente (noniid).")
    ap.add_argument("--min-train-per-class", type=int, default=20, help="Min por classe no TRAIN por cliente (noniid).")
    ap.add_argument("--min-test-per-class", type=int, default=5, help="Min por classe no TEST por cliente (noniid).")

    ap.add_argument("--noniid-strict", action="store_true",
                    help="Se ligado, falha se não conseguir satisfazer constraints do noniid.")
    # quais variantes gerar
    ap.add_argument("--variants", default="iid,noniid",
                    help="Lista separada por vírgula. Ex: iid,noniid (default) ou apenas iid")
    args = ap.parse_args()

    cv_root = Path(args.cv_root)
    folds = sorted([p for p in cv_root.glob("fold_*") if p.is_dir()])

    if args.fold is not None:
        folds = [cv_root / f"fold_{args.fold}"]
        if not folds[0].exists():
            raise FileNotFoundError(f"Fold não encontrado: {folds[0]}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ("iid", "noniid"):
            raise ValueError(f"Variant inválida: {v}. Use iid e/ou noniid.")

    for fold_dir in folds:
        central_dir = fold_dir / "dataset_central"
        if not central_dir.exists():
            raise FileNotFoundError(f"dataset_central não existe em: {central_dir}")

        df_train = pd.read_csv(central_dir / "data_train.csv")
        df_test = pd.read_csv(central_dir / "data_test.csv")

        label_col = "Label"
        id_col = "instance_id"

        # sanity: central train/test não podem compartilhar instance_id
        inter = set(df_train[id_col]).intersection(set(df_test[id_col]))
        if inter:
            raise RuntimeError(f"[LEAK] {fold_dir.name}: central train/test compartilham ids. Ex: {list(inter)[:10]}")

        # ============================================================
        # IID: split do TRAIN e split do TEST (separados, mas ambos disjuntos e completos)
        # ============================================================
        if "iid" in variants:
            train_parts_iid = split_iid_stratified(df_train, args.n_clients, label_col, args.seed)
            test_parts_iid = split_iid_stratified(df_test, args.n_clients, label_col, args.seed + 12345)

            # validações: disjunção + união exata
            ensure_disjoint_by_id(train_parts_iid, id_col=id_col)
            ensure_disjoint_by_id(test_parts_iid, id_col=id_col)
            ensure_union_equals(train_parts_iid, df_train, id_col=id_col, name=f"{fold_dir.name}:iid_train")
            ensure_union_equals(test_parts_iid, df_test, id_col=id_col, name=f"{fold_dir.name}:iid_test")

            write_variant(fold_dir, "iid", train_parts_iid, test_parts_iid, args.n_clients)

        # ============================================================
        # NONIID: split do TRAIN e split do TEST, ambos com constraints
        # ============================================================
        if "noniid" in variants:
            train_parts_noniid = split_noniid_dirichlet_with_constraints(
                df_train, args.n_clients, label_col, args.seed,
                alpha_size=args.alpha_size,
                alpha_class=args.alpha_class,
                min_total=args.min_train_total,
                min_per_class=args.min_train_per_class,
                strict=bool(args.noniid_strict),
            )

            test_parts_noniid = split_noniid_dirichlet_with_constraints(
                df_test, args.n_clients, label_col, args.seed + 7777,
                alpha_size=args.alpha_size,
                alpha_class=args.alpha_class,
                min_total=args.min_test_total,
                min_per_class=args.min_test_per_class,
                strict=bool(args.noniid_strict),
            )

            # validações: disjunção + união exata
            ensure_disjoint_by_id(train_parts_noniid, id_col=id_col)
            ensure_disjoint_by_id(test_parts_noniid, id_col=id_col)
            ensure_union_equals(train_parts_noniid, df_train, id_col=id_col, name=f"{fold_dir.name}:noniid_train")
            ensure_union_equals(test_parts_noniid, df_test, id_col=id_col, name=f"{fold_dir.name}:noniid_test")

            write_variant(fold_dir, "noniid", train_parts_noniid, test_parts_noniid, args.n_clients)

        # checagens finais: clientes train ⊂ central train e clientes test ⊂ central test
        train_ids = set(df_train[id_col].tolist())
        test_ids = set(df_test[id_col].tolist())
        for variant in variants:
            fed_variant_dir = fold_dir / "dataset_fed" / variant
            for i in range(args.n_clients):
                cdir = fed_variant_dir / f"client{i}"
                ctrain = pd.read_csv(cdir / "data_train.csv")
                ctest = pd.read_csv(cdir / "data_test.csv")
                if not set(ctrain[id_col]).issubset(train_ids):
                    raise RuntimeError(f"[ERROR] {fold_dir.name}/{variant}/client{i}: train fora do central train.")
                if not set(ctest[id_col]).issubset(test_ids):
                    raise RuntimeError(f"[ERROR] {fold_dir.name}/{variant}/client{i}: test fora do central test.")

        print(f"[OK] {fold_dir.name}: dataset_fed gerado em {fold_dir / 'dataset_fed'} | variants={variants} | test=partitioned")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
