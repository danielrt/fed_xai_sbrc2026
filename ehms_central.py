#!/usr/bin/env python3
"""
ehms_central.py (MODIFICADO)

Agora aceita:
- --dataset-dir: pasta contendo data_train.csv e data_test.csv do fold
- --out-dir: pasta onde salvar best_model.npz e metrics_best.txt (e checkpoints)

Exemplo:
python3 ehms_central.py --dataset-dir cv/fold_0/dataset_central --out-dir training_results/k_fold0/central
"""

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configurações gerais
# ============================================================

RANDOM_STATE = 42
BATCH_SIZE = 256
N_EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4

PATIENCE = 10
MIN_DELTA = 0.0
SAVE_EVERY = 5

LABEL_COL = "Label"
ID_COL = "instance_id"


# ============================================================
# Dataset e modelo
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device, threshold: float = 0.5, eps: float = 1e-12) -> Dict:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(y_batch.cpu().numpy())

    all_probs = np.vstack(all_probs).ravel()
    all_labels = np.vstack(all_labels).ravel().astype("int64")
    preds = (all_probs >= threshold).astype("int64")

    cm = confusion_matrix(all_labels, preds, labels=[0, 1]).astype(np.int64)

    total = float(np.sum(cm))
    acc = float(np.trace(cm) / (total + eps))

    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0).astype(np.float64) - tp
    fn = np.sum(cm, axis=1).astype(np.float64) - tp

    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    f1_per_class = (2.0 * precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + eps)

    support = np.sum(cm, axis=1).astype(np.float64)
    w = support / (np.sum(support) + eps)

    p_macro = float(np.mean(precision_per_class))
    r_macro = float(np.mean(recall_per_class))
    f1_macro = float(np.mean(f1_per_class))

    p_weighted = float(np.sum(precision_per_class * w))
    r_weighted = float(np.sum(recall_per_class * w))
    f1_weighted = float(np.sum(f1_per_class * w))

    tp_sum = float(np.sum(tp))
    p_micro = float(tp_sum / (total + eps))
    r_micro = float(tp_sum / (total + eps))
    f1_micro = float((2.0 * p_micro * r_micro) / (p_micro + r_micro + eps))

    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = float("nan")

    return {
        "confusion_matrix": cm.tolist(),

        "accuracy": acc,
        "precision": p_macro,
        "recall": r_macro,
        "f1": f1_macro,

        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f1_weighted,

        "micro_precision": p_micro,
        "micro_recall": r_micro,
        "micro_f1": f1_micro,

        "support_per_class": support.astype(int).tolist(),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),

        "auc": auc,
    }


def save_model_npz(model: nn.Module, out_path: Path) -> None:
    params = list(model.parameters())
    payload = {f"p{i}": p.detach().cpu().numpy() for i, p in enumerate(params)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)


def format_metrics_report(metrics: Dict, client_name: str = "GLOBAL") -> str:
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    sp = metrics["support_per_class"]
    ppc = metrics["precision_per_class"]
    rpc = metrics["recall_per_class"]
    fpc = metrics["f1_per_class"]

    lines = []
    lines.append("============================================================")
    lines.append(f"METRICS REPORT — CLIENT: {client_name}")
    lines.append("============================================================\n")

    lines.append("[ GLOBAL METRICS ]")
    lines.append(f"- accuracy            : {metrics['accuracy']:.4f}")
    lines.append(f"- precision (macro)   : {metrics['precision']:.4f}")
    lines.append(f"- recall (macro)      : {metrics['recall']:.4f}")
    lines.append(f"- f1 (macro)          : {metrics['f1']:.4f}")
    lines.append(f"- weighted_precision  : {metrics['weighted_precision']:.4f}")
    lines.append(f"- weighted_recall     : {metrics['weighted_recall']:.4f}")
    lines.append(f"- weighted_f1         : {metrics['weighted_f1']:.4f}")
    lines.append(f"- micro_precision     : {metrics['micro_precision']:.4f}")
    lines.append(f"- micro_recall        : {metrics['micro_recall']:.4f}")
    lines.append(f"- micro_f1            : {metrics['micro_f1']:.4f}")
    lines.append(f"- auc                 : {metrics['auc']:.4f}")

    lines.append("\n[ PER-CLASS METRICS ]")
    lines.append("Support:")
    for i, s in enumerate(sp):
        lines.append(f"  - class_{i}        : {int(s)}")

    lines.append("\nPrecision:")
    for i, v in enumerate(ppc):
        lines.append(f"  - class_{i}        : {float(v):.4f}")

    lines.append("\nRecall:")
    for i, v in enumerate(rpc):
        lines.append(f"  - class_{i}        : {float(v):.4f}")

    lines.append("\nF1-score:")
    for i, v in enumerate(fpc):
        lines.append(f"  - class_{i}        : {float(v):.4f}")

    lines.append("\n[ CONFUSION MATRIX ]")
    if cm.shape == (2, 2):
        lines.append("                   pred_0    pred_1")
        lines.append(f"true_0              {cm[0,0]:>5d}    {cm[0,1]:>5d}")
        lines.append(f"true_1              {cm[1,0]:>5d}    {cm[1,1]:>5d}")

    lines.append("\n\n============================================================\n")
    return "\n".join(lines)


def save_metrics_txt(metrics: Dict, out_path: Path, client_name: str = "GLOBAL") -> None:
    out_path.write_text(format_metrics_report(metrics, client_name=client_name), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Pasta do fold com data_train.csv e data_test.csv")
    ap.add_argument("--out-dir", required=True, help="Pasta de saída para resultados do fold (central)")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = dataset_dir / "data_train.csv"
    test_csv = dataset_dir / "data_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Esperado: {train_csv} e {test_csv}")

    # Seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Carregar dados
    print(f"Lendo treino: {train_csv}")
    df_train = pd.read_csv(train_csv)
    print(f"Lendo teste : {test_csv}")
    df_test = pd.read_csv(test_csv)

    excluded = {LABEL_COL, ID_COL}
    features = [c for c in df_train.columns if c not in excluded]
    features_test = [c for c in df_test.columns if c not in excluded]
    if features_test != features:
        raise RuntimeError("Features de treino e teste não batem.\n"
                           f"Treino: {features}\nTeste: {features_test}")

    X_train = df_train[features].values.astype("float32")
    y_train = df_train[LABEL_COL].values.astype("int64")
    X_test = df_test[features].values.astype("float32")
    y_test = df_test[LABEL_COL].values.astype("int64")

    train_ds = TabularDataset(X_train, y_train)
    test_ds = TabularDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsando device:", device)

    model = EHMSANN(input_dim=X_train.shape[1]).to(device)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight_value = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    print(f"\nClasse positiva (1): {n_pos}")
    print(f"Classe negativa (0): {n_neg}")
    print(f"pos_weight usado na BCEWithLogitsLoss: {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1_macro = -np.inf
    best_state_dict = None
    no_improve = 0

    # onde salvar checkpoints
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n================= INICIANDO TREINAMENTO CENTRAL =================")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, test_loader, device)

        f1_macro = metrics["f1"]
        improved = (f1_macro > best_f1_macro + MIN_DELTA)

        if improved:
            best_f1_macro = f1_macro
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                  f"acc={metrics['accuracy']:.4f} | auc={metrics['auc']:.4f} | "
                  f"f1_macro={metrics['f1']:.4f} | no_improve={no_improve}/{args.patience}")

        if epoch % args.save_every == 0:
            save_model_npz(model, ckpt_dir / f"model_{epoch}.npz")
            save_metrics_txt(metrics, ckpt_dir / f"metrics_{epoch}.txt", client_name="GLOBAL")

        if no_improve >= args.patience:
            print(f"\nEarly stopping: melhor F1 macro={best_f1_macro:.4f} (parou na época {epoch}).")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_model_path = out_dir / "best_model.npz"
    best_metrics_path = out_dir / "metrics_best.txt"

    save_model_npz(model, best_model_path)
    best_metrics_eval = evaluate(model, test_loader, device)
    save_metrics_txt(best_metrics_eval, best_metrics_path, client_name="GLOBAL")

    print(f"\n[OK] Best model salvo em: {best_model_path}")
    print(f"[OK] Métricas do best model em: {best_metrics_path}")
    print(f"[OK] Saída do fold em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
