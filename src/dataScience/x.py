# pip index versions torch 2>&1 | head -5
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Config ───────────────────────────────────────────────────────────────────
LOOKBACK    = 200
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 1e-3
DROPOUT     = 0.3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "isSTR", "isITR", "isLTR",
    "barsSinceITR", "barsSinceLTR",
    "hiDTK_LTR", "lowDTK_LTR", "closeDTK_LTR",
    "hiDTK_ITR",  "lowDTK_ITR",  "closeDTK_ITR",
]
LABEL_COL = "Y"

DATA_PATH  = Path(__file__).parents[2] / "data" / "mlData" / "trainData" / "202603-BTCUSD-15m-train.jsonl"
MODEL_PATH = Path(__file__).parents[2] / "data" / "mlData" / "202603-BTCUSD-15m-lstm.pt"


# ── Data ─────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Stream JSONL line-by-line → numpy arrays. Remaps Y: -1→0, 1→1."""
    X_rows, y_rows = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            X_rows.append([row[c] for c in FEATURE_COLS])
            y_rows.append(0.0 if row[LABEL_COL] == -1 else 1.0)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


class SlidingWindowDataset(Dataset):
    """Creates (lookback, features) sequences on-the-fly — no pre-allocation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        self.X       = X
        self.y       = y
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.X) - self.lookback

    def __getitem__(self, idx: int):
        x_seq = torch.from_numpy(self.X[idx : idx + self.lookback])  # (lookback, features)
        label  = torch.tensor(self.y[idx + self.lookback])
        return x_seq, label


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = 11, hidden1: int = 64, hidden2: int = 32,
                 dense: int = 16, dropout: float = DROPOUT):
        super().__init__()
        self.lstm1  = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1  = nn.Dropout(dropout)
        self.lstm2  = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2  = nn.Dropout(dropout)
        self.dense  = nn.Linear(hidden2, dense)
        self.relu   = nn.ReLU()
        self.out    = nn.Linear(dense, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, features)
        out, _ = self.lstm1(x)          # (batch, lookback, 64)
        out     = self.drop1(out)
        out, _ = self.lstm2(out)         # (batch, lookback, 32)
        out     = out[:, -1, :]          # last timestep → (batch, 32)
        out     = self.drop2(out)
        out     = self.relu(self.dense(out))        # (batch, 16)
        return torch.sigmoid(self.out(out)).squeeze(1)  # (batch,)


# ── Training ──────────────────────────────────────────────────────────────────
def train() -> None:
    print(f"Device : {DEVICE}")
    print(f"Loading : {DATA_PATH.name}")

    X, y = load_jsonl(DATA_PATH)
    print(f"Rows    : {len(X):,}  |  Features: {X.shape[1]}")

    dataset = SlidingWindowDataset(X, y, LOOKBACK)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=0,
    )

    model     = LSTMClassifier().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params  : {n_params:,}")
    print(f"Seqs    : {len(dataset):,}  |  Batches/epoch: {len(loader)}")
    print(f"Epochs  : {EPOCHS}  |  Lookback: {LOOKBACK}  |  Batch: {BATCH_SIZE}")
    print("-" * 55)
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Acc':>7}  {'TP%':>6}  {'TN%':>6}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        tp = tn = fp = fn = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            predicted   = (preds >= 0.5).float()

            tp += ((predicted == 1) & (y_batch == 1)).sum().item()
            tn += ((predicted == 0) & (y_batch == 0)).sum().item()
            fp += ((predicted == 1) & (y_batch == 0)).sum().item()
            fn += ((predicted == 0) & (y_batch == 1)).sum().item()

        total    = tp + tn + fp + fn
        avg_loss = total_loss / total
        acc      = (tp + tn) / total * 100
        tp_rate  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        tn_rate  = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0

        print(f"{epoch:>6d}  {avg_loss:>8.4f}  {acc:>6.2f}%  {tp_rate:>5.1f}%  {tn_rate:>5.1f}%")

    print("-" * 55)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved   : {MODEL_PATH.name}")


if __name__ == "__main__":
    train()
