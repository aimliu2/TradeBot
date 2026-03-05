# Training Notes

## 2026-03-04 — BTCUSD 15m LSTM Model (v1)

### Model Architecture
```
Input: (batch, lookback=200, features=11)
  ↓
LSTM(64) → Dropout(0.3)
  ↓
LSTM(32) → Dropout(0.3) → last timestep
  ↓
Dense(16, ReLU)
  ↓
Dense(1, sigmoid) → threshold 0.5 → predict -1 or 1
```

### Config
| Setting | Value |
|---|---|
| Lookback | 200 bars |
| Batch size | 256 |
| Epochs | 30 |
| Learning rate | 1e-3 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Loss | BCELoss (binary cross-entropy) |
| Device | auto (cuda / cpu) |

### Data
- Source: `data/mlData/trainData/202603-BTCUSD-15m-train.jsonl`
- Labels remapped: -1 → 0, 1 → 1
- Streaming: line-by-line JSONL, sliding window sequences on-the-fly
- ~70,229 sequences after applying lookback window

### Features (11)
`isSTR`, `isITR`, `isLTR`, `barsSinceITR`, `barsSinceLTR`,
`hiDTK_LTR`, `lowDTK_LTR`, `closeDTK_LTR`, `hiDTK_ITR`, `lowDTK_ITR`, `closeDTK_ITR`

### Design Decisions
- **LSTM over Transformer**: ~70K rows is moderate; Transformer needs 200K+
- **sigmoid + BCELoss over tanh + MSE**: log penalty forces sharp predictions, avoids mushy outputs near 0
- **2 LSTM layers**: sweet spot for this data size; 3+ layers risk overfitting
- **Dropout after LSTM output** (not recurrent_dropout): faster training, sufficient regularization
- **Dense(16) as translator layer**: compresses LSTM representation before binary decision
- **ReLU in dense**: shallow layer, low dying neuron risk, simplest default

### Output
- Model saved to: `data/mlData/202603-BTCUSD-15m-lstm.pt`
- Script: `src/dataScience/x.py`

### Notes
- Python 3.14 may not have a PyTorch wheel yet — downgrade to 3.11/3.12 if `pip install torch` fails
- Run: `python src/dataScience/x.py`
