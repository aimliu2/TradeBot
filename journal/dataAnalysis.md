# Data Analysis Journal

## 2026-03-04 — PACF Analysis: Sequence Length

### Setup
- Dataset: BTCUSD 15m, 70,929 rows
- Tool: `statsmodels.tsa.stattools.pacf`, method=ywm, max_lags=250
- 95% confidence band: ±0.0074 (tight due to large N)

### Results

```
Y (target)       last significant lag: 216  (54.0 hrs)
closeDTK_LTR     last significant lag: 250  (62.5 hrs)
closeDTK_ITR     last significant lag: 249  (62.2 hrs)
barsSinceLTR     last significant lag: 240  (60.0 hrs)
```

### Interpretation

| Feature | Root cause | Informationally meaningful? |
|---|---|---|
| `barsSinceLTR` lag 240 | LTR counter increments every bar until reset (avg 60 bars) | Structural — expected |
| `closeDTK_LTR/ITR` lag ~250 | DTK is nearly constant while key level doesn't update (98.4% of bars) | Artifact of level persistence, not 250 bars of independent signal |
| `Y` lag 216 | Market regime (trending/ranging) persists ~2–3 days | Genuine — most meaningful result |

### Key finding
- **100-bar window was too short** — misses the regime context visible in Y's 216-bar autocorrelation.
- DTK PACF tails (~250) are structural artifacts (same value repeated during level stasis), not 250 independent signal bars.
- Y's 216-bar cutoff is the binding constraint for genuine predictive information.

### Recommended sequence length: 150–200 bars

| Option | Rationale |
|---|---|
| 100 | Too short — misses Y regime signal |
| 150–200 | Covers Y autocorrelation, one full LTR cycle, avoids DTK artifact padding |
| 250 | Over-parameterized — DTK contribution is mostly repetition |

### Next step
Empirical sweep `[100, 150, 200, 250]` on validation loss to confirm optimal sequence length experimentally.
