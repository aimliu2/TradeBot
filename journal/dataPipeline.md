# Data pipeline
raw rows (70,929) → scale → sliding windows → split into train/val sequences

```python
# Production Pipeline
scaler = joblib.load('scaler.pkl')   # fixed mean/std from training

# fetch 1000 rows for performance/buffering reasons
raw_1000 = fetch_latest_candles(n=1000)

# slice the window you need
window = raw_1000[-100:]             # (100, 12)

# scale only what you feed the model
window_scaled = scaler.transform(window[feature_cols])  # (100, 12)

prediction = model.predict(window_scaled[np.newaxis])   # (1, 100, 12)
```

# Key rules:
- Always use saved_scaler.transform() — **never refit on live data**
- Scale only the 100 rows you feed the model (transform is row-independent)
- Refitting on live data = distribution mismatch with training



# Distribution shift
- If production mean/std has shifted from training, model performance degrades.
- This is called covariate shift — model sees normalized values outside its training range.

# Why DTK features are resilient:
- closeDTK_LTR = (close - LTRkeylv) / LTRkeylv — already domain-normalized
- removes absolute price level (BTC $50k vs $100k)
- relative distance from structure is statistically stable across time
- barsSinceITR/LTR is most vulnerable — shifts with market regime (trending vs ranging)

# Mitigation:
- Retrain periodically (e.g. monthly), refresh scaler
- Monitor barsSince* features first — early warning signal for drift

# Simple drift monitor
```python
live_mean = window[feature_cols].mean()
train_mean = pd.Series(scaler.mean_, index=feature_cols)

drift = (live_mean - train_mean).abs() / train_mean.abs()
if (drift > 0.5).any():
    print("Warning: feature distribution has shifted >50%")
```