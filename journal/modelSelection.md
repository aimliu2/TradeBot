# 2 LSTM layers + 1 dense hidden layer
# LSTM Unit perlayer : 32-128

# Input shape: (batch, lookback=150, features=11)
#        ↓
# LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(16) optional
#        ↓
# Dense(1) + sigmoid  + remap labels → threshold at 0.5 → predict -1 or 1
# 
# binary_crossentropy as loss function (binary classifier)

# Depth guidelines
Too shallow → underfits, can't capture temporal patterns
Too deep → overfits, vanishing gradients, slow training