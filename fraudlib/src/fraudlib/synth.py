import numpy as np, pandas as pd

def make_transactions(n=200_000, fraud_rate=0.002, seed=42):
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate
    amount = np.where(is_fraud,
                      rng.lognormal(5.5, 1.2, n),   # fraud skews larger
                      rng.lognormal(3.5, 1.0, n))
    return pd.DataFrame({
        "customer_id": rng.integers(1, 5000, n),
        "amount": amount.round(2),
        "ts": pd.Timestamp("2026-01-01")
              + pd.to_timedelta(rng.integers(0, 90*24*3600, n), unit="s"),
        "channel": rng.choice(["web", "app", "pos"], n, p=[.3, .4, .3]),
        "is_fraud": is_fraud.astype(int),
    })