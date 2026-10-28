import numpy as np
import pandas as pd

def datasets():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({'age': rng.normal(38, 9, 500).round(1), 'income': rng.normal(52000, 12000, 500).round(), 'channel': rng.choice(['web', 'store', 'app'], 500, p=[.5,.3,.2]), 'actual': rng.integers(0, 2, 500)})
    ref['prediction'] = np.where(rng.random(500) < .82, ref.actual, 1-ref.actual)
    cur = pd.DataFrame({'age': rng.normal(43, 9, 500).round(1), 'income': rng.normal(54000, 12000, 500).round(), 'channel': rng.choice(['web', 'store', 'app'], 500, p=[.3,.2,.5]), 'actual': rng.integers(0, 2, 500)})
    cur['prediction'] = np.where(rng.random(500) < .76, cur.actual, 1-cur.actual)
    return ref, cur
