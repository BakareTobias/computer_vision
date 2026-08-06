import pandas as pd
import numpy as np

df = pd.read_csv('ML_pipeline/ASL_alphabet/Z.csv')
TARGET_LEN = 27

def downsample_group(g, n):
    if len(g) <= n:
        return g  # nothing to do if already at/below target
    idx = np.linspace(0, len(g) - 1, n).round().astype(int)
    idx = np.unique(idx)  # guard against duplicate indices on rounding
    return g.iloc[idx]

parts = [downsample_group(g, TARGET_LEN) for _, g in df.groupby('clip_id')]
downsampled = pd.concat(parts, ignore_index=True)
downsampled.to_csv('Z_downsampled.csv', index=False)
#print(downsampled['clip_id'].value_counts().sort_index())