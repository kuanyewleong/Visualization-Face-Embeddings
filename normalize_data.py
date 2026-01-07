import pandas as pd

# Load CSV (no header)
df = pd.read_csv("Result_3D.csv", header=None)

# Columns to normalize: B, C, D → indices 1, 2, 3
cols = [1, 2, 3]

# Min-Max normalization
df[cols] = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())

# Round to 6 decimal places
df[cols] = df[cols].round(6)

# Save result
df.to_csv("normalized.csv", index=False, header=False)
