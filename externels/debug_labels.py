import pandas as pd
import numpy as np

path = "features/train/mapping_train.csv"  # change to val/test if needed
df = pd.read_csv(path)

cols = ["label_block", "label_prolong", "label_soundrep", "label_wordrep", "label_interjection", "label_no_stutter"]
labels = df[cols].values

print("Shape:", labels.shape)
print("Min:", np.min(labels), "Max:", np.max(labels))
print("\nUnique values per column:")
for c in cols:
    print(f"{c}: {df[c].unique()[:10]}")
