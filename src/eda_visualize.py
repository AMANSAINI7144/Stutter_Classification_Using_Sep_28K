import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional: use UMAP if installed
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

BASE = Path.cwd()
FEATURES_DIR = BASE / "features"
RESULTS_DIR = BASE / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
for d in (FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["label_block","label_prolong","label_soundrep","label_wordrep","label_interjection","label_no_stutter"]

def load_mapping(split):
    path = FEATURES_DIR / split / f"mapping_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def plot_label_distribution():
    # load all splits
    dfs = {}
    for s in ("train","val","test"):
        try:
            dfs[s] = load_mapping(s)
        except FileNotFoundError:
            print(f"[WARN] mapping for {s} missing, skipping")
    # counts per label
    counts = {}
    for s, df in dfs.items():
        counts[s] = df[LABEL_COLS].apply(lambda c: (c>0).sum()).to_dict()
    counts_df = pd.DataFrame(counts).T
    counts_df.to_csv(TABLE_DIR / "label_counts.csv")
    # plot stacked bars (presence counts)
    plt.figure(figsize=(10,5))
    counts_df.plot(kind='bar', stacked=False)
    plt.title("Label counts per split (presence > 0)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_counts_per_split.png")
    plt.close()
    print("[INFO] Saved label_counts_per_split.png")

def label_cooccurrence_heatmap(split="train"):
    df = load_mapping(split)
    # binary matrix
    binmat = (df[LABEL_COLS] > 0).astype(int)
    co = binmat.T.dot(binmat)  # co-occurrence counts
    co.to_csv(TABLE_DIR / f"cooccurrence_{split}.csv")
    plt.figure(figsize=(8,6))
    sns.heatmap(co, annot=True, fmt="d", cmap="viridis")
    plt.title(f"Label co-occurrence matrix ({split})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"cooccurrence_{split}.png")
    plt.close()
    print(f"[INFO] Saved cooccurrence_{split}.png")

def sample_and_save_spectrograms(split="train", samples_per_label=4):
    df = load_mapping(split)
    # for each label, pick samples where that label > 0 and save spectrogram images
    for lab in LABEL_COLS:
        ids = df[df[lab] > 0]["ClipId"].values
        if len(ids) == 0:
            continue
        sel = ids[:samples_per_label]  # deterministic pick; you can randomize
        lab_dir = FIG_DIR / f"spect_examples_{split}" / lab
        lab_dir.mkdir(parents=True, exist_ok=True)
        for clipid in sel:
            feat_path = FEATURES_DIR / split / f"{clipid}.npy"
            if not feat_path.exists():
                continue
            feat = np.load(feat_path)  # shape (n_mels, T)
            plt.figure(figsize=(6,3))
            plt.imshow(feat, aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{split}:{lab}:{clipid}")
            plt.xlabel("Time frames")
            plt.ylabel("Mel bins")
            plt.tight_layout()
            outp = lab_dir / f"{clipid}.png"
            plt.savefig(outp)
            plt.close()
    print(f"[INFO] Saved example spectrograms to {FIG_DIR / ('spect_examples_'+split)}")

def average_spectrogram_per_label(split="train"):
    df = load_mapping(split)
    for lab in LABEL_COLS:
        ids = df[df[lab] > 0]["ClipId"].values
        if len(ids) == 0:
            continue
        # accumulate mean
        sum_spec = None
        cnt = 0
        for clipid in tqdm(ids, desc=f"avg spec {lab}"):
            p = FEATURES_DIR / split / f"{clipid}.npy"
            if not p.exists(): 
                continue
            arr = np.load(p)
            if sum_spec is None:
                sum_spec = np.zeros_like(arr, dtype=np.float64)
            # ensure same shape
            sum_spec += arr
            cnt += 1
            if cnt >= 2000:
                break
        if cnt == 0:
            continue
        avg = sum_spec / cnt
        plt.figure(figsize=(6,3))
        plt.imshow(avg, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f"Average log-mel for {lab} (n={cnt})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"avg_spec_{split}_{lab}.png")
        plt.close()
    print(f"[INFO] Saved average spectrograms to {FIG_DIR}")

def rms_and_loudness_histogram(split="train"):
    df = load_mapping(split)
    rms_list = []
    group_list = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="compute rms"):
        clipid = str(row.ClipId)
        feat_path = FEATURES_DIR / split / f"{clipid}.npy"
        if not feat_path.exists():
            continue
        arr = np.load(feat_path)  # log-mel in dB
        # approximate RMS by converting from dB back to power then average
        linear = 10 ** (arr / 10.0)
        rms = np.sqrt(np.mean(linear))
        # sum all label values to determine stutter vs fluent
        total_labels = (row.label_block + row.label_prolong + row.label_soundrep +
                        row.label_wordrep + row.label_interjection)
        group = "stutter" if total_labels > 0 else "fluent"
        rms_list.append(rms)
        group_list.append(group)

    outdf = pd.DataFrame({"rms": rms_list, "group": group_list})
    outdf.to_csv(TABLE_DIR / f"rms_{split}.csv", index=False)

    plt.figure(figsize=(6, 4))
    sns.boxplot(x="group", y="rms", data=outdf)
    plt.title(f"RMS Energy Distribution ({split})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"rms_box_{split}.png")
    plt.close()
    print(f"[INFO] Saved RMS distributions: {FIG_DIR / f'rms_box_{split}.png'}")

def embedding_tsne(split="train", n_samples=2000, method="pca_tsne"):
    df = load_mapping(split)
    # sample subset for speed
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=42).reset_index(drop=True)
    feats = []
    labels = []
    for clipid, *labs in tqdm(df[["ClipId"]+LABEL_COLS].itertuples(index=False), total=len(df), desc="load feats"):
        p = FEATURES_DIR / split / f"{clipid}.npy"
        if not p.exists():
            continue
        arr = np.load(p)
        feats.append(arr.flatten())
        labels.append(int(sum(labs) > 0))  # binary: stutter vs fluent
    feats = np.array(feats)
    # PCA -> TSNE
    pca = PCA(n_components=50, random_state=42)
    pcs = pca.fit_transform(feats)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    X2 = tsne.fit_transform(pcs)
    plt.figure(figsize=(7,6))
    plt.scatter(X2[:,0], X2[:,1], c=labels, cmap="coolwarm", s=6, alpha=0.7)
    plt.title(f"t-SNE embeddings ({split})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"tsne_{split}.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved t-SNE plot to {FIG_DIR / f'tsne_{split}.png'}")

def per_episode_counts(split="train"):
    df = load_mapping(split)
    # attempt to extract episode key from rel_path filename stem (part before final _clipid)
    def stem_to_key(path):
        name = Path(path).name
        stem = os.path.splitext(name)[0]
        # remove trailing _<clipid>
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
        return stem
    df["episode_key"] = df["feature_path" if "feature_path" in df.columns else "rel_path"].apply(lambda x: stem_to_key(x))
    counts = df["episode_key"].value_counts().reset_index()
    counts.columns = ["episode_key", "count"]
    counts.to_csv(TABLE_DIR / f"per_episode_counts_{split}.csv", index=False)
    plt.figure(figsize=(10,4))
    counts.head(30).plot(kind='bar', x="episode_key", y="count", legend=False)
    plt.title(f"Top 30 episodes by clip count ({split})")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"per_episode_top30_{split}.png")
    plt.close()
    print(f"[INFO] Saved per-episode counts plot to {FIG_DIR}/per_episode_top30_{split}.png")

# --------------- main ---------------
def main():
    print("[START] EDA & Visualization")
    plot_label_distribution()
    label_cooccurrence_heatmap("train")
    sample_and_save_spectrograms("train", samples_per_label=4)
    average_spectrogram_per_label("train")
    rms_and_loudness_histogram("train")
    embedding_tsne("train", n_samples=1500)
    per_episode_counts("train")
    print("[DONE] EDA files are in results/figures and tables in results/tables")

if __name__ == "__main__":
    main()