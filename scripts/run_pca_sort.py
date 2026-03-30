# run_pca_sort.py
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA

# ===== Paths ======
FEATURE_DIR = Path.home() / "Microscopy/Cohort1_TPH2/features"
SORT_OUT = FEATURE_DIR / "sorting"
SORT_OUT.mkdir(exist_ok=True)

feature_files = sorted(FEATURE_DIR.glob("features_*.csv"))
all_results = []

for fpath in feature_files:
    rat_name = fpath.stem.replace("features_", "")
    print(f"\nSorting {rat_name}")

    # Load per-rat feature CSV
    df = pd.read_csv(fpath, header=None)

    section_names = df.iloc[:,0].values   # first column = image stem
    X = df.iloc[:,1:].values              # rest = feature vector

    # PCA embedding
    pca = PCA(n_components=1)
    coords = pca.fit_transform(X).flatten()

    order = coords.argsort()
    ordered_names = [section_names[i] for i in order]

    out_df = pd.DataFrame({
        "rat": rat_name,
        "image": ordered_names,
        "predicted_order": range(1, len(ordered_names)+1),
        "pc1_value": coords[order]
    })

    per_rat_csv = SORT_OUT / f"predicted_order_{rat_name}.csv"
    out_df.to_csv(per_rat_csv, index=False)
    print(f"  Saved per-rat predicted order: {per_rat_csv}")

    all_results.append(out_df)

# Combined master CSV
if all_results:
    final = pd.concat(all_results, ignore_index=True)
    master_csv = SORT_OUT / "predicted_order.csv"
    final.to_csv(master_csv, index=False)
    print(f"\nCombined master predicted_order.csv saved: {master_csv}")

print("\nPCA-based sorting complete.")