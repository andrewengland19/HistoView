import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

HOME = Path.home()
WORK = HOME / "SectionSorter"

FEATURES = WORK / "features/features.csv"
ORDER = WORK / "sorting/predicted_order.csv"
OUT = WORK / "diagnostics"
OUT.mkdir(exist_ok=True)

feat = pd.read_csv(FEATURES)
order = pd.read_csv(ORDER)

feat["rat"] = feat["image"].str.split("_").str[0]

for rat in order["rat"].unique():

    print(f"heatmap: {rat}", flush=True)

    rat_order = order[order["rat"] == rat]
    rat_feat = feat[feat["rat"] == rat]

    rat_feat = rat_feat.set_index("image").loc[rat_order["image"]]

    X = rat_feat.drop(columns=["rat"]).values

    sim = cosine_similarity(X)

    plt.figure(figsize=(6,6))
    plt.imshow(sim, cmap="viridis")
    plt.title(f"Section similarity – {rat}")
    plt.colorbar(label="cosine similarity")
    plt.xlabel("section index")
    plt.ylabel("section index")

    plt.tight_layout()
    plt.savefig(OUT / f"similarity_heatmap_{rat}.png")
    plt.close()

print("Heatmaps generated.")