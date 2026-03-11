import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from pathlib import Path

# Load embeddings
embeddings = torch.load(
    '/data/mwu11/LLM_affinity/embedding/output/qwen3_embedding_8b_itc_embeddings.pt',
    map_location='cpu',
    weights_only=True,
)

# Load and filter tabular data
data = pd.read_csv(
    '/data/mwu11/LLM_affinity/BindingDB/non_null_spr_itc_structured_description_subset.csv',
    low_memory=False,
)
success_path = Path('/data/mwu11/boltz/BindingDB/boltz_results_yaml_affinity_input/success.txt')
if success_path.exists():
    with open(success_path, 'r') as f:
        success_ids = set(line.strip() for line in f if line.strip())
    data = data[data['BindingDB Reactant_set_id'].astype(str).isin(success_ids)].reset_index(drop=True)

# Exclude Kd with inequality signs
data = data[~data['Kd (nM)'].astype(str).str.contains('[<>]', regex=True)]
# Convert Kd to pKd
data['y'] = -np.log10(data['Kd (nM)'].astype(float) * 1e-9)

# Match embeddings with filtered data
ids_with_embeddings = []
emb_list = []
y_list = []
for _, row in data.iterrows():
    rid = int(row['BindingDB Reactant_set_id'])
    if rid in embeddings:
        ids_with_embeddings.append(rid)
        emb_list.append(embeddings[rid].numpy())
        y_list.append(row['y'])

X = np.stack(emb_list)
y = np.array(y_list)
print(f"Matched samples: {len(y)}")

# UMAP reduction
reducer = umap.UMAP(n_components=2, random_state=42)
X_2d = reducer.fit_transform(X)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('pKd')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP of ITC Assay Context Embeddings (Qwen3-8B)')
plt.tight_layout()
plt.savefig('/data/mwu11/LLM_affinity/embedding/output/umap_itc_embeddings.png', dpi=150)
print("Saved to embedding/output/umap_itc_embeddings.png")
plt.show()
