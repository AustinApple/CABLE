"""Embed structured descriptions from ITC/SPR extraction results.

Usage:
    python embedding/embed_descriptions.py --model pubmedbert --assay_type itc
    python embedding/embed_descriptions.py --model qwen3      --assay_type spr
    python embedding/embed_descriptions.py --model qwen3      --assay_type all
"""

import argparse
import copy
import glob
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = "/data/mwu11/LLM_affinity/qwen_test"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

ASSAY_DIRS = {
    "spr": {
        "dir": os.path.join(BASE_DIR, "spr_extraction_results_two_step_bindingdb"),
        "label": "Surface Plasmon Resonance (SPR)",
    },
    "itc": {
        "dir": os.path.join(BASE_DIR, "itc_extraction_results_two_step_bindingdb"),
        "label": "Isothermal Titration Calorimetry (ITC)",
    },
}

MODEL_CONFIG = {
    "qwen3": {
        "name": "Qwen/Qwen3-Embedding-8B",
        "output_prefix": "qwen3_embedding_8b",
        "kwargs": {"device_map": "auto"},
        "tokenizer_kwargs": {"padding_side": "left"},
    },
    "pubmedbert": {
        "name": "NeuML/pubmedbert-base-embeddings",
        "output_prefix": "pubmedbert",
        "kwargs": {},
        "tokenizer_kwargs": {},
    },
}


def load_entries(assay_type):
    """Load entries with non-null structured_description for the given assay type."""
    assay_types = [assay_type] if assay_type != "all" else list(ASSAY_DIRS.keys())
    entries = []

    for atype in assay_types:
        cfg = ASSAY_DIRS[atype]
        for filepath in sorted(glob.glob(os.path.join(cfg["dir"], "*.json"))):
            with open(filepath) as f:
                data = json.load(f)
            for _key, entry in data.items():
                sd = entry.get("structured_description")
                if sd is None:
                    continue
                # Augment with assay_type and binding_affinity_type
                sd_augmented = copy.deepcopy(sd)
                sd_augmented["assay_type"] = cfg["label"]
                affinity_data = entry.get("affinity_data")
                if affinity_data and affinity_data.get("type"):
                    sd_augmented["binding_affinity_type"] = affinity_data["type"]
                entries.append({
                    "pmid": entry.get("pmid"),
                    "reactant_set_id": entry.get("reactant_set_id"),
                    "assay_type": atype,
                    "text": json.dumps(sd_augmented),
                })

    return entries


def embed(texts, model_key, batch_size=32):
    """Load model and encode texts."""
    cfg = MODEL_CONFIG[model_key]
    print(f"Loading model: {cfg['name']}")
    model = SentenceTransformer(
        cfg["name"],
        model_kwargs=cfg["kwargs"],
        tokenizer_kwargs=cfg["tokenizer_kwargs"],
    )
    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings)


def save_results(entries, embeddings, model_key, assay_type):
    """Save embeddings as a PyTorch dict {reactant_set_id: embedding} per assay type."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = MODEL_CONFIG[model_key]["output_prefix"]

    assay_types = [assay_type] if assay_type != "all" else list(ASSAY_DIRS.keys())
    for atype in assay_types:
        indices = [i for i, e in enumerate(entries) if e["assay_type"] == atype]
        if not indices:
            continue
        emb_dict = {
            entries[i]["reactant_set_id"]: torch.tensor(embeddings[i])
            for i in indices
        }
        out_path = os.path.join(OUTPUT_DIR, f"{prefix}_{atype}_embeddings.pt")
        torch.save(emb_dict, out_path)
        print(f"Saved {len(emb_dict)} embeddings to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Embed structured descriptions")
    parser.add_argument(
        "--model", required=True, choices=list(MODEL_CONFIG.keys()),
        help="Embedding model to use",
    )
    parser.add_argument(
        "--assay_type", required=True, choices=["itc", "spr", "all"],
        help="Assay type to process",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    args = parser.parse_args()

    entries = load_entries(args.assay_type)
    print(f"Loaded {len(entries)} entries with non-null structured_description")
    if not entries:
        print("No entries found. Exiting.")
        return

    texts = [e["text"] for e in entries]
    embeddings = embed(texts, args.model, batch_size=args.batch_size)
    save_results(entries, embeddings, args.model, args.assay_type)
    print("Done.")


if __name__ == "__main__":
    main()
