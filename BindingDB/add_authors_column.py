"""Add Authors column to BindingDB_assay_description.tsv from BindingDB_All_202601.tsv by matching PMIDs."""

import pandas as pd

# Read only PMID and Authors columns from the large file to save memory
print("Reading Authors from BindingDB_All_202601.tsv...")
df_all = pd.read_csv(
    "/data/mwu11/LLM_affinity/BindingDB/BindingDB_All_202601.tsv",
    sep="\t",
    usecols=["PMID", "Authors"],
    low_memory=False,
)
df_all["PMID"] = df_all["PMID"].astype(float).astype("Int64").astype(str)

# Get unique PMID -> Authors mapping
pmid_to_authors = df_all.drop_duplicates(subset="PMID").set_index("PMID")["Authors"]
print(f"Found {len(pmid_to_authors)} unique PMIDs with Authors.")

# Read assay description file
print("Reading BindingDB_assay_description.tsv...")
df_desc = pd.read_csv(
    "/data/mwu11/LLM_affinity/BindingDB/BindingDB_assay_description.tsv",
    sep="\t",
    low_memory=False,
)
df_desc["PMID"] = df_desc["PMID"].astype(float).astype("Int64").astype(str)

# Map Authors by PMID
df_desc["Authors"] = df_desc["PMID"].map(pmid_to_authors)

matched = df_desc["Authors"].notna().sum()
total = len(df_desc)
print(f"Matched {matched}/{total} rows with Authors ({matched/total*100:.1f}%).")

# Save
df_desc.to_csv(
    "/data/mwu11/LLM_affinity/BindingDB/BindingDB_assay_description.tsv",
    sep="\t",
    index=False,
)
print("Saved updated BindingDB_assay_description.tsv.")