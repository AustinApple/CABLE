"""
Split BindingDB assay descriptions by assay type and create per-assay
PMID list files. Each file contains deduplicated PMIDs for that assay type.
A PMID can appear in multiple files if the paper uses multiple assay types.
"""

import os
import re
import pandas as pd
from categorize_assays import categorize_description

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'assay_pmid_lists')
INPUT_FILE = os.path.join(os.path.dirname(__file__), 'BindingDB_assay_description.tsv')


def sanitize_filename(category_name):
    """Convert assay category name to a filesystem-safe filename."""
    # Remove special chars except alphanumeric, spaces, underscores
    name = re.sub(r'[/\\()]+', ' ', category_name)
    name = re.sub(r'[^a-zA-Z0-9\s_]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name + '.txt'


def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, sep='\t', usecols=['PMID', 'DESCRIPTION'])
    print(f"Total rows: {len(df):,}")

    # Drop rows with null PMIDs
    df = df.dropna(subset=['PMID'])
    df['PMID'] = df['PMID'].astype(int)
    print(f"Rows with valid PMID: {len(df):,}")

    # Categorize each row by description
    print("Categorizing descriptions...")
    unique_descs = df['DESCRIPTION'].drop_duplicates()
    desc_to_assay = {desc: categorize_description(desc) for desc in unique_descs}
    df['ASSAY_TYPE'] = df['DESCRIPTION'].map(desc_to_assay)
    print(f"Unique assay types found: {df['ASSAY_TYPE'].nunique()}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Group by assay type and write PMID lists
    print(f"\nWriting PMID lists to {OUTPUT_DIR}/")
    print(f"{'Assay Type':<55} {'PMIDs':>8}")
    print("-" * 65)

    total_pmids = 0
    for assay_type, group in sorted(df.groupby('ASSAY_TYPE'),
                                     key=lambda x: -x[1]['PMID'].nunique()):
        pmids = sorted(group['PMID'].unique())
        filename = sanitize_filename(assay_type)
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, 'w') as f:
            for pmid in pmids:
                f.write(f"{pmid}\n")

        print(f"  {assay_type:<55} {len(pmids):>8,}")
        total_pmids += len(pmids)

    print("-" * 65)
    print(f"  {'Total (with cross-file duplicates)':<55} {total_pmids:>8,}")
    print(f"  {'Total unique PMIDs across all files':<55} {df['PMID'].nunique():>8,}")
    print(f"\nDone! {df['ASSAY_TYPE'].nunique()} files created in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()