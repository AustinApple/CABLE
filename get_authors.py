"""Fetch authors for a list of PMIDs using Biopython's Entrez API."""

import sys
import time
from Bio import Entrez

# NCBI requires an email address
Entrez.email = "ming.hsiu.wu@uth.tmc.edu" 


def get_authors(pmids, batch_size=200):
    """Fetch authors for a list of PMIDs.

    Args:
        pmids: List of PMID strings.
        batch_size: Number of PMIDs to fetch per request.

    Returns:
        Dict mapping PMID -> list of author name strings.
    """
    results = {}

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        id_str = ",".join(batch)

        handle = Entrez.efetch(db="pubmed", id=id_str, retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        for article in records["PubmedArticle"]:
            medline = article["MedlineCitation"]
            pmid = str(medline["PMID"])
            authors = []
            author_list = medline["Article"].get("AuthorList", [])
            for author in author_list:
                last = author.get("LastName", "")
                initials = author.get("Initials", "")
                if last:
                    authors.append(f"{last}, {initials}" if initials else last)
            results[pmid] = authors

        # Respect NCBI rate limit (3 requests/sec without API key)
        time.sleep(0.4)

    return results


if __name__ == "__main__":
    # Example: pass PMIDs as command-line arguments
    # Usage: python get_authors.py 31635932 9925731
    if len(sys.argv) > 1:
        pmids = sys.argv[1:]
    else:
        # Default demo PMIDs
        pmids = ["31635932", "9925731"]

    authors_map = get_authors(pmids)
    for pmid, authors in authors_map.items():
        print(f"PMID {pmid}: {'; '.join(authors)}")