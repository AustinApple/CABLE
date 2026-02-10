"""
Assay Extraction Agent Package

This package provides tools for extracting assay descriptions from scientific papers
using vision-language models (Qwen3-VL).

Modules:
- assay_extraction_agent_qwen: Main agent class for extraction
- pubmed_utils: PubMed and PMC fetching utilities
- pdf_utils: PDF and document conversion utilities
- prompts: Prompt templates for the agent
"""

from .assay_extraction_agent_qwen import AssayExtractionAgentQwen
from .pubmed_utils import PubMedFetcher, CitationSearcher
from .pdf_utils import DocumentConverter

__all__ = [
    "AssayExtractionAgentQwen",
    "PubMedFetcher",
    "CitationSearcher",
    "DocumentConverter",
]
