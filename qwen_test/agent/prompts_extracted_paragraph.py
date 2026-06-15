"""
Prompt templates for the assay extraction agent.
"""


def get_assay_extraction_prompt(assay_description: str) -> str:
    """
    Get the prompt for extracting assay descriptions from a paper.

    Args:
        assay_description: Brief assay description from BindingDB

    Returns:
        Formatted prompt string
    """
    return f"""You are an expert scientific reader analyzing a research paper.

Task: Find and extract the COMPLETE ORIGINAL PARAGRAPH that describes the following assay:

Assay Description: {assay_description}

Instructions:
1. Search through the paper to find where this assay is described (focus on Experimental Methods or Methods section first if available)
2. Extract the COMPLETE ORIGINAL paragraph(s) that contain the full experimental protocol
3. Include ALL details: concentrations, conditions, temperatures, times, buffers, equipment, etc.
4. Do NOT summarize or paraphrase - provide the exact text from the paper
5. If the description spans multiple paragraphs, include all relevant paragraphs
6. Indicate the location in the paper (e.g., "Methods section, page X" or "Results section")
7. If the paper references a previous publication for methodology details (e.g., "as described previously", "following [ref]", "according to [author]"):

   CRITICAL STEPS FOR REFERENCE EXTRACTION:
   a) First, note the EXACT reference number mentioned in the methods text (e.g., if text says "as previously described [53]", the number is 53)
   b) Go to the References/Bibliography section at the END of the paper
   c) Find the reference entry that starts with EXACTLY that number (e.g., "53." or "(53)")
   d) VERIFY the number matches before copying - do NOT copy a different numbered reference
   e) Copy the FULL citation including: authors, title, journal name, year, volume, and page numbers

   Example: If methods say "following the protocol in [53]", find entry "53. Smith, J.; Jones, A. Title of Paper. J. Med. Chem. 2020, 63, 1234-1245." and copy that EXACT entry.

   IMPORTANT: Reference numbers in the text MUST match the number you extract from the reference list. If you cannot find the exact numbered reference, write "Reference [X] not found in bibliography" where X is the number.

Output format (JSON):
{{
    "original_paragraph": "The complete extracted text from the paper...",
    "location": "Where this text was found in the paper (section name, page, etc.)",
    "confidence": "high/medium/low - your confidence that this is the correct passage",
    "reference_number_in_text": "The exact reference number found in the methods text (e.g., '53' or '12,13'). Use 'none' if no reference is cited.",
    "references_previous": "The COMPLETE citation from the References section that matches the reference_number_in_text. MUST start with the same number. If no reference, use 'none'"
}}

If you cannot find a matching description, return:
{{
    "original_paragraph": "NOT FOUND",
    "location": "N/A",
    "confidence": "N/A",
    "reference_number_in_text": "none",
    "references_previous": "none"
}}

Please respond ONLY with valid JSON, no other text."""
