# Two-Step Assay Extraction Pipeline

## Overview

The two-step pipeline extracts structured assay information from scientific papers using a vision-language model (Qwen3.5-27B). Given a PubMed ID (PMID) and a brief assay description from BindingDB, it locates the relevant experimental paragraphs in the paper and then extracts structured parameters (instrument, sensor chip, buffer, kinetics, etc.).

**Why two steps?** Step 1 requires a vision model because it reads PDF page images. Step 2 only processes the extracted text, so it can run without images — cheaper, faster, and optionally on a different (text-only) model.

```
Input (PMID + assay description)
        |
  [Step 1] Vision model reads PDF images -> original_paragraph
        |     (falls back to MinerU markdown on-demand if vision misses)
        |
  [Step 2] Text-only model reads extracted text -> structured_description
        |
Output (JSON per PMID)
```

---

## PDF → Markdown Conversion (MinerU, on-demand)

**File:** `pdf_utils.py` — `pdf_to_markdown()`; called from `base_extraction_agent.py` — `_get_paper_markdown()` / `_get_full_paper_markdown()`.

MinerU conversion is **lazy**: it runs only when a downstream step actually needs the markdown. The current test scripts (e.g. `rba_test_two_step_all.py`, `fpa_test_two_step.py`) do **not** call the bulk `preconvert_pdfs()` helper — every conversion is triggered on first use, then cached.

Trigger points:
1. **Step 1 markdown fallback** — `assay_extraction_agent_two_step.py:372` calls `_get_full_paper_markdown(pmid)` only when the vision model fails to locate the paragraph in PDF images.
2. **Reference resolution** — `base_extraction_agent.py:615` calls `_get_paper_markdown(pmid)` to read the bibliography section when chasing cited papers.

Implementation details:
- Runs as a subprocess (`/data/mwu11/miniconda3/envs/mineru/bin/mineru`).
- Results are disk-cached under `<pdf_dir>/markdown_cache/<pmid>/auto/<pmid>.md`, and additionally memory-cached in `self._markdown_cache`. Each PMID is converted at most once per run.
- `preconvert_pdfs()` (bulk pre-pass) is still available in `pdf_utils.py` for callers that want to convert everything up-front before the model is loaded — useful when MinerU and Qwen would otherwise contend for GPU VRAM — but it is opt-in.

**Error handling:**
| Condition | Behavior |
|---|---|
| MinerU binary not found | Prints error, returns `None` |
| Subprocess times out (default 300s) | Prints timeout warning, returns `None` |
| PDF file doesn't exist | Skips with log message |
| Any other subprocess error | Prints error, returns `None` |

---

## Step 1: Paragraph Extraction (`extract_paragraph`)

**File:** `assay_extraction_agent_two_step.py` — `TwoStepAssayExtractionAgent.extract_paragraph()`

This step finds and extracts the relevant experimental paragraphs from the paper. It follows a cascading search strategy with multiple fallback paths.

### Search Order

```
1. Vision model on main paper PDF images
       |
       |-- Found? --> [Chase supplementary if mentioned] --> [Chase references if cited] --> Return
       |
2. Markdown text fallback (full paper text via MinerU)
       |
       |-- Found? --> [Chase supplementary if mentioned] --> [Chase references if cited] --> Return
       |
3. Supplementary materials (fetched from PubMed Central)
       |
       |-- Found in any supplementary file? --> [Chase references if cited] --> Return
       |
4. Referenced literature (fetch cited papers, recursive up to max_reference_depth)
       |
       |-- Found in any referenced paper? --> Return
       |
5. Not found --> Return empty result
```

### 1.1 Vision Model on Main Paper

- PDF is converted to page images at 150 DPI using PyMuPDF.
- Images are cached in memory per `(PMID, max_pages)` key to avoid re-conversion.
- If the PDF is not found locally, the agent attempts to fetch it from PubMed Central (Europe PMC direct link -> PMC article page -> OA API).
- The vision model processes all page images together with the extraction prompt.
- The model returns JSON with `original_paragraph` (a dict of `location -> text`), `confidence`, `reference_number_in_text`, and `references_previous`.

### 1.2 Markdown Text Fallback

If the vision model returns empty `original_paragraph`:
- MinerU is invoked on-demand (if it hasn't been run for this PMID already), and the full paper markdown is sent to the model as text-only input.
- This catches cases where the vision model misses content (e.g., complex layouts, small text).

### 1.3 Supplementary Material Chase

Triggered in two scenarios:

**A. Paragraph found but mentions supplementary** — the extracted paragraph text contains patterns like "Supplemental Experimental Procedures", "Supporting Information", "Table S1", "Figure S2". The agent fetches supplementary files and searches them, **combining** any found content with the main paragraph.

**B. Paragraph not found in main paper** — the agent fetches supplementary files as a standalone search before trying references.

How supplementary files are handled:
- Fetched from PubMed Central using the PMC article page (scraping `href` links to `/bin/*.pdf|docx|xlsx|zip`).
- Tried from PMC first, then Europe PMC as fallback.
- Maximum 5 supplementary files downloaded per paper.
- Files exceeding 50 pages (`MAX_SUPPLEMENTARY_PAGES`) are skipped.
- Supported formats: PDF (direct), DOCX (converted to PDF via pypandoc+tectonic).

### 1.4 Reference Chase

Triggered in two scenarios:

**A. Paragraph found but references previous work** — the model output indicates `references_previous != "none"` (e.g., "as described previously [26]"). The agent resolves the cited paper's PMID and fetches/searches it, **combining** both papers' paragraphs.

**B. Paragraph not found anywhere** — the model's response may still contain a `references_previous` field pointing to where the methodology was originally published. The agent follows that reference.

Reference resolution priority:
1. **Markdown-based** (most reliable): Parse the reference number from `reference_number_in_text`, look up the full citation in the paper's bibliography (via MinerU markdown), then search PubMed by DOI/title/author.
2. **Direct PMID extraction**: Regex for `PMID:12345678` or PubMed URLs in the text.
3. **Citation-based PubMed search**: Extract author/title/DOI from free text and query PubMed (DOI search -> title search -> free text search with author+journal+year+volume+page).

Reference search is recursive up to `max_reference_depth` (default 1, meaning only direct references are followed). Up to 3 referenced PMIDs are tried per paper.

---

## Step 2: Structured Description Extraction (`fill_structured_description`)

**File:** `assay_extraction_agent_two_step.py` — `TwoStepAssayExtractionAgent.fill_structured_description()`

Takes the `original_paragraph` dict from Step 1, combines all text entries, and sends them to the model (text-only, no images) to extract structured assay parameters.

Extracted fields include:
- **instrument**: manufacturer, model
- **sensor_chip**: type, manufacturer
- **immobilization**: ligand name, strategy, density (RU), concentration
- **analyte**: description, concentration range
- **assay_conditions**: buffer, pH, assay type, flow rate, temperature, association/dissociation time, regeneration
- **data_analysis**: reference surface, subtraction method, fitting model, software

If the input text is empty or whitespace-only, returns `structured_description: null` immediately without querying the model.

---

## Error Handling Summary

### PDF / Image Errors

| Condition | Result |
|---|---|
| PDF not found locally and cannot be fetched from PMC | Returns `{"error": "PDF not found and could not be fetched"}` |
| PDF exists but conversion to images fails | Returns `{"error": "PDF conversion failed"}` |
| DOCX supplementary fails to convert to PDF | Skipped, logged |
| Supplementary file exceeds 50 pages | Skipped, logged to chase log |

### Model / Parsing Errors

| Condition | Result |
|---|---|
| Model output is not valid JSON | `_parse_response` returns `None`, treated as "not found" |
| Model returns `original_paragraph: {}` (empty dict) | `_is_paragraph_found` returns `False`, triggers fallback chain |
| Model returns `original_paragraph` with only `"error"` key | `_is_paragraph_found` returns `False` |
| Model returns `"NOT FOUND"` string | `_is_paragraph_found` returns `False` |
| Any exception during Step 1 | Caught, returns `{"error": "<exception message>"}` with full traceback printed |
| Any exception during Step 2 | Caught, returns `{"structured_description": null}` with full traceback printed |

### Network / API Errors

| Condition | Result |
|---|---|
| NCBI ID converter API fails | Cannot get PMC ID, supplementary/paper fetch skipped |
| PubMed Central article page unreachable | No supplementary files fetched |
| Supplementary file download fails (all mirrors) | Logged, file skipped |
| Paper PDF download fails (all 3 strategies) | Returns `None`, treated as "PDF not found" |
| PubMed search for referenced PMID fails | Logged to chase log, reference skipped |
| MinerU markdown conversion fails | Markdown fallback unavailable, reference resolution falls back to text-based methods |

### Reference Resolution Errors

| Condition | Result |
|---|---|
| `reference_number_in_text` given but not found in markdown bibliography | Falls back to `references_previous` text, then citation-based search |
| Citation resolved to PMID but PDF unavailable | Logged to chase log as "resolved but PDF unavailable" |
| Citation resolved to PMID but paragraph not found in that paper | Logged to chase log as "resolved but paragraph not found" |
| No PMID found from any citation strategy | Logged to chase log as "unresolved citation" |
| Reference depth exceeds `max_reference_depth` | Reference search stops, returns current result |

---

## Caching Strategy

| Cache | Scope | Storage |
|---|---|---|
| PDF-to-image conversion | In-memory per session | `_image_cache` dict keyed by `"<pmid>_<max_pages>"` |
| MinerU markdown | Disk + in-memory | Disk: `<pdf_dir>/markdown_cache/<pmid>/auto/<pmid>.md`; Memory: `_markdown_cache` dict |
| Description-level extraction | In-memory per PMID | `description_cache` in `process_dataframe_to_json` — same `DESCRIPTION` text reuses Step 1+2 results |
| PMID-level JSON output | Disk | `<output_dir>/<pmid>.json` — if file exists and is complete, entire PMID is skipped |

---

## Resume Support

The pipeline supports resuming interrupted runs:

- **In `process_dataframe_to_json`**: If `<pmid>.json` already exists on disk, the PMID is skipped entirely.
- **In the test script (`fpa_test_two_step.py`)**: Existing JSON files are checked for completeness — if any entry has an empty `original_paragraph` or `null` `structured_description`, the PMID is re-run.

---

## Token Usage Tracking

Every model query (both vision and text-only) records:
- Input tokens
- Output tokens
- Total tokens
- Request count

Accumulated totals are printed after each query and summarized at the end of a batch run via `_print_token_summary()`.

---

## Key Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen3-VL-30B-A3B-Thinking` | Vision model for Step 1 |
| `text_model_name` | `None` (uses vision model) | Optional separate text model for Step 2 |
| `temperature` | `0.0` | Greedy decoding (deterministic) |
| `search_supplementary` | `True` | Whether to fetch/search supplementary materials |
| `search_references` | `True` | Whether to follow references to other papers |
| `max_reference_depth` | `1` | How deep to follow reference chains |
| `MAX_SUPPLEMENTARY_PAGES` | `50` | Skip supplementary files larger than this |
| `paragraph_prompt_fn` | SPR prompt | Custom prompt function for Step 1 |
| `structured_prompt_fn` | SPR prompt | Custom prompt function for Step 2 |

---

## Class Hierarchy

```
BaseAssayExtractionAgent (base_extraction_agent.py)
  ├── Model loading, _query_model (vision), _parse_response
  ├── PDF/image handling, markdown conversion
  ├── Reference resolution (_resolve_reference_pmids)
  ├── Reference chase logic (_check_and_fetch_references, _search_references_not_found)
  ├── Supplementary mention detection (_paragraph_mentions_supplementary)
  └── Chase log management (_log_chase_miss)
        |
        v
TwoStepAssayExtractionAgent (assay_extraction_agent_two_step.py)
  ├── _query_text_model (text-only queries for Step 2)
  ├── extract_paragraph (Step 1)
  ├── fill_structured_description (Step 2)
  ├── extract_assay_description (combined Step 1 + Step 2)
  └── process_dataframe_to_json (batch processing)

Utilities:
  ├── PubMedFetcher (pubmed_utils.py) — PDF/supplementary fetching, PMID<->PMC conversion
  ├── CitationSearcher (pubmed_utils.py) — PMID resolution from citations
  ├── DocumentConverter (pdf_utils.py) — PDF/DOCX to images
  └── pdf_to_markdown / preconvert_pdfs (pdf_utils.py) — MinerU wrapper
```

---

## Output Format

Each PMID produces a JSON file (`<pmid>.json`) containing entries keyed by `reactant_set_id`:

```json
{
    "12345": {
        "reactant_set_id": 12345,
        "pmid": 9876543,
        "protein": "Target protein name",
        "ligand": {"smiles": "CC(=O)..."},
        "affinity_data": [{"type": "Kd", "value": 150.0, "relation": "=", "unit": "nM"}],
        "DESCRIPTION": "Original assay description from BindingDB",
        "search_path": ["main", "supplementary", "reference", "main"],
        "supplementary_source": ["paper_supp_1.pdf"],
        "references_previous": "Full citation text or 'none'",
        "original_paragraph": {
            "Materials and Methods": "The SPR experiments were performed...",
            "[Supp paper_supp_1.pdf] Table S1": "Detailed buffer conditions..."
        },
        "structured_description": {
            "instrument": {"manufacturer": "Cytiva", "model": "Biacore T200"},
            "sensor_chip": {"type": "CM5", "manufacturer": "Cytiva"},
            ...
        }
    }
}
```

The `search_path` array records the locations searched in order, providing an audit trail of how the extraction was obtained.
