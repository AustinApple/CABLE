# Assay Extraction Agents

Two agents for extracting complete assay descriptions from scientific papers using vision-language models.

## Agents

### 1. Gemini Flash 2.5 Agent ([assay_extraction_agent.py](assay_extraction_agent.py))

Uses Google's Gemini Flash 2.5 API for extraction.

**Pros:**
- No local GPU required
- Handles PDFs directly (no conversion needed)
- Fast inference
- Low cost ($0.075 per 1M input tokens, $0.30 per 1M output tokens)
- Automatic token usage tracking with cost estimates

**Cons:**
- Requires API key
- Requires internet connection
- Rate limits apply

**Usage:**
```python
from agent.assay_extraction_agent import AssayExtractionAgent

agent = AssayExtractionAgent(
    api_key="YOUR_API_KEY",
    pdf_dir="./downloaded_paper_kd"
)

result = agent.extract_assay_description(
    pmid="12345678",
    assay_description="Brief description from BindingDB"
)

# Or process a dataframe
results_df = agent.process_dataframe(
    df=data,
    pmid_col="PMID",
    description_col="DESCRIPTION",
    limit=10,
    delay=1.0  # Delay between API calls
)

agent.cleanup()  # Delete uploaded files
```

### 2. Qwen3-VL Agent ([assay_extraction_agent_qwen.py](assay_extraction_agent_qwen.py))

Uses open-source Qwen3-VL model running locally.

**Pros:**
- Free to use (no API costs)
- No rate limits
- Works offline
- Full control over model
- Privacy (data stays local)

**Cons:**
- Requires powerful GPU (recommended: 24GB+ VRAM for 7B model)
- Slower inference than API
- Requires PDF to image conversion
- Higher memory usage

**Available Models:**
- `Qwen/Qwen3-VL-2B-Instruct` - Smallest (5GB VRAM), fastest
- `Qwen/Qwen3-VL-4B-Instruct` - Small (10GB VRAM)
- `Qwen/Qwen3-VL-8B-Instruct` - Medium (20GB VRAM), recommended
- `Qwen/Qwen3-VL-32B-Instruct` - Large (65GB VRAM)
- `Qwen/Qwen2.5-VL-7B-Instruct` - Alternative (Qwen2.5 series)

**Usage:**
```python
from agent.assay_extraction_agent_qwen import AssayExtractionAgentQwen

agent = AssayExtractionAgentQwen(
    model_name="Qwen/Qwen3-VL-8B-Instruct",  # Choose based on GPU
    pdf_dir="./downloaded_paper_kd",
    torch_dtype="bfloat16"  # or "float16", "float32"
)

result = agent.extract_assay_description(
    pmid="12345678",
    assay_description="Brief description from BindingDB",
    max_pages=10  # Limit pages to process
)

# Or process a dataframe
results_df = agent.process_dataframe(
    df=data,
    pmid_col="PMID",
    description_col="DESCRIPTION",
    limit=10,
    max_pages=10,  # Limit pages per PDF
    delay=0.0  # No delay needed for local model
)

agent.cleanup()  # Clean up GPU memory
```

## Installation

### For Gemini Agent
```bash
pip install google-genai pandas
```

### For Qwen3-VL Agent
```bash
pip install transformers>=4.37.0 torch>=2.0.0 accelerate pymupdf pillow pandas
```

## Output Format

Both agents return the same format:

```python
{
    "pmid": "12345678",
    "assay_description": "Original brief description",
    "original_paragraph": "Complete extracted paragraph from paper...",
    "location": "Methods section, page 3",
    "confidence": "high"  # or "medium", "low", "N/A"
}
```

## Token Usage Tracking

Both agents track token usage:

**Gemini Agent:**
- Shows per-query and accumulated token counts
- Provides cost estimates based on current pricing
- Example output:
  ```
  [This Query] Input: 45,234 | Output: 512 | Total: 45,746
  [Accumulated] Input: 123,456 | Output: 2,048 | Total: 125,504 | Requests: 3
  Estimated Cost: $0.0104
  ```

**Qwen3-VL Agent:**
- Shows token counts (no cost, local model)
- Tracks input/output tokens for analysis
- Example output:
  ```
  [This Query] Input: 45,234 | Output: 512 | Total: 45,746
  [Accumulated] Input: 123,456 | Output: 2,048 | Total: 125,504 | Requests: 3
  ```

## Choosing Between Agents

**Use Gemini Flash 2.5 if:**
- You have limited GPU resources
- You want faster processing
- You need to process large batches quickly
- Cost is acceptable ($0.075-0.30 per 1M tokens)

**Use Qwen3-VL if:**
- You have a powerful GPU available
- You want zero API costs
- You need offline processing
- Privacy is important (data stays local)
- You want full control over the model

## Performance Tips

### Gemini Agent
1. Use `delay` parameter to avoid rate limiting
2. Cache uploaded files (automatic in the agent)
3. Process in batches to optimize API usage

### Qwen3-VL Agent
1. Limit `max_pages` to reduce memory usage
2. Use `bfloat16` for best performance on modern GPUs
3. Process smaller batches if running out of memory
4. Consider using smaller model variants for faster inference

## Example Scripts

- [spr_test.py](../spr_test.py) - Gemini agent example
- [spr_test_qwen.py](../spr_test_qwen.py) - Qwen3-VL agent example

Both scripts demonstrate:
- Loading data from BindingDB
- Filtering by PMID
- Processing with the agent
- Saving results to TSV
