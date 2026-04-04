# CIM Document PDF Parser & Extractor

This tool is designed to accurately parse large Confidential Information Memorandums (CIMs) ranging from 80-150 pages, extract and organize tables and text, and finally leverage an LLM to reliably extract granular financial measurements per explicit client requirements.

## Overview
1.  **PDF Parsing**: Leveraging `pdfplumber`, the script systematically analyzes and formats both raw text blocks and embedded tables (into DataFrames/JSON format) efficiently.
2.  **Context Keyword Filtering**: Due to the size of typical CIMs, feeding a 150-page PDF to an LLM context is expensive and causes hallucinations. This script intelligently detects keyword pages regarding financials, drastically trimming context length without losing data.
3.  **LLM Formatting Engine**: Utilizing OpenAI's `gpt-4o` or Anthropic's `claude-3-haiku/sonnet`, it synthesizes required metrics explicitly in valid `JSON` via standard or specialized extraction prompting methods.

## Installation

```powershell
# 1. Ensure Python is installed
python --version

# 2. Install Dependencies
pip install -r requirements.txt
```

## Usage Example

Before running the tool, you must configure your API tokens since it communicates with hosted LLM providers to perform intelligent field parsing from tables.

### Setting API Keys
**For OpenAI:**
```powershell
$env:OPENAI_API_KEY="sk-..."
```
**For Anthropic:**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

### Running the Analyzer
The script takes arguments via command-line:

```powershell
python cim_extractor.py --pdf "project_alpha_cim.pdf" --req "Extract historical Gross Margin and operating income for 2021, 2022, 2023. Additionally extract projected SG&A for 2024 and 2025. Please label everything explicitly and ensure numeric values."
```

### Full Options:
*   `--pdf`: Path to the local 80-150 page CIM PDF document.
*   `--req`: Flexible client instructions denoting which fields and periods are desired.
*   `--provider`: Model provider (default is `openai`, can use `anthropic`).
*   `--model`: Default is `gpt-4o`. For fast testing with anthropic, use `claude-3-haiku-20240307`. For high quality, use `claude-3-5-sonnet-20240620`.

## Output
A `JSON` artifact is printed to stdout and saved locally alongside the PDF as `{filename}_extracted.json`.
