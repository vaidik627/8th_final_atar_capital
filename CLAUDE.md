# CIM Financial Extraction System — Project Context

## What This Project Does

Automatically extracts financial data from large CIM (Confidential Information Memorandum) PDFs (50–180 pages) and fills a client Excel template (`Prebid V31 Template (1).xlsx`) for M&A deal analysis.

**Client:** Atar Capital (private equity / M&A firm)
**Production LLM:** Anthropic Claude (API key in .env)
**Dev/Test LLM:** NVIDIA API — `meta/llama-3.3-70b-instruct` (free)

---

## API Strategy

| API | When to use |
|-----|------------|
| NVIDIA LLaMA 70B | Development, testing new fields — free |
| Claude Haiku (`claude-haiku-4-5-20251001`) | Field verification — cheap |
| Claude Sonnet (`claude-sonnet-4-6`) | Client demo — best quality |

---

## Key Files

| File | Purpose |
|------|---------|
| `cim_extractor.py` | Main script — PDF parsing + LLM extraction + year mapping |
| `.env` | API keys (NVIDIA_API_KEY, ANTHROPIC_API_KEY) |
| `Prebid V31 Template (1).xlsx` | Client Excel template to fill |
| `extracted_results/` | Output JSONs — `*_extracted.json` (raw) and `*_excel_mapped.json` (mapped) |
| `CLAUDE.md` | This file — project context for new chat sessions |

---

## Architecture

```
CIM PDF
  → CIMParser (pdfplumber) — extracts all text + tables page by page
  → find_financial_sections() — keyword filter, keeps only relevant pages
  → LLMExtractor.extract_fields() — sends filtered content to LLM, returns JSON
  → map_to_excel_columns() — converts year labels to FY19/FY20/FY21/Year_1-5
  → [TODO] ExcelWriter — fills the Prebid template
```

---

## Excel Template Structure

- File: `Prebid V31 Template (1).xlsx`, Sheet: `Sheet1`
- Units: **$ in 000s (thousands)**
- Columns: I=FY19, J=FY20, K=FY21 (historical) | L=Year 1 … P=Year 5 (projections)

### 3 Field Types — CRITICAL

**Type 1 — AI Extraction (LLM fills these):**

| Row | Field | JSON Key | Notes |
|-----|-------|----------|-------|
| 7 | Net Revenue | `Total_Revenue` | Also: net revenue, total net revenue |
| 10 | Gross Margin | `Gross_Margin` | = "Contribution Margin" in many CIM PDFs |
| 13 | SG&A | `SGA` | Only if explicitly labeled — null if not found |
| 15 | Operating Income | `Operating_Income` | Nested: {value, source, formula, confidence} |
| 18 | Adj. EBITDA | `Adj_EBITDA` | Multi-variant priority logic |
| — | EBITDA (reported) | `EBITDA` | Reference field only — no Excel row, client reference |
| 16-17 | 1X Adjustments | TODO | Next field |
| 21 | Other expense/(income) | TODO | |
| 23 | Depreciation | TODO | |

**Type 2 — Formula (DO NOT overwrite):**
- Growth Rate (row 8), GM% (row 11), Operating Income (row 15)
- Adj. EBITDA (row 18), EBITDA% (row 19), Taxable Income (row 24)
- Taxes, Net Income, Debt Service, FCCR, MOIC — all formula-driven

**Type 3 — Manual (client fills, never touch):**
- Sources section (AR, Inventory, M&E, Building & Land, Term Loans, Seller Note, Earnout)
- EBITDA multiple (C26, C27), Exit multiple (C40), Tax rate (H26), interest rates (H39-H41)
- CAPEX (row 32), Management Fees (row 53)

---

## LLM Year Label Convention

LLM must output period keys in this exact format:
- Historical/Actual: `YYYY_A` — any A/a suffix in doc = actual, even recent years
- Projected: `YYYY_E` — F/E/P/Est/Proj/Forecast/Budget suffixes in doc
- Budget (`_B` in doc) → output as `YYYY_E`
- TTM: `TTM_YYYY`

`map_to_excel_columns()` maps:
- Last 3 historical years → FY19, FY20, FY21
- Projected years → Year_1 through Year_5
- TTM preserved as own `TTM` key

---

## System Prompt Rules (numbered)

1. Output ONE valid JSON
2. Keys match client requirements
3. null (not string) if not found
4. Distinguish historical vs projected
5. All values in $000s — check table headers
6. Year labels: YYYY_A / YYYY_E / TTM_YYYY only
7. Extract from P&L tables ONLY — not charts, bullets, narrative
8. Check units per table independently
9. SGA aliases: SG&A, G&A, Operating Expenses, Overhead, Opex
10. Operating Income: 4-step priority (extract → EBITDA-D&A → GP-OpEx → Rev-COGS-OpEx)
11. Adj. EBITDA: priority Adj. EBITDA > PF Adj. EBITDA > plain EBITDA; CAD = no conversion
12. EBITDA (reported): plain/reported EBITDA only — NEVER use Adj. EBITDA; null if only adjusted exists

---

## PDF Test Files & Known Patterns

| PDF | EBITDA Label | Units | Notes |
|-----|-------------|-------|-------|
| Project Chimera (1).pdf | Reported EBITDA → Adjusted EBITDA | $000s | Page 55 P&L, page 60 adjustments |
| Project Network_CIP_Atar Capital.pdf | EBITDA → PF Adjusted EBITDA | $M | No D&A line, Operating Income = null |
| Project Palm_CIM_(Atar Capital).pdf | Adj. EBITDA (simple row) | $M | Page 71 |
| Project Smores - CIM_2026 (Atar).pdf | Plain EBITDA only | CAD millions | CAD currency, FY suffix |

---

## Current Step-by-Step Plan

| Step | Task | Status |
|------|------|--------|
| 1 | Revenue + Gross Margin | DONE |
| 2 | SG&A | DONE |
| 3 | Operating Income (4-step logic) | DONE |
| 4 | Adj. EBITDA (multi-variant logic) | DONE |
| 5 | EBITDA reported/plain (reference field) | DONE |
| 6 | 1X Adjustments | NEXT |
| 6 | Depreciation + Other expense | TODO |
| 7 | Excel writer | TODO |

---

## Working Rules

- Add extraction fields **one at a time** — never batch
- Always verify extracted values against the source PDF after adding a field
- Use LLaMA for dev, Haiku for verification, Sonnet for demo
- SGA: extract only if explicitly labeled — null if not found (Option B)
- Operating Income returns nested object `{value, source, formula, confidence}` per year
- When editing prompts: only touch the relevant rule, keywords, output format, --req — never rewrite other rules

---

## Test Commands

```powershell
cd "C:\Users\vaidi\Desktop\jjj"

# LLaMA (free - dev)
python cim_extractor.py --pdf "Project Chimera (1).pdf" --provider nvidia --model "meta/llama-3.3-70b-instruct"

# Claude Haiku (cheap - verify)
python cim_extractor.py --pdf "Project Chimera (1).pdf" --provider anthropic --model "claude-haiku-4-5-20251001"

# Claude Sonnet (demo)
python cim_extractor.py --pdf "Project Chimera (1).pdf" --provider anthropic --model "claude-sonnet-4-6"
```
