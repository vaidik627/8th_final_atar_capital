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
| Kimi K2.5 `moonshotai/kimi-k2.5` | Via NVIDIA API; thinking model — `extra_body={"chat_template_kwargs":{"thinking":True}}`, temp=1.0, top_p=1.0, max_tokens=16384 |
| Claude Haiku (`claude-haiku-4-5-20251001`) | Field verification — cheap |
| Claude Sonnet (`claude-sonnet-4-6`) | Client demo — best quality |
| DeepSeek `deepseek-chat` / `deepseek-reasoner` | Via `https://api.deepseek.com/v1`; OpenAI client; `DEEPSEEK_API_KEY` in .env; `response_format: json_object` |
| Ollama (local/cloud) | GLM-Z1, GLM4, Qwen, DeepSeek-R1, Llama, Gemma, Mistral, Phi4; "custom..." option in UI |

---

## Key Files

| File | Purpose |
|------|---------|
| `cim_extractor.py` | Main script — PDF parsing + LLM extraction + year mapping |
| `.env` | API keys (NVIDIA_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY) |
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
  → ExcelWriter — fills the Prebid template (DONE)
```

---

## Provider Notes

- `nvidia`, `openai`, `ollama`, `deepseek` all use OpenAI Python client with custom `base_url`
- Kimi K2.5 detected via `_is_kimi = self.model_name == "moonshotai/kimi-k2.5"` — special params
- `<think>...</think>` blocks stripped in `_extract_json_from_text()` at Step 0 via regex
- Ollama "custom..." option in model dropdown shows a text input for any model name

---

## Excel Template Structure

- File: `Prebid V31 Template (1).xlsx`, Sheet: `Sheet1`
- Units: **$ in 000s (thousands)**
- Columns: I=FY19, J=FY20, K=FY21 (historical) | L=Year 1 … P=Year 5 (projections)
- L13:P13 cleared before write (removes hardcoded SGA values)
- L32:P32 cleared before write (removes hardcoded CAPEX values)

### Excel Fill Patterns

**`_write_field(row, force_negative=False)`** — standard pattern for all time-series fields:
- Historical: right-aligned into last N of I/J/K (2 actuals → J+K; 1 actual → K only; I blank if unused)
- Projection: fill real extracted proj data left-to-right; remaining proj slots get hist avg
- `force_negative=True` for CAPEX

**`_write_field_formula_proj(row)`** — Interest_Expense (row 22), WC_Change (row 33):
- Historical: same right-align logic
- Projection: write only if extracted — no avg fill (proj cells formula-driven in template)

### All Input Cells

| Cell | Field | Pattern | Notes |
|------|-------|---------|-------|
| B1 | Deal name | scalar | |
| I4:P4 | Year headers | dynamic | |
| I7:P7 | Total_Revenue | `_write_field` | |
| I10:P10 | Gross_Margin | `_write_field` | |
| I13:P13 | SGA | `_write_field` | L13:P13 cleared first |
| I16:P16 | Onex_Adjustments | `_write_field` | NO formulas in I16:P16 (verified) |
| I21:P21 | Other_Expense | `_write_field` | |
| I22:K22 | Interest_Expense | `_write_field_formula_proj` | hist right-align; proj only if extracted |
| I23:P23 | Depreciation | `_write_field` | |
| I32:P32 | CAPEX | `_write_field`, negative | L32:P32 cleared first |
| I33:K33 | WC_Change | `_write_field_formula_proj` | |
| C7 | AR | scalar | |
| C8 | Inventory | scalar | |
| C9 | ME_Equipment | scalar | |
| C11 | Building_Land | scalar | |

### 3 Field Types — CRITICAL

**Type 1 — AI Extraction (LLM fills these):**

| Row | Field | JSON Key | Notes |
|-----|-------|----------|-------|
| 7 | Net Revenue | `Total_Revenue` | Also: net revenue, total net revenue |
| 10 | Gross Margin | `Gross_Margin` | = "Contribution Margin" in many CIM PDFs |
| 13 | SG&A | `SGA` | Only if explicitly labeled — null if not found |
| 18 | Adj. EBITDA | `Adj_EBITDA` | Multi-variant priority logic |
| — | EBITDA (reported) | `EBITDA` | Reference field only — no Excel row |
| 16 | 1X Adjustments | `Onex_Adjustments` | Full hist+proj |
| 21 | Other expense/(income) | `Other_Expense` | Chimera only; null for other 3 |
| 22 | Interest Expense | `Interest_Expense` | Historical years (YYYY_A) ONLY |
| 23 | Depreciation | `Depreciation` | Hist + projected |
| 32 | CAPEX | `CAPEX` | Output as negative; Total Capex row only |
| 33 | Working Capital Change | `WC_Change` | Change in NWC year-over-year |

**Type 2 — Formula (DO NOT overwrite):**
- Growth Rate (row 8), GM% (row 11), Operating Income (row 15)
- Adj. EBITDA (row 18), EBITDA% (row 19), Taxable Income (row 24)
- Taxes, Net Income, Debt Service, FCCR, MOIC — all formula-driven

**Type 3 — Manual (client fills, never touch):**
- Sources section (AR, Inventory, M&E, Building & Land, Term Loans, Seller Note, Earnout)
- EBITDA multiple (C26, C27), Exit multiple (C40), Tax rate (H26), interest rates (H39-H41)
- Management Fees (row 53)

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
13. Other_Expense: only if explicitly labeled; no bundled-interest rows; sign: parentheses=negative
14. Interest_Expense: historical years (YYYY_A/TTM) ONLY — null for YYYY_E; no bundled rows; no balance sheet
15. Depreciation: historical + projected both; standalone "Depreciation" or "D&A" row; NEVER derive; null if not found
16. CAPEX: output as negative; Total Capex row only; base row not incl. one-time items; hist + projected
17. WC_Change: year-over-year change in NWC; output negative if NWC increases (cash use); hist + projected; labels vary

---

## PDF Test Files & Known Patterns

| PDF | EBITDA Label | Units | Depreciation | Interest Expense |
|-----|-------------|-------|-------------|-----------------|
| Project Chimera (1).pdf | Reported EBITDA → Adjusted EBITDA | $000s | YES — standalone "Depreciation" row, page 55; 2022A–2028F | null (no P&L line) |
| Project Network_CIP_Atar Capital.pdf | EBITDA → PF Adjusted EBITDA | $M | null (P&L stops at EBITDA) | null |
| Project Palm_CIM_(Atar Capital).pdf | Adj. EBITDA (simple row) | $M | null (P&L stops at EBITDA) | null |
| Project Smores - CIM_2026 (Atar).pdf | Plain EBITDA only | CAD millions | null (P&L stops at EBITDA) | null |

### Depreciation Values — Project Chimera (verified, page 55, $000s)
| Year | Value |
|------|-------|
| 2022_A | 2,559 |
| 2023_A | 2,604 |
| 2024_A | 2,438 |
| 2025_A | 2,305 |
| 2026_E | 2,063 |
| 2027_E | 1,968 |
| 2028_E | 1,872 |

---

## Results UI — Key Behaviors

**Section render order (top to bottom):**
1. Company Summary — always first (`insertBefore(summaryEl, sections.firstChild)`)
2. Company KPIs — directly after Summary (`summaryEl.after(kpisEl)`)
3. Revenue by Segment, Customer Concentration, Geography
4. Management Team, Growth Initiatives, Market Intelligence
5. Investment Recommendation — always last

**Company Summary card:** `summary-section-wrap` (dark bg, gold top border 2px) → `summary-eyebrow` ("Executive Brief") → `summary-text`

**Risk scorecard notes:** `white-space: normal; line-height: 1.5` — full text, no ellipsis

**Null auto-hiding:** All sections auto-hide when data is null (return empty fragment). Financial table rows: hide if all years null for that field (TODO — not yet implemented).

---

## Current Step-by-Step Plan

| Step | Task | Status |
|------|------|--------|
| 1 | Revenue + Gross Margin | DONE |
| 2 | SG&A | DONE |
| 3 | Operating Income (4-step logic) | DONE — removed (formula cell) |
| 4 | Adj. EBITDA (multi-variant logic) | DONE |
| 5 | EBITDA reported/plain (reference field) | DONE |
| 6 | Other expense/(income) — Rule 13 | DONE |
| 7 | Interest Expense — Rule 14 (historical only) | DONE |
| 8 | Depreciation — Rule 15 (hist + projected) | DONE |
| 9 | CAPEX — Rule 16 (output negative) | DONE |
| 10 | Working Capital Change — Rule 17 | DONE |
| 11 | 1X Adjustments | DONE |
| 12 | Excel writer (partial-projection fill) | DONE |
| 13 | DeepSeek + Kimi K2.5 providers | DONE |
| 14 | Ollama model expansion + custom input | DONE |
| 15 | Summary card at top + redesign | DONE |
| 16 | Risk scorecard note wrap | DONE |
| 17 | Hide null rows in financial table | TODO |
| 18 | GCV OCR in Flask UI | TODO |

---

## Working Rules

- Add extraction fields **one at a time** — never batch
- Always verify extracted values against the source PDF after adding a field
- Use LLaMA for dev, Haiku for verification, Sonnet for demo
- SGA: extract only if explicitly labeled — null if not found (Option B)
- When editing prompts: only touch the relevant rule, keywords, output format, --req — never rewrite other rules
- Test in `cim_extractor.py` first — only propagate to `app.py`/UI after confirmation

---

## Test Commands

```powershell
cd "C:\Users\vaidi\Desktop\jjj"

# LLaMA (free - dev)
python cim_extractor.py --pdf "Project Chimera (1).pdf" --provider nvidia --model "meta/llama-3.3-70b-instruct"

# Kimi K2.5 (NVIDIA - thinking model)
python cim_extractor.py --pdf "Project Chimera (1).pdf" --provider nvidia --model "moonshotai/kimi-k2.5"

# DeepSeek
python cim_extractor.py --pdf "Project Chimera (1).pdf" --provider deepseek --model "deepseek-chat"

# Claude Haiku (cheap - verify)
python cim_extractor.py --pdf "Project Chimera (1).pdf" --provider anthropic --model "claude-haiku-4-5-20251001"

# Claude Sonnet (demo)
python cim_extractor.py --pdf "Project Chimera (1).pdf" --provider anthropic --model "claude-sonnet-4-6"
```
