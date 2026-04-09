"""
CIM (Confidential Information Memorandum) Document Parser & Extraction System
Targeted for large (80-150+ pages) financial PDFs. Extracts text and tables,
filters for financial sections, and utilizes an LLM to reliably extract 
fields based on variable client requirements.
"""

import os
import re
import json
import time
import logging
import datetime
import pdfplumber
import pandas as pd
from typing import List, Dict, Any, Optional
import argparse

# PyMuPDF — local fallback for pages where pdfplumber returns near-empty text
try:
    import fitz as _fitz
    _FITZ_AVAILABLE = True
except ImportError:
    _fitz = None
    _FITZ_AVAILABLE = False

# Google Cloud Vision (optional — only used if gcv_key_path provided)
try:
    from google.cloud import vision as _gcv
    _GCV_AVAILABLE = True
except ImportError:
    _GCV_AVAILABLE = False

# Load .env file if present
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _extract_json_from_text(text: str) -> str:
    """
    Robustly extract JSON from LLM output.
    Strategy 1: ```json ... ``` fences.
    Strategy 2: ``` ... ``` fences (no language tag).
    Strategy 3: find the first '{' and last '}' (raw JSON, no fences).
    Falls back to returning the original text so json.loads can raise a clear error.
    """
    # Strategy 1 & 2 — markdown fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    # Strategy 3 — bare JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text

class CIMParser:
    # Minimum character count from pdfplumber to consider a page text-sufficient.
    # Pages with fewer chars are treated as image-heavy and sent to Google Vision OCR.
    _OCR_TEXT_THRESHOLD = 80

    def __init__(self, pdf_path: str, gcv_key_path: str = None, gcv_api_key: str = None):
        self.pdf_path = pdf_path
        self.extracted_text = []
        self.extracted_tables = []
        self._gcv_client = None       # service account client
        self._gcv_api_key = None      # simple API key (REST)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        # Service account JSON path (existing method)
        if gcv_key_path:
            if not _GCV_AVAILABLE:
                logging.warning("google-cloud-vision or PyMuPDF not installed — OCR disabled.")
            elif not os.path.exists(gcv_key_path):
                logging.warning(f"GCV key file not found at {gcv_key_path} — OCR disabled.")
            else:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcv_key_path
                self._gcv_client = _gcv.ImageAnnotatorClient()
                logging.info("Google Cloud Vision OCR enabled (service account).")

        # Simple API key (REST endpoint method)
        elif gcv_api_key:
            if not _GCV_AVAILABLE:
                logging.warning("PyMuPDF not installed — OCR disabled.")
            else:
                self._gcv_api_key = gcv_api_key
                logging.info("Google Cloud Vision OCR enabled (API key).")

    def _extract_page_fitz(self, page_num: int) -> str:
        """Extract text from a single page using PyMuPDF — local fallback, no cloud needed."""
        try:
            doc = _fitz.open(self.pdf_path)
            try:
                page = doc[page_num - 1]
                return page.get_text("text").strip()
            finally:
                doc.close()
        except Exception as e:
            logging.warning(f"PyMuPDF fallback failed on page {page_num}: {e}")
            return ""

    def _ocr_page_api_key(self, page_num: int) -> str:
        """Render PDF page to PNG via PyMuPDF, send to Vision REST API using simple API key."""
        import base64, urllib.request, urllib.error
        try:
            doc = _fitz.open(self.pdf_path)
            try:
                page = doc[page_num - 1]
                mat = _fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
                pix = page.get_pixmap(matrix=mat)
                content = pix.tobytes("png")
            finally:
                doc.close()

            # Encode image as base64 for REST API
            image_b64 = base64.b64encode(content).decode("utf-8")

            # Build REST request body
            request_body = json.dumps({
                "requests": [{
                    "image": {"content": image_b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
                }]
            }).encode("utf-8")

            url = f"https://vision.googleapis.com/v1/images:annotate?key={self._gcv_api_key}"
            req = urllib.request.Request(
                url, data=request_body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            responses = result.get("responses", [])
            if not responses:
                return ""
            annotation = responses[0].get("fullTextAnnotation", {})
            return annotation.get("text", "")

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            logging.warning(f"GCV API key OCR HTTP error on page {page_num}: {e.code} {body[:200]}")
            return ""
        except Exception as e:
            logging.warning(f"GCV API key OCR failed on page {page_num}: {e}")
            return ""

    def _ocr_page(self, page_num: int) -> str:
        """Render a single PDF page to PNG via PyMuPDF and run Google Vision OCR on it."""
        # Route to API key method if that's what's configured
        if self._gcv_api_key:
            return self._ocr_page_api_key(page_num)
        try:
            doc = _fitz.open(self.pdf_path)
            try:
                page = doc[page_num - 1]              # fitz is 0-indexed
                mat = _fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
                pix = page.get_pixmap(matrix=mat)
                content = pix.tobytes("png")
            finally:
                doc.close()  # always close even if pixmap fails

            image = _gcv.Image(content=content)
            response = self._gcv_client.document_text_detection(image=image)
            if response.error.message:
                logging.warning(f"GCV error on page {page_num}: {response.error.message}")
                return ""
            return response.full_text_annotation.text or ""
        except Exception as e:
            logging.warning(f"OCR failed on page {page_num}: {e}")
            return ""

    def parse_pdf(self) -> None:
        """
        Parses the entire PDF, extracting both text blocks and tabular structures page by page.
        Suitable for 80-150 page documents.
        """
        logging.info(f"Opening PDF: {self.pdf_path} for parsing...")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"Total pages detected: {total_pages}. Starting extraction...")
            
            for index, page in enumerate(pdf.pages):
                page_num = index + 1
                
                # 1. Text Extraction
                try:
                    text = page.extract_text() or ""
                    text = text.strip()

                    # If page has very little text, try fallbacks before giving up
                    if len(text) < self._OCR_TEXT_THRESHOLD:
                        # Fallback 1 — PyMuPDF (local, no cloud, no API key needed)
                        if _FITZ_AVAILABLE:
                            fitz_text = self._extract_page_fitz(page_num)
                            if len(fitz_text) > len(text):
                                logging.info(f"Page {page_num}: PyMuPDF recovered {len(fitz_text)} chars (pdfplumber gave {len(text)}).")
                                text = fitz_text
                        # Fallback 2 — GCV OCR (cloud, only if explicitly enabled)
                        if len(text) < self._OCR_TEXT_THRESHOLD and self._gcv_client:
                            logging.info(f"Page {page_num}: still low text ({len(text)} chars) — running GCV OCR...")
                            ocr_text = self._ocr_page(page_num)
                            if ocr_text.strip():
                                logging.info(f"Page {page_num}: GCV OCR recovered {len(ocr_text)} chars.")
                                text = ocr_text.strip()

                    if text:
                        self.extracted_text.append({
                            "page": page_num,
                            "text": text
                        })
                except Exception as e:
                    logging.warning(f"Failed to extract text on page {page_num}: {e}")

                # 2. Table Extraction
                try:
                    tables = page.extract_tables()
                    for t_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            # Use Pandas to structure it
                            headers = table[0]
                            # Clean up empty or None headers to prevent Pandas errors
                            headers = [str(col).strip() if col else f"Col_{i}" for i, col in enumerate(headers)]
                            
                            df = pd.DataFrame(table[1:], columns=headers)
                            df = df.fillna("")  # Remove None values
                            
                            self.extracted_tables.append({
                                "page": page_num,
                                "table_index": t_idx + 1,
                                "data": df.to_dict(orient="records")
                            })
                except Exception as e:
                    logging.warning(f"Failed to extract tables on page {page_num}: {e}")

                if page_num % 20 == 0 or page_num == total_pages:
                    logging.info(f"Processed {page_num}/{total_pages} pages...")

        logging.info("PDF parsing completed successfully.")

    def find_financial_sections(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Filters the extracted PDF content, returning only pages/sections that contain
        important financial keywords. This reduces LLM token usage and prevents hallucinations.
        """
        if not keywords:
            keywords = [
                "revenue", "total revenue", "net revenue",
                "gross margin", "total gross margin",
                "contribution margin", "gross profit",
                "sg&a", "sga", "selling general", "selling, general",
                "general & administrative", "general and administrative",
                "operating expenses", "total operating expenses",
                "overhead", "opex",
                "operating income", "operating profit", "income from operations", "ebit",
                "depreciation", "amortization", "d&a",
                "adjusted ebitda", "adj. ebitda", "adj ebitda",
                "reported ebitda", "normalized ebitda",
                "ebitda adjustments", "management adjustments",
                "income statement", "financial performance",
                "historical financials", "projected financials",
                "other expense", "other income", "other (income)", "other income/expense",
                "other income / expenses", "non-operating expense", "non-operating income",
                "interest expense", "interest expense, net", "net interest expense",
                "interest and debt expense", "finance costs", "finance charges",
                "interest charges", "interest on debt",
                "capex", "capital expenditure", "capital expenditures", "total capex",
                "capital spending", "purchases of property", "pp&e",
                "change in working capital", "change in nwc", "changes in working capital",
                "working capital change", "(growth) / decline in net working capital",
                "increase in working capital", "decrease in working capital",
                "change in operating working capital",
                "accounts receivable", "account receivable", "trade receivables",
                "receivables, net", "accounts receivable, net", "net receivables",
                "balance sheet", "current assets", "working capital detail",
                "inventory", "inventories", "inventory, net", "stock",
                "raw materials", "finished goods", "work in progress", "work-in-progress",
                "total inventory",
                # Fixed Asset Schedule keywords (for ME_Equipment and Building_Land extraction)
                "fixed asset", "fixed assets", "fixed asset schedule",
                "property, plant", "property plant", "pp&e", "ppe",
                "machinery & equipment", "machinery and equipment", "m&e",
                "warehouse equipment", "manufacturing equipment", "production equipment",
                "plant & equipment", "plant and equipment",
                "building & land", "building and land", "land and building",
                "accumulated depreciation", "gross fixed assets", "total fixed assets",
                # Business overview keywords (for Company_Summary extraction)
                "business overview", "company overview", "executive summary",
                "company description", "business description", "business profile",
                "company profile", "about the company", "about the business",
                "products and services", "product and service", "business model",
                "company history", "key customers", "end markets", "end-markets",
                "investment highlights", "investment overview", "transaction overview",
                "overview", "introduction",
                # Market intelligence keywords (for Market_Intelligence extraction)
                "competitive landscape", "competition", "competitors", "competitive position",
                "competitive overview", "competitive dynamics", "key competitors",
                "market overview", "industry overview", "market size", "market opportunity",
                "total addressable market", "addressable market", "tam", "sam", "som",
                "market growth", "market growth rate", "industry growth", "cagr",
                "market share", "market position", "key players", "major players",
                "industry players", "industry participants", "industry dynamics",
                "market trends", "industry trends", "competitive advantages",
                "barriers to entry", "market leadership", "fragmented market",
                "industry revenue", "industry sales", "total industry", "dealer industry",
                "industry outlook", "industry shipments", "unit shipments",
                # Additional year label formats used in some CIMs
                "ltm", "ntm", "last twelve months", "next twelve months",
                "trailing twelve", "annualized",
                "fy20", "fy21", "fy22", "fy23", "fy24", "fy25", "fy26", "fy27",
                "fiscal 20", "fiscal year",
                # Revenue aliases not already covered
                "net sales", "total sales", "gross revenue", "total net revenue",
                "product revenue", "service revenue", "subscription revenue",
                "recurring revenue", "arr", "mrr", "contract revenue",
                # EBITDA / margin aliases
                "pro forma", "proforma", "pf adj", "pf ebitda",
                "normalized ebitda", "run-rate", "run rate",
                "adjusted gross profit", "adjusted gross margin",
                # Debt / leverage (for future CIMs with richer balance sheets)
                "total debt", "net debt", "long-term debt", "senior debt",
                "term loan", "revolver", "credit facility",
                # Cash flow
                "free cash flow", "fcf", "unlevered free cash flow",
                "cash from operations", "operating cash flow",
            ]
            
        logging.info(f"Filtering content over {len(self.extracted_text)} text pages and {len(self.extracted_tables)} tables...")
        relevant_content = []
        relevant_pages = set()
        
        # Keyword matching on text
        for text_block in self.extracted_text:
            text_lower = text_block["text"].lower()
            if any(kw in text_lower for kw in keywords):
                relevant_content.append({
                    "type": "text", 
                    "page": text_block["page"], 
                    "content": text_block["text"].replace('\n', ' ')
                })
                relevant_pages.add(text_block["page"])
                
        # Retrieve all tables present on relevant pages OR containing financial keywords in their data
        for table_block in self.extracted_tables:
            table_str = json.dumps(table_block["data"]).lower()
            if table_block["page"] in relevant_pages or any(kw in table_str for kw in keywords):
                relevant_content.append({
                    "type": "table", 
                    "page": table_block["page"], 
                    "content": table_block["data"]
                })

        # Sort by page number to maintain reading order
        relevant_content.sort(key=lambda x: x["page"])
        if not relevant_content:
            logging.warning("No relevant financial sections found. The PDF may be image-only or use unsupported formatting. LLM will receive no context — all fields will likely be null.")
        logging.info(f"Identified {len(relevant_content)} relevant standalone sections (text/tables).")
        return relevant_content

class LLMExtractor:
    def __init__(self, api_key: str, provider: str = "openai", model_name: str = "gpt-4o"):
        self.api_key = api_key
        self.provider = provider.lower()
        self.model_name = model_name
        
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        elif self.provider == "nvidia":
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=self.api_key
                )
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        elif self.provider == "ollama":
            try:
                from openai import OpenAI
                base_url = os.getenv("OLLAMA_BASE_URL", "https://openapi.ollama.ai/v1")
                self.client = OpenAI(base_url=base_url, api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        else:
            raise ValueError("Unsupported provider. Choose 'openai', 'anthropic', 'nvidia', or 'ollama'.")

    def map_to_excel_columns(self, extracted_data: Dict) -> Dict:
        """
        Converts extracted year labels (e.g. '2021_A', '2024_E', 'TTM_2024') into
        Excel template column labels (FY19, FY20, FY21, Year_1 ... Year_5).

        Logic:
          - Historical (_A) years: sort ascending, map the last 3 to FY19 → FY20 → FY21
            (FY21 = most recent historical year, FY19 = oldest of the 3)
          - TTM: treated as an extra historical data point (slotted after FY21 if present,
            otherwise appended as a note key 'TTM')
          - Projected (_E) years: sort ascending, map up to 5 years to Year_1 … Year_5
        """
        HIST_LABELS = ["FY19", "FY20", "FY21"]
        PROJ_LABELS = ["Year_1", "Year_2", "Year_3", "Year_4", "Year_5"]

        excel_mapped = {}

        for field, periods in extracted_data.items():
            if not isinstance(periods, dict):
                excel_mapped[field] = periods
                continue

            historical: Dict[int, Any] = {}
            projected:  Dict[int, Any] = {}
            ttm_value = None

            for period_key, value in periods.items():
                key_upper = period_key.upper()
                year_match = re.search(r'(\d{4})', period_key)
                year = int(year_match.group(1)) if year_match else None

                if "TTM" in key_upper or "LTM" in key_upper:
                    ttm_value = value
                elif key_upper.endswith("_A") and year:
                    historical[year] = value
                elif key_upper.endswith("_E") and year:
                    projected[year] = value
                elif year:
                    # Fallback: guess by year — ≤ current year = historical, else projected
                    _current_year = datetime.date.today().year
                    if year <= _current_year:
                        historical[year] = value
                    else:
                        projected[year] = value

            mapped: Dict[str, Any] = {}

            # Map historical → FY19 / FY20 / FY21
            sorted_hist = sorted(historical.items())          # [(year, val), ...]
            recent_hist = sorted_hist[-3:]                    # keep latest 3
            for i, label in enumerate(HIST_LABELS):
                offset = i - (3 - len(recent_hist))          # align to rightmost slot
                if offset >= 0:
                    mapped[label] = recent_hist[offset][1]
                else:
                    mapped[label] = None

            # TTM gets its own key (informational, not mapped to a column)
            if ttm_value is not None:
                mapped["TTM"] = ttm_value

            # Map projected → Year_1 … Year_5
            sorted_proj = sorted(projected.items())
            for i, label in enumerate(PROJ_LABELS):
                if i < len(sorted_proj):
                    mapped[label] = sorted_proj[i][1]
                else:
                    mapped[label] = None

            excel_mapped[field] = mapped

        return excel_mapped

    @staticmethod
    def _normalize_nulls(result: Dict) -> Dict:
        """
        Convert year-keyed objects where ALL year values are null into scalar null.
        e.g. {"2023_A": null, "2024_A": null} → null
        Leaves Market_Intelligence, Investment_Recommendation, Company_Summary untouched.
        """
        SKIP_KEYS = {"Investment_Recommendation", "Company_Summary", "Revenue_By_Segment", "Customer_Concentration", "Revenue_By_Geography", "Management_Team", "Growth_Initiatives"}
        year_pat = re.compile(r'^\d{4}_(A|E)$|^TTM_\d{4}$|^TTM$|^LTM_\d{4}$|^LTM$')
        normalized = {}
        for key, value in result.items():
            # Market_Intelligence: collapse to null if all six sub-fields are null
            if key == "Market_Intelligence":
                if isinstance(value, dict) and all(
                    value.get(k) is None for k in ("market_size", "market_growth_rate", "market_position", "competitors", "industry_tailwinds", "barriers_to_entry")
                ):
                    normalized[key] = None
                else:
                    normalized[key] = value
            # Company_KPIs: collapse to null if all sub-fields are null
            elif key == "Company_KPIs":
                kpi_keys = ("founded_year", "total_employees", "num_locations", "countries_of_operation", "capacity_utilization")
                if isinstance(value, dict) and all(value.get(k) is None for k in kpi_keys):
                    normalized[key] = None
                else:
                    normalized[key] = value
            elif key in SKIP_KEYS or not isinstance(value, dict):
                normalized[key] = value
            else:
                year_vals = {k: v for k, v in value.items() if year_pat.match(k)}
                if year_vals and all(v is None for v in year_vals.values()):
                    normalized[key] = None
                else:
                    normalized[key] = value
        return normalized

    def extract_fields(self, relevant_content: List[Dict[str, Any]], client_requirements: str) -> Optional[Dict]:
        """
        Sends the filtered context and client instruction to the LLM to dynamically extract JSON.
        """
        # Format context for the prompt securely
        context_str = ""
        for item in relevant_content:
            context_str += f"\n--- Page {item['page']} ({item['type'].upper()}) ---\n"
            if item["type"] == "table":
                context_str += json.dumps(item["content"], indent=2) + "\n"
            else:
                context_str += str(item["content"]) + "\n"
                
        # To avoid exceeding max bounds, truncate if ridiculously large. (E.g. limit to 100k chars ~25k tokens roughly)
        if len(context_str) > 300000:
            logging.warning("Context is overly large, truncating cautiously...")
            context_str = context_str[:300000] + "\n...[TRUNCATED]"

        system_prompt = (
            "You are an elite financial M&A analyst. Your task is to process excerpts "
            "from a Confidential Information Memorandum (CIM) to extract financial metrics.\n"
            "You will be given unstructured text and semi-structured tabular data.\n\n"
            "CRITICAL RULES:\n"
            "0. TOTAL REVENUE EXTRACTION RULES:\n"
            "   Extract the top-line revenue figure per year from the main P&L / income statement table.\n"
            "   VALID LABELS — accept ANY of these (case-insensitive):\n"
            "     'Revenue', 'Revenues', 'Net Revenue', 'Net Revenues', 'Total Revenue', 'Total Revenues',\n"
            "     'Net Sales', 'Total Net Sales', 'Gross Sales', 'Total Sales', 'Sales',\n"
            "     'Net Sales Revenue', 'Total Net Revenue',\n"
            "     'Service Revenue', 'Product Revenue', 'Subscription Revenue',\n"
            "     'Net Patient Revenue', 'Net Operating Revenue',\n"
            "     'Gross Billings less Returns & Allowances', 'Net Billings'\n"
            "   PRIORITY: If both 'Gross Sales' and 'Net Sales' appear in the same table, use 'Net Sales'.\n"
            "   If both 'Revenue' and 'Net Revenue' appear, use 'Net Revenue'.\n"
            "   Always use the TOP-LINE summary row — never a segment sub-row.\n"
            "   NEVER extract from charts, bullet points, or narrative text.\n"
            "   NEVER use a gross-to-net adjustment row (e.g. 'Returns & Allowances') as revenue.\n"
            "   Normalise to $000s. Output null per year if not found.\n"
            "1. Output exactly ONE valid JSON object.\n"
            "2. Ensure Keys inside JSON perfectly map what the client requested.\n"
            "3. Do NOT hallucinate data. If a requested metric/year/value is NOT present in the context, explicitly use null (no quotes).\n"
            "4. Distinguish carefully between historical (actual) vs projected (estimated/forecast) metrics.\n"
            "5. OUTPUT UNITS: Always output all numeric values in THOUSANDS ($000s). "
            "   If the document reports in millions, multiply by 1000. "
            "   If in actual dollars, divide by 1000. "
            "   Check every table header (e.g. '$ in thousands', '$ in millions') before outputting.\n"
            "6. YEAR LABEL FORMAT: Use exactly this convention for period keys:\n"
            "   - Historical/Actual years: 'YYYY_A' (e.g. '2021_A', '2022_A', '2023_A')\n"
            "     A suffix in the document (e.g. '25A', 'Dec-25A', '2025A') always means ACTUAL — label it YYYY_A regardless of how recent the year is.\n"
            "   - Projected/Estimated/Forecast years: 'YYYY_E' (e.g. '2024_E', '2025_E')\n"
            "     Only use _E if the document explicitly marks the period as E, F, P, Est., Proj., Forecast, or Budget.\n"
            "   - Budget years: also use 'YYYY_E' (e.g. '2025_B' in document → output as '2025_E')\n"
            "   - Trailing Twelve Months / Last Twelve Months: 'TTM_YYYY' (e.g. 'TTM_2024')\n"
            "     IMPORTANT: 'LTM' in the document means the same thing as 'TTM' — always output as 'TTM_YYYY'.\n"
            "     Example: 'LTM Sep-2024' → output key 'TTM_2024'.\n"
            "   - NON-CALENDAR / FISCAL YEAR companies: Some companies have fiscal years ending on dates other\n"
            "     than December 31 (e.g. FY ending March 31, June 30, September 30, October 31).\n"
            "     RULE: Use the CALENDAR YEAR in which the fiscal year ENDS as YYYY.\n"
            "     Examples:\n"
            "       'FY2024' ending March 31, 2024  → '2024_A'\n"
            "       'FY24' ending June 30, 2024     → '2024_A'\n"
            "       'Apr-24A' (year ending Apr 2024) → '2024_A'\n"
            "       'Sep-23A' (year ending Sep 2023) → '2023_A'\n"
            "       'FY ending Oct 2023'             → '2023_A'\n"
            "     If the document uses labels like 'FY22', 'FY2022', or 'Year ended [Month] 20XX',\n"
            "     extract the 4-digit year from the label and apply the _A or _E suffix accordingly.\n"
            "   - Do NOT use '2023A', '2024E', 'FY2023' or any other format in the output.\n"
            "7. EXTRACT FROM TABLES ONLY: Only extract numeric values from structured financial tables (P&L tables, income statement tables). "
            "   Do NOT extract numbers from charts, graphs, bullet points, narrative text, or illustrative figures. "
            "   If a year appears only in a chart or narrative but not in a financial table, do NOT include it.\n"
            "8. UNITS AND CURRENCY — check each table independently:\n"
            "   SCALE: If one table uses thousands and another uses millions, normalise BOTH to $000s before outputting.\n"
            "   CURRENCY HANDLING:\n"
            "     - USD ($): normalise to $000s as usual.\n"
            "     - CAD (Canadian dollars, C$, CA$): output the number as-is in thousands — do NOT convert to USD.\n"
            "     - EUR (€): output the number as-is in thousands — do NOT convert to USD.\n"
            "     - GBP (£): output the number as-is in thousands — do NOT convert to USD.\n"
            "     - AUD (A$): output the number as-is in thousands — do NOT convert to USD.\n"
            "     - Any other currency: output the number as-is in thousands — do NOT convert to USD.\n"
            "   SANITY CHECK: If extracted Revenue seems 1,000x larger or smaller than expected given the\n"
            "     stated units, re-check the table header — you likely read a millions table as thousands or vice versa.\n"
            "9. SG&A EXTRACTION RULES:\n"
            "   Extract ONLY if a row is explicitly labeled with one of these names in a structured financial table:\n"
            "   VALID LABELS (extract these):\n"
            "   - 'SG&A', 'SGA', 'Unadjusted SG&A', 'Adjusted SG&A'\n"
            "   - 'Selling, General & Administrative', 'Selling, General and Administrative'\n"
            "   - 'Selling & Marketing', 'Selling Expenses'\n"
            "   - 'G&A', 'General & Administrative', 'General and Administrative'\n"
            "   - 'Overhead' (only if it is a standalone labeled row, not a subtotal)\n"
            "   HARD RULES:\n"
            "   - NEVER extract a row labeled 'Operating Expenses', 'Total Operating Expenses', or 'Opex' as SG&A.\n"
            "     These are calculated subtotal rows (Gross Profit - Operating Income) and will produce wrong values.\n"
            "   - NEVER infer or calculate SG&A from other lines. Only extract if the label is explicit.\n"
            "   - NEVER use 'Any expense line between gross margin and EBITDA' as a rule — that is too vague.\n"
            "   - Do NOT confuse SG&A with COGS or direct costs — those are captured in Gross Margin.\n"
            "   - If no row with a valid SG&A label exists in any financial table → output null for ALL years.\n"
            "   - SG&A is always output as a POSITIVE number (it is a cost/expense).\n"
            "     If the table shows SG&A in parentheses e.g. (13.2) or as a negative → convert to positive.\n"
            "     Example: table shows (13,200) or -13,200 → output 13200.\n"
            "10. ADJ. EBITDA EXTRACTION LOGIC:\n"
            "   OUTPUT: flat numeric value per year (same format as Total_Revenue). Use null if not found.\n\n"
            "   PRIORITY ORDER — attempt each step in order, stop at first success:\n\n"
            "   STEP 1 — DIRECT EXTRACTION (preferred):\n"
            "     Scan P&L tables for a row explicitly labeled one of (in this priority):\n"
            "       a) 'Adjusted EBITDA' or 'Adj. EBITDA' or 'Adj EBITDA' or 'EBITDA (Adjusted)' or 'EBITDA (as adjusted)'\n"
            "       b) 'Management Adjusted EBITDA' or 'Mgmt. Adj. EBITDA' or 'Management EBITDA'\n"
            "       c) 'Normalized EBITDA' or 'Normalised EBITDA'\n"
            "       d) 'Run-Rate EBITDA' or 'Run Rate EBITDA' — treat as equivalent to Adjusted EBITDA\n"
            "       e) 'EBITDA before one-time items' or 'EBITDA excl. one-time items' or 'EBITDA ex. non-recurring'\n"
            "       f) 'EBITDA' (plain, no qualifier) — ONLY if no adjusted variant exists anywhere in the document\n"
            "     Rules:\n"
            "       - If BOTH 'Adjusted EBITDA' AND 'PF Adjusted EBITDA' (Pro Forma) exist → use 'Adjusted EBITDA' (NOT Pro Forma).\n"
            "       - If ONLY 'PF Adjusted EBITDA' or 'Pro Forma Adjusted EBITDA' or 'Pro Forma EBITDA' exists → use it.\n"
            "       - If a table is labeled 'Pro Forma Adjusted Income Statement' — this contains the Adj. EBITDA figures, extract from there.\n"
            "       - CROSS-PERIOD RULE: If 'Adjusted EBITDA' covers only historical years and 'PF Adjusted EBITDA'\n"
            "         covers projected years in different tables, combine both — use Adj. EBITDA for historical\n"
            "         years and PF Adj. EBITDA for projected years to ensure all periods are populated.\n"
            "       - NEVER extract from charts, dashboards, or bullet points — only structured financial tables.\n\n"
            "   STEP 2 — CALCULATE: Reported EBITDA + Total Management Adjustments:\n"
            "     Only attempt if Step 1 failed.\n"
            "     Conditions — ALL must be true in the SAME table:\n"
            "       a) A row labeled 'Reported EBITDA', 'EBITDA' (unadjusted), or 'EBITDA before adjustments' is present.\n"
            "       b) A row labeled 'Total Management Adjustments', 'Total Adjustments', or 'Total Add-backs' is present.\n"
            "     → Adj. EBITDA = Reported EBITDA + Total Management Adjustments.\n"
            "     Note: Adjustments are typically positive (add-backs); verify sign before calculating.\n\n"
            "   HARD RULES:\n"
            "     - NEVER use Operating Income or Net Income as a proxy for Adj. EBITDA.\n"
            "     - NEVER mix values from different tables or pages.\n"
            "     - Convert parentheses to negative: (1,200) = -1200.\n"
            "     - Units: normalise to $000s. Check table header for '$ in thousands' vs '$ in millions' vs 'CAD millions'.\n"
            "       If currency is CAD (Canadian dollars), still output the number as-is in thousands — do NOT convert to USD.\n"
            "     - If a document has multiple EBITDA adjustment schedules (e.g. management + synergy), use ONLY the\n"
            "       management-adjusted figure, NOT the synergy/pro forma layer unless it is the only one available.\n"
            "     - If a year has no extractable or calculable Adj. EBITDA → output null.\n"
            "11. EBITDA (REPORTED/PLAIN) EXTRACTION LOGIC:\n"
            "   This is the unadjusted/reported EBITDA — BEFORE any management add-backs or adjustments.\n"
            "   OUTPUT: flat numeric value per year. Use null if not found.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (in priority):\n"
            "       a) 'Reported EBITDA'\n"
            "       b) 'EBITDA' (plain, with no qualifier like Adjusted/Normalized/Pro Forma)\n"
            "       c) 'EBITDA before adjustments' or 'EBITDA before add-backs'\n"
            "     STEP 2 — If no plain EBITDA row exists but an adjustment table shows 'EBITDA' as the starting\n"
            "       line before listing adjustments → extract that value as Reported EBITDA.\n"
            "   HARD RULES:\n"
            "     - NEVER use Adjusted EBITDA, Adj. EBITDA, Normalized EBITDA, or PF EBITDA for this field.\n"
            "     - NEVER use Operating Income or Net Income as a proxy.\n"
            "     - NEVER extract from charts, dashboards, or bullet points — only structured financial tables.\n"
            "     - NEVER mix values from different tables.\n"
            "     - Convert parentheses to negative: (1,200) = -1200.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If a document only has Adjusted EBITDA and no plain EBITDA → output null for this field.\n"
            "     - If a year has no plain/reported EBITDA → output null for that year.\n"
            "12. OTHER EXPENSE / (INCOME) EXTRACTION LOGIC:\n"
            "   This is the non-operating 'Other expense / (income)' line (Excel row 21).\n"
            "   OUTPUT: flat numeric value per year. Use null if not found.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Other expense / (income)' or 'Other (income) / expense'\n"
            "       b) 'Other income / expenses' or 'Other income/expense'\n"
            "       c) 'Other income, net' or 'Other expense, net'\n"
            "       d) 'Non-operating expense' or 'Non-operating income'\n"
            "       e) 'Other income (expense)' or 'Other expense (income)'\n"
            "     STEP 2 — Also check EBITDA adjustment schedules: if a line explicitly labeled\n"
            "       'Other Income / Expenses' or 'Other expense' appears inside the adjustment table, extract it.\n"
            "   HARD RULES:\n"
            "     - Extract ONLY if the row is explicitly labeled as above in a structured financial table.\n"
            "     - NEVER extract if the label also includes 'interest' (e.g. 'Interest and other expense') —\n"
            "       that is Interest Expense, not Other Expense.\n"
            "     - NEVER derive or calculate this field from other lines.\n"
            "     - NEVER extract from charts, dashboards, bullet points, or narrative text.\n"
            "     - Sign convention: expense shown in parentheses = negative number.\n"
            "       If shown as positive (no parentheses) in an expense section → keep positive.\n"
            "     - Convert parentheses to negative: (1,296) = -1296.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - Extract ONLY for years where the value is explicitly stated in the table.\n"
            "     - If the line is not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year.\n"
            "13. INTEREST EXPENSE EXTRACTION LOGIC:\n"
            "   This is the interest paid on debt — a non-operating P&L line (Excel row 22).\n"
            "   OUTPUT: flat numeric value per year. HISTORICAL YEARS ONLY (YYYY_A or TTM_YYYY).\n"
            "           Do NOT output any projected/estimated years (YYYY_E) — output null for those.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Interest expense'\n"
            "       b) 'Interest expense, net'\n"
            "       c) 'Net interest expense'\n"
            "       d) 'Interest and debt expense'\n"
            "       e) 'Finance costs' or 'Finance charges' (common in Canadian/UK documents)\n"
            "       f) 'Interest charges' or 'Interest on debt'\n"
            "   HARD RULES:\n"
            "     - Extract ONLY if the row is explicitly labeled as one of the above in a structured financial table.\n"
            "     - NEVER extract a row labeled 'Interest and other expense' or 'Interest and other income' —\n"
            "       those are bundled and cannot be separated; output null.\n"
            "     - NEVER derive or calculate interest expense from other line items.\n"
            "     - NEVER extract from charts, dashboards, bullet points, or narrative text.\n"
            "     - NEVER extract from the Balance Sheet (e.g. 'Accrued Interest') — P&L only.\n"
            "     - HISTORICAL YEARS ONLY: If the table shows projected years (YYYY_E), skip them — output null.\n"
            "       Only extract for years with an actual/historical suffix (YYYY_A or TTM_YYYY).\n"
            "     - If a document shows multiple debt instruments separately (Senior debt interest,\n"
            "       Subordinated note interest, etc.) — use ONLY the 'Total interest expense' subtotal row.\n"
            "     - Sign convention: interest expense is typically a positive number (a cost to the business).\n"
            "       If shown in parentheses (e.g. (2,500)) → convert to -2500 and output as-is.\n"
            "     - Convert parentheses to negative: (2,500) = -2500.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year.\n"
            "14. DEPRECIATION EXTRACTION LOGIC:\n"
            "   This is the depreciation (non-cash) charge from the P&L (Excel row 23).\n"
            "   OUTPUT: flat numeric value per year. Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Depreciation'\n"
            "       b) 'Depreciation & Amortization' or 'Depreciation and Amortization' or 'D&A'\n"
            "       c) 'Depreciation of fixed assets' or 'Depreciation of property, plant & equipment'\n"
            "     STEP 2 — If not in the main P&L, also check EBITDA bridge tables where D&A appears\n"
            "       as an add-back line (e.g. 'Operating Income + D&A = EBITDA' format).\n"
            "   HARD RULES:\n"
            "     - Extract ONLY if the row is explicitly labeled as above in a structured financial table.\n"
            "     - If the document shows 'Depreciation' and 'Amortization' on SEPARATE rows, extract\n"
            "       only the 'Depreciation' row — do NOT add them together unless the label says 'D&A'.\n"
            "     - NEVER extract 'Amortization of acquired intangibles' or 'Amortization of intangibles'\n"
            "       as Depreciation — those are post-acquisition accounting items, not operating depreciation.\n"
            "     - NEVER derive or calculate depreciation from other line items.\n"
            "     - NEVER extract from charts, dashboards, bullet points, or narrative text.\n"
            "     - Extract BOTH historical (YYYY_A) AND projected (YYYY_E) years — unlike Interest Expense.\n"
            "     - A value of 0 (zero) is valid — output 0, not null.\n"
            "     - Convert parentheses to negative: (1,200) = -1200.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year.\n"
            "15. CAPEX EXTRACTION LOGIC:\n"
            "   Capital expenditures — cash spent on fixed assets (Excel row 32). Output as NEGATIVE numbers.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Look for a row explicitly labeled one of (case-insensitive):\n"
            "       a) 'Total Capex' or 'Total Capital Expenditures' or 'Total CapEx'\n"
            "       b) 'Capital Expenditures' or 'Capex' (standalone summary row)\n"
            "       c) 'Capital Spending' or 'Purchases of property, plant & equipment'\n"
            "   HARD RULES:\n"
            "     - ALWAYS use the 'Total Capex' subtotal row — NEVER sum individual breakdown lines\n"
            "       (e.g. Warehouse Equipment, Building Improvements, Maintenance, Growth, Other).\n"
            "     - If the document shows TWO Total Capex rows (e.g. 'Total Capex' and\n"
            "       'Total Capex (incl. Plant Consolidations)' or similar with one-time items) —\n"
            "       use the FIRST 'Total Capex' row (base recurring capex). NEVER use the inflated\n"
            "       version that includes one-time plant consolidations or non-recurring items.\n"
            "     - OUTPUT AS NEGATIVE: multiply the extracted value by -1 before outputting.\n"
            "       E.g. if the table shows 2,490 → output -2490. If already in parentheses (2,490) → output -2490.\n"
            "       Do NOT double-negate: if source shows (2,490) it is already a cash outflow → output -2490.\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "     - NEVER derive capex from fixed asset schedule changes (e.g. delta in gross PP&E).\n"
            "     - Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - Zero (0) is a valid value — output 0, not null.\n"
            "     - If not present in the document → output null for all years.\n"
            "     - If a year has no value in an otherwise present row → output null for that year.\n"
            "16. WORKING CAPITAL CHANGE EXTRACTION LOGIC:\n"
            "   Year-over-year change in Net Working Capital (Excel row 33).\n"
            "   Sign convention: NWC INCREASE = cash outflow = NEGATIVE. NWC DECREASE = cash inflow = POSITIVE.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years where available.\n\n"
            "   STEP 1 — PREFERRED: Look for an explicitly labeled change/movement row (case-insensitive):\n"
            "       a) 'Change in NWC' or 'Change in Working Capital' or 'Changes in Working Capital'\n"
            "       b) 'Change in Operating Working Capital' or 'Net Working Capital Change'\n"
            "       c) '(Growth) / Decline in Net Working Capital' or 'Growth / (Decline) in Net Working Capital'\n"
            "       d) 'Increase / (Decrease) in Working Capital' or '(Increase) / Decrease in Working Capital'\n"
            "       e) 'Working Capital Change'\n"
            "     If found → extract all years from that row directly. Do NOT also apply Step 2.\n\n"
            "   STEP 2 — FALLBACK (only if Step 1 finds nothing): Derive from NWC balance rows.\n"
            "     Condition: a table contains a row labeled 'Net Working Capital', 'Adjusted Net Working Capital',\n"
            "       or 'NWC' with values for 2 or more years.\n"
            "     Method: WC_Change(year N) = NWC(year N) - NWC(year N-1)\n"
            "       - First year in the table → output null (no prior period exists).\n"
            "       - Apply to ALL consecutive year pairs found in the table (historical and projected).\n"
            "       - If only 1 year of NWC balance data exists → output null for all years (cannot derive).\n"
            "     Sign check after derivation:\n"
            "       - Result is POSITIVE → NWC increased → cash outflow → output as NEGATIVE.\n"
            "       - Result is NEGATIVE → NWC decreased → cash inflow → output as POSITIVE.\n"
            "       (Reverse the sign of the raw arithmetic result to match cash flow convention.)\n\n"
            "   SIGN RULES for Step 1 (explicit row) — WORKED EXAMPLES:\n"
            "     Universal rule: parentheses in the source value always convert to negative.\n\n"
            "     EXAMPLE A — label: 'Change in NWC' or 'Change in Working Capital':\n"
            "       Source shows  9,900  (no parentheses) → NWC fell → cash inflow → output  9900\n"
            "       Source shows (9,900) (parentheses)    → NWC rose → cash outflow → output -9900\n\n"
            "     EXAMPLE B — label: '(Growth) / Decline in Net Working Capital':\n"
            "       The label word 'Growth' is in parentheses meaning growth is the negative direction.\n"
            "       Source shows  5,000  (no parentheses) → this is a Decline → NWC fell → cash inflow → output  5000\n"
            "       Source shows (5,000) (parentheses)    → this is a Growth  → NWC rose → cash outflow → output -5000\n\n"
            "     EXAMPLE C — label: 'Increase / (Decrease) in Working Capital':\n"
            "       Source shows  3,000  (no parentheses) → NWC increased → cash outflow → output -3000\n"
            "       Source shows (3,000) (parentheses)    → NWC decreased → cash inflow  → output  3000\n"
            "       NOTE: For this label type only, positive raw value = NWC increase = NEGATIVE output.\n"
            "             Reverse the sign relative to Example A.\n\n"
            "     SUMMARY: Always ask 'did NWC go UP or DOWN?' — UP = negative output, DOWN = positive output.\n"
            "     When in doubt, apply: parentheses in the value → negative output; no parentheses → positive output.\n\n"
            "   HARD RULES (both steps):\n"
            "     - NEVER derive from individual component changes (AR, Inventory, Payables separately).\n"
            "     - NEVER extract a single NWC balance value as WC_Change.\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "     - Normalise units to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If Step 1 and Step 2 both fail → output null for all years.\n"
            "17. ACCOUNTS RECEIVABLE (AR) EXTRACTION LOGIC:\n"
            "   The AR balance used as ABL collateral in the Sources section (Excel C7).\n"
            "   OUTPUT: a SINGLE numeric value — the most recent actual balance only (NOT a time series).\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Find a balance sheet or NWC table containing AR balances across multiple periods.\n"
            "       Look for rows labeled (case-insensitive):\n"
            "         a) 'Accounts Receivable' or 'Account Receivable'\n"
            "         b) 'Accounts Receivable, net' or 'Trade Receivables'\n"
            "         c) 'Receivables, net' or 'Net Receivables' or 'AR'\n"
            "     STEP 2 — Identify the MOST RECENT ACTUAL period in the table:\n"
            "         'Actual' means a year column whose label ends in 'A' or 'a' (e.g. 2024A, FY24A, Sep-24A).\n"
            "         Priority order: TTM_YYYY > highest YYYY_A > YYYY_B (budget, last resort) > YYYY_E (last resort)\n"
            "         *** CRITICAL WARNING: CIM tables often extend 5-8 years of projections beyond the actuals.\n"
            "             The LAST column in the table is almost always a PROJECTED year (E/B/F suffix).\n"
            "             NEVER use it as 'most recent'. Scan left from the right until you find the last 'A' column.\n"
            "             EXAMPLE: columns = 2023A, 2024A, 2025B, 2026E, 2027E, 2028E, 2029E, 2030E\n"
            "                      → most recent actual = 2024A (second column), NOT 2030E (last column). ***\n"
            "         Use the value from that single actual period only.\n"
            "   HARD RULES:\n"
            "     - Output ONE single number — not a dict of years. Format: \"AR\": 6147\n"
            "     - ALWAYS use net value (after allowance for doubtful accounts) if available.\n"
            "       If only gross AR is available, use gross.\n"
            "     - NEVER extract AR from a P&L, income statement, or cash flow adjustments table.\n"
            "       AR must come from a balance sheet or NWC schedule.\n"
            "     - NEVER use AR movement/change values — only the balance (stock) amount.\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "     - If multiple balance sheet snapshots exist (e.g. Dec-23A, Dec-24A, Dec-25A),\n"
            "       always pick the most recent one (Dec-25A in this example).\n"
            "     - If only a single snapshot is shown (e.g. 'As of Sep-25'), use that value.\n"
            "     - Normalise to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If AR is not found in any balance sheet or NWC table → output null.\n"
            "18. INVENTORY EXTRACTION LOGIC:\n"
            "   The Inventory balance used as ABL collateral in the Sources section (Excel C8).\n"
            "   OUTPUT: a SINGLE numeric value — the most recent actual balance only (NOT a time series).\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Find a balance sheet or NWC table containing inventory balances.\n"
            "       Look for rows labeled (case-insensitive):\n"
            "         a) 'Inventory' or 'Inventories' or 'Inventory, net'\n"
            "         b) 'Stock' (common in UK/Canadian documents)\n"
            "         c) 'Total Inventory' (if broken into components)\n"
            "         d) 'Raw Materials', 'Work in Progress', 'Finished Goods' — BUT only if a\n"
            "            'Total Inventory' subtotal row also exists; use the subtotal, never sum manually.\n"
            "     STEP 2 — Identify the MOST RECENT ACTUAL period in the table:\n"
            "         'Actual' means a year column whose label ends in 'A' or 'a' (e.g. 2024A, FY24A).\n"
            "         Priority order: TTM_YYYY > highest YYYY_A > YYYY_B (budget, last resort) > YYYY_E (last resort)\n"
            "         *** CRITICAL WARNING: CIM tables often extend 5-8 projected years beyond the actuals.\n"
            "             The LAST column in the table is almost always a PROJECTED year (E/B/F suffix).\n"
            "             NEVER use it as 'most recent'. Scan left from the right until you find the last 'A' column.\n"
            "             EXAMPLE: columns = 2023A, 2024A, 2025B, 2026E, ..., 2030E\n"
            "                      → most recent actual = 2024A (second column), NOT 2030E (last column). ***\n"
            "         Use the value from that single actual period only.\n"
            "   HARD RULES:\n"
            "     - Output ONE single number — not a dict of years. Format: \"Inventory\": 13512\n"
            "     - Zero (0) is a valid value — service businesses with no physical goods output 0, not null.\n"
            "     - ALWAYS use net value (after obsolescence/write-down reserve) if shown.\n"
            "       If only gross inventory is available, use gross.\n"
            "     - If inventory is broken into components (Raw Materials / WIP / Finished Goods),\n"
            "       use the 'Total Inventory' subtotal row ONLY. NEVER sum the components yourself.\n"
            "     - NEVER extract inventory movement/change values from a cash flow statement.\n"
            "       Inventory must come from a balance sheet or NWC schedule (a stock/balance, not a flow).\n"
            "     - NEVER extract from narrative text, bullet points, or charts.\n"
            "       Example: 'inventory reduction from $34M in 2022' in narrative → ignore.\n"
            "     - If multiple balance sheet snapshots exist, always pick the most recent period.\n"
            "     - If only a single snapshot is shown (e.g. 'As of Sep-25'), use that value.\n"
            "     - Normalise to $000s. If CAD, output as-is without USD conversion.\n"
            "     - If Inventory is not found in any balance sheet or NWC table → output null.\n"
            "19. MARKET INTELLIGENCE EXTRACTION LOGIC:\n"
            "   Extract competitive landscape and market intelligence data from the CIM.\n"
            "   OUTPUT: structured object with these 6 sub-fields (each independently null if not found):\n"
            "   {\n"
            "     \"market_size\": string or null,\n"
            "     \"market_growth_rate\": string or null,\n"
            "     \"market_position\": string or null,\n"
            "     \"competitors\": array of strings or null,\n"
            "     \"industry_tailwinds\": array of strings or null,\n"
            "     \"barriers_to_entry\": array of strings or null\n"
            "   }\n\n"
            "   SUB-FIELD RULES:\n\n"
            "   A) market_size — overall industry/market size as stated in the document:\n"
            "     Accept: TAM, Addressable Market, Industry Revenue, Total Market, '[Industry] Market', etc.\n"
            "     Extract figure + units + year if stated (e.g. '$48.9 billion USD (2024)').\n"
            "     Range → preserve as-is. Multi-segment → total if stated, else list segments.\n"
            "     Qualitative only ('large', 'growing') → null. Not found → null.\n\n"
            "   B) market_growth_rate — stated CAGR or growth rate:\n"
            "     Extract rate + time horizon if given (e.g. '~8% CAGR (2024–2028)').\n"
            "     Range → preserve as-is. Qualitative only → null. Not found → null.\n\n"
            "   C) market_position — company's stated competitive rank or position:\n"
            "     SOURCE: investment highlights, executive summary, competitive overview.\n"
            "     Examples: '#1 manufacturer in North America', 'Top 4 security technology partner', 'market leader in specialty polymers'.\n"
            "     Extract the most specific and prominent position claim. One string only.\n"
            "     NEVER infer position — only extract if explicitly stated in the document.\n"
            "     Null if no position claim found.\n\n"
            "   D) competitors — named competitor companies:\n"
            "     Extract only company/brand names explicitly named as competitors.\n"
            "     Exclude the company itself. Max 12. Array of strings.\n"
            "     Null if no named competitors found (not an empty array).\n\n"
            "   E) industry_tailwinds — key demand/growth drivers for the industry:\n"
            "     SOURCE: market overview, investment highlights, industry trends sections.\n"
            "     Extract 2–5 distinct tailwinds explicitly stated as growth drivers (e.g. '5G densification', 'aging baby boomers driving RV demand', 'EV charging infrastructure buildout').\n"
            "     Each tailwind: short descriptive phrase (max 10 words), as stated or closely paraphrased.\n"
            "     Array of strings. Null if no tailwinds section present.\n\n"
            "   F) barriers_to_entry — factors that protect the company from new competition:\n"
            "     SOURCE: competitive advantages, investment highlights, competitive positioning sections.\n"
            "     Extract 2–5 distinct barriers explicitly cited (e.g. 'long-term MSA contracts', 'proprietary formulations held as trade secrets', 'certification requirements', 'switching costs').\n"
            "     Each barrier: short descriptive phrase (max 10 words).\n"
            "     Array of strings. Null if no barriers stated.\n\n"
            "   GLOBAL HARD RULES for Market_Intelligence:\n"
            "     - NEVER hallucinate or infer data not in the document.\n"
            "     - NEVER extract from financial tables — narrative and overview sections only.\n"
            "     - NEVER include financial metrics (revenue, EBITDA, margins).\n"
            "     - If entire market/competitive section absent → output null for the whole field.\n"
            "     - Otherwise output the full object with null for any missing sub-field.\n"
            "20. 1X ADJUSTMENTS EXTRACTION LOGIC:\n"
            "   These are non-recurring / one-time add-backs in the EBITDA bridge table that reconcile\n"
            "   Reported EBITDA to Adjusted EBITDA. Extract the TOTAL adjustment amount per year.\n"
            "   OUTPUT: flat numeric value per year (positive = net add-back). Historical years ONLY (YYYY_A / TTM_YYYY).\n"
            "   Do NOT extract projected (YYYY_E) years — projected adjustments are always $0 by convention.\n\n"
            "   STEP 1 — DIRECT TOTAL ROW (preferred):\n"
            "     Scan EBITDA bridge / adjustment schedule tables for a row explicitly labeled:\n"
            "     'Total Adjustments', 'Total Management Adjustments', 'Total normalizations',\n"
            "     'Net Adjustments', 'Total Add-backs', 'Total 1x Items'.\n"
            "     If found → extract the per-year values from that row.\n\n"
            "   STEP 2 — DERIVE: Adj_EBITDA minus EBITDA (fallback only):\n"
            "     Only attempt if Step 1 fails (no explicit total row found).\n"
            "     Conditions — BOTH must be present in the SAME table or from already-extracted values:\n"
            "       a) Adj_EBITDA value for the year\n"
            "       b) Plain EBITDA (Reported EBITDA) value for the year\n"
            "     Formula: Onex_Adjustments = Adj_EBITDA - EBITDA\n"
            "     Only compute for years where BOTH values are non-null.\n\n"
            "   STEP 3 — NULL:\n"
            "     If neither Step 1 nor Step 2 succeeds → output null.\n\n"
            "   HARD RULES:\n"
            "     - NEVER use individual adjustment line items (e.g. 'Management Fees', 'Legal') — use the TOTAL only.\n"
            "     - NEVER extract Pro Forma / synergy adjustment tiers — Management adjustments only.\n"
            "     - Sign: positive = net add-back (increases EBITDA). If total row shows parentheses → negative.\n"
            "     - HISTORICAL ONLY: do NOT output YYYY_E values (projected 1x = $0, not meaningful).\n"
            "     - If projected years show $0 in the total row, output null for those years (not 0).\n"
            "     - NEVER extract from P&L rows labeled 'Other Expense' or 'Other Income' unless in bridge table.\n"
            "     - NEVER mix adjustment tiers (e.g. do not add Management + PF + Synergy together).\n"
            "     - YEAR LABEL ALIGNMENT (CRITICAL): Use the SAME year key suffix (_A or _E) that the main P&L\n"
            "       uses for that calendar year — NOT what the adjustment bridge table header shows.\n"
            "       EXAMPLE: If the P&L labels 2023 as '2023E' (projected) but the adjustment table header\n"
            "       says '2023A' (because those adjustments were reported) → STILL output '2023_A' ONLY if\n"
            "       that year already appears as '_A' in Total_Revenue or Adj_EBITDA. If those fields use\n"
            "       '2023_E', then 2023 is a projected year — skip it entirely for Onex_Adjustments (output null).\n"
            "       RULE: A year key used in Onex_Adjustments MUST already exist as YYYY_A in at least one\n"
            "       other field (Total_Revenue, Gross_Margin, Adj_EBITDA). If not → do not output that year.\n"
            "22. COMPANY KPIs EXTRACTION LOGIC:\n"
            "   Extract key operational snapshot metrics from the company overview / 'by the numbers' section.\n"
            "   OUTPUT: structured object with exactly these sub-fields (all scalars, NOT time series):\n"
            "     founded_year: integer — the year the company was founded/established. null if not stated.\n"
            "     total_employees: integer — total headcount (FTEs). Use most recent figure. null if not stated.\n"
            "     num_locations: integer — number of physical office/facility/plant locations. null if not stated.\n"
            "     countries_of_operation: integer — number of countries the company operates in. null if not stated.\n"
            "     capacity_utilization: string — current production/facility utilization as stated (e.g. '~60%', '35%'). null if not stated.\n"
            "   SOURCE: company overview, executive summary, 'by the numbers' box, facilities section, or intro pages.\n"
            "   HARD RULES:\n"
            "     - Only extract what is explicitly stated — NEVER infer or calculate.\n"
            "     - founded_year: use original founding year, NOT acquisition year by current owner.\n"
            "     - total_employees: use the most recent headcount figure. If range given ('200-250'), use midpoint.\n"
            "     - num_locations: count only distinct physical locations (offices, plants, warehouses). Not virtual/digital.\n"
            "     - If the entire section is absent → output null for the whole Company_KPIs field.\n"
            "     - If only some sub-fields are found, output an object with found values and null for missing ones.\n"
            "28. MACHINERY & EQUIPMENT (M&E) EXTRACTION LOGIC:\n"
            "   The gross book value of Machinery & Equipment assets used as ABL collateral (Sources section C9).\n"
            "   OUTPUT: a SINGLE numeric value — the most recent actual period only (NOT a time series).\n\n"
            "   STEP 1 — SOURCE TABLE: Find a Fixed Asset Schedule, PP&E Schedule, or Property Plant & Equipment table.\n"
            "     These tables typically appear in the financial appendix or balance sheet supplemental schedules.\n"
            "     Look for rows labeled (case-insensitive, accept any of these):\n"
            "       a) 'Machinery & Equipment' or 'Machinery and Equipment' or 'M&E'\n"
            "       b) 'Plant & Equipment' or 'Plant and Equipment'\n"
            "       c) 'Equipment' (standalone row in a fixed asset schedule)\n"
            "       d) 'Warehouse Equipment' or 'Manufacturing Equipment' or 'Production Equipment'\n"
            "       e) 'Furniture, Fixtures & Equipment' or 'FF&E' (only if no other M&E row exists)\n"
            "       f) 'Machinery' (standalone row)\n"
            "   STEP 2 — PERIOD SELECTION: Identify the MOST RECENT ACTUAL period:\n"
            "     'Actual' = column labeled with suffix A or a (e.g. 2024A, Dec-24A, FY24A).\n"
            "     Priority: TTM_YYYY > latest YYYY_A > YYYY_B > YYYY_E (only absolute last resort).\n"
            "     *** CRITICAL WARNING: Fixed asset schedules often extend 5-8 projected years (E/F/B).\n"
            "         The LAST column is almost always projected — NEVER use it as most recent.\n"
            "         Example: columns = 2022A, 2023A, 2024A, 2025B, 2026E ... 2030E\n"
            "         → most recent actual = 2024A (third column), NOT 2030E (last column). ***\n"
            "   STEP 3 — VALUE SELECTION:\n"
            "     a) Use GROSS value (before depreciation/accumulated amortization) if the schedule shows both\n"
            "        gross and net. Gross value = the original cost, before any write-downs.\n"
            "     b) If only NET value (after depreciation) is shown, use net and note it is net.\n"
            "     c) If the row is broken into sub-components (e.g. 'Warehouse Equipment' and 'Office Equipment'\n"
            "        separately), use the individual row for the best M&E match — do NOT sum them.\n"
            "     d) If a 'Total Fixed Assets' or 'Total PP&E' subtotal row exists — NEVER use it for this field.\n"
            "        Always use the individual M&E row only.\n"
            "   HARD RULES:\n"
            "     - SOURCE: Fixed Asset Schedule / PP&E Schedule ONLY.\n"
            "       NEVER extract from: P&L, Income Statement, Cash Flow Statement, CAPEX tables,\n"
            "       narrative text, bullet points, charts, or appraisal summaries (unless they explicitly\n"
            "       state a book value for M&E as a standalone line).\n"
            "     - NEVER use CAPEX line items (e.g. 'Warehouse Equipment $864' from a CAPEX table)\n"
            "       as M&E — CAPEX rows are annual spending amounts, not asset balances.\n"
            "     - NEVER sum multiple rows to create M&E — use only the single best-matching row.\n"
            "     - NEVER extract 'Office Furniture & Equipment' or 'Leasehold Improvements' as M&E\n"
            "       unless no other M&E row exists anywhere in the document.\n"
            "     - NEVER extract 'Total Fixed Assets (Gross)' or 'Total Fixed Assets (Net)' — that is\n"
            "       the total, not M&E. We need the M&E line item only.\n"
            "     - A value of 0 (zero) is valid — output 0, not null.\n"
            "     - Normalise to $000s. If document is in $M, multiply by 1000. If CAD, output as-is.\n"
            "     - If no M&E row exists in any fixed asset schedule → output null.\n"
            "     - Output ONE single number. Format: \"ME_Equipment\": 14067\n\n"
            "29. BUILDING & LAND EXTRACTION LOGIC:\n"
            "   The gross book value of Building & Land real estate assets used as ABL collateral (Sources section C11).\n"
            "   OUTPUT: a SINGLE numeric value — the most recent actual period only (NOT a time series).\n\n"
            "   STEP 1 — SOURCE TABLE: Find a Fixed Asset Schedule, PP&E Schedule, or Property Plant & Equipment table.\n"
            "     Look for rows labeled (case-insensitive, accept any of these):\n"
            "       a) 'Building & Land' or 'Building and Land' or 'Buildings & Land'\n"
            "       b) 'Building' (standalone row — distinct from 'Building Improvements')\n"
            "       c) 'Land' (standalone row)\n"
            "       d) 'Real Estate' or 'Property' (standalone row in a fixed asset schedule)\n"
            "       e) 'Land and Building' or 'Land & Buildings'\n"
            "       f) 'Leasehold' or 'Leasehold Property' (only if company owns the property, not renting)\n"
            "     IMPORTANT DISTINCTION:\n"
            "       'Building Improvements' or 'Leasehold Improvements' is NOT the same as 'Building'.\n"
            "       Building Improvements = renovation/upgrade costs (tenant improvements).\n"
            "       Building = the actual owned real property (structure + land value).\n"
            "       ONLY extract 'Building Improvements' if NO 'Building' or 'Land' row exists AND the company\n"
            "       clearly owns the property (not a tenant). Otherwise output null for Building & Land.\n"
            "   STEP 2 — PERIOD SELECTION: Identify the MOST RECENT ACTUAL period:\n"
            "     'Actual' = column labeled with suffix A or a (e.g. 2024A, Dec-24A).\n"
            "     Priority: TTM_YYYY > latest YYYY_A > YYYY_B > YYYY_E.\n"
            "     *** CRITICAL WARNING: Same as M&E — the last column in a fixed asset table is almost\n"
            "         always projected. Scan LEFT from the right to find the last 'A'-suffixed column. ***\n"
            "   STEP 3 — VALUE SELECTION:\n"
            "     a) Use GROSS value (before accumulated depreciation) if both gross and net are shown.\n"
            "     b) If 'Building' and 'Land' are on SEPARATE rows — add them together for Building_Land output.\n"
            "        EXCEPTION: if only one of the two rows has a value and the other is null/zero, use the available one.\n"
            "     c) If only NET (post-depreciation) value is available, use net.\n"
            "     d) NEVER use 'Total Fixed Assets' subtotal — use the Building/Land individual rows only.\n"
            "   HARD RULES:\n"
            "     - SOURCE: Fixed Asset Schedule / PP&E Schedule ONLY.\n"
            "       NEVER extract from: P&L, CAPEX tables, narrative descriptions, facility square footage tables,\n"
            "       real estate listings, or appraisal summaries (unless they state a book value for building/land).\n"
            "     - NEVER confuse 'Building Improvements' with 'Building' — they are different rows.\n"
            "       Building Improvements = tenant improvements (capex spend); Building = owned asset book value.\n"
            "     - NEVER extract square footage, number of locations, or lease terms as Building & Land value.\n"
            "     - NEVER extract from a real estate overview table (which lists sq footage and headcount per site).\n"
            "     - A value of 0 (zero) is valid — output 0, not null.\n"
            "     - Normalise to $000s. If document is in $M, multiply by 1000. If CAD, output as-is.\n"
            "     - If no Building or Land row exists in any fixed asset schedule → output null.\n"
            "     - Output ONE single number. Format: \"Building_Land\": 3250\n"
            "27. GROWTH INITIATIVES EXTRACTION LOGIC:\n"
            "   Extract the company's key strategic growth initiatives or pillars.\n"
            "   OUTPUT: array of objects — [{\"title\": \"Short Title\", \"description\": \"1-2 sentence summary\", \"impact\": \"$ or % figure or null\"}, ...]\n"
            "   SOURCE: growth strategy section, investment highlights, strategic initiatives pages.\n"
            "   HARD RULES:\n"
            "     - title: short label for the initiative (max 6 words), as stated or summarized from the document.\n"
            "     - description: 1–2 sentence factual summary of what the initiative entails. No promotional language.\n"
            "     - impact: ONLY if an explicit $ or % financial impact is stated for that initiative (e.g. '$2.0M EBITDA impact', '15% of sales'). Otherwise null.\n"
            "     - Max 8 initiatives. If more, take the most prominent ones.\n"
            "     - Do NOT include financial table data as initiatives.\n"
            "     - NEVER invent initiatives not in the document.\n"
            "     - Output null if no growth strategy section is present in the document.\n"
            "26. MANAGEMENT TEAM EXTRACTION LOGIC:\n"
            "   Extract the key management team members from the CIM.\n"
            "   OUTPUT: array of objects — [{\"name\": \"Full Name\", \"title\": \"Job Title\", \"experience\": \"X years\" or null}, ...]\n"
            "   SOURCE: management team section, leadership overview, executive team pages.\n"
            "   HARD RULES:\n"
            "     - Extract only named individuals with an explicit job title.\n"
            "     - experience: extract ONLY if a total years of experience figure is explicitly stated for that person (e.g. '25+ years'). Otherwise null.\n"
            "     - Max 10 people. If more, take the most senior (C-suite and VP level first).\n"
            "     - Do NOT include board members, advisors, or investors — only operating management.\n"
            "     - Do NOT include prior employer names or education in the output — name, title, experience only.\n"
            "     - NEVER invent people not in the document.\n"
            "     - Output null if no management team section is present in the document.\n"
            "25. REVENUE BY GEOGRAPHY EXTRACTION LOGIC:\n"
            "   Extract the company's revenue breakdown by geographic region as percentages.\n"
            "   OUTPUT: array of objects — [{\"region\": \"Name\", \"pct\": number}, ...] sorted descending by pct.\n"
            "   SOURCE: company overview, geographic breakdown charts/tables, revenue by region/country/state sections.\n"
            "   HARD RULES:\n"
            "     - Use most recent actual year data available.\n"
            "     - Regions can be countries, states, continents, or named territories — use whatever the document uses.\n"
            "     - Values are percentages (0–100). Must sum to ~100% (allow ±3% for rounding).\n"
            "     - Max 10 regions. If more, group smallest as 'Other'.\n"
            "     - Single-geography companies (operates only in one country/region, no breakdown given) → output null.\n"
            "     - Do NOT confuse revenue by geography with revenue by end market or segment.\n"
            "     - NEVER invent geographic data not in the document.\n"
            "     - Output null if no geographic revenue breakdown exists in the document.\n"
            "24. CUSTOMER CONCENTRATION EXTRACTION LOGIC:\n"
            "   Extract the revenue concentration across customer tiers as percentages.\n"
            "   OUTPUT: array of objects — [{\"tier\": \"Label\", \"pct\": number}, ...] sorted descending by pct.\n"
            "   SOURCE: customer overview, customer concentration charts/tables, 'by the numbers' section.\n"
            "   PREFERRED FORMAT — tier buckets (use if available):\n"
            "     e.g. Top 1 Customer, Top 2-5, Top 6-10, All Others — extract the % each bucket represents.\n"
            "   FALLBACK FORMAT — if only named customer % listed:\n"
            "     Use customer aliases (Customer A, Customer B, etc.) or generic labels (Largest Customer, etc.).\n"
            "   HARD RULES:\n"
            "     - Values are percentages (0–100). Must sum to ~100% (allow ±3% for rounding).\n"
            "     - If a total 'All Others' or 'Remaining' bucket is stated → include it as a tier.\n"
            "     - If only top-N % is given with no 'others' breakdown → derive 'All Others' = 100 - top-N %.\n"
            "     - Max 8 tiers. Group smallest into 'All Others' if more than 8.\n"
            "     - Use most recent actual year data.\n"
            "     - NEVER invent concentration data not in the document.\n"
            "     - Output null if no customer concentration data exists anywhere in the document.\n"
            "23. REVENUE BY SEGMENT EXTRACTION LOGIC:\n"
            "   Extract the company's revenue breakdown by business segment as percentages.\n"
            "   OUTPUT: array of objects — [{\"segment\": \"Name\", \"pct\": number}, ...] sorted descending by pct.\n"
            "   Use the MOST RECENT actual year (YYYY_A) revenue split available.\n"
            "   SOURCE: company overview, segment summary pages, revenue breakdown charts/tables.\n"
            "   HARD RULES:\n"
            "     - Segments must be distinct business units/divisions — NOT product categories or geographies.\n"
            "     - Values are percentages (0–100). If document shows $ amounts, compute % of total.\n"
            "     - Percentages must sum to ~100% (allow ±2% for rounding).\n"
            "     - Max 8 segments. If more, group smallest ones as 'Other'.\n"
            "     - Single-segment companies (no breakdown) → output null.\n"
            "     - If only projected (YYYY_E) split available and no actual → use it but prefer actual.\n"
            "     - NEVER invent segment names not in the document.\n"
            "     - Output null if no segment revenue breakdown exists anywhere in the document.\n"
            "21. COMPANY SUMMARY EXTRACTION LOGIC:\n"
            "   A concise analyst-written memo summary — business description PLUS key financial anchors.\n"
            "   OUTPUT: a single string. Target length: 220–270 words. Must be in English.\n\n"
            "   EXTRACTION RULES:\n"
            "     STEP 1 — Read the business overview, executive summary, company profile, or\n"
            "       investment highlights section (typically the first 5–15 pages of the CIM).\n"
            "     STEP 2 — Write a factual, neutral analyst memo covering ALL of the following:\n"
            "       a) What the company does — core business, products/services, platforms or formats\n"
            "       b) Key end markets or customer segments served\n"
            "       c) Business model (how it makes money — recurring contracts, direct sales, dealer network, etc.)\n"
            "       d) Geographic presence if mentioned\n"
            "       e) Competitive advantages or moat (brand, contracts, switching costs, etc.)\n"
            "       f) FINANCIAL ANCHORS — include ALL of the following that are available in the CIM:\n"
            "            - Most recent actual year revenue (e.g. 'FY2025 revenue of $X')\n"
            "            - Current or most recent EBITDA and EBITDA margin\n"
            "            - Peak revenue year and peak revenue value if the company has grown then contracted\n"
            "            - Gross margin % if stated\n"
            "            - Revenue growth rate (YoY % for most recent year)\n"
            "            - Any projected EBITDA or revenue target if explicitly stated in the CIM\n"
            "          If a financial figure is not present in the CIM, omit it — do NOT fabricate.\n"
            "       g) Any key operational metrics (headcount, facility size, utilization, units, etc.)\n"
            "       h) Brief note on recent performance trend (growth, contraction, recovery) with quantification\n"
            "   HARD RULES:\n"
            "     - Write as a neutral third-party analyst — no promotional language, no exclamation marks.\n"
            "     - Do NOT quote raw CIM text verbatim — paraphrase into clean analytical prose.\n"
            "     - DO include specific financial figures where available — they make the summary useful.\n"
            "     - Do NOT reference the CIM document itself ('this document states...', 'per the CIM...').\n"
            "     - Keep it strictly 220–270 words — do not go below 200 or above 290.\n"
            "     - If no business overview or company description section is present → output null.\n"
            "     - Output format: plain string, no bullet points, no markdown headers.\n"
            "     - Lead with the company name and core business, then end markets, then financials, then outlook."
        )

        user_prompt = (
            f"### CLIENT REQUIREMENTS ###\n{client_requirements}\n\n"
            f"### FINANCIAL CONTEXT (EXTRACTED) ###\n{context_str}\n\n"
            "### OUTPUT FORMAT INSTRUCTION ###\n"
            "Return JSON with this structure (all values in $000s, year keys in YYYY_A / YYYY_E / TTM_YYYY format):\n"
            "{\n"
            "  \"Total_Revenue\": {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"Gross_Margin\":   {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"SGA\":            {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"Adj_EBITDA\":     {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"EBITDA\":         {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  \"Other_Expense\":     {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, ...},\n"
            "  \"Interest_Expense\":  {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, ...},\n"
            "  NOTE for Interest_Expense: historical years (YYYY_A / TTM_YYYY) ONLY — null for any YYYY_E.\n"
            "  \"Depreciation\":      {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  NOTE for Depreciation: include BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "  \"CAPEX\":             {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  NOTE for CAPEX: output as NEGATIVE numbers (e.g. -2490, not 2490). Both historical and projected.\n"
            "  Use 'Total Capex' row only — never breakdown lines. If two Total Capex rows exist, use the first (base, not incl. one-time items).\n"
            "  \"WC_Change\":         {\"2021_A\": value, \"2022_A\": value, \"2023_A\": value, \"2024_E\": value, ...},\n"
            "  NOTE for WC_Change: NWC increase = NEGATIVE (cash outflow), NWC decrease = POSITIVE (cash inflow).\n"
            "  STEP 1: extract from explicit labeled change row if present.\n"
            "  STEP 2 fallback: if no change row, derive from NWC balance rows (need 2+ years; first year = null).\n"
            "  Derivation sign: WC_Change = NWC(N) - NWC(N-1), then REVERSE sign for cash flow convention.\n"
            "  Both historical and projected years. If only 1 NWC balance year exists → null.\n"
            "  \"Onex_Adjustments\": {\"2021_A\": value, \"2022_A\": value, ...},\n"
            "  NOTE for Onex_Adjustments: HISTORICAL years (YYYY_A / TTM_YYYY) ONLY — never output YYYY_E.\n"
            "  STEP 1: find explicit total row ('Total Adjustments', 'Total normalizations', 'Total Management Adjustments').\n"
            "  STEP 2 fallback: derive as Adj_EBITDA - EBITDA per year (only if both non-null).\n"
            "  Positive = net add-back. Projected $0 rows → output null, not 0. Null if no bridge table exists.\n"
            "  YEAR LABEL ALIGNMENT: Only output a year key YYYY_A if that same year already appears as YYYY_A\n"
            "  in Total_Revenue or Adj_EBITDA. If the main P&L labels a year as YYYY_E, that year is projected\n"
            "  — do NOT output it here even if the adjustment bridge shows values for it. This prevents\n"
            "  duplicate year columns (e.g. both 2023_A and 2023_E) appearing in the output.\n"
            "  \"AR\": single_number_or_null,\n"
            "  NOTE for AR: ONE single value — last 'A'-suffixed actual year only (e.g. 2024A not 2030E). NOT a time series.\n"
            "  NEVER use projected (E/B/F) columns. Table may extend to 2030E — use only the last actual column.\n"
            "  \"Inventory\": single_number_or_null,\n"
            "  NOTE for Inventory: ONE single value — last 'A'-suffixed actual year only (e.g. 2024A not 2030E). NOT a time series.\n"
            "  NEVER use projected (E/B/F) columns. Table may extend to 2030E — use only the last actual column.\n"
            "  Zero (0) is valid for service businesses. Use Total Inventory subtotal if broken into components.\n"
            "  NEVER extract from narrative text or cash flow movements — balance sheet/NWC table only.\n"
            "  \"ME_Equipment\": single_number_or_null,\n"
            "  NOTE for ME_Equipment: ONE single value — gross book value from Fixed Asset / PP&E Schedule only.\n"
            "  Most recent ACTUAL period (last 'A'-suffixed column). NEVER use projected columns.\n"
            "  NEVER use Total Fixed Assets subtotal. NEVER use CAPEX table values.\n"
            "  Labels: 'Machinery & Equipment', 'M&E', 'Plant & Equipment', 'Warehouse Equipment', 'Equipment' (standalone in PP&E).\n"
            "  Normalise to $000s. Output null if no M&E row found in any fixed asset schedule.\n"
            "  \"Building_Land\": single_number_or_null,\n"
            "  NOTE for Building_Land: ONE single value — gross book value from Fixed Asset / PP&E Schedule only.\n"
            "  Most recent ACTUAL period (last 'A'-suffixed column). NEVER use projected columns.\n"
            "  NEVER use Total Fixed Assets subtotal. NEVER confuse 'Building Improvements' with 'Building'.\n"
            "  If 'Building' and 'Land' are separate rows — ADD them together for this field.\n"
            "  Labels: 'Building & Land', 'Building', 'Land', 'Real Estate', 'Property' (standalone in PP&E schedule).\n"
            "  Normalise to $000s. Output null if no Building or Land row found in any fixed asset schedule.\n"
            "  \"Market_Intelligence\": {\n"
            "    \"market_size\": \"string with figure + units + year\" or null,\n"
            "    \"market_growth_rate\": \"string with CAGR + horizon\" or null,\n"
            "    \"market_position\": \"string — company's stated rank/position\" or null,\n"
            "    \"competitors\": [\"Name A\", \"Name B\", ...] or null,\n"
            "    \"industry_tailwinds\": [\"tailwind 1\", \"tailwind 2\", ...] or null,\n"
            "    \"barriers_to_entry\": [\"barrier 1\", \"barrier 2\", ...] or null\n"
            "  },\n"
            "  NOTE for Market_Intelligence: 6 sub-fields, each independently null. Source = narrative sections only.\n"
            "  market_position = single string of explicit rank/position claim. industry_tailwinds/barriers_to_entry = short phrases max 10 words each.\n"
            "  If entire section absent → null for whole field. Otherwise output object with nulls for missing sub-fields.\n"
            "  \"Company_Summary\": \"string of 220-270 words or null\"\n"
            "  NOTE for Company_Summary: plain English analyst prose, 220-270 words, no bullet points.\n"
            "  Must include: what company does, end markets, business model, geography, competitive position,\n"
            "  AND key financial anchors (current revenue, EBITDA/margin, growth rate, gross margin, any peak/trough if relevant).\n"
            "  Source: business overview / executive summary sections of the CIM for narrative;\n"
            "  pull financial figures from P&L tables or financial summary pages.\n"
            "  Output null if no overview section is present.\n"
            "  \"Revenue_By_Segment\": [{\"segment\": \"Name\", \"pct\": number}, ...] or null,\n"
            "  NOTE for Revenue_By_Segment: array of segment % breakdown from most recent actual year.\n"
            "  Segments = distinct business divisions only. Values sum to ~100. Null if single-segment or not found.\n"
            "  \"Customer_Concentration\": [{\"tier\": \"Label\", \"pct\": number}, ...] or null,\n"
            "  NOTE for Customer_Concentration: array of customer revenue concentration tiers.\n"
            "  Preferred: bucket tiers (Top 1, Top 2-5, Top 6-10, All Others). Fallback: named customer aliases.\n"
            "  Values sum to ~100. Derive 'All Others' if not stated. Null if no concentration data found.\n"
            "  \"Revenue_By_Geography\": [{\"region\": \"Name\", \"pct\": number}, ...] or null,\n"
            "  NOTE for Revenue_By_Geography: array of geographic revenue % breakdown, most recent actual year.\n"
            "  Regions = countries, states, continents or named territories as stated in document.\n"
            "  Values sum to ~100. Max 10 entries. Null if single-geography or no breakdown found.\n"
            "  \"Management_Team\": [{\"name\": \"Full Name\", \"title\": \"Job Title\", \"experience\": \"X years or null\"}, ...] or null,\n"
            "  NOTE for Management_Team: named executives with explicit titles only. Max 10, C-suite/VP first.\n"
            "  experience = only if explicitly stated years figure for that person — otherwise null.\n"
            "  No board members, advisors, or investors. Null if no management section present.\n"
            "  \"Growth_Initiatives\": [{\"title\": \"Short Title\", \"description\": \"1-2 sentence summary\", \"impact\": \"$ or % or null\"}, ...] or null,\n"
            "  NOTE for Growth_Initiatives: key strategic growth pillars from growth strategy section. Max 8.\n"
            "  title = short label (max 6 words). description = factual 1-2 sentence summary. impact = explicit $ or % only.\n"
            "  Null if no growth strategy section present.\n"
            "  \"Company_KPIs\": {\n"
            "    \"founded_year\": integer_or_null,\n"
            "    \"total_employees\": integer_or_null,\n"
            "    \"num_locations\": integer_or_null,\n"
            "    \"countries_of_operation\": integer_or_null,\n"
            "    \"capacity_utilization\": \"string_or_null\"\n"
            "  }\n"
            "  NOTE for Company_KPIs: all scalars from company overview / 'by the numbers' section.\n"
            "  founded_year = original founding year (not PE acquisition year). total_employees = most recent headcount.\n"
            "  num_locations = distinct physical locations only. capacity_utilization = stated % string (e.g. '~60%').\n"
            "  If entire section absent → null. If only some sub-fields found → object with nulls for missing ones.\n"
            "}\n"
            "Include every period found in the document. Use null for any metric not found."
        )

        _RETRY_DELAYS = [5, 15, 30]  # seconds between attempts
        _RETRYABLE = ("rate limit", "timeout", "connection", "overloaded",
                      "503", "529", "502", "too many requests", "server error")

        logging.info(f"Sending extraction request to {self.provider.upper()} ({self.model_name})...")

        for attempt in range(3):
            try:
                if self.provider in ["openai", "nvidia", "ollama"]:
                    kwargs = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.2 if self.provider == "nvidia" else 0.0,
                    }

                    if self.provider == "openai":
                        kwargs["response_format"] = {"type": "json_object"}
                    elif self.provider == "ollama":
                        kwargs["temperature"] = 0.1
                        kwargs["max_tokens"] = 8192
                    else:
                        kwargs["top_p"] = 0.7
                        kwargs["max_tokens"] = 8192

                    response = self.client.chat.completions.create(**kwargs)
                    output = response.choices[0].message.content
                    output = _extract_json_from_text(output)
                    return self._normalize_nulls(json.loads(output.strip()))

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=8192,
                        temperature=0.0,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": f"{user_prompt}\n\nPlease output only JSON wrapped in ```json ... ``` blocks."}
                        ]
                    )

                    if response.stop_reason == "max_tokens":
                        logging.warning("Anthropic response was cut off (max_tokens). JSON may be incomplete.")

                    output_text = response.content[0].text
                    output_text = _extract_json_from_text(output_text)
                    return self._normalize_nulls(json.loads(output_text.strip()))

            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error (attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    return None
                delay = _RETRY_DELAYS[attempt]
                logging.warning(f"Retrying in {delay}s...")
                time.sleep(delay)

            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in _RETRYABLE) and attempt < 2:
                    delay = _RETRY_DELAYS[attempt]
                    logging.warning(f"LLM transient error (attempt {attempt+1}/3): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logging.error(f"LLM extraction failed: {type(e).__name__}: {e}")
                    return None

        return None

    def generate_investment_recommendation(self, extracted_data: Dict, deal_value: float) -> Optional[Dict]:
        """
        Makes a second LLM call using the already-extracted data + deal value
        to produce a structured M&A investment recommendation with IRR scenarios,
        leverage analysis, and a risk scorecard.
        """
        system_prompt = (
            "You are a senior M&A analyst at a private equity firm. "
            "You have been given extracted financial and market data from a CIM and an enterprise value the client is considering paying. "
            "Produce a rigorous, honest PE investment recommendation.\n\n"
            "CRITICAL RULES:\n"
            "1. Base your recommendation strictly on the data provided — do NOT assume or invent figures.\n"
            "2. If a key metric is null/missing, reason from what IS available and flag the gap with its severity.\n"
            "3. Adjust verdict for business TRAJECTORY not just entry multiple — a declining business at 9x is worse than a growing one at 11x.\n"
            "4. Be direct — this is for a sophisticated PE client.\n"
            "5. Output ONE valid JSON object and nothing else.\n\n"
            "VERDICT OPTIONS: 'Strong Buy', 'Buy', 'Caution', 'Pass', 'Insufficient Data'\n"
            "CONFIDENCE OPTIONS: 'High', 'Medium', 'Low'\n\n"
            "VERDICT GUIDANCE:\n"
            "  < 6x EV/EBITDA  → lean Strong Buy (if stable/growing business)\n"
            "  6–9x            → Buy\n"
            "  9–12x           → Caution\n"
            "  > 12x           → Pass (unless exceptional growth justifies premium)\n"
            "  DOWNGRADE verdict by one level if: revenue declining >15%, EBITDA collapsing, interest expense unknown at likely high leverage\n"
            "  UPGRADE verdict by one level if: market leadership, strong multi-year growth trajectory, high FCF conversion\n"
            "  EBITDA null AND Revenue null → 'Insufficient Data'\n\n"
            "CONFIDENCE GUIDANCE:\n"
            "  High   — EBITDA + Revenue + 3 or more supporting fields present\n"
            "  Medium — EBITDA or Revenue present + at least 1 supporting field\n"
            "  Low    — Only 1 key metric present, rest null\n\n"
            "IRR SCENARIOS — compute 3 scenarios using this exact math:\n"
            "  entry_ev = deal_value (given in $000s)\n"
            "  hold_years = 5 (standard PE hold)\n"
            "  For each scenario estimate:\n"
            "    exit_ebitda ($000s): your best projection of EBITDA at end of hold period\n"
            "    exit_multiple: realistic EV/EBITDA exit multiple for this business\n"
            "  Then calculate (show your arithmetic):\n"
            "    exit_ev = exit_ebitda × exit_multiple\n"
            "    moic = exit_ev / entry_ev   (all-equity, unlevered)\n"
            "    irr_pct = round((moic ** (1.0 / hold_years) - 1) * 100, 1)\n"
            "  Scenario definitions:\n"
            "    base:     achieves projected EBITDA, exits at sector-fair multiple\n"
            "    upside:   beats projections by 15-25%, exits at premium multiple\n"
            "    downside: misses targets by 20-30%, exits at discount multiple\n"
            "  If EBITDA is entirely null → set irr_scenarios to null.\n\n"
            "LEVERAGE ANALYSIS:\n"
            "  Use most recent Adj_EBITDA (prefer actual year; fall back to first projected).\n"
            "  debt_capacity_multiple: 4.0 (standard PE leverage)\n"
            "  implied_debt ($000s): 4.0 × EBITDA\n"
            "  implied_equity_check ($000s): entry_ev - implied_debt\n"
            "  fcf_estimate ($000s): EBITDA + CAPEX (CAPEX is negative) — subtract WC_Change if it is negative (cash drain); else EBITDA only\n"
            "  debt_service_note: plain-English flag — mention if interest expense unknown, if FCF barely covers debt, or if NWC is a persistent drain\n"
            "  Set all fields to null if EBITDA is null.\n\n"
            "RISK SCORECARD — score each dimension 1–5:\n"
            "  1 = Very Weak, 2 = Weak, 3 = Neutral, 4 = Strong, 5 = Very Strong\n"
            "  Dimensions (always output all 5 in this order):\n"
            "  1. Revenue Quality — trend, concentration risk, predictability\n"
            "  2. EBITDA Quality  — margin level, margin trend, cash conversion\n"
            "  3. Market Position — competitive moat, leadership, barriers to entry\n"
            "  4. Leverage Risk   — if interest expense null → score 2 (unknown risk); else assess coverage\n"
            "  5. Execution Risk  — projection dependency, turnaround required, management depth\n"
            "  Format: [{\"dimension\": \"Revenue Quality\", \"score\": N, \"note\": \"one-line rationale\"}, ...]\n\n"
            "DATA GAPS — only gaps relevant to the investment decision:\n"
            "  severity levels:\n"
            "    critical — deal-breaker level unknowns (interest expense/debt, missing EBITDA, no revenue)\n"
            "    moderate — important but workable gaps (WC data, depreciation, capex trend)\n"
            "    minor    — useful but non-critical (management experience, geography breakdown)\n"
            "  Format: [{\"field\": \"field_name\", \"severity\": \"critical|moderate|minor\", \"impact\": \"one-line why it matters\"}, ...]\n\n"
            "OUTPUT FORMAT — return exactly this JSON structure:\n"
            "{\n"
            "  \"verdict\": \"Strong Buy|Buy|Caution|Pass|Insufficient Data\",\n"
            "  \"confidence\": \"High|Medium|Low\",\n"
            "  \"ev_ebitda_multiple\": number_or_null,\n"
            "  \"ev_revenue_multiple\": number_or_null,\n"
            "  \"irr_scenarios\": {\n"
            "    \"base\":     {\"exit_multiple\": number, \"exit_ebitda\": number, \"hold_years\": 5, \"moic\": number, \"irr_pct\": number},\n"
            "    \"upside\":   {\"exit_multiple\": number, \"exit_ebitda\": number, \"hold_years\": 5, \"moic\": number, \"irr_pct\": number},\n"
            "    \"downside\": {\"exit_multiple\": number, \"exit_ebitda\": number, \"hold_years\": 5, \"moic\": number, \"irr_pct\": number}\n"
            "  } or null,\n"
            "  \"leverage_analysis\": {\n"
            "    \"debt_capacity_multiple\": number_or_null,\n"
            "    \"implied_debt\": number_or_null,\n"
            "    \"implied_equity_check\": number_or_null,\n"
            "    \"fcf_estimate\": number_or_null,\n"
            "    \"debt_service_note\": \"string_or_null\"\n"
            "  },\n"
            "  \"risk_scorecard\": [\n"
            "    {\"dimension\": \"Revenue Quality\", \"score\": 1-5, \"note\": \"...\"},\n"
            "    {\"dimension\": \"EBITDA Quality\",  \"score\": 1-5, \"note\": \"...\"},\n"
            "    {\"dimension\": \"Market Position\", \"score\": 1-5, \"note\": \"...\"},\n"
            "    {\"dimension\": \"Leverage Risk\",   \"score\": 1-5, \"note\": \"...\"},\n"
            "    {\"dimension\": \"Execution Risk\",  \"score\": 1-5, \"note\": \"...\"}\n"
            "  ],\n"
            "  \"key_positives\": [\"concise evidence-based point\", ...],\n"
            "  \"key_risks\": [\"concise evidence-based point\", ...],\n"
            "  \"data_gaps\": [{\"field\": \"name\", \"severity\": \"critical|moderate|minor\", \"impact\": \"...\"}],\n"
            "  \"rationale\": \"2-3 sentence plain-English summary of the recommendation\"\n"
            "}\n"
            "key_positives and key_risks: 2-4 points each, specific and evidence-based (cite numbers where available).\n"
            "ev_ebitda_multiple: round to 1 decimal. Use most recent actual Adj_EBITDA; if null use first projected.\n"
            "ev_revenue_multiple: round to 2 decimals. Same period priority.\n"
            "irr_scenarios: unlevered (all-equity) return — mention in rationale if leverage would meaningfully change the picture.\n"
            "If both EBITDA and Revenue are null → verdict must be 'Insufficient Data', irr_scenarios and leverage_analysis must be null."
        )

        user_prompt = (
            f"DEAL VALUE (Enterprise Value): ${deal_value:,.0f} thousand  (= ${deal_value/1000:,.1f}M)\n"
            f"Use this as entry_ev = {deal_value:.0f} ($000s) for all IRR calculations.\n"
            f"All financial values below are in $000s (thousands).\n\n"
            f"EXTRACTED FINANCIAL DATA:\n{json.dumps(extracted_data, indent=2)}\n\n"
            "Generate the investment recommendation JSON. "
            "Compute IRR scenarios arithmetically as instructed — show moic and irr_pct for all 3 scenarios."
        )

        logging.info(f"Generating investment recommendation (deal value: ${deal_value:,.0f})...")
        try:
            if self.provider in ["openai", "nvidia", "ollama"]:
                kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                }
                if self.provider == "openai":
                    kwargs["response_format"] = {"type": "json_object"}
                elif self.provider == "nvidia":
                    kwargs["top_p"] = 0.7
                    kwargs["max_tokens"] = 6144
                elif self.provider == "ollama":
                    kwargs["max_tokens"] = 6144

                response = self.client.chat.completions.create(**kwargs)
                output = response.choices[0].message.content
                output = _extract_json_from_text(output)
                return json.loads(output.strip())

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=6144,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": f"{user_prompt}\n\nOutput only JSON wrapped in ```json ... ``` blocks."}
                    ]
                )
                if response.stop_reason == "max_tokens":
                    logging.warning("Anthropic recommendation response was cut off (max_tokens).")
                output_text = response.content[0].text
                output_text = _extract_json_from_text(output_text)
                return json.loads(output_text.strip())

        except Exception as e:
            logging.error(f"Error generating investment recommendation: {type(e).__name__}: {e}")
            return None

DEFAULT_CLIENT_REQ = (
            "Extract these financial metrics for ALL available periods found in the document:\n"
            "1) Total_Revenue — Top-line revenue from the main P&L / income statement table. Follow Rule 0.\n"
            "   VALID labels (accept any, case-insensitive): 'Revenue', 'Revenues', 'Net Revenue', 'Net Revenues',\n"
            "   'Total Revenue', 'Total Revenues', 'Net Sales', 'Total Net Sales', 'Gross Sales', 'Total Sales',\n"
            "   'Sales', 'Service Revenue', 'Product Revenue', 'Subscription Revenue',\n"
            "   'Net Patient Revenue', 'Net Operating Revenue', 'Net Billings'.\n"
            "   PRIORITY: If both 'Gross Sales' and 'Net Sales' exist in the same table → use 'Net Sales'.\n"
            "   If both 'Revenue' and 'Net Revenue' exist → use 'Net Revenue'.\n"
            "   Always use the top-line summary row — never a segment sub-row or breakdown line.\n"
            "   Output null per year if not found in any financial table.\n"
            "2) Gross_Margin — The revenue-minus-direct-costs line from the P&L.\n"
            "   VALID labels (accept any, case-insensitive):\n"
            "     'Gross Profit', 'Gross Margin', 'Total Gross Profit', 'Total Gross Margin',\n"
            "     'Gross Profit Margin', 'Net Gross Margin',\n"
            "     'Contribution Margin', 'Total Contribution Margin', 'CM'\n"
            "   PRIORITY — if multiple rows exist in the same table, use this order (top = highest priority):\n"
            "     1st: 'Gross Profit' or 'Gross Margin' (or any 'Gross ...' variant)\n"
            "     2nd: 'Contribution Margin' or 'CM' (only if no Gross Profit/Margin row exists)\n"
            "   REASON: Gross Margin = Revenue minus COGS only. Contribution Margin may deduct additional\n"
            "     variable costs and can differ materially. Never use both — always pick the higher-priority label.\n"
            "   NEVER extract a percentage row (e.g. 'Gross Margin %') — extract the dollar value row only.\n"
            "   NEVER calculate or derive — only extract if a matching labeled row exists in a financial table.\n"
            "   Output null per year if no matching row is found.\n"
            "3) SGA — Selling, General & Administrative expenses. Follow Rule 9 strictly.\n"
            "   VALID labels: 'SG&A', 'SGA', 'Unadjusted SG&A', 'Selling, General & Administrative',\n"
            "   'G&A', 'General & Administrative', 'Selling & Marketing', 'Overhead' (standalone only).\n"
            "   NEVER extract rows labeled 'Operating Expenses' or 'Total Operating Expenses' for this field.\n"
            "   NEVER calculate or infer SG&A — only extract if explicitly labeled in a financial table.\n"
            "   Output as POSITIVE number always — if table shows parentheses or negative, convert to positive.\n"
            "   Output null for ALL years if no valid SG&A label is found in the document.\n"
            "4) Adj_EBITDA — Adjusted EBITDA per year. Follow the strict priority logic in Rule 10.\n"
            "   Labels to look for (in priority order):\n"
            "     'Adjusted EBITDA', 'Adj. EBITDA', 'Adj EBITDA', 'EBITDA (Adjusted)', 'EBITDA (as adjusted)',\n"
            "     'Management Adjusted EBITDA', 'Mgmt. Adj. EBITDA', 'Management EBITDA',\n"
            "     'Normalized EBITDA', 'Normalised EBITDA',\n"
            "     'Run-Rate EBITDA', 'Run Rate EBITDA',\n"
            "     'EBITDA before one-time items', 'EBITDA excl. one-time items',\n"
            "     'PF Adjusted EBITDA', 'Pro Forma Adjusted EBITDA', 'Pro Forma EBITDA' (only if no non-PF adjusted variant exists).\n"
            "   If only plain 'EBITDA' exists with no adjustments anywhere in the document, use that.\n"
            "   Output null per year if not found.\n"
            "5) EBITDA — Reported/plain EBITDA per year (BEFORE any adjustments). Follow Rule 11.\n"
            "   Only extract rows explicitly labeled 'Reported EBITDA', 'EBITDA' (no qualifier), or\n"
            "   'EBITDA before adjustments'. NEVER use Adjusted EBITDA for this field.\n"
            "   Output null if only Adjusted EBITDA exists in the document.\n"
            "6) Other_Expense — Other expense / (income) per year. Follow Rule 12.\n"
            "   Labels to look for: 'Other expense/(income)', 'Other (income)/expense', 'Other income/expenses',\n"
            "   'Other income, net', 'Non-operating expense'. Also check EBITDA adjustment tables.\n"
            "   Do NOT extract if label includes 'interest'. Sign: expense=positive, income=negative.\n"
            "   Output null if not explicitly labeled in a financial table.\n"
            "7) Interest_Expense — Interest paid on debt per year. Follow Rule 13.\n"
            "   Labels: 'Interest expense', 'Interest expense, net', 'Net interest expense',\n"
            "   'Finance costs', 'Finance charges', 'Interest charges', 'Interest on debt'.\n"
            "   CRITICAL: Extract HISTORICAL years only (YYYY_A / TTM_YYYY). Output null for YYYY_E.\n"
            "   Do NOT extract if bundled with 'other' (e.g. 'Interest and other expense').\n"
            "   Do NOT extract from balance sheet items like 'Accrued Interest'.\n"
            "   If multiple debt lines present, use 'Total interest expense' subtotal only.\n"
            "   Output null if not explicitly labeled in a P&L financial table.\n"
            "8) Depreciation — Depreciation (non-cash charge) per year. Follow Rule 14.\n"
            "   Labels: 'Depreciation', 'Depreciation & Amortization', 'D&A', 'Depreciation and Amortization'.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   If 'Depreciation' and 'Amortization' are on separate rows, extract only 'Depreciation'.\n"
            "   NEVER extract 'Amortization of intangibles' for this field.\n"
            "   Zero (0) is a valid value — output 0, not null.\n"
            "   Output null if not explicitly labeled in a financial table.\n"
            "9) CAPEX — Capital expenditures per year. Follow Rule 15.\n"
            "   Labels: 'Total Capex', 'Total Capital Expenditures', 'Capital Expenditures', 'Capex'.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   OUTPUT AS NEGATIVE: value from table is 2,490 → output -2490.\n"
            "   Use ONLY 'Total Capex' subtotal row — NEVER individual breakdown lines (Maintenance, Growth, etc.).\n"
            "   If two Total Capex rows exist (base vs. incl. one-time items), use the FIRST base row.\n"
            "   NEVER derive from fixed asset schedule changes. NEVER extract from narrative text.\n"
            "   Zero is a valid value — output 0, not null.\n"
            "   Output null if not found in a structured financial table.\n"
            "10) WC_Change — Year-over-year change in Net Working Capital. Follow Rule 16 (two-step logic).\n"
            "   STEP 1 (preferred): extract from explicit labeled change row if present.\n"
            "     Labels: 'Change in NWC', 'Change in Working Capital', 'Changes in Working Capital',\n"
            "     '(Growth) / Decline in Net Working Capital', 'Change in Operating Working Capital'.\n"
            "   STEP 2 (fallback): if no change row exists, derive from NWC balance rows:\n"
            "     WC_Change(year N) = NWC(year N) - NWC(year N-1), then REVERSE sign.\n"
            "     First year in table = null. Requires 2+ NWC balance years. If only 1 year → null.\n"
            "   Sign convention always: NWC increase = NEGATIVE. NWC decrease = POSITIVE.\n"
            "   Extract BOTH historical (YYYY_A/TTM_YYYY) AND projected (YYYY_E) years.\n"
            "   NEVER derive from individual components (AR, Inventory, Payables separately).\n"
            "   Output null if both steps fail (no change row and fewer than 2 NWC balance years).\n"
            "11) AR — Accounts Receivable balance for ABL collateral (Sources section C7). Follow Rule 17.\n"
            "   Labels: 'Accounts Receivable', 'Account Receivable, net', 'Trade Receivables', 'Receivables, net'.\n"
            "   OUTPUT: ONE single number (not a time series) — the most recent ACTUAL balance only.\n"
            "   'Actual' = last year column ending in 'A' or 'a' (e.g. 2024A). NEVER use E/B/F projected columns.\n"
            "   WARNING: If table spans e.g. 2023A–2030E, the most recent actual is 2024A, NOT the last column.\n"
            "   Period priority: TTM > latest YYYY_A > YYYY_B > YYYY_E (only if no actual exists).\n"
            "   Source: balance sheet or NWC schedule table ONLY — never from P&L or cash flow table.\n"
            "   Use net value (after allowance) if available. Normalise to $000s.\n"
            "   Output null if AR not found in any balance sheet or NWC table.\n"
            "12) Inventory — Inventory balance for ABL collateral (Sources section C8). Follow Rule 18.\n"
            "   Labels: 'Inventory', 'Inventories', 'Inventory, net', 'Stock', 'Total Inventory'.\n"
            "   OUTPUT: ONE single number — the most recent ACTUAL balance only (not a time series).\n"
            "   'Actual' = last year column ending in 'A' or 'a'. NEVER use E/B/F projected columns.\n"
            "   WARNING: If table spans e.g. 2023A–2030E, the most recent actual is 2024A, NOT the last column.\n"
            "   Period priority: TTM > latest YYYY_A > YYYY_B > YYYY_E (only if no actual exists).\n"
            "   Source: balance sheet or NWC schedule table ONLY — never from P&L or cash flow table.\n"
            "   Zero (0) is valid — service businesses with no physical goods output 0, not null.\n"
            "   If broken into components (Raw Materials/WIP/Finished Goods), use 'Total Inventory' subtotal only.\n"
            "   Use net value (after obsolescence reserve) if shown. Normalise to $000s.\n"
            "   NEVER extract from narrative text (e.g. 'inventory reduced from $34M') — table values only.\n"
            "   Output null if Inventory not found in any balance sheet or NWC table.\n"
            "13) Market_Intelligence — Competitive landscape and market intelligence. Follow Rule 19.\n"
            "   OUTPUT: object with 6 sub-fields, each independently null:\n"
            "     market_size: string with figure + units + year (e.g. '$48.9B (2024)') — null if not stated.\n"
            "     market_growth_rate: string with CAGR + horizon (e.g. '~8% CAGR 2024-2028') — null if qualitative only.\n"
            "     market_position: single string of company's explicit rank/position claim — null if none stated.\n"
            "     competitors: array of named competitor strings (max 12) — null if no named competitors.\n"
            "     industry_tailwinds: array of 2-5 short phrases (max 10 words each) for demand drivers — null if absent.\n"
            "     barriers_to_entry: array of 2-5 short phrases for competitive moat factors — null if absent.\n"
            "   SOURCE: narrative sections only — market overview, competitive landscape, investment highlights.\n"
            "   If entire section absent → null for whole field. Otherwise object with nulls for missing sub-fields.\n"
            "   NEVER invent data. NEVER extract from financial tables.\n"
            "14) Onex_Adjustments — Total non-recurring / 1x adjustment amount per year. Follow Rule 20.\n"
            "   OUTPUT: flat numeric value per year (positive = net add-back). Historical years ONLY (YYYY_A / TTM_YYYY).\n"
            "   STEP 1: find an explicit total row in the EBITDA bridge table:\n"
            "     Labels: 'Total Adjustments', 'Total normalizations', 'Total Management Adjustments',\n"
            "     'Net Adjustments', 'Total Add-backs'.\n"
            "   STEP 2 fallback: derive as Adj_EBITDA - EBITDA for each year (only if both non-null).\n"
            "   NEVER use individual line items — use the total only.\n"
            "   NEVER include PF or synergy adjustment tiers — Management adjustments only.\n"
            "   Projected years (YYYY_E): do NOT output even if $0 appears — output null.\n"
            "   Output null if no EBITDA bridge table exists in the document.\n"
            "   YEAR LABEL ALIGNMENT (CRITICAL): Only output YYYY_A keys that also exist as YYYY_A in\n"
            "   Total_Revenue or Adj_EBITDA. If the main P&L calls a year YYYY_E, skip it here entirely.\n"
            "   NEVER create a year key in Onex_Adjustments that does not already appear as YYYY_A in\n"
            "   other fields — this causes duplicate columns (e.g. both 2023_A and 2023_E) in the output.\n"
            "15) Company_Summary — A concise analyst-written description of the company. Follow Rule 21.\n"
            "   Source: executive summary, business overview, company profile, or investment highlights\n"
            "   sections (typically the first 5–15 pages of the CIM).\n"
            "   Length: 200–250 words. Plain English prose — no bullet points, no financial figures.\n"
            "   Neutral tone: describe what the company does, its end markets, business model,\n"
            "   geographic presence, and competitive position.\n"
            "   Output null if no business overview section is present in the document.\n"
            "16) Revenue_By_Segment — Revenue % split by business segment. Follow Rule 23.\n"
            "   OUTPUT: array [{\"segment\": \"Name\", \"pct\": number}] sorted desc by pct, or null.\n"
            "   Use most recent actual year split. Segments = distinct divisions only. Values sum ~100.\n"
            "   Null if single-segment company or no breakdown found.\n"
            "17) Customer_Concentration — Revenue % by customer tier. Follow Rule 24.\n"
            "   OUTPUT: array [{\"tier\": \"Label\", \"pct\": number}] sorted desc by pct, or null.\n"
            "   Prefer bucket tiers (Top 1, Top 2-5, Top 6-10, All Others). Derive 'All Others' = 100 - top-N if not stated.\n"
            "   Null if no customer concentration data found.\n"
            "18) Revenue_By_Geography — Revenue % by geographic region. Follow Rule 25.\n"
            "   OUTPUT: array [{\"region\": \"Name\", \"pct\": number}] sorted desc by pct, or null.\n"
            "   Use most recent actual year. Regions as stated (countries/states/continents). Values sum ~100.\n"
            "   Null if single-geography company or no geographic breakdown found.\n"
            "19) Management_Team — Key executives. Follow Rule 26.\n"
            "   OUTPUT: array [{\"name\": \"Full Name\", \"title\": \"Job Title\", \"experience\": \"X years\" or null}], or null.\n"
            "   Max 10 people, C-suite/VP first. experience only if explicitly stated. No board/advisors.\n"
            "   Null if no management team section present.\n"
            "20) Growth_Initiatives — Strategic growth pillars. Follow Rule 27.\n"
            "   OUTPUT: array [{\"title\": \"Short Title\", \"description\": \"1-2 sentence summary\", \"impact\": \"$ or % or null\"}], or null.\n"
            "   Max 8 initiatives. title = short label. impact = explicit stated $ or % figure only, else null.\n"
            "   Null if no growth strategy section present.\n"
            "   OUTPUT: array [{\"segment\": \"Name\", \"pct\": number}] sorted desc by pct, or null.\n"
            "   Use most recent actual year split. Segments = distinct divisions only. Values sum ~100.\n"
            "   Null if single-segment company or no breakdown found.\n"
            "17) Company_KPIs — Operational snapshot metrics. Follow Rule 22.\n"
            "   OUTPUT: object with: founded_year (int), total_employees (int), num_locations (int),\n"
            "   countries_of_operation (int), capacity_utilization (string e.g. '~60%').\n"
            "   Source: company overview, 'by the numbers' box, facilities section, intro pages.\n"
            "   All scalars — NOT time series. founded_year = original founding, not PE acquisition year.\n"
            "   If section absent → null. If partial → object with nulls for missing sub-fields.\n"
            "21) ME_Equipment — Gross book value of Machinery & Equipment assets. Follow Rule 28.\n"
            "   SOURCE: Fixed Asset Schedule / PP&E Schedule ONLY — never CAPEX table, P&L, or narrative.\n"
            "   OUTPUT: ONE single number — most recent ACTUAL period (last 'A'-suffixed column).\n"
            "   VALID labels: 'Machinery & Equipment', 'M&E', 'Plant & Equipment', 'Warehouse Equipment',\n"
            "   'Manufacturing Equipment', 'Production Equipment', 'Equipment' (standalone in PP&E table).\n"
            "   Use GROSS value (before accumulated depreciation) if both gross and net are shown.\n"
            "   NEVER use 'Total Fixed Assets' or any subtotal row — individual M&E row only.\n"
            "   NEVER extract CAPEX spending amounts (e.g. annual capex rows) as M&E book value.\n"
            "   NEVER sum multiple asset rows — use the single best-matching M&E row only.\n"
            "   Period warning: fixed asset tables often extend to 2030E — most recent actual is NOT\n"
            "   the last column. Scan left to find the last 'A'-suffixed column.\n"
            "   Normalise to $000s. Output null if not found in any fixed asset schedule.\n"
            "22) Building_Land — Gross book value of Building & Land real estate assets. Follow Rule 29.\n"
            "   SOURCE: Fixed Asset Schedule / PP&E Schedule ONLY — never CAPEX table, P&L, or narrative.\n"
            "   OUTPUT: ONE single number — most recent ACTUAL period (last 'A'-suffixed column).\n"
            "   VALID labels: 'Building & Land', 'Building and Land', 'Building', 'Land', 'Real Estate',\n"
            "   'Property' (standalone in PP&E), 'Land and Building'.\n"
            "   CRITICAL DISTINCTION: 'Building Improvements' ≠ 'Building'.\n"
            "   'Building Improvements' = tenant renovation costs. 'Building' = owned real property.\n"
            "   Only use 'Building Improvements' if NO 'Building' or 'Land' row exists AND company owns property.\n"
            "   If 'Building' and 'Land' are SEPARATE rows → ADD them together for a single output value.\n"
            "   Use GROSS value (before accumulated depreciation) if both gross and net are shown.\n"
            "   NEVER use 'Total Fixed Assets' or any subtotal row.\n"
            "   NEVER extract square footage, headcount tables, or lease descriptions as dollar values.\n"
            "   Period warning: same as ME_Equipment — last column is usually projected, scan left for 'A'.\n"
            "   Normalise to $000s. Output null if not found in any fixed asset schedule.\n\n"
            "Output all numeric values in $000s (thousands). "
            "If currency is CAD, output as-is without USD conversion. "
            "If a metric does not exist in the document, use null."
)

def main():
    parser = argparse.ArgumentParser(description="Parse CIM PDF and extract specific financial fields via LLM.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the CIM PDF file (e.g., sample_cim.pdf).")
    parser.add_argument(
        "--req",
        type=str,
        default=DEFAULT_CLIENT_REQ,
        help="Description of exactly what you want extracted from the financials."
    )
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic", "nvidia", "ollama"], help="LLM Provider to use (openai, anthropic, nvidia, or ollama).")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (e.g., gpt-4o, claude-3-5-sonnet-20240620, meta/llama-3.3-70b-instruct)")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides environment variable).")
    parser.add_argument("--deal-value", type=float, default=None, help="Enterprise value / deal price being considered (in USD). If provided, generates an investment recommendation.")
    parser.add_argument("--gcv-key", type=str, default=None, help="Path to Google Cloud Vision service account JSON key. Enables OCR for image-heavy PDF pages.")
    parser.add_argument("--gcv-api-key", type=str, default=None, help="Google Cloud Vision simple API key. Alternative to --gcv-key. Can also be set via GCV_API_KEY env var.")
    args = parser.parse_args()

    # Authentication — --api-key flag takes priority, then environment variable
    if args.api_key:
        api_key = args.api_key
    else:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "nvidia": "NVIDIA_API_KEY", "ollama": "OLLAMA_API_KEY"}
        api_key_env = env_var_map[args.provider]
        api_key = os.getenv(api_key_env)

    if not api_key:
        env_var_map = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "nvidia": "NVIDIA_API_KEY", "ollama": "OLLAMA_API_KEY"}
        api_key_env = env_var_map[args.provider]
        logging.error(f"Missing API key. Provide it via --api-key or set the {api_key_env} environment variable.")
        exit(1)

    # Execution flow
    try:
        # GCV API key: --gcv-api-key flag takes priority, then GCV_API_KEY env var
        gcv_api_key = args.gcv_api_key or os.getenv("GCV_API_KEY", "")
        gcv_api_key = gcv_api_key if gcv_api_key and gcv_api_key != "your-gcv-api-key-here" else None
        cim = CIMParser(args.pdf, gcv_key_path=args.gcv_key, gcv_api_key=gcv_api_key)
        cim.parse_pdf()

        relevant_sections = cim.find_financial_sections()
        if not relevant_sections:
            logging.warning("No standard financial keywords found in document. Passing all tables to LLM.")
            # Fallback - just pass tables if we couldn't filter by keywords
            relevant_sections = [t for t in cim.extracted_tables]

        extractor = LLMExtractor(api_key=api_key, provider=args.provider, model_name=args.model)
        result = extractor.extract_fields(relevant_sections, client_requirements=args.req)

        # Generate investment recommendation if deal value provided
        if result and args.deal_value is not None:
            recommendation = extractor.generate_investment_recommendation(result, args.deal_value)
            if recommendation:
                # Insert recommendation between Market_Intelligence and Company_Summary
                ordered = {}
                for key in result:
                    if key == "Company_Summary":
                        ordered["Investment_Recommendation"] = recommendation
                    ordered[key] = result[key]
                # If Company_Summary was not present, append at end before it
                if "Investment_Recommendation" not in ordered:
                    ordered["Investment_Recommendation"] = recommendation
                result = ordered

        # Output the result
        if result:
            print("\n" + "="*50)
            print("EXTRACTED FINANCIAL DATA (raw)")
            print("="*50)
            print(json.dumps(result, indent=4))

            output_dir = "extracted_results"
            os.makedirs(output_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(args.pdf))[0]

            raw_file = os.path.join(output_dir, f"{base_name}_extracted.json")
            with open(raw_file, 'w') as f:
                json.dump(result, f, indent=4)

            logging.info(f"Raw JSON saved to {raw_file}")
        else:
            logging.error("The LLM failed to return a valid extraction or threw an error.")
            
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    main()
