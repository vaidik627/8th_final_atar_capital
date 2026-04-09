"""
CIM Extraction Test Suite — Atar Capital
=========================================
Validates extracted JSON output against verified ground truth
for all 4 known CIM PDFs.

Usage:
  # Validate existing extracted JSONs (fast — no API call needed)
  python test_extraction.py

  # Run fresh extraction then validate (slow — calls LLM)
  python test_extraction.py --run --provider nvidia --model meta/llama-3.3-70b-instruct

  # Test a single PDF
  python test_extraction.py --pdf chimera
  python test_extraction.py --pdf network
  python test_extraction.py --pdf palm
  python test_extraction.py --pdf smores
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Ground Truth ─────────────────────────────────────────────────────────────
# Values verified manually against source PDFs.
# Numeric tolerance: 5% (LLMs may round slightly differently)

GROUND_TRUTH = {

    "chimera": {
        "label": "Project Chimera (1).pdf",
        "json":  "extracted_results/Project Chimera (1)_extracted.json",
        "pdf":   "Project Chimera (1).pdf",
        "units": "$000s",
        "tests": [
            # ── Financial dict fields — must be dict with year keys ───────────
            {"field": "Total_Revenue",      "check": "is_dict",    "desc": "Revenue is a dict"},
            {"field": "Total_Revenue",      "check": "has_actuals", "min_count": 3, "desc": "Revenue has ≥3 actual years"},
            {"field": "Total_Revenue",      "check": "has_proj",    "min_count": 1, "desc": "Revenue has projected years"},
            {"field": "Gross_Margin",       "check": "is_dict",    "desc": "Gross Margin is a dict"},
            {"field": "Gross_Margin",       "check": "has_actuals", "min_count": 3, "desc": "Gross Margin has ≥3 actual years"},
            {"field": "SGA",                "check": "is_dict",    "desc": "SGA is a dict"},
            {"field": "Adj_EBITDA",         "check": "is_dict",    "desc": "Adj_EBITDA is a dict"},
            {"field": "Adj_EBITDA",         "check": "has_actuals", "min_count": 3, "desc": "Adj_EBITDA has ≥3 actual years"},
            {"field": "Adj_EBITDA",         "check": "has_proj",    "min_count": 1, "desc": "Adj_EBITDA has projected years"},
            {"field": "EBITDA",             "check": "is_dict",    "desc": "EBITDA (reported) is a dict"},
            {"field": "CAPEX",              "check": "is_dict",    "desc": "CAPEX is a dict"},
            {"field": "CAPEX",              "check": "all_negative","desc": "CAPEX values all negative (Rule 16)"},
            {"field": "Onex_Adjustments",   "check": "is_dict",    "desc": "Onex_Adjustments is a dict"},
            # ── Depreciation — exact values (page 55) ────────────────────────
            {"field": "Depreciation",       "check": "is_dict",    "desc": "Depreciation is a dict"},
            {"field": "Depreciation",       "check": "exact_value", "year": "2022_A", "value": 2559,  "tol": 0.05, "desc": "Depreciation 2022A = 2,559"},
            {"field": "Depreciation",       "check": "exact_value", "year": "2023_A", "value": 2604,  "tol": 0.05, "desc": "Depreciation 2023A = 2,604"},
            {"field": "Depreciation",       "check": "exact_value", "year": "2024_A", "value": 2438,  "tol": 0.05, "desc": "Depreciation 2024A = 2,438"},
            {"field": "Depreciation",       "check": "exact_value", "year": "2025_A", "value": 2305,  "tol": 0.05, "desc": "Depreciation 2025A = 2,305"},
            {"field": "Depreciation",       "check": "exact_value", "year": "2026_E", "value": 2063,  "tol": 0.05, "desc": "Depreciation 2026E = 2,063"},
            {"field": "Depreciation",       "check": "exact_value", "year": "2027_E", "value": 1968,  "tol": 0.05, "desc": "Depreciation 2027E = 1,968"},
            {"field": "Depreciation",       "check": "exact_value", "year": "2028_E", "value": 1872,  "tol": 0.05, "desc": "Depreciation 2028E = 1,872"},
            # ── Fixed Asset Schedule — exact scalar values (page 62) ─────────
            {"field": "ME_Equipment",       "check": "exact_scalar", "value": 14634,  "tol": 0.05, "desc": "ME_Equipment = 14,634 (Dec-25A)"},
            {"field": "Building_Land",      "check": "exact_scalar", "value": 3250,   "tol": 0.05, "desc": "Building_Land = 3,250 (Dec-25A)"},
            # ── Balance sheet scalars ─────────────────────────────────────────
            {"field": "AR",                 "check": "is_positive_scalar", "desc": "AR is a positive number"},
            {"field": "Inventory",          "check": "is_positive_scalar", "desc": "Inventory is a positive number"},
            # ── Year key format (YYYY_A / YYYY_E) ────────────────────────────
            {"field": "Total_Revenue",      "check": "valid_year_keys", "desc": "Revenue year keys are YYYY_A/YYYY_E format"},
            {"field": "Adj_EBITDA",         "check": "valid_year_keys", "desc": "EBITDA year keys are YYYY_A/YYYY_E format"},
            # ── Narrative / qualitative fields ───────────────────────────────
            {"field": "Company_Summary",    "check": "is_non_empty_string", "desc": "Company Summary is a string"},
            {"field": "Company_Summary",    "check": "min_word_count", "count": 150,  "desc": "Company Summary ≥150 words"},
            {"field": "Market_Intelligence","check": "is_dict",    "desc": "Market_Intelligence is a dict"},
            {"field": "Revenue_By_Segment", "check": "is_list_or_null", "desc": "Revenue_By_Segment is list or null"},
            {"field": "Management_Team",    "check": "is_list_or_null", "desc": "Management_Team is list or null"},
            {"field": "Growth_Initiatives", "check": "is_list_or_null", "desc": "Growth_Initiatives is list or null"},
        ],
    },

    "network": {
        "label": "Project Network_CIP_Atar Capital.pdf",
        "json":  "extracted_results/Project Network_CIP_Atar Capital_extracted.json",
        "pdf":   "Project Network_CIP_Atar Capital.pdf",
        "units": "$M",
        "tests": [
            {"field": "Total_Revenue",  "check": "is_dict",         "desc": "Revenue is a dict"},
            {"field": "Total_Revenue",  "check": "has_actuals",      "min_count": 1, "desc": "Revenue has actual years"},
            {"field": "Adj_EBITDA",     "check": "is_dict",         "desc": "Adj_EBITDA is a dict"},
            {"field": "CAPEX",          "check": "is_dict",         "desc": "CAPEX is a dict (Network has capex)"},
            {"field": "CAPEX",          "check": "all_negative",     "desc": "CAPEX values all negative"},
            # ── Service business — no fixed assets ───────────────────────────
            {"field": "ME_Equipment",   "check": "is_null",         "desc": "ME_Equipment = null (service business)"},
            {"field": "Building_Land",  "check": "is_null",         "desc": "Building_Land = null (service business)"},
            {"field": "Depreciation",   "check": "is_null",         "desc": "Depreciation = null (P&L stops at EBITDA)"},
            # ── TTM key handling ─────────────────────────────────────────────
            {"field": "Total_Revenue",  "check": "has_ttm",         "desc": "Revenue has TTM key"},
            {"field": "AR",             "check": "is_positive_scalar","desc": "AR is a positive number"},
            {"field": "Company_Summary","check": "is_non_empty_string","desc": "Company Summary is a string"},
            {"field": "Total_Revenue",  "check": "valid_year_keys",  "desc": "Revenue year keys valid format"},
        ],
    },

    "palm": {
        "label": "Project Palm_CIM_(Atar Capital).pdf",
        "json":  "extracted_results/Project Palm_CIM_(Atar Capital)_extracted.json",
        "pdf":   "Project Palm_CIM_(Atar Capital).pdf",
        "units": "$M",
        "tests": [
            {"field": "Total_Revenue",  "check": "is_dict",          "desc": "Revenue is a dict"},
            {"field": "Total_Revenue",  "check": "has_actuals",       "min_count": 1, "desc": "Revenue has actual years"},
            {"field": "Total_Revenue",  "check": "has_proj",          "min_count": 3, "desc": "Revenue has ≥3 projected years"},
            {"field": "Adj_EBITDA",     "check": "is_dict",          "desc": "Adj_EBITDA is a dict"},
            # ── Asset-light carve-out — no fixed assets ───────────────────
            {"field": "ME_Equipment",   "check": "is_null",          "desc": "ME_Equipment = null (asset-light)"},
            {"field": "Building_Land",  "check": "is_null",          "desc": "Building_Land = null (asset-light)"},
            {"field": "Depreciation",   "check": "is_null",          "desc": "Depreciation = null (P&L stops at EBITDA)"},
            {"field": "AR",             "check": "is_positive_scalar","desc": "AR is a positive number"},
            {"field": "Inventory",      "check": "is_positive_scalar","desc": "Inventory is a positive number"},
            {"field": "Company_Summary","check": "is_non_empty_string","desc": "Company Summary is a string"},
            {"field": "Total_Revenue",  "check": "valid_year_keys",   "desc": "Revenue year keys valid format"},
        ],
    },

    "smores": {
        "label": "Project Smores - CIM_2026 (Atar).pdf",
        "json":  "extracted_results/Project Smores - CIM_2026 (Atar)_extracted.json",
        "pdf":   "Project Smores - CIM_2026 (Atar).pdf",
        "units": "CAD M",
        "tests": [
            {"field": "Total_Revenue",  "check": "is_dict",          "desc": "Revenue is a dict"},
            {"field": "Total_Revenue",  "check": "has_actuals",       "min_count": 3, "desc": "Revenue has ≥3 actual years"},
            {"field": "Adj_EBITDA",     "check": "is_dict",          "desc": "Adj_EBITDA is a dict"},
            # ── No fixed asset schedule ───────────────────────────────────
            {"field": "ME_Equipment",   "check": "is_null",          "desc": "ME_Equipment = null (no fixed asset schedule)"},
            {"field": "Building_Land",  "check": "is_null",          "desc": "Building_Land = null (no fixed asset schedule)"},
            {"field": "Depreciation",   "check": "is_null",          "desc": "Depreciation = null (P&L stops at EBITDA)"},
            {"field": "CAPEX",          "check": "is_null",          "desc": "CAPEX = null"},
            # ── CAD — no currency conversion expected ─────────────────────
            {"field": "Total_Revenue",  "check": "valid_year_keys",   "desc": "Revenue year keys valid format"},
            {"field": "AR",             "check": "is_positive_scalar","desc": "AR is a positive number"},
            {"field": "Inventory",      "check": "is_positive_scalar","desc": "Inventory is a positive number"},
            {"field": "Company_Summary","check": "is_non_empty_string","desc": "Company Summary is a string"},
        ],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

YEAR_KEY_RE = re.compile(r"^(\d{4}_[AE]|TTM_\d{4})$")

def _within_tol(actual, expected, tol):
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= tol


def run_check(test, data):
    """Run a single test dict against the extracted data dict. Returns (passed, message)."""
    field   = test["field"]
    check   = test["check"]
    value   = data.get(field)

    if check == "is_dict":
        ok = isinstance(value, dict) and len(value) > 0
        return ok, f"Expected dict, got {type(value).__name__}" if not ok else ""

    if check == "is_null":
        ok = value is None
        return ok, f"Expected null, got {repr(value)}" if not ok else ""

    if check == "is_non_empty_string":
        ok = isinstance(value, str) and len(value.strip()) > 0
        return ok, f"Expected non-empty string, got {type(value).__name__}" if not ok else ""

    if check == "min_word_count":
        count = test["count"]
        words = len(value.split()) if isinstance(value, str) else 0
        ok = words >= count
        return ok, f"Expected ≥{count} words, got {words}" if not ok else ""

    if check == "is_positive_scalar":
        ok = isinstance(value, (int, float)) and value > 0
        return ok, f"Expected positive number, got {repr(value)}" if not ok else ""

    if check == "is_list_or_null":
        ok = value is None or isinstance(value, list)
        return ok, f"Expected list or null, got {type(value).__name__}" if not ok else ""

    if check == "has_actuals":
        min_c = test.get("min_count", 1)
        if not isinstance(value, dict):
            return False, "Field is not a dict"
        actuals = [k for k in value if k.endswith("_A")]
        ok = len(actuals) >= min_c
        return ok, f"Expected ≥{min_c} actual (_A) keys, got {len(actuals)}: {actuals}" if not ok else ""

    if check == "has_proj":
        min_c = test.get("min_count", 1)
        if not isinstance(value, dict):
            return False, "Field is not a dict"
        proj = [k for k in value if k.endswith("_E")]
        ok = len(proj) >= min_c
        return ok, f"Expected ≥{min_c} projected (_E) keys, got {len(proj)}: {proj}" if not ok else ""

    if check == "has_ttm":
        if not isinstance(value, dict):
            return False, "Field is not a dict"
        ttm = [k for k in value if k.startswith("TTM")]
        ok = len(ttm) > 0
        return ok, f"Expected TTM key, found none. Keys: {list(value.keys())}" if not ok else ""

    if check == "valid_year_keys":
        if not isinstance(value, dict):
            return False, "Field is not a dict"
        bad = [k for k in value if not YEAR_KEY_RE.match(k)]
        ok = len(bad) == 0
        return ok, f"Invalid year key format: {bad}" if not ok else ""

    if check == "all_negative":
        if not isinstance(value, dict):
            return False, "Field is not a dict"
        bad = {k: v for k, v in value.items() if v is not None and v > 0}
        ok = len(bad) == 0
        return ok, f"Expected all negative, positive values found: {bad}" if not ok else ""

    if check == "exact_value":
        year = test["year"]
        expected = test["value"]
        tol = test.get("tol", 0.05)
        if not isinstance(value, dict):
            return False, "Field is not a dict"
        actual = value.get(year)
        if actual is None:
            return False, f"Key '{year}' not found. Available: {list(value.keys())}"
        ok = _within_tol(actual, expected, tol)
        return ok, f"Expected {expected} (±{int(tol*100)}%), got {actual}" if not ok else ""

    if check == "exact_scalar":
        expected = test["value"]
        tol = test.get("tol", 0.05)
        if value is None:
            return False, "Value is null"
        if not isinstance(value, (int, float)):
            return False, f"Expected scalar number, got {type(value).__name__}: {value}"
        ok = _within_tol(value, expected, tol)
        return ok, f"Expected {expected} (±{int(tol*100)}%), got {value}" if not ok else ""

    return False, f"Unknown check type: {check}"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_suite(suite_key, data, label, units):
    suite = GROUND_TRUTH[suite_key]
    tests = suite["tests"]

    PASS = "\033[92m PASS\033[0m"
    FAIL = "\033[91m FAIL\033[0m"
    SKIP = "\033[93m SKIP\033[0m"

    print(f"\n{'='*70}")
    print(f"  PDF   : {label}")
    print(f"  Units : {units}")
    print(f"  Tests : {len(tests)}")
    print(f"{'='*70}")

    passed = 0
    failed = 0
    skipped = 0
    failures = []

    for test in tests:
        desc = test["desc"]
        field = test["field"]

        if field not in data:
            print(f"  [{SKIP}]  {desc}")
            skipped += 1
            continue

        ok, msg = run_check(test, data)
        if ok:
            print(f"  [{PASS}]  {desc}")
            passed += 1
        else:
            print(f"  [{FAIL}]  {desc}")
            print(f"           → {msg}")
            failed += 1
            failures.append((desc, msg))

    pct = round(passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
    status = "PASS" if failed == 0 else "FAIL"
    color = "\033[92m" if failed == 0 else "\033[91m"

    print(f"\n  Result : {color}{status}\033[0m  —  {passed}/{passed+failed} checks passed ({pct}%)")
    if skipped:
        print(f"  Skipped: {skipped} (field not in JSON)")
    if failures:
        print(f"\n  Failed checks:")
        for desc, msg in failures:
            print(f"    • {desc}: {msg}")

    return passed, failed, skipped


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  ERROR: JSON not found — {path}")
        print("  Run extraction first or use --run flag.")
        return None
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON — {e}")
        return None


def run_fresh_extraction(suite_key, provider, model):
    """Run cim_extractor on the PDF and return extracted JSON."""
    import subprocess
    suite = GROUND_TRUTH[suite_key]
    pdf = suite["pdf"]
    if not os.path.exists(pdf):
        print(f"  ERROR: PDF not found — {pdf}")
        return None
    print(f"  Running extraction: {pdf} via {provider}/{model} ...")
    result = subprocess.run(
        [sys.executable, "cim_extractor.py", "--pdf", pdf, "--provider", provider, "--model", model],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR during extraction:\n{result.stderr[-500:]}")
        return None
    # Load the output JSON
    out_path = f"extracted_results/{Path(pdf).stem.replace(' ', '_')}_extracted.json"
    return load_json(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CIM Extraction Test Suite — Atar Capital")
    parser.add_argument("--pdf",      default="all", choices=["all","chimera","network","palm","smores"],
                        help="Which PDF to test (default: all)")
    parser.add_argument("--run",      action="store_true",
                        help="Run fresh extraction before testing (requires API)")
    parser.add_argument("--provider", default="nvidia",
                        help="LLM provider for --run (default: nvidia)")
    parser.add_argument("--model",    default="meta/llama-3.3-70b-instruct",
                        help="LLM model for --run")
    args = parser.parse_args()

    suites = list(GROUND_TRUTH.keys()) if args.pdf == "all" else [args.pdf]

    total_pass = total_fail = total_skip = 0

    for key in suites:
        suite = GROUND_TRUTH[key]

        if args.run:
            data = run_fresh_extraction(key, args.provider, args.model)
        else:
            data = load_json(suite["json"])

        if data is None:
            continue

        p, f, s = run_suite(key, data, suite["label"], suite["units"])
        total_pass += p
        total_fail += f
        total_skip += s

    print(f"\n{'='*70}")
    print(f"  TOTAL  :  {total_pass} passed  |  {total_fail} failed  |  {total_skip} skipped")
    overall = "ALL PASS" if total_fail == 0 else f"{total_fail} FAILURES"
    color = "\033[92m" if total_fail == 0 else "\033[91m"
    print(f"  STATUS :  {color}{overall}\033[0m")
    print(f"{'='*70}\n")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
