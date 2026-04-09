"""
LlamaParse PDF Extraction Test
Atar Capital — CIM Financial Extraction Project

Tests LlamaParse against our known CIM PDFs and verifies
that financial tables, year headers, and key values extract cleanly.

Setup:
  pip install llama-parse
  Get free API key: https://cloud.llamaindex.ai

Usage:
  python test_llamaparse.py --key YOUR_API_KEY --pdf "Project Chimera (1).pdf"
  python test_llamaparse.py --key YOUR_API_KEY --all
"""

import argparse
import sys
import os
import re

# ── Known ground-truth values from our verified extractions ─────────────────
GROUND_TRUTH = {
    "Project Chimera (1).pdf": {
        "desc": "Chimera — $000s, 2022A-2028F, Fixed Asset Schedule p62, Depreciation p55",
        "must_contain": [
            "2022", "2023", "2024", "2025", "2026", "2027", "2028",  # year columns
            "Revenue", "EBITDA",                                        # key rows
            "Depreciation",                                             # Rule 15
            "14,634",  "14634",                                        # ME_Equipment (Dec-25A)
            "3,250",   "3250",                                         # Building_Land
            "2,559",   "2559",                                         # Depreciation 2022A
        ],
        "check_tables": True,
    },
    "Project Network_CIP_Atar Capital.pdf": {
        "desc": "Network — $M, service business, no fixed assets",
        "must_contain": ["Revenue", "EBITDA"],
        "check_tables": True,
    },
    "Project Palm_CIM_(Atar Capital).pdf": {
        "desc": "Palm — $M, carve-out, asset-light",
        "must_contain": ["Revenue", "EBITDA"],
        "check_tables": True,
    },
    "Project Smores - CIM_2026 (Atar).pdf": {
        "desc": "Smores — CAD millions, no conversion, no fixed asset schedule",
        "must_contain": ["Revenue", "EBITDA", "CAD"],
        "check_tables": True,
    },
}

PDF_FILES = list(GROUND_TRUTH.keys())


def check_ground_truth(pdf_name, full_text):
    """Check how many known values are present in extracted text."""
    gt = GROUND_TRUTH.get(pdf_name)
    if not gt:
        return
    print("\n  ── Ground Truth Checks ──")
    hits = 0
    for val in gt["must_contain"]:
        found = val.lower() in full_text.lower()
        status = "✓" if found else "✗ MISSING"
        print(f"    [{status}] '{val}'")
        if found:
            hits += 1
    total = len(gt["must_contain"])
    pct = round(hits / total * 100)
    print(f"\n  Score: {hits}/{total} ({pct}%) — {'PASS' if pct >= 70 else 'FAIL'}")


def show_table_sample(full_text, keyword, context_lines=6):
    """Find and print lines around a keyword to check table quality."""
    lines = full_text.splitlines()
    for i, line in enumerate(lines):
        if keyword.lower() in line.lower():
            start = max(0, i - 1)
            end = min(len(lines), i + context_lines)
            snippet = "\n".join(f"    {l}" for l in lines[start:end])
            return snippet
    return f"    ('{keyword}' not found in extracted text)"


def count_tables(full_text):
    """Rough count of markdown tables in output."""
    table_rows = [l for l in full_text.splitlines() if l.strip().startswith("|")]
    return len(table_rows)


def run_test(pdf_path, api_key):
    pdf_name = os.path.basename(pdf_path)
    gt = GROUND_TRUTH.get(pdf_name, {})

    print(f"\n{'='*70}")
    print(f"  PDF : {pdf_name}")
    print(f"  Info: {gt.get('desc', 'unknown')}")
    print(f"{'='*70}")

    if not os.path.exists(pdf_path):
        print(f"  ERROR: File not found — {pdf_path}")
        return

    # ── Parse ────────────────────────────────────────────────────────────────
    try:
        from llama_parse import LlamaParse
    except ImportError:
        print("  ERROR: llama-parse not installed. Run: pip install llama-parse")
        sys.exit(1)

    print("  Parsing with LlamaParse (markdown mode)...")
    try:
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            verbose=False,
            language="en",
        )
        documents = parser.load_data(pdf_path)
    except Exception as e:
        print(f"  ERROR during parse: {e}")
        return

    # ── Combine all pages ────────────────────────────────────────────────────
    full_text = "\n".join(doc.text for doc in documents)
    total_chars = len(full_text)
    total_pages = len(documents)
    table_row_count = count_tables(full_text)

    print(f"  Pages returned : {total_pages}")
    print(f"  Total chars    : {total_chars:,}")
    print(f"  Markdown table rows found: {table_row_count}")

    if total_chars < 500:
        print("  WARNING: Very little text extracted — may be scanned/image PDF")

    # ── Ground truth checks ──────────────────────────────────────────────────
    if pdf_name in GROUND_TRUTH:
        check_ground_truth(pdf_name, full_text)

    # ── Show financial table samples ─────────────────────────────────────────
    print("\n  ── Financial Table Samples ──")

    print(f"\n  > Around 'Revenue':")
    print(show_table_sample(full_text, "Revenue"))

    print(f"\n  > Around 'EBITDA':")
    print(show_table_sample(full_text, "EBITDA"))

    if pdf_name == "Project Chimera (1).pdf":
        print(f"\n  > Around 'Depreciation':")
        print(show_table_sample(full_text, "Depreciation"))

        print(f"\n  > Around 'Machinery' or 'Equipment':")
        snippet = show_table_sample(full_text, "Machinery")
        if "not found" in snippet:
            snippet = show_table_sample(full_text, "Equipment")
        print(snippet)

    # ── First 800 chars of raw output ────────────────────────────────────────
    print(f"\n  ── Raw output (first 800 chars) ──")
    print(full_text[:800])

    # ── Save full output to file ─────────────────────────────────────────────
    out_path = pdf_name.replace(".pdf", "_llamaparse.txt").replace(" ", "_")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\n  Full output saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Test LlamaParse against Atar Capital CIM PDFs")
    parser.add_argument("--key",  required=True, help="LlamaParse API key from cloud.llamaindex.ai")
    parser.add_argument("--pdf",  default=None,  help="Specific PDF filename to test")
    parser.add_argument("--all",  action="store_true", help="Test all 4 known CIM PDFs")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.all:
        pdfs = [os.path.join(base_dir, p) for p in PDF_FILES]
    elif args.pdf:
        pdfs = [os.path.join(base_dir, args.pdf) if not os.path.isabs(args.pdf) else args.pdf]
    else:
        # Default: test Chimera only (our most verified PDF)
        pdfs = [os.path.join(base_dir, "Project Chimera (1).pdf")]
        print("No --pdf specified. Defaulting to Project Chimera (1).pdf")
        print("Use --all to test all 4 PDFs, or --pdf 'filename.pdf' for a specific one.")

    for pdf in pdfs:
        run_test(pdf, args.key)

    print(f"\n{'='*70}")
    print("  Done. Check the _llamaparse.txt files for full output quality review.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
