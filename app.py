"""
CIM Financial Intelligence — Flask Server
Atar Capital · M&A Deal Analysis Platform
"""

import os
import sys
import json
import uuid
import logging
import tempfile
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, make_response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cim_extractor import CIMParser, LLMExtractor, _load_env, DEFAULT_CLIENT_REQ

_load_env()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False  # preserve extraction field order in API responses

jobs: dict = {}
jobs_lock = threading.Lock()

PROVIDER_ENV = {
    "nvidia":    "NVIDIA_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "ollama":    "OLLAMA_API_KEY",
}

DEFAULT_REQ = DEFAULT_CLIENT_REQ


class _LogHandler(logging.Handler):
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id

    def emit(self, record):
        with jobs_lock:
            if self.job_id in jobs:
                jobs[self.job_id]["log"].append(self.format(record))


def _run_extraction(job_id, pdf_path, provider, model, api_key, deal_value=None):
    handler = _LogHandler(job_id)
    handler.setFormatter(logging.Formatter("%(levelname)s — %(message)s"))
    logging.getLogger().addHandler(handler)

    def _set(status, **kw):
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = status
                jobs[job_id].update(kw)

    try:
        _set("parsing")
        cim = CIMParser(pdf_path)
        cim.parse_pdf()

        _set("filtering")
        relevant = cim.find_financial_sections() or list(cim.extracted_tables)

        _set("extracting")
        extractor = LLMExtractor(api_key=api_key, provider=provider, model_name=model)
        raw = extractor.extract_fields(relevant, client_requirements=DEFAULT_REQ)
        if raw is None:
            raise RuntimeError("LLM returned no data. Check API key and model.")

        # Generate investment recommendation if deal value provided
        if deal_value:
            try:
                dv_float = float(str(deal_value).replace(',', '').replace('$', '').strip())
                _set("recommending")
                recommendation = extractor.generate_investment_recommendation(raw, dv_float)
                if recommendation:
                    ordered = {}
                    for key in raw:
                        if key == "Company_Summary":
                            ordered["Investment_Recommendation"] = recommendation
                        ordered[key] = raw[key]
                    if "Investment_Recommendation" not in ordered:
                        ordered["Investment_Recommendation"] = recommendation
                    raw = ordered
            except Exception as e:
                logging.warning(f"Investment recommendation skipped: {e}")

        out_dir = Path("extracted_results")
        out_dir.mkdir(exist_ok=True)
        base = Path(pdf_path).stem.replace(" ", "_")
        raw_path = str(out_dir / f"{base}_extracted.json")
        Path(raw_path).write_text(json.dumps(raw, indent=4))

        _set("done", raw=raw, raw_path=raw_path)

    except Exception as exc:
        logging.error(f"Extraction error: {exc}")
        _set("error", error=str(exc))
    finally:
        logging.getLogger().removeHandler(handler)
        try:
            os.unlink(pdf_path)
        except OSError:
            pass


@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/results")
def results():
    resp = make_response(render_template("results.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/extract", methods=["POST"])
def api_extract():
    pdf_file = request.files.get("pdf")
    provider  = request.form.get("provider", "nvidia")
    model     = request.form.get("model", "meta/llama-3.3-70b-instruct")
    api_key   = request.form.get("api_key", "").strip()
    # deal_name and deal_value are metadata only — stored in job for display
    deal_name  = request.form.get("deal_name", "").strip()
    deal_value = request.form.get("deal_value", "").strip()

    if not pdf_file or not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file."}), 400
    if not api_key:
        api_key = os.getenv(PROVIDER_ENV.get(provider, ""), "")
    if not api_key:
        return jsonify({"error": f"API key missing for provider '{provider}'."}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_file.save(tmp.name)
    tmp.close()

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status": "queued", "log": [], "raw": None, "error": None,
            "deal_name": deal_name, "deal_value": deal_value,
            "filename": pdf_file.filename,
        }

    threading.Thread(
        target=_run_extraction,
        args=(job_id, tmp.name, provider, model, api_key, deal_value),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id}), 202


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "log":    job["log"][-20:],
        "error":  job.get("error"),
    })


@app.route("/api/result/<job_id>")
def api_result(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Not finished"}), 400
    return jsonify({
        "raw":        job["raw"],
        "deal_name":  job["deal_name"],
        "deal_value": job["deal_value"],
        "filename":   job["filename"],
    })


@app.route("/api/download/<job_id>")
def api_download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Result not ready"}), 400
    path = job.get("raw_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    import io
    return send_file(
        io.BytesIO(json.dumps(job["raw"], indent=4).encode()),
        mimetype="application/json",
        as_attachment=True,
        download_name="extracted.json",
    )


@app.route("/api/env-keys")
def api_env_keys():
    return jsonify({p: bool(os.getenv(v, "")) for p, v in PROVIDER_ENV.items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    print("\n  CIM Intelligence · Atar Capital")
    print("  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
