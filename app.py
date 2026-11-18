"""
Hotel Booking Cancellation Prediction - Flask app (single file).

* Loads the pre-trained scikit-learn Pipeline (hotel_cancellation_model.pkl). No retraining.
* The Pipeline already contains imputation + one-hot encoding + RandomForest, so the app
  only has to send the 27 RAW columns in the exact order stored in `feature_names_in_`.
* Dropdown options are read from the fitted OneHotEncoder, so they always match training.
* Model output: 0 = Not Cancelled, 1 = Cancelled.

Run locally : python app.py
Production  : gunicorn app:app
Vercel      : exposes the top-level `app` object.
"""
import calendar
import logging
import math
import os
import threading
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from werkzeug.exceptions import HTTPException

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hotel_cancellation_model.pkl"
OUTCOMES = {0: "Not Cancelled", 1: "Cancelled"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hotel-app")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # inputs are tiny; reject anything bigger

# --------------------------------------------------------------------------------------
# Field definitions (UI + validation). Names must match the model's training columns.
# Numeric limits are sanity limits for form validation only; they are not used by the model.
# --------------------------------------------------------------------------------------
NUMERIC = {
    # name: (label, min, max, step, hint)
    "lead_time": ("Lead time (days)", 0, 737, 1, "Days between booking and arrival"),
    "arrival_date_year": ("Arrival year", 2015, 2017, 1, "Years present in the dataset: 2015-2017"),
    "arrival_date_week_number": ("Arrival week number", 1, 53, 1, "ISO week of the year (1-53)"),
    "arrival_date_day_of_month": ("Arrival day of month", 1, 31, 1, "1-31"),
    "stays_in_weekend_nights": ("Weekend nights", 0, 20, 1, "Saturday / Sunday nights"),
    "stays_in_week_nights": ("Week nights", 0, 50, 1, "Monday - Friday nights"),
    "adults": ("Adults", 0, 10, 1, ""),
    "children": ("Children", 0, 10, 1, ""),
    "babies": ("Babies", 0, 10, 1, ""),
    "previous_cancellations": ("Previous cancellations", 0, 30, 1, "Earlier bookings this guest cancelled"),
    "previous_bookings_not_canceled": ("Previous bookings not cancelled", 0, 80, 1, ""),
    "booking_changes": ("Booking changes", 0, 25, 1, "Changes made to the booking"),
    "days_in_waiting_list": ("Days in waiting list", 0, 400, 1, ""),
    "adr": ("Average daily rate (ADR)", 0, 1000, 0.01, "Average price per night"),
    "required_car_parking_spaces": ("Car parking spaces", 0, 8, 1, ""),
    "total_of_special_requests": ("Special requests", 0, 5, 1, ""),
}
INT_FIELDS = {k for k in NUMERIC if k != "adr"}

CATEGORICAL_LABELS = {
    "hotel": "Hotel type",
    "arrival_date_month": "Arrival month",
    "meal": "Meal plan",
    "country": "Country",
    "market_segment": "Market segment",
    "distribution_channel": "Distribution channel",
    "reserved_room_type": "Reserved room type",
    "assigned_room_type": "Assigned room type",
    "deposit_type": "Deposit type",
    "customer_type": "Customer type",
}
OPTION_TEXT = {  # friendly display text; the submitted value is always the trained category
    "meal": {"BB": "BB - Bed & Breakfast", "HB": "HB - Half board", "FB": "FB - Full board",
             "SC": "SC - Self catering"},
    "country": {"AUS": "Australia", "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria",
                "CHE": "Switzerland", "CHN": "China", "CZE": "Czechia", "DEU": "Germany",
                "ESP": "Spain", "FRA": "France", "GBR": "United Kingdom", "IRL": "Ireland",
                "ITA": "Italy", "MEX": "Mexico", "NLD": "Netherlands", "PRT": "Portugal",
                "ROU": "Romania", "USA": "United States"},
}
BINARY = {"is_repeated_guest": ("Repeat guest?", [("0", "No"), ("1", "Yes")])}

SECTIONS = [
    ("Booking", ["hotel", "lead_time", "deposit_type", "customer_type", "adr"]),
    ("Arrival date", ["arrival_date_year", "arrival_date_month", "arrival_date_week_number",
                      "arrival_date_day_of_month"]),
    ("Stay", ["stays_in_weekend_nights", "stays_in_week_nights", "meal",
              "required_car_parking_spaces", "total_of_special_requests"]),
    ("Guests", ["adults", "children", "babies", "country", "is_repeated_guest"]),
    ("Channel & rooms", ["market_segment", "distribution_channel", "reserved_room_type",
                         "assigned_room_type", "booking_changes", "days_in_waiting_list"]),
    ("Guest history", ["previous_cancellations", "previous_bookings_not_canceled"]),
]

# --------------------------------------------------------------------------------------
# Model loading (lazy, thread-safe, never crashes the import)
# --------------------------------------------------------------------------------------
_state = {"model": None, "error": None, "categories": {}}
_lock = threading.Lock()


def _read_categories(model):
    """Return {column: [trained categories]} from the fitted OneHotEncoder(s)."""
    cats = {}
    pre = model.named_steps["preprocessor"]
    for _, transformer, columns in pre.transformers_:
        last = transformer.steps[-1][1] if hasattr(transformer, "steps") else transformer
        if hasattr(last, "categories_"):
            for col, values in zip(columns, last.categories_):
                cats[col] = [str(v) for v in values]
    return cats


def get_model():
    if _state["model"] is None and _state["error"] is None:
        with _lock:
            if _state["model"] is None and _state["error"] is None:
                try:
                    model = joblib.load(MODEL_PATH)
                    features = list(model.feature_names_in_)
                    cats = _read_categories(model)
                    known = set(NUMERIC) | set(BINARY) | set(cats)
                    missing = [f for f in features if f not in known]
                    if missing:
                        raise RuntimeError(f"No form field defined for model features: {missing}")
                    _state.update(model=model, categories=cats)
                    log.info("Model loaded: %d features", len(features))
                except Exception as exc:  # noqa: BLE001
                    log.exception("Model failed to load")
                    _state["error"] = f"{type(exc).__name__}: {exc}"
    return _state["model"]


def build_sections():
    """Turn SECTIONS into renderable field dicts, using the model's trained categories."""
    cats = _state["categories"]
    out = []
    for title, names in SECTIONS:
        fields = []
        for n in names:
            if n in NUMERIC:
                label, lo, hi, step, hint = NUMERIC[n]
                fields.append(dict(name=n, label=label, kind="number", min=lo, max=hi,
                                   step=step, hint=hint))
            elif n in BINARY:
                label, opts = BINARY[n]
                fields.append(dict(name=n, label=label, kind="select", options=opts, hint=""))
            elif n in cats:
                texts = OPTION_TEXT.get(n, {})
                opts = [(v, texts.get(v, v)) for v in cats[n]]
                if n == "country":
                    opts.sort(key=lambda o: o[1])
                fields.append(dict(name=n, label=CATEGORICAL_LABELS.get(n, n), kind="select",
                                   options=opts, hint=""))
        out.append((title, fields))
    return out


# --------------------------------------------------------------------------------------
# Validation + prediction
# --------------------------------------------------------------------------------------
def validate(payload, model):
    """Return (row_dict, errors). Nothing is guessed: every model feature must be supplied."""
    row, errors = {}, {}
    cats = _state["categories"]
    for name in model.feature_names_in_:
        raw = payload.get(name)
        raw = "" if raw is None else str(raw).strip()
        if raw == "":
            errors[name] = "This field is required."
            continue
        if name in NUMERIC:
            _, lo, hi, _, _ = NUMERIC[name]
            try:
                val = float(raw)
                if not math.isfinite(val):
                    raise ValueError
            except ValueError:
                errors[name] = "Enter a valid number."
                continue
            if name in INT_FIELDS:
                if not val.is_integer():
                    errors[name] = "Enter a whole number."
                    continue
                val = int(val)
            if not lo <= val <= hi:
                errors[name] = f"Must be between {lo} and {hi}."
                continue
            row[name] = val
        elif name in BINARY:
            if raw not in {"0", "1"}:
                errors[name] = "Choose Yes or No."
            else:
                row[name] = int(raw)
        else:
            if raw not in cats.get(name, []):
                errors[name] = "Choose one of the listed options."
            else:
                row[name] = raw

    if not errors:  # cross-field checks
        if row["adults"] + row["children"] + row["babies"] < 1:
            errors["adults"] = "At least one guest is required."
        month_no = list(calendar.month_name).index(row["arrival_date_month"])
        last_day = calendar.monthrange(row["arrival_date_year"], month_no)[1]
        if row["arrival_date_day_of_month"] > last_day:
            errors["arrival_date_day_of_month"] = (
                f"{row['arrival_date_month']} {row['arrival_date_year']} has only {last_day} days.")
    return row, errors


def run_prediction(row, model):
    # Column order is taken from the trained model itself -> exact training order.
    frame = pd.DataFrame([row], columns=list(model.feature_names_in_))
    pred = int(model.predict(frame)[0])
    proba = model.predict_proba(frame)[0]
    classes = [int(c) for c in model.classes_]
    p_cancel = float(proba[classes.index(1)])
    return {"prediction": pred, "label": OUTCOMES[pred],
            "probability_cancelled": round(p_cancel, 4),
            "probability_not_cancelled": round(1.0 - p_cancel, 4)}


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    model = get_model()
    return render_template_string(PAGE, sections=build_sections() if model else [],
                                  load_error=_state["error"])


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    model = get_model()
    if model is None:
        return jsonify(ok=False, error="The model could not be loaded on the server."), 503
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if not isinstance(payload, dict):
        return jsonify(ok=False, error="Send the booking details as JSON or form data."), 400
    row, errors = validate(payload, model)
    if errors:
        return jsonify(ok=False, error="Please correct the highlighted fields.", errors=errors), 422
    try:
        return jsonify(ok=True, **run_prediction(row, model))
    except Exception:  # noqa: BLE001
        log.exception("Prediction failed")
        return jsonify(ok=False, error="Prediction failed. Please check your inputs."), 500


@app.route("/health", methods=["GET"])
def health():
    model = get_model()
    body = {"status": "ok" if model else "error", "model_loaded": model is not None,
            "n_features": len(model.feature_names_in_) if model else None}
    return jsonify(body), (200 if model else 503)


@app.errorhandler(HTTPException)
def http_error(err):
    return jsonify(ok=False, error=err.description or err.name), err.code


@app.errorhandler(Exception)
def unhandled_error(err):
    log.exception("Unhandled error")
    return jsonify(ok=False, error="Internal server error."), 500


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    return resp


# --------------------------------------------------------------------------------------
# UI (single template, no external assets -> works offline and on any host)
# --------------------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hotel Booking Cancellation Predictor</title>
<meta name="color-scheme" content="light dark">
<script>
  try { var t = localStorage.getItem("theme") ||
        (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
        document.documentElement.setAttribute("data-theme", t); } catch (e) {}
</script>
<style>
:root{--bg:#f6f3ee;--card:#fff;--ink:#1f2937;--muted:#6b7280;--line:#e5e0d6;--gold:#b8860b;
--gold-ink:#fff;--accent:#1e3a5f;--ok:#15803d;--ok-bg:#e8f5ec;--bad:#b91c1c;--bad-bg:#fdecec;
--shadow:0 10px 30px rgba(31,41,55,.08);--field:#fbfaf7}
[data-theme=dark]{--bg:#0f141b;--card:#171e28;--ink:#e8ecf1;--muted:#9aa5b4;--line:#283140;
--gold:#d4a73a;--gold-ink:#1a1405;--accent:#8fb3e0;--ok:#4ade80;--ok-bg:#12281b;--bad:#f87171;
--bad-bg:#2d1618;--shadow:0 10px 30px rgba(0,0,0,.45);--field:#111823}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
transition:background .2s,color .2s}
.wrap{max-width:1040px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 0 20px}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:44px;height:44px;border-radius:12px;background:var(--gold);color:var(--gold-ink);display:grid;place-items:center}
h1{font:600 clamp(1.25rem,3.2vw,1.75rem)/1.2 Georgia,"Times New Roman",serif;margin:0}
.sub{color:var(--muted);font-size:.9rem;margin:2px 0 0}
.btn{cursor:pointer;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;
padding:8px 14px;font:inherit;font-size:.9rem;display:inline-flex;align-items:center;gap:8px}
.btn:hover{border-color:var(--gold)}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);padding:22px;margin-bottom:18px}
.card h2{font:600 1.05rem Georgia,serif;margin:0 0 14px;color:var(--accent);display:flex;align-items:center;gap:8px}
.card h2:before{content:"";width:4px;height:18px;background:var(--gold);border-radius:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px 16px}
label{display:block;font-size:.85rem;font-weight:600;margin-bottom:5px}
input,select{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--field);
color:var(--ink);font:inherit;min-height:44px}
input:focus,select:focus{outline:2px solid var(--gold);outline-offset:1px}
.hint{color:var(--muted);font-size:.76rem;margin-top:3px}
.err{color:var(--bad);font-size:.78rem;margin-top:3px;min-height:0}
.invalid input,.invalid select{border-color:var(--bad)}
.actions{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.primary{background:var(--gold);color:var(--gold-ink);border:0;font-weight:700;padding:13px 28px;border-radius:12px;
cursor:pointer;font-size:1rem}
.primary:disabled{opacity:.6;cursor:wait}
.alert{padding:12px 14px;border-radius:12px;background:var(--bad-bg);color:var(--bad);margin-bottom:16px;display:none}
.alert.show{display:block}
#result{display:none;border-radius:18px;padding:22px;margin-bottom:18px;border:1px solid var(--line);scroll-margin-top:16px}
#result.show{display:block}
#result.ok{background:var(--ok-bg);border-color:var(--ok)}
#result.bad{background:var(--bad-bg);border-color:var(--bad)}
.verdict{font:700 1.6rem Georgia,serif;margin:0}
#result.ok .verdict{color:var(--ok)}#result.bad .verdict{color:var(--bad)}
.bar{height:12px;background:var(--line);border-radius:99px;overflow:hidden;margin:14px 0 6px}
.bar>i{display:block;height:100%;width:0;background:var(--bad);transition:width .6s}
.legend{display:flex;justify-content:space-between;font-size:.85rem;color:var(--muted)}
.note{font-size:.8rem;color:var(--muted);margin-top:12px}
footer{color:var(--muted);font-size:.8rem;text-align:center;margin-top:8px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="brand">
    <div class="logo" aria-hidden="true">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v16"/><path d="M9 21v-4h6v4"/><path d="M8 7h2M14 7h2M8 11h2M14 11h2"/></svg>
    </div>
    <div><h1>Hotel Booking Cancellation Predictor</h1>
    <p class="sub">Random Forest model - estimates whether a booking will be cancelled</p></div>
  </div>
  <button class="btn" id="themeBtn" type="button" aria-label="Toggle light and dark mode">
    <span id="themeIcon">&#9790;</span><span id="themeText">Dark</span></button>
</header>

{% if load_error %}
<div class="alert show" role="alert">The prediction model could not be loaded: {{ load_error }}</div>
{% else %}
<div id="alert" class="alert" role="alert"></div>
<div id="result" tabindex="-1" aria-live="polite">
  <p class="sub" style="margin:0 0 4px">Prediction</p>
  <p class="verdict" id="verdict"></p>
  <div class="bar"><i id="barFill"></i></div>
  <div class="legend"><span id="pStay"></span><span id="pCancel"></span></div>
  <p class="note">Model output: 0 = Not Cancelled, 1 = Cancelled. The percentages are the model's class probabilities, not a guarantee.</p>
</div>

<form id="form" novalidate>
  {% for title, fields in sections %}
  <section class="card">
    <h2>{{ title }}</h2>
    <div class="grid">
      {% for f in fields %}
      <div class="field" data-field="{{ f.name }}">
        <label for="{{ f.name }}">{{ f.label }}</label>
        {% if f.kind == "select" %}
        <select id="{{ f.name }}" name="{{ f.name }}" required>
          <option value="" selected disabled>Select...</option>
          {% for value, text in f.options %}<option value="{{ value }}">{{ text }}</option>{% endfor %}
        </select>
        {% else %}
        <input id="{{ f.name }}" name="{{ f.name }}" type="number" inputmode="decimal"
               min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}" required>
        {% endif %}
        {% if f.hint %}<div class="hint">{{ f.hint }}</div>{% endif %}
        <div class="err" aria-live="polite"></div>
      </div>
      {% endfor %}
    </div>
  </section>
  {% endfor %}
  <div class="actions">
    <button class="primary" id="submitBtn" type="submit">Predict cancellation</button>
    <button class="btn" id="exampleBtn" type="button">Fill example</button>
    <button class="btn" id="resetBtn" type="reset">Reset</button>
  </div>
</form>
{% endif %}
<footer>Built with Flask and scikit-learn</footer>
</div>

<script>
(function () {
  var root = document.documentElement;
  function paintTheme() {
    var dark = root.getAttribute("data-theme") === "dark";
    document.getElementById("themeIcon").innerHTML = dark ? "&#9728;" : "&#9790;";
    document.getElementById("themeText").textContent = dark ? "Light" : "Dark";
  }
  paintTheme();
  document.getElementById("themeBtn").addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    paintTheme();
  });

  var form = document.getElementById("form");
  if (!form) return;
  var alertBox = document.getElementById("alert"), result = document.getElementById("result");
  var submitBtn = document.getElementById("submitBtn");

  // Demo helper only: sample values to test the deployment quickly. Safe to delete.
  var EXAMPLE = {hotel:"City Hotel",lead_time:120,arrival_date_year:2016,arrival_date_month:"August",
    arrival_date_week_number:33,arrival_date_day_of_month:14,stays_in_weekend_nights:2,
    stays_in_week_nights:3,adults:2,children:0,babies:0,meal:"BB",country:"PRT",
    market_segment:"Online TA",distribution_channel:"TA/TO",is_repeated_guest:0,
    previous_cancellations:0,previous_bookings_not_canceled:0,reserved_room_type:"A",
    assigned_room_type:"A",booking_changes:0,deposit_type:"No Deposit",days_in_waiting_list:0,
    customer_type:"Transient",adr:95,required_car_parking_spaces:0,total_of_special_requests:1};

  function clearErrors() {
    alertBox.className = "alert"; alertBox.textContent = "";
    form.querySelectorAll(".field").forEach(function (f) {
      f.classList.remove("invalid"); f.querySelector(".err").textContent = "";
    });
  }
  function showErrors(map, message) {
    var first = null;
    Object.keys(map || {}).forEach(function (name) {
      var box = form.querySelector('[data-field="' + name + '"]');
      if (!box) return;
      box.classList.add("invalid"); box.querySelector(".err").textContent = map[name];
      if (!first) first = box;
    });
    alertBox.textContent = message || "Something went wrong."; alertBox.className = "alert show";
    if (first) { first.scrollIntoView({behavior: "smooth", block: "center"});
                 var el = first.querySelector("input,select"); if (el) el.focus({preventScroll: true}); }
  }

  document.getElementById("exampleBtn").addEventListener("click", function () {
    clearErrors();
    Object.keys(EXAMPLE).forEach(function (k) { var el = form.elements[k]; if (el) el.value = EXAMPLE[k]; });
  });
  form.addEventListener("reset", function () { clearErrors(); result.className = ""; });

  form.addEventListener("submit", function (ev) {
    ev.preventDefault(); clearErrors(); result.className = "";
    var data = {};
    new FormData(form).forEach(function (v, k) { data[k] = v; });
    submitBtn.disabled = true; submitBtn.textContent = "Predicting...";
    fetch("/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)})
      .then(function (r) { return r.json().then(function (j) { return {status: r.status, body: j}; }); })
      .then(function (res) {
        var b = res.body;
        if (!b.ok) { showErrors(b.errors, b.error); return; }
        var cancelled = b.prediction === 1;
        result.className = "show " + (cancelled ? "bad" : "ok");
        document.getElementById("verdict").textContent = cancelled
          ? "Likely to be cancelled" : "Likely to be kept (not cancelled)";
        document.getElementById("barFill").style.width = (b.probability_cancelled * 100).toFixed(1) + "%";
        document.getElementById("pStay").textContent = "Not cancelled: " + (b.probability_not_cancelled * 100).toFixed(1) + "%";
        document.getElementById("pCancel").textContent = "Cancelled: " + (b.probability_cancelled * 100).toFixed(1) + "%";
        result.scrollIntoView({behavior: "smooth", block: "start"}); result.focus({preventScroll: true});
      })
      .catch(function () { showErrors({}, "Network error. Please try again."); })
      .finally(function () { submitBtn.disabled = false; submitBtn.textContent = "Predict cancellation"; });
  });
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)