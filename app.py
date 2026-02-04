"""
Streamlit app for ADE Classifier and Relational Extraction.
Loads models from local folders or from Google Drive (if folder IDs are set).
"""
import os
import re
import random
import time
import html
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import requests
except ImportError:
    requests = None

# Paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(_BASE_DIR, "classifier")
RELATIONAL_PATH = os.path.join(_BASE_DIR, "relational_extraction")
TEST_CSV_PATH = os.path.join(_BASE_DIR, "test (1).csv")
MAX_LEN = 128

# Google Drive folder IDs (hardcoded; override with env CLASSIFIER_DRIVE_ID / RELATIONAL_DRIVE_ID if needed).
CLASSIFIER_DRIVE_ID = os.environ.get("CLASSIFIER_DRIVE_ID", "").strip() or "13eHMflUnsfbUfmi6BznkQwX8VrA2Dmx8"
RELATIONAL_DRIVE_ID = os.environ.get("RELATIONAL_DRIVE_ID", "").strip() or "1Oz25YsSMwwJvQQc8S338KN7s_8hpw_kP"
_MODEL_CACHE = os.path.join(_BASE_DIR, ".drive_model_cache")


def _ensure_model_path(name: str, drive_id, local_path: str) -> str:
    """Return path to model: use local if present, else download from Drive to cache."""
    cache_subdir = os.path.join(_MODEL_CACHE, name)
    if local_path and os.path.isdir(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
        return local_path
    if drive_id:
        os.makedirs(_MODEL_CACHE, exist_ok=True)
        if not os.path.exists(os.path.join(cache_subdir, "config.json")):
            try:
                import gdown
                gdown.download_folder(
                    id=drive_id,
                    output=cache_subdir,
                    quiet=False,
                    use_cookies=False,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download {name} model from Google Drive: {e}") from e
        if os.path.exists(os.path.join(cache_subdir, "config.json")):
            return cache_subdir
    raise FileNotFoundError(
        f"Model '{name}' not found. Use local folder '{local_path}' or set {name.upper()}_DRIVE_ID (and share the folder with 'Anyone with the link')."
    )


@st.cache_resource
def load_classifier():
    """Load ADE classifier from local path or Google Drive."""
    path = _ensure_model_path("classifier", CLASSIFIER_DRIVE_ID, CLASSIFIER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_relational():
    """Load relation extraction model from local path or Google Drive."""
    path = _ensure_model_path("relational_extraction", RELATIONAL_DRIVE_ID, RELATIONAL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def fetch_openfda_meddrapt(drug: str, reaction: str, limit: int = 50):
    """Fetch MedDRA preferred terms from OpenFDA for this drug–reaction pair. Returns list of terms or [] on error."""
    if not requests:
        return []
    base_url = "https://api.fda.gov/drug/event.json"
    search = (
        f'patient.drug.medicinalproduct:"{drug}" AND '
        f'patient.reaction.reactionmeddrapt:"{reaction}"'
    )
    params = {"search": search, "limit": limit}
    try:
        r = requests.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    terms = set()
    for rec in data.get("results", []):
        for rxn in rec.get("patient", {}).get("reaction", []):
            pt = rxn.get("reactionmeddrapt")
            if pt:
                terms.add(pt)
    return sorted(terms)


def wrap_entities(text: str, drug: str, ade: str) -> str:
    """Wrap drug and ADE in markers [DRUG]...[/DRUG] and [ADE]...[/ADE] (first occurrence each)."""
    try:
        text = re.sub(f"({re.escape(drug)})", r"[DRUG]\1[/DRUG]", text, flags=re.IGNORECASE, count=1)
        text = re.sub(f"({re.escape(ade)})", r"[ADE]\1[/ADE]", text, flags=re.IGNORECASE, count=1)
    except Exception:
        pass
    return text


def predict_ade_batch(texts, tokenizer, model, device, batch_size=16):
    """Classify whether each clinical note has ADE (1) or not (0)."""
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
    return all_preds


def run_relation_extraction(row_df, rel_tokenizer, rel_model, device):
    """
    For one row (clinical_note, drugs, conditions), run relation model on each (drug, condition) pair.
    Returns DataFrame with columns: drug, disease, is_valid_ade, confidence, sentence_view.
    """
    context = str(row_df.get("clinical_note", ""))
    drugs_str = str(row_df.get("drugs", ""))
    conditions_str = str(row_df.get("conditions", ""))

    drugs = [d.strip() for d in drugs_str.split("|") if d.strip()]
    diseases = [c.strip() for c in conditions_str.split("|") if c.strip()]

    if not context or context == "nan":
        return pd.DataFrame()

    final_results = []
    for d in drugs:
        for c in diseases:
            marked_text = wrap_entities(context, d, c)
            inputs = rel_tokenizer(
                marked_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
            ).to(device)
            with torch.no_grad():
                logits = rel_model(**inputs).logits
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = torch.softmax(logits, dim=-1)[0][1].item()

            final_results.append({
                "drug": d,
                "disease": c,
                "is_valid_ade": "Yes" if prediction == 1 else "No",
                "confidence": round(confidence, 4),
                "sentence_view": marked_text[:200] + ("..." if len(marked_text) > 200 else ""),
            })

    return pd.DataFrame(final_results)


def run_test():
    """Pick a random row from test (1).csv, run both models, return results."""
    if not os.path.exists(TEST_CSV_PATH):
        return None, None, None, "Test file not found: test (1).csv"

    df = pd.read_csv(TEST_CSV_PATH)
    if df.empty:
        return None, None, None, "Test CSV is empty."

    # Normalize column names (in case of extra spaces)
    df.columns = df.columns.str.strip()
    required = ["clinical_note", "drugs", "conditions"]
    for col in required:
        if col not in df.columns:
            return None, None, None, f"Missing column: {col}"

    row_idx = random.randint(0, len(df) - 1)
    row = df.iloc[row_idx]

    try:
        cls_tokenizer, cls_model, device = load_classifier()
        rel_tokenizer, rel_model, _ = load_relational()

        # Step 1: ADE classification on full note
        has_ade_list = predict_ade_batch([str(row["clinical_note"])], cls_tokenizer, cls_model, device)
        has_ade = int(has_ade_list[0])

        # Step 2: Relation extraction for each (drug, condition) pair
        results_df = run_relation_extraction(row, rel_tokenizer, rel_model, device)

        # Build summary row for display
        summary_row = pd.DataFrame([{
            "clinical_note": row["clinical_note"],
            "has_ade": has_ade,
            "drugs": row["drugs"],
            "conditions": row["conditions"],
            "ade_relations_ground_truth": row.get("ade_relations", ""),
        }])

        return summary_row, results_df, row_idx, None
    except Exception as e:
        return None, None, None, str(e)


# --- Streamlit UI ---
st.set_page_config(page_title="Pharmacovigilance Pipeline - IQVIA Panel Demo", layout="wide")
st.title("🏥 Pharmacovigilance Pipeline - IQVIA Panel Demo")
st.markdown('<p style="font-size: 1.35rem; margin-top: 0.25rem; margin-bottom: 0.5rem;">Team: Mridul · Harshita Jena · Sahiti Amirapu</p>', unsafe_allow_html=True)

# BONUS feature: always kept up (right below title, before button and results)
st.markdown('<span style="color: #2e7d32; font-weight: 700;">BONUS feature</span>', unsafe_allow_html=True)
st.checkbox(" Enable OpenFDA Integration", value=False, key="openfda", help="Fetch FDA MedDRA reaction terms for validated drug–ADE pairs (requires network).")

# Test button
if st.button("🧪 Test on random clinical note", type="primary", use_container_width=True):
    t0 = time.perf_counter()
    with st.spinner("Loading models and running on a random test row..."):
        summary_row, results_df, row_idx, err = run_test()
        elapsed = time.perf_counter() - t0
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)

    if err:
        st.error(err)
    else:
        has_ade_val = summary_row["has_ade"].iloc[0]
        raw_note = summary_row["clinical_note"].iloc[0]
        if pd.isna(raw_note):
            note = ""
        else:
            note = str(raw_note).strip()
        drugs_str = str(summary_row["drugs"].iloc[0])
        conditions_str = str(summary_row["conditions"].iloc[0])
        gt_relations = str(summary_row["ade_relations_ground_truth"].iloc[0] or "")

        # --- 1. Clinical note first (primary input) ---
        st.subheader("1. Clinical note")
        if note:
            # Show full note in a readable block; original box size, larger font inside
            note_escaped = html.escape(note).replace("\n", "<br>")
            note_box_html = (
                '<div style="'
                "background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;"
                "padding: 0.75rem 1rem; max-height: 400px; overflow-y: auto;"
                "font-size: 1.1rem; line-height: 1.6; color: #1a1a1a;"
                '">' + note_escaped + "</div>"
            )
            st.markdown(note_box_html, unsafe_allow_html=True)
        else:
            st.warning("No clinical note in this row.")
        st.divider()

        # --- At-a-glance metrics ---
        yes_count = len(results_df[results_df["is_valid_ade"] == "Yes"]) if results_df is not None and not results_df.empty else 0
        pair_count = len(results_df) if results_df is not None else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Has ADE", "Yes" if has_ade_val == 1 else "No")
        m2.metric("Pairs checked", pair_count)
        m3.metric("Validated ADEs", yes_count)

        # --- Final verdict (when ADE): only drugs/conditions with is_valid_ade == "Yes" ---
        if has_ade_val == 1:
            drugs_display = "—"
            conditions_display = "—"
            if results_df is not None and not results_df.empty:
                yes_df = results_df[results_df["is_valid_ade"] == "Yes"]
                if not yes_df.empty:
                    validated_drugs = yes_df["drug"].unique().tolist()
                    validated_conditions = yes_df["disease"].unique().tolist()
                    # Format as "a, b & c" for both
                    def _fmt_list(items):
                        if not items:
                            return "—"
                        if len(items) == 1:
                            return html.escape(items[0])
                        if len(items) == 2:
                            return html.escape(f"{items[0]} & {items[1]}")
                        return html.escape(", ".join(items[:-1]) + " & " + items[-1])
                    drugs_display = _fmt_list(validated_drugs)
                    conditions_display = _fmt_list(validated_conditions)
            verdict_html = (
                '<div style="'
                "background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%);"
                "border-left: 4px solid #c62828;"
                "border-radius: 8px;"
                "padding: 1rem 1.2rem;"
                "margin: 0.5rem 0;"
                "box-shadow: 0 1px 3px rgba(198,40,40,0.12);"
                '">'
                '<p style="margin: 0; color: #b71c1c; font-size: 1.08rem; line-height: 1.5;">'
                '<span style="font-size: 1.28em;">⚠️</span> '
                '<strong style="color: #c62828; font-size: 1.12em;">Final verdict</strong><br><br>'
                '<span style="font-size: 1.04em;">'
                '💊 Patient is found to have an ADE caused by '
                f'<strong style="color: #b71c1c;">{drugs_display}</strong>; '
                'the ADE experienced is '
                f'<strong style="color: #b71c1c;">{conditions_display}</strong>.'
                '</span></p></div>'
            )
            st.markdown(verdict_html, unsafe_allow_html=True)
        else:
            # --- Final verdict (no ADE): green — format drugs as "a, b & c" ---
            raw_drugs = [d.strip() for d in (drugs_str or "").split("|") if d.strip()] if drugs_str and drugs_str != "nan" else []
            if len(raw_drugs) == 0:
                medicine_display = "—"
            elif len(raw_drugs) == 1:
                medicine_display = html.escape(raw_drugs[0])
            elif len(raw_drugs) == 2:
                medicine_display = html.escape(f"{raw_drugs[0]} & {raw_drugs[1]}")
            else:
                medicine_display = html.escape(", ".join(raw_drugs[:-1]) + " & " + raw_drugs[-1])
            no_ade_html = (
                '<div style="'
                "background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);"
                "border-left: 4px solid #2e7d32;"
                "border-radius: 8px;"
                "padding: 1rem 1.2rem;"
                "margin: 0.5rem 0;"
                "box-shadow: 0 1px 3px rgba(46,125,50,0.12);"
                '">'
                '<p style="margin: 0; color: #1b5e20; font-size: 1.08rem; line-height: 1.5;">'
                '<span style="font-size: 1.28em;">✅</span> '
                '<strong style="color: #2e7d32; font-size: 1.12em;">Final verdict</strong><br><br>'
                '<span style="font-size: 1.04em;">'
                '💊 Patient took '
                f'<strong style="color: #1b5e20;">{medicine_display}</strong> '
                'and no ADE was experienced.'
                '</span></p></div>'
            )
            st.markdown(no_ade_html, unsafe_allow_html=True)

        st.divider()

        # --- 2. Drugs, conditions, ground truth ---
        st.subheader("2. Input & ADE classification")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<span style="color: green; font-weight: 700; font-size: 1.1em;">Drugs</span>', unsafe_allow_html=True)
            st.text(drugs_str if drugs_str and drugs_str != "nan" else "—")
        with c2:
            st.markdown("**Conditions**")
            st.text(conditions_str if conditions_str and conditions_str != "nan" else "—")
        with c3:
            st.markdown("**Ground truth (ade_relations)**")
            st.text(gt_relations if gt_relations and gt_relations != "nan" else "—")
        st.divider()

        # --- 3. Pair validation ---
        st.subheader("3. Drug–condition pair validation")
        if results_df is not None and not results_df.empty:
            display_df = results_df.copy()
            enable_openfda = st.session_state.get("openfda", False)
            openfda_log = []  # Log to verify if FDA API is working
            if enable_openfda and requests is not None:
                yes_mask = results_df["is_valid_ade"] == "Yes"
                if yes_mask.any():
                    display_df["openfda_meddrapt"] = [[] for _ in range(len(display_df))]
                    with st.spinner("Fetching OpenFDA MedDRA reaction terms for validated pairs..."):
                        for i, idx in enumerate(results_df.index[yes_mask], 1):
                            drug = str(results_df.at[idx, "drug"]).strip()
                            reaction = str(results_df.at[idx, "disease"]).strip()
                            openfda_log.append(f"[{i}] Calling FDA API: drug=\"{drug}\", reaction=\"{reaction}\"")
                            try:
                                terms = fetch_openfda_meddrapt(drug, reaction)
                                display_df.at[idx, "openfda_meddrapt"] = terms
                                n = len(terms)
                                if n:
                                    openfda_log.append(f"    → OK: {n} MedDRA term(s) found.")
                                else:
                                    openfda_log.append(f"    → No terms (no FDA reports or API returned empty).")
                            except Exception as e:
                                display_df.at[idx, "openfda_meddrapt"] = []
                                openfda_log.append(f"    → Error: {e}")
                            time.sleep(0.2)
                    openfda_log.insert(0, "OpenFDA enabled. Check log below to verify FDA API is working.")
                    # Show as comma-separated string for dataframe display
                    display_df["openfda_meddrapt"] = display_df["openfda_meddrapt"].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) and x else "—"
                    )
                else:
                    openfda_log.append("OpenFDA enabled but no validated ADE pairs to query.")
            elif enable_openfda and not requests:
                openfda_log.append("OpenFDA enabled but `requests` not installed. Install with: pip install requests")
            if openfda_log:
                with st.expander("OpenFDA log (check if FDA is working)"):
                    st.text("\n".join(openfda_log))
            col_config = {
                "confidence": st.column_config.NumberColumn(format="%.2f"),
                "sentence_view": st.column_config.TextColumn("sentence (preview)", width="medium"),
            }
            if "openfda_meddrapt" in display_df.columns:
                col_config["openfda_meddrapt"] = st.column_config.TextColumn("OpenFDA MedDRA terms", width="large")
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config=col_config,
            )
            yes_df = results_df[results_df["is_valid_ade"] == "Yes"]
            if not yes_df.empty:
                with st.expander("Strongest Yes predictions (top 15 by confidence)"):
                    st.dataframe(
                        yes_df.sort_values("confidence", ascending=False).head(15),
                        use_container_width=True,
                        hide_index=True,
                        column_config={"confidence": st.column_config.NumberColumn(format="%.2f")},
                    )
        else:
            st.info("No drug–condition pairs to check (empty drugs or conditions).")
