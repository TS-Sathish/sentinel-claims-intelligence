from pathlib import Path
import sys
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import APP_NAME, DATASET_PATH, MODEL_DIR, RISK_REVIEW_THRESHOLD, TARGET_COLUMN
from src.data.loader import load_dataset, excluded_columns
from src.ml.pipeline import load_global_explanation, load_artifacts
from src.services.scoring import score_claim
from src.services.narrative import narrative
from src.database.store import recent_events
from src.database.store import save_event

PAGES = ["Executive Overview", "Claims Explorer", "Claim Investigation", "Model Performance",
         "Explainability", "Data Quality", "Audit History", "Settings"]
st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {--navy:#13243d;--blue:#2866a5;--ink:#1d2b40;--muted:#64748b;--line:#e3e9f1;--canvas:#f5f7fa}
html,body,[class*="css"] {font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp {background:var(--canvas)} .block-container {padding-top:1.4rem;max-width:1500px}
[data-testid="stSidebar"] {background:#122039} [data-testid="stSidebar"] * {color:#f2f6fb}
[data-testid="stSidebar"] [data-testid="stRadio"] label {font-size:.94rem}
.hero {background:linear-gradient(115deg,#122039,#24558a);padding:1.45rem 1.8rem;border-radius:15px;color:#fff;margin:0 0 1.1rem}
.hero h1 {font:800 1.75rem Manrope,sans-serif;color:#fff;margin:0}.hero p {color:#d7e4f4;margin:.35rem 0 0}
.kpi {background:#fff;padding:1rem 1.1rem;border:1px solid var(--line);border-radius:12px;min-height:102px}
.k-label {font-size:.76rem;color:var(--muted);font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.k-value {font:800 1.65rem Manrope,sans-serif;color:var(--navy);margin:.32rem 0 .18rem}.k-note {font-size:.78rem;color:var(--muted)}
.section {font:750 1.08rem Manrope,sans-serif;color:var(--navy);margin:1rem 0 .55rem}
div[data-testid="stMetric"] {background:white;padding:.85rem 1rem;border:1px solid var(--line);border-radius:12px}
.notice {background:#edf4fb;border-left:4px solid #4380b5;padding:.8rem 1rem;border-radius:5px;color:#29415f}
</style>
<div class="hero"><h1>Sentinel Claims Intelligence</h1><p>Explainable risk triage for insurance claims · evidence-led · auditable · human-reviewed</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## SENTINEL")
    st.caption("CLAIMS INTELLIGENCE")
    page = st.radio("WORKSPACE", PAGES, label_visibility="collapsed", key="workspace_navigation")
    st.divider()
    st.caption("CONFIGURED DATASET")
    st.write(DATASET_PATH.name)
    st.caption("MODEL")
    st.write("Ready" if (MODEL_DIR / "manifest.json").exists() else "Not trained")


def kpi(label, value, note):
    st.markdown(f'<div class="kpi"><div class="k-label">{label}</div><div class="k-value">{value}</div><div class="k-note">{note}</div></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading configured dataset…")
def dataset(path, stamp):
    return load_dataset(Path(path), TARGET_COLUMN)


loaded = None
load_error = None
if DATASET_PATH.exists():
    try:
        loaded = dataset(str(DATASET_PATH), DATASET_PATH.stat().st_mtime_ns)
    except Exception as exc:
        load_error = str(exc)

if load_error:
    st.error(f"Dataset could not be loaded: {load_error}")
elif loaded is None:
    st.info(f"No dataset found at `{DATASET_PATH}`. Download an authorized public CSV as described in README.md. Audit History, Model Performance, Explainability, and Settings remain available when their artifacts exist.")

X = y = raw = target = None
if loaded:
    X, y, raw, target = loaded
trained = (MODEL_DIR / "metrics.joblib").exists() and (MODEL_DIR / "manifest.json").exists()
metrics = joblib.load(MODEL_DIR / "metrics.joblib") if trained else None

if page == "Executive Overview":
    if raw is None:
        st.info("Load a public dataset to populate the overview. No example metrics are shown.")
    else:
        c = st.columns(4)
        for col, args in zip(c, [("Claims in source", f"{len(raw):,}", "Loaded local CSV"),
                                 ("Positive labels", f"{y.mean():.1%}", f"Target: {target}"),
                                 ("Usable features", f"{X.shape[1]}", "After documented exclusions"),
                                 ("Model status", "Trained" if trained else "Not trained", "Local artifacts")]):
            with col: kpi(*args)
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown('<div class="section">Label distribution</div>', unsafe_allow_html=True)
            chart = y.map({0: "No fraud label", 1: "Fraud label"}).value_counts().rename_axis("Label").reset_index(name="Claims")
            fig = px.bar(chart, x="Label", y="Claims", color="Label", text="Claims",
                         color_discrete_map={"No fraud label": "#85a9cc", "Fraud label": "#d68a79"})
            fig.update_layout(template="plotly_white", showlegend=False, margin=dict(t=15, b=10), height=330)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown('<div class="section">Claim amount / vehicle value distribution</div>', unsafe_allow_html=True)
            amount_col = next((col for col in X.columns if str(col).lower() in {"claimamount", "vehicleprice", "totalclaim"}), None)
            if amount_col:
                counts = X[amount_col].fillna("Unknown").astype(str).value_counts().head(20).rename_axis("Value band").reset_index(name="Claims")
                st.plotly_chart(px.bar(counts, x="Value band", y="Claims", template="plotly_white").update_layout(height=330, margin=dict(t=15,b=10)), use_container_width=True)
            else:
                st.caption("No recognized claim amount or vehicle value field is available in this dataset.")
        month_col = next((col for col in raw.columns if str(col).lower() in {"date", "claimdate", "accidentdate"}), None)
        if month_col:
            dates = pd.to_datetime(raw[month_col], errors="coerce")
            if dates.notna().any():
                trend = pd.DataFrame({"month": dates.dt.to_period("M").astype(str), "label": y.values}).groupby("month").agg(claims=("label", "size"), positive_rate=("label", "mean")).reset_index()
                st.markdown('<div class="section">Label trend</div>', unsafe_allow_html=True)
                st.plotly_chart(px.line(trend, x="month", y="positive_rate", markers=True, template="plotly_white", labels={"positive_rate":"Positive label share","month":"Month"}), use_container_width=True)
        st.markdown('<div class="section">Data status</div>', unsafe_allow_html=True)
        st.write(f"Dataset: `{DATASET_PATH}` · target: `{target}` · missing cells: {int(raw.isna().sum().sum()):,} · duplicate rows: {int(raw.duplicated().sum()):,}")
        if trained:
            st.markdown('<div class="section">Highest score source rows</div>', unsafe_allow_html=True)
            try:
                classifier, _, _, manifest = load_artifacts(MODEL_DIR)
                scores = classifier.predict_proba(X)[:, 1]
                top = pd.DataFrame({"Source row": X.index, "Fraud probability": scores})
                top["Review priority"] = top["Fraud probability"].map(lambda value: "High" if value >= RISK_REVIEW_THRESHOLD else ("Elevated" if value >= .35 else "Standard"))
                st.dataframe(top.sort_values("Fraud probability", ascending=False).head(12).style.format({"Fraud probability": "{:.1%}"}), use_container_width=True, hide_index=True)
                st.caption(f"Reference scoring of the configured source file with {manifest['model_version']}. These rows are from the model's training/evaluation dataset and are not an operational queue or held-out performance measure.")
            except Exception as exc:
                st.info(f"Risk queue is unavailable: {type(exc).__name__}")
        st.caption("Historical label prevalence may not represent a current portfolio or verified fraud rate.")

elif page == "Claims Explorer":
    st.markdown('<div class="section">Claims Explorer</div>', unsafe_allow_html=True)
    if raw is None:
        st.info("Load a dataset to search, filter, sort, and export claims.")
    else:
        view = raw.copy()
        q = st.text_input("Search visible fields", placeholder="Search values")
        f1, f2, f3 = st.columns(3)
        with f1:
            selected_target = st.selectbox("Label filter", ["All", "No fraud label", "Fraud label"])
        with f2:
            filter_col = st.selectbox("Optional field filter", ["None"] + [str(col) for col in raw.columns if raw[col].nunique(dropna=True) <= 40])
        with f3:
            sort_col = st.selectbox("Sort by", [str(col) for col in raw.columns])
        if selected_target != "All":
            view = view.loc[y == (1 if selected_target == "Fraud label" else 0)]
        if filter_col != "None":
            values = ["All"] + sorted(view[filter_col].dropna().astype(str).unique().tolist())
            chosen = st.selectbox(f"{filter_col} value", values)
            if chosen != "All": view = view.loc[view[filter_col].astype(str) == chosen]
        if q:
            mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False, na=False)).any(axis=1)
            view = view.loc[mask]
        ascending = st.toggle("Ascending sort", value=False)
        view = view.sort_values(sort_col, ascending=ascending, kind="stable")
        page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1)
        pages = max(1, (len(view) + page_size - 1) // page_size)
        page_idx = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
        st.caption(f"Showing {min(len(view), page_size)} of {len(view):,} filtered rows · page {page_idx} of {pages}")
        st.dataframe(view.iloc[(page_idx-1)*page_size:page_idx*page_size], use_container_width=True, hide_index=True)
        st.download_button("Export filtered results (CSV)", view.to_csv(index=False), "claims_filtered.csv", "text/csv")

elif page == "Claim Investigation":
    st.markdown('<div class="section">Claim Investigation</div>', unsafe_allow_html=True)
    if X is None:
        st.info("Load a dataset to select a claim. You can still use the API with a trained model.")
    elif not trained:
        st.warning("Model artifacts are not available. Train with `python -m src.training.train` before scoring.")
    else:
        source_idx = st.selectbox("Select source row for review", X.index.tolist(), format_func=lambda i: f"Row {i}")
        row = X.loc[[source_idx]]
        st.dataframe(X.loc[[source_idx]], use_container_width=True, hide_index=True)
        if st.button("Score selected claim", type="primary"):
            try:
                result = score_claim(row.iloc[0].to_dict())
                result["narrative"] = narrative(result, row.iloc[0].to_dict())
                save_event(f"source-row-{source_idx}", result)
                st.session_state["last_investigation"] = {"result": result, "row": int(source_idx)}
            except Exception as exc:
                st.error(f"Scoring could not be completed: {exc}")
        assessment = st.session_state.get("last_investigation")
        if assessment:
            result = assessment["result"]
            a,b,c,d = st.columns(4)
            a.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
            b.metric("Anomaly score", f"{result['anomaly_score']:.3f}")
            c.metric("Anomaly signal", "Flagged" if result["anomaly_flag"] else "Not flagged")
            d.metric("Review priority", result["risk_band"])
            st.caption(f"Review threshold: {result['threshold']:.2f} · model: {result['model_version']}")
            st.markdown("#### Explanation")
            exp = result["explanation"]
            if exp.get("available"):
                eframe = pd.DataFrame(exp["contributions"])
                eframe["feature"] = eframe["feature"].str.replace(r"^(num|cat)__", "", regex=True).str.replace("_", " ")
                st.plotly_chart(px.bar(eframe.sort_values("value"), x="value", y="feature", color="direction", orientation="h", template="plotly_white", color_discrete_map={"increases score":"#c87868","decreases score":"#6393bd"}).update_layout(height=max(260, 29*len(eframe))), use_container_width=True)
            else:
                st.info(exp.get("reason", "SHAP explanation unavailable."))
            st.markdown("#### Review note")
            st.text(result["narrative"])
            st.markdown('<div class="notice">Risk indicators support human review; they do not establish fraud or determine a claim outcome.</div>', unsafe_allow_html=True)

elif page == "Model Performance":
    st.markdown('<div class="section">Held-out Model Performance</div>', unsafe_allow_html=True)
    if not trained:
        st.info("Train models to display measured held-out performance. No placeholder metrics are shown.")
    else:
        model_choice = st.radio("Compare model", ["XGBoost", "Logistic Regression baseline"], horizontal=True)
        m = metrics["xgboost" if model_choice == "XGBoost" else "baseline"]
        cards = st.columns(5)
        for col,key in zip(cards,["precision","recall","f1","pr_auc","roc_auc"]):
            with col: kpi(key.replace("_", " "), f"{m[key]:.3f}", "Held-out split")
        left,right = st.columns(2)
        with left:
            st.markdown('<div class="section">Precision–recall curve</div>', unsafe_allow_html=True)
            curve = m["pr_curve"]
            st.plotly_chart(go.Figure(go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines", line=dict(color="#2866a5",width=3))).update_layout(template="plotly_white",xaxis_title="Recall",yaxis_title="Precision",height=350), use_container_width=True)
        with right:
            st.markdown('<div class="section">ROC curve</div>', unsafe_allow_html=True)
            curve = m["roc_curve"]
            fig = go.Figure(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", line=dict(color="#347d70",width=3)))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color="#9aa6b2",dash="dash"),showlegend=False))
            st.plotly_chart(fig.update_layout(template="plotly_white",xaxis_title="False positive rate",yaxis_title="True positive rate",height=350), use_container_width=True)
        st.markdown('<div class="section">Confusion matrix at 0.50 threshold</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=["Actual no fraud label", "Actual fraud label"], columns=["Predicted no fraud label", "Predicted fraud label"]), use_container_width=True)
        st.markdown('<div class="section">Threshold trade-offs</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(m["threshold_analysis"]), use_container_width=True, hide_index=True)
        st.caption(f"Train rows: {metrics['train_rows']:,} · held-out rows: {metrics['test_rows']:,} · held-out positive-label share: {metrics['test_fraud_rate']:.1%}. PR-AUC and precision depend on prevalence.")

elif page == "Explainability":
    st.markdown('<div class="section">Model Explainability</div>', unsafe_allow_html=True)
    explanation = load_global_explanation(MODEL_DIR)
    if not explanation.get("available"):
        st.info(explanation.get("reason", "Global SHAP values are unavailable."))
    else:
        st.caption(explanation["method"] + " · values are model contributions, not causal effects.")
        frame = pd.DataFrame(explanation["features"])
        frame["feature"] = frame["feature"].str.replace(r"^(num|cat)__", "", regex=True).str.replace("_", " ")
        st.plotly_chart(px.bar(frame.sort_values("importance"), x="importance", y="feature", orientation="h", template="plotly_white", labels={"importance":"Mean absolute SHAP value","feature":"Transformed feature"}).update_layout(height=max(380, len(frame)*27)), use_container_width=True)
        st.caption("Local SHAP values for an investigated row are available on the Claim Investigation page.")

elif page == "Data Quality":
    st.markdown('<div class="section">Data Quality</div>', unsafe_allow_html=True)
    if raw is None:
        st.info("Load a dataset to show missingness, duplicates, class balance, and feature profile.")
    else:
        a,b,c,d = st.columns(4)
        for col,args in zip([a,b,c,d],[("Rows",f"{len(raw):,}","Source file"),("Columns",str(raw.shape[1]),"Including target"),("Missing cells",f"{raw.isna().sum().sum():,}","Source file"),("Duplicate rows",f"{raw.duplicated().sum():,}","Source file")]):
            with col: kpi(*args)
        profile = pd.DataFrame({"Column":raw.columns.astype(str),"Type":raw.dtypes.astype(str).values,"Missing":raw.isna().sum().values,"Missing %":(raw.isna().mean().values*100).round(2),"Unique":raw.nunique(dropna=True).values})
        st.dataframe(profile.sort_values("Missing",ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Export data quality report", profile.to_csv(index=False), "data_quality_report.csv", "text/csv")
        st.markdown("#### Label balance")
        st.write({"target":target,"counts":y.value_counts().sort_index().to_dict(),"positive_share":float(y.mean())})
        st.markdown("#### Excluded and leakage-risk columns")
        exclusions = excluded_columns(raw.columns, target)
        st.dataframe(pd.DataFrame([{"Column":name,"Reason":reason} for name,reason in exclusions.items()]), use_container_width=True, hide_index=True)
        st.caption("See docs/DATA_DICTIONARY.md for the full field notes and source limitations.")

elif page == "Audit History":
    st.markdown('<div class="section">Scoring Audit History</div>', unsafe_allow_html=True)
    try:
        events = recent_events(200)
        if not events:
            st.info("No scoring events have been recorded in the local audit database yet.")
        else:
            frame = pd.DataFrame(events).drop(columns=["result_json","explanation_json"], errors="ignore")
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.download_button("Export audit history (CSV)", frame.to_csv(index=False), "audit_history.csv", "text/csv")
    except Exception:
        st.info("Audit history is not available yet. The local database will be initialized when a score is saved.")

else:
    st.markdown('<div class="section">Settings & Status</div>', unsafe_allow_html=True)
    st.text_input("Dataset path", str(DATASET_PATH), disabled=True)
    st.text_input("Detected target", target or "Not detected", disabled=True)
    st.number_input("Review threshold", min_value=0.0, max_value=1.0, value=float(RISK_REVIEW_THRESHOLD), disabled=True, help="Configured via RISK_REVIEW_THRESHOLD; informational setting.")
    st.write("Model status:", "Trained" if trained else "Not trained")
    if trained:
        try:
            manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
            st.write("Model version:", manifest["model_version"])
            st.write("Trained at (UTC):", manifest["trained_at_utc"])
            st.write("Dataset SHA-256:", manifest["dataset_sha256"])
            st.write("Target:", manifest.get("target"))
        except Exception:
            st.warning("Model manifest could not be read. Retrain the model.")
    st.write("Narrative provider:", "OpenAI configured" if __import__("os").getenv("LLM_PROVIDER", "local").lower()=="openai" and __import__("os").getenv("OPENAI_API_KEY") else "Local fallback")
    st.caption("Provider status does not expose secrets. Configure settings in your local .env file and restart the app.")
    st.markdown("See [Security and privacy](../docs/SECURITY_AND_PRIVACY.md) before real claim use.")
