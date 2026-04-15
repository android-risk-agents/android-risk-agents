import base64
import os
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

# Page config
st.set_page_config(page_title="Android Risk Executive Dashboard", layout="wide")

def svg_to_b64_img(path: str, height: int = 56) -> str:
    """Read an SVG file and return an <img> tag with base64-encoded src."""
    try:
        svg_bytes = Path(path).read_bytes()
        b64 = base64.b64encode(svg_bytes).decode("utf-8")
        return (
            f'<img src="data:image/svg+xml;base64,{b64}" '
            f'style="height:{height}px; object-fit:contain;" />'
        )
    except Exception:
        return ""

# ── Header with logos ─────────────────────────────────────────────────────────
ASSETS_DIR = Path(__file__).parent / "assets"
uic_img_tag        = svg_to_b64_img(ASSETS_DIR / "UIC Logo.SVG",         height=64)
transunion_img_tag = svg_to_b64_img(ASSETS_DIR / "transunion logo.svg",  height=38)

col_logo_left, col_title, col_logo_right = st.columns([1, 4, 1])

with col_logo_left:
    if uic_img_tag:
        st.markdown(
            f'<div style="display:flex;align-items:center;height:80px;">{uic_img_tag}</div>',
            unsafe_allow_html=True,
        )

with col_title:
    st.markdown(
        """
        <div style="text-align:center; padding:8px 0 4px 0;">
            <h2 style="margin:0; font-size:1.45rem; font-weight:700; line-height:1.25;">
                Android Risk Agents – Executive Risk Intelligence Dashboard
            </h2>
            <p style="margin:4px 0 0 0; color:#888; font-size:0.88rem;">
                Strategic Risk Posture &nbsp;·&nbsp; Exposure Concentration
                &nbsp;·&nbsp; Change Intelligence &nbsp;·&nbsp; Actionability
            </p>
            <p style="margin:4px 0 0 0; color:#aaa; font-size:0.78rem;">
                UIC Liautaud Graduate School of Business &nbsp;|&nbsp;
                IDS 560 &nbsp;|&nbsp; Group 2 &nbsp;|&nbsp;
                Industry Partner: TransUnion
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_logo_right:
    if transunion_img_tag:
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:flex-end;height:80px;">'
            f'{transunion_img_tag}</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Env / Supabase ────────────────────────────────────────────────────────────
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

INSIGHTS_TABLE = os.getenv("INSIGHTS_TABLE", "insights")
DEFAULT_FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "500"))

# ── RAG / embedding env ───────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
VECTOR_RPC_MATCH = os.getenv("VECTOR_RPC_MATCH", "match_vector_chunks")
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "vector_chunks")
FINGERPRINT_VECTOR_RPC = os.getenv("FINGERPRINT_VECTOR_RPC", "match_fingerprint_library_chunks")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_STRUCTURED_TOP_K = int(os.getenv("RAG_STRUCTURED_TOP_K", "3"))
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.30"))
FINGERPRINT_ENABLED = os.getenv("FINGERPRINT_ENABLED", "true").lower() == "true"


@st.cache_resource
def get_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE key in .env file")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ── General helpers ───────────────────────────────────────────────────────────

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def risk_band(score: int) -> str:
    if score >= 4:
        return "High"
    if score == 3:
        return "Medium"
    return "Low"


def stable_key(row: pd.Series) -> str:
    return f"{row.get('component', '')}//{row.get('kind', '')}//{row.get('title', '')}"


def df_to_csv_bytes(d: pd.DataFrame) -> bytes:
    return d.to_csv(index=False).encode("utf-8")


def badge_delta(new_val, old_val, fmt="{:.2f}"):
    if old_val is None or pd.isna(old_val):
        return ""
    try:
        dv = float(new_val) - float(old_val)
    except Exception:
        return ""
    if dv > 0:
        return f" 🔺{fmt.format(dv)}"
    if dv < 0:
        return f" 🔻{fmt.format(abs(dv))}"
    return " —"


def normalize_actions(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            if isinstance(v, list):
                out.extend([f"{k}: {vv}" for vv in v])
            else:
                out.append(f"{k}: {v}")
        return out
    return [str(x)]


def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def normalize_risk_score(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        val = float(x)
    except Exception:
        return np.nan
    if val <= 1.0:
        val = round(val * 5)
        val = max(1, min(5, int(val)))
        return val
    return int(max(1, min(5, round(val))))


def normalize_confidence(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        val = float(x)
    except Exception:
        return np.nan
    if val > 1.0:
        val = val / 100.0
    return round(max(0.0, min(1.0, val)), 2)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "url": "",
        "title": "",
        "summary": "",
        "risk_score": np.nan,
        "confidence": np.nan,
        "created_at": pd.NaT,
        "component": "Unknown",
        "kind": "unknown",
        "category": "unknown",
        "recommended_actions": None,
        "status": "New",
        "owner": "",
        "due_date": "",
        "notes": "",
        "snapshot_id": "snapshot_unknown",
        "source_id": "",
        "source_name": "Unknown",
    }
    out = df.copy()
    for col, default in needed.items():
        if col not in out.columns:
            out[col] = default
    return out


# ── Supabase data fetch ───────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_insights_from_supabase(limit: int = DEFAULT_FETCH_LIMIT) -> pd.DataFrame:
    sb = get_supabase()

    insights_rows = (
        sb.table("insights")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )
    df_insights = pd.DataFrame(insights_rows or [])

    source_rows = (
        sb.table("sources")
        .select("id, url, name")
        .execute()
        .data
    )
    df_sources = pd.DataFrame(source_rows or [])

    if not df_sources.empty and "source_id" in df_insights.columns:
        src_map = df_sources.rename(columns={"url": "source_url"})
        if "name" in src_map.columns:
            src_map = src_map.rename(columns={"name": "source_name"})

        df_insights = df_insights.merge(
            src_map,
            left_on="source_id",
            right_on="id",
            how="left",
        )

        if "source_name" not in df_insights.columns:
            df_insights["source_name"] = (
                df_insights.get("source_url", df_insights["source_id"].astype(str))
            )

    rec_rows = (
        sb.table("recommendations")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )
    df_recs = pd.DataFrame(rec_rows or [])

    if not df_recs.empty:
        df_recs = df_recs.rename(columns={"final_risk_score": "risk_score"})
        df_recs["title"] = df_recs.get("title", "Recommendation")
        df_recs["summary"] = df_recs.get("recommendation_text", "")
        df_recs["component"] = df_recs.get("source_id", "Recommendation Engine")
        df_recs["kind"] = df_recs.get("category", "recommendation")
        df_recs["source_name"] = df_recs.get("source_name", "Recommendation Engine")
        df_recs["confidence"] = df_recs["confidence"].apply(normalize_confidence)
        df_recs["risk_score"] = df_recs["risk_score"].apply(normalize_risk_score)
        df_recs["created_at"] = pd.to_datetime(df_recs["created_at"], errors="coerce")

    if not df_insights.empty:
        df_insights = ensure_columns(df_insights)
        df_insights["confidence"] = df_insights["confidence"].apply(normalize_confidence)
        df_insights["risk_score"] = df_insights["risk_score"].apply(normalize_risk_score)
        df_insights["created_at"] = pd.to_datetime(df_insights["created_at"], errors="coerce")

    df_combined = pd.concat([df_insights, df_recs], ignore_index=True)
    df_combined = df_combined.dropna(subset=["created_at", "risk_score", "confidence"])

    if "source_name" not in df_combined.columns:
        df_combined["source_name"] = df_combined.get("source_id", "Unknown").fillna("Unknown")
    else:
        df_combined["source_name"] = df_combined["source_name"].fillna("Unknown")

    return df_combined


def cluster_titles(d: pd.DataFrame, threshold=0.86):
    titles = d["title"].fillna("").tolist()
    cluster_id = [-1] * len(titles)
    cid = 0
    for i in range(len(titles)):
        if cluster_id[i] != -1:
            continue
        cluster_id[i] = cid
        for j in range(i + 1, len(titles)):
            if cluster_id[j] == -1 and sim(titles[i], titles[j]) >= threshold:
                cluster_id[j] = cid
        cid += 1
    out = d.copy()
    out["cluster_id"] = cluster_id
    return out


def compute_diff(df_old: pd.DataFrame, df_new: pd.DataFrame):
    old_map = {stable_key(r): r for _, r in df_old.iterrows()}
    new_map = {stable_key(r): r for _, r in df_new.iterrows()}

    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    common_keys = old_keys & new_keys

    added = pd.DataFrame([new_map[k] for k in added_keys]) if added_keys else pd.DataFrame(columns=df_new.columns)
    removed = pd.DataFrame([old_map[k] for k in removed_keys]) if removed_keys else pd.DataFrame(columns=df_old.columns)

    changed_rows = []
    persisting_rows = []
    for k in common_keys:
        o = old_map[k]
        n = new_map[k]
        changed = False
        for col in ["risk_score", "confidence", "summary"]:
            if str(o.get(col)) != str(n.get(col)):
                changed = True
                break
        if changed:
            row = n.copy()
            row["old_risk_score"] = o.get("risk_score")
            row["old_confidence"] = o.get("confidence")
            row["old_summary"] = o.get("summary")
            row["delta_risk"] = float(n.get("risk_score") or 0) - float(o.get("risk_score") or 0)
            row["delta_conf"] = float(n.get("confidence") or 0) - float(o.get("confidence") or 0)
            changed_rows.append(row)
        else:
            persisting_rows.append(n)

    changed_df = pd.DataFrame(changed_rows) if changed_rows else pd.DataFrame(
        columns=list(df_new.columns) + ["old_risk_score", "old_confidence", "old_summary", "delta_risk", "delta_conf"]
    )
    persisting_df = pd.DataFrame(persisting_rows) if persisting_rows else pd.DataFrame(columns=df_new.columns)
    return added, removed, changed_df, persisting_df


# ── Sidebar: data source label ────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Data source")
    st.caption("Supabase")

try:
    all_df = fetch_insights_from_supabase(DEFAULT_FETCH_LIMIT)
    data_mode = "Supabase"
    if all_df.empty:
        raise RuntimeError(f"No rows found in table '{INSIGHTS_TABLE}'.")
except Exception as e:
    st.error(f"Failed to load data from Supabase: {e}")
    st.stop()

st.caption(f"Data source: **{data_mode}**")

if "triage" not in st.session_state:
    st.session_state.triage = {}


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filters")

    source_names = sorted(
        all_df["source_name"].fillna("Unknown").astype(str).unique().tolist()
    )
    if not source_names:
        source_names = ["Unknown"]
        all_df["source_name"] = "Unknown"

    default_source = source_names[0]
    selected_source = st.selectbox(
        "Source",
        source_names,
        index=source_names.index(default_source),
    )

    source_df = all_df[
        all_df["source_name"].fillna("Unknown").astype(str) == str(selected_source)
    ].copy()

    snapshots = sorted(
        [s for s in source_df["snapshot_id"].dropna().astype(str).unique().tolist() if s]
    )
    if not snapshots:
        snapshots = ["snapshot_unknown"]
        source_df["snapshot_id"] = "snapshot_unknown"

    default_new = snapshots[-1]
    default_old = snapshots[-2] if len(snapshots) >= 2 else snapshots[0]

    enable_compare = st.toggle("Compare snapshots", value=(len(snapshots) >= 2))

    if enable_compare and len(snapshots) >= 2:
        snap_new = st.selectbox("New snapshot", snapshots, index=snapshots.index(default_new))
        snap_old = st.selectbox("Old snapshot", snapshots, index=snapshots.index(default_old))
    else:
        snap_new = st.selectbox("Snapshot", snapshots, index=snapshots.index(default_new))
        snap_old = None

    st.divider()

    max_dt = source_df["created_at"].max() if not source_df.empty else all_df["created_at"].max()
    days = st.number_input("Recent days", min_value=1, max_value=365, value=14, step=1)
    start_dt = max_dt - timedelta(days=int(days)) if pd.notna(max_dt) else datetime.utcnow() - timedelta(days=14)

    min_risk = st.slider("Min risk score", min_value=1, max_value=5, value=3, step=1)
    min_conf = st.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    comps = sorted(source_df["component"].fillna("Unknown").astype(str).unique().tolist()) if not source_df.empty else []
    kinds = sorted(source_df["kind"].fillna("unknown").astype(str).unique().tolist()) if not source_df.empty else []
    sel_comps = st.multiselect("Component", comps, default=comps)
    sel_kinds = st.multiselect("Kind", kinds, default=kinds)

    q = st.text_input("Search (title/summary)", value="")
    high_only = st.toggle("High only (risk ≥4)", value=False)
    sort_mode = st.selectbox("Sort", ["Most recent", "Highest risk", "Highest confidence"], index=0)

    st.divider()
    dedup = st.toggle("Deduplicate / cluster similar titles", value=False)
    cluster_threshold = st.slider("Cluster threshold", 0.70, 0.95, 0.86, 0.01, disabled=not dedup)

    st.divider()
    row_limit = st.slider("Max rows in lists", 10, 200, 50, 10)

    st.markdown("---")
    st.markdown("### 👥 Project Team")
    st.markdown("""
• **Ishdeep Singh**  
🔗 [LinkedIn](https://www.linkedin.com/in/ishdeepsgh/)

• **Ishan Singh**  
🔗 [LinkedIn](https://www.linkedin.com/in/ishansingh98/)

• **Kiran Kumar**  
🔗 [LinkedIn](https://www.linkedin.com/in/kiran-kumar-srinivasa-37b57017b/)

• **Rameen Shakeel**  
🔗 [LinkedIn](https://www.linkedin.com/in/rameen-shakeel-16442a221/)

• **Seungmin Pack**  
🔗 [LinkedIn](https://www.linkedin.com/in/seungminpack/)

• **Hadiqa Shah**  
🔗 [LinkedIn](https://www.linkedin.com/in/hadiqa-shah/)
    """)


# ── Filter dataframe ──────────────────────────────────────────────────────────
base_df = all_df[
    all_df["source_name"].fillna("Unknown").astype(str) == str(selected_source)
].copy()

df_new = base_df[base_df["snapshot_id"].astype(str) == str(snap_new)].copy()
df_old = base_df[base_df["snapshot_id"].astype(str) == str(snap_old)].copy() if snap_old else None

df = df_new.copy()
df = df[df["created_at"] >= pd.to_datetime(start_dt)]
df = df[df["risk_score"] >= min_risk]
df = df[df["confidence"] >= min_conf]
if sel_comps:
    df = df[df["component"].isin(sel_comps)]
if sel_kinds:
    df = df[df["kind"].isin(sel_kinds)]
if q.strip():
    qq = q.strip().lower()
    df = df[df["title"].astype(str).str.lower().str.contains(qq) | df["summary"].astype(str).str.lower().str.contains(qq)]
if high_only:
    df = df[df["risk_score"] >= 4]

if dedup and len(df) > 0:
    df = cluster_titles(df, threshold=float(cluster_threshold))
    df = df.sort_values(["risk_score", "confidence", "created_at"], ascending=[False, False, False])
    df = df.drop_duplicates(subset=["cluster_id"], keep="first")

if sort_mode == "Highest risk":
    df = df.sort_values(["risk_score", "confidence", "created_at"], ascending=[False, False, False])
elif sort_mode == "Highest confidence":
    df = df.sort_values(["confidence", "risk_score", "created_at"], ascending=[False, False, False])
else:
    df = df.sort_values(["created_at"], ascending=[False])

df = df.head(int(row_limit)).copy()

diff_added = None
diff_removed = None
diff_changed = None
diff_persisting = None
if enable_compare and df_old is not None and not df_old.empty:
    diff_added, diff_removed, diff_changed, diff_persisting = compute_diff(df_old, df_new)


def apply_triage(row):
    k = stable_key(row)
    t = st.session_state.triage.get(k, {})
    for col in ["owner", "status", "due_date", "notes"]:
        if col in t:
            row[col] = t[col]
    return row


if len(df) > 0:
    df = df.apply(apply_triage, axis=1)


# ── KPIs ──────────────────────────────────────────────────────────────────────
total_insights = len(df)
avg_risk = round(df["risk_score"].mean(), 2) if total_insights else 0
max_risk = int(df["risk_score"].max()) if total_insights else 0
high_risk_count = int((df["risk_score"] >= 4).sum()) if total_insights else 0
risk_variance = round(float(df["risk_score"].var()), 2) if total_insights > 1 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Insights (filtered)", total_insights)
c2.metric("Average Risk Score", avg_risk)
c3.metric("Maximum Risk Score", max_risk)
c4.metric("High Risk Items (≥4)", high_risk_count)
c5.metric("Risk Variance Index", risk_variance)

st.markdown("---")

high_risk_pct = round((high_risk_count / total_insights) * 100, 1) if total_insights else 0.0
if high_risk_pct > 30:
    risk_status = "⚠️ Elevated Risk Exposure"
elif high_risk_pct > 15:
    risk_status = "🟡 Moderate Risk Exposure"
else:
    risk_status = "🟢 Controlled Risk Exposure"

diff_added = None
diff_removed = None
diff_changed = None
diff_persisting = None


# ── Executive Risk Overview ───────────────────────────────────────────────────
df_exec = all_df.copy()
df_exec = df_exec[df_exec["risk_score"] > 0]
df_exec["risk_score"] = pd.to_numeric(df_exec["risk_score"], errors="coerce")
df_exec["confidence"] = pd.to_numeric(df_exec["confidence"], errors="coerce")
df_exec["weighted_risk"] = df_exec["risk_score"] * df_exec["confidence"]
df_exec = df_exec.drop_duplicates(subset=["title"])

top_items = (
    df_exec
    .groupby("title", as_index=False)
    .agg({"risk_score": "max", "confidence": "max", "weighted_risk": "max", "category": "first"})
    .sort_values("weighted_risk", ascending=False)
    .head(3)
)

st.markdown("## 🔎 Executive Risk Overview")

total_high = len(df_exec[df_exec["risk_score"] >= 4])

dominant_domain = (
    df_exec.groupby("category")["weighted_risk"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

st.markdown(f"""
• **{total_high} high-risk issues** currently active across the ecosystem  
• Risk exposure is **primarily concentrated in the {dominant_domain} domain**  
• Top risk drivers contributing to weighted exposure:
""")

for _, row in top_items.iterrows():
    st.markdown(
        f"    • **[HIGH] {row['title']}** "
        f"(Risk {row['risk_score']}, Confidence {round(row['confidence'], 2)})"
    )

st.markdown("• Immediate prioritization recommended for high-risk + high-confidence items to mitigate validated exposure")


# ── Charts ────────────────────────────────────────────────────────────────────
st.header("Executive Intelligence Overview")

st.subheader("1️⃣ Confidence-Weighted Risk Exposure")
weighted_total = df_exec["weighted_risk"].sum()
avg_weighted = df_exec["weighted_risk"].mean()
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Weighted Risk Exposure", round(weighted_total, 2))
with col2:
    st.metric("Average Weighted Risk per Insight", round(avg_weighted, 2))

fig_weighted = px.bar(
    df_exec.sort_values("weighted_risk", ascending=False),
    x="title", y="weighted_risk", color="category",
    title="Risk Exposure per Insight (Risk × Confidence)",
)
st.plotly_chart(fig_weighted, use_container_width=True)

st.subheader("2️⃣ Risk–Confidence Priority Quadrant")
median_risk = df_exec["risk_score"].median()
median_conf = df_exec["confidence"].median()
fig_quad = go.Figure()
fig_quad.add_trace(
    go.Scatter(
        x=df_exec["confidence"],
        y=df_exec["risk_score"],
        mode="markers",
        marker=dict(
            size=df_exec["weighted_risk"] * 4,
            color=df_exec["weighted_risk"],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Weighted Risk"),
            opacity=0.85,
            line=dict(width=1, color="white"),
        ),
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "Risk: %{y}<br>"
            "Confidence: %{x}<br>"
            "Weighted Risk: %{marker.color:.2f}<extra></extra>"
        ),
        customdata=df_exec["title"],
    )
)
fig_quad.add_vline(x=median_conf, line_dash="dash", line_color="gray")
fig_quad.add_hline(y=median_risk, line_dash="dash", line_color="gray")
fig_quad.update_layout(title="High Impact Zone = Top Right Quadrant", xaxis_title="Confidence", yaxis_title="Risk Score", height=550)
fig_quad.update_yaxes(range=[0.5, 5.5])
fig_quad.update_xaxes(range=[0.6, 1.02])
st.plotly_chart(fig_quad, use_container_width=True)

st.subheader("3️⃣ Cumulative Risk Exposure Curve")
sorted_df = df_exec.sort_values("weighted_risk", ascending=False).reset_index(drop=True)
sorted_df["cumulative_exposure"] = sorted_df["weighted_risk"].cumsum()
fig_curve = px.line(
    sorted_df, x=sorted_df.index + 1, y="cumulative_exposure",
    markers=True, title="How Quickly Risk Concentrates Across Insights",
)
fig_curve.update_layout(xaxis_title="Top N Insights", yaxis_title="Cumulative Weighted Risk")
st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("4️⃣ Risk Concentration by Category")
category_risk = (
    df_exec.groupby("category")["weighted_risk"]
    .sum().reset_index().sort_values("weighted_risk", ascending=False)
)
fig_cat = px.bar(category_risk, x="category", y="weighted_risk", title="Total Weighted Risk by Category")
st.plotly_chart(fig_cat, use_container_width=True)


# ── Triage queue + detail panel ───────────────────────────────────────────────
st.subheader("🧰 Action Queue (Top items to triage today)")
sel_row = None
if total_insights:
    queue = df.copy().sort_values(["risk_score", "confidence", "created_at"], ascending=[False, False, False]).head(10)
    st.caption("Click an item below to see details and assign actions/owner/status.")

    labels = []
    for _, r in queue.iterrows():
        labels.append(f"[{risk_band(int(r['risk_score']))}] {r['title']} • {r['component']} • conf {r['confidence']}")

    selected = st.radio("Top 10", labels, label_visibility="collapsed")
    sel_idx = labels.index(selected)
    sel_row = queue.iloc[sel_idx]
else:
    st.info("No items available under the current filters.")

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    if sel_row is None:
        st.markdown("### No selected item")
    else:
        st.markdown(f"### {sel_row['title']}")
    st.write(sel_row.get("summary", ""))

    if sel_row.get("url"):
        st.markdown(
            f"""
            <div style="margin-top:8px; margin-bottom:16px;">
                <a href="{sel_row['url']}" target="_blank"
                   style="font-weight:600;font-size:15px;color:#1f6feb;text-decoration:none;">
                   View Official Release ↗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    a1, a2, a3 = st.columns(3)
    a1.metric("Risk", int(sel_row["risk_score"]))
    a2.metric("Confidence", float(sel_row["confidence"]))
    a3.metric("Band", risk_band(int(sel_row["risk_score"])))

    st.markdown("#### Recommended actions")
    for act in (sel_row.get("recommended_actions") or []):
        st.checkbox(str(act), value=False)

    if enable_compare and diff_changed is not None and len(diff_changed):
        key = stable_key(sel_row)
        changed_hit = diff_changed.apply(lambda r: stable_key(r) == key, axis=1)
        if changed_hit.any():
            rowc = diff_changed[changed_hit].iloc[0]
            st.markdown("#### What changed since previous snapshot")
            st.write(
                f"- Risk: {rowc.get('old_risk_score')} → {rowc.get('risk_score')}"
                f"{badge_delta(rowc.get('risk_score'), rowc.get('old_risk_score'), fmt='{:.0f}')}"
            )
            st.write(
                f"- Confidence: {rowc.get('old_confidence')} → {rowc.get('confidence')}"
                f"{badge_delta(rowc.get('confidence'), rowc.get('old_confidence'), fmt='{:.2f}')}"
            )
            if str(rowc.get("old_summary")) != str(rowc.get("summary")):
                st.write("- Summary updated")

    with st.expander("Raw record"):
        st.json(sel_row.to_dict())

with right:
    st.markdown("### Triage")
    k = stable_key(sel_row)
    existing = st.session_state.triage.get(k, {})
    owner = st.text_input("Owner", value=existing.get("owner", sel_row.get("owner", "")), key=f"owner_{k}")
    status_options = ["New", "In review", "Mitigated", "Closed", "Won't fix"]
    existing_status = existing.get("status", sel_row.get("status", "New"))
    status = st.selectbox(
        "Status", status_options,
        index=status_options.index(existing_status) if existing_status in status_options else 0,
        key=f"status_{k}",
    )
    due_date = st.text_input("Due date (YYYY-MM-DD)", value=existing.get("due_date", sel_row.get("due_date", "")), key=f"due_{k}")
    notes = st.text_area("Notes", value=existing.get("notes", sel_row.get("notes", "")), height=140, key=f"notes_{k}")

    if st.button("Save triage", type="primary"):
        st.session_state.triage[k] = {"owner": owner, "status": status, "due_date": due_date, "notes": notes}
        st.success("Saved.")

    st.markdown("---")
    st.markdown("### Export")
    st.download_button("Download current filtered CSV", data=df_to_csv_bytes(df), file_name="filtered_insights.csv", mime="text/csv")


# ── High-risk table ───────────────────────────────────────────────────────────
st.subheader("📌 High-Risk Insight Breakdown (filtered)")
top_risks = df[df["risk_score"] >= 4].copy().sort_values(
    by=["risk_score", "confidence", "created_at"], ascending=[False, False, False]
)
if len(top_risks) == 0:
    st.info("No high-risk items under current filters.")
else:
    display_cols = [
        "title", "component", "kind", "source_name",
        "risk_score", "confidence", "created_at", "status", "owner", "due_date"
    ]
    st.dataframe(
        top_risks[[c for c in display_cols if c in top_risks.columns]],
        use_container_width=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# RAG CHATBOT  –  Supabase-only, fingerprint-first routing
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## Ask the assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Embedding ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def embed_query(text: str):
    model = get_embedder()
    try:
        vec = model.encode([text], normalize_embeddings=True, prompt_name="query")[0]
    except TypeError:
        try:
            vec = model.encode([text], normalize_embeddings=True)[0]
        except Exception:
            vec = model.encode([f"search_query: {text}"], normalize_embeddings=True)[0]
    except Exception:
        vec = model.encode([f"search_query: {text}"], normalize_embeddings=True)[0]
    return vec.tolist()


# ── Small utilities ───────────────────────────────────────────────────────────
def to_iso_str(x):
    try:
        ts = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts):
            return ""
        return str(ts)
    except Exception:
        return str(x or "")


def safe_similarity(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def truncate_at_sentence(text: str, max_chars: int = 500) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in [". ", "! ", "? "]:
        last = truncated.rfind(sep)
        if last != -1:
            return truncated[:last + 1]
    last_space = truncated.rfind(" ")
    if last_space != -1:
        return truncated[:last_space] + "."
    return truncated + "."


def build_structured_chunk_text(row: dict) -> str:
    fields = [
        f"Title: {row.get('title', '')}",
        f"Summary: {row.get('summary', '')}",
        f"Recommendation: {row.get('recommendation_text', '')}",
        f"Rationale: {row.get('rationale', '')}",
        f"Risk Score: {row.get('risk_score', row.get('final_risk_score', ''))}",
        f"Confidence: {row.get('confidence', '')}",
        f"Priority: {row.get('priority', '')}",
        f"Category: {row.get('category', '')}",
        f"Component: {row.get('component', row.get('source_id', ''))}",
        f"Kind: {row.get('kind', '')}",
        f"Created At: {row.get('created_at', '')}",
    ]
    actions = normalize_actions(row.get("recommended_actions"))
    if actions:
        fields.append("Recommended Actions: " + " | ".join(actions))
    return "\n".join(fields)


# ── normalize_hit  (enriched fingerprint branch) ─────────────────────────────
def normalize_hit(source_type: str, row: dict, similarity=None) -> dict:
    if source_type == "fingerprint_library_chunks":
        # Prefer file-level identifiers for the display title
        title = (
            row.get("file_name")
            or row.get("file_path")
            or row.get("chunk_title")
            or "Fingerprint Evidence"
        )
        # Build a rich, file-oriented chunk text from all RPC-returned fields
        parts = []
        for label, key in [
            ("File",        "file_name"),
            ("Path",        "file_path"),
            ("Module",      "module_name"),
            ("Repo",        "repo_name"),
            ("Category",    "category"),
            ("Chunk Title", "chunk_title"),
            ("Summary",     "chunk_summary"),
            ("Code/Text",   "chunk_text"),
        ]:
            val = row.get(key, "")
            if val:
                parts.append(f"{label}: {val}")
        chunk_text = "\n".join(parts) if parts else "No content available."

    elif source_type == "vector_chunks":
        raw_chunk = row.get("chunk_text", "") or ""
        title = raw_chunk[:80].replace("\n", " ").strip() or "Knowledge Chunk"
        chunk_text = raw_chunk

    else:
        title = (
            row.get("title")
            or row.get("file_name")
            or row.get("name")
            or source_type
        )
        chunk_text = (
            row.get("chunk_text")
            or row.get("chunk_summary")
            or row.get("summary")
            or row.get("recommendation_text")
            or row.get("content")
            or row.get("text")
            or build_structured_chunk_text(row)
        )

    risk_score = normalize_risk_score(row.get("risk_score", row.get("final_risk_score")))
    if isinstance(risk_score, float) and pd.isna(risk_score):
        risk_score = 0

    confidence = normalize_confidence(row.get("confidence"))
    if isinstance(confidence, float) and pd.isna(confidence):
        confidence = 0.0

    raw_created_at = row.get("created_at")
    created_at_str = to_iso_str(raw_created_at) if raw_created_at else ""

    source_id = row.get("source_id") or row.get("file_id") or row.get("id") or ""

    return {
        "source_type":    source_type,
        "title":          title,
        "chunk_text":     str(chunk_text or ""),
        "score":          safe_similarity(
            similarity if similarity is not None
            else row.get("similarity", row.get("score", 0.0))
        ),
        "risk_score":     safe_int(risk_score, default=0),
        "confidence":     float(confidence) if confidence else 0.0,
        "source_id":      str(source_id),
        "snapshot_id":    row.get("snapshot_id", ""),
        "kind":           row.get("kind") or row.get("category") or source_type,
        "component": (
            row.get("component")
            or row.get("module_name")
            or row.get("repo_name")
            or row.get("source_id")
            or row.get("file_name")
            or ""
        ),
        "category":       row.get("category") or source_type,
        "created_at":     created_at_str,
        "has_created_at": bool(created_at_str),
        "metadata":       row,
    }


# ── Low-level RPC helper ──────────────────────────────────────────────────────
def vector_rpc_call(rpc_name: str, query_embedding: list, match_count: int):
    sb = get_supabase()
    payload_options = [
        {"query_embedding": query_embedding, "match_count": match_count},
        {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter_source_id": None,
            "filter_kind": None,
        },
    ]
    last_err = None
    for payload in payload_options:
        try:
            resp = sb.rpc(rpc_name, payload).execute()
            data = getattr(resp, "data", None) or []
            if isinstance(data, list):
                return data
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    return []


# ── Individual retrievers ─────────────────────────────────────────────────────
def retrieve_vector_chunks(question: str, top_k: int = 5) -> list:
    query_embedding = embed_query(question)
    try:
        rows = vector_rpc_call(VECTOR_RPC_MATCH, query_embedding, top_k)
    except Exception as e:
        st.warning(f"Vector RPC failed: {e}")
        return []
    hits = []
    for r in rows:
        sim_score = safe_similarity(r.get("similarity", r.get("score", 0.0)))
        if sim_score < MIN_SIMILARITY_THRESHOLD:
            continue
        hits.append(normalize_hit("vector_chunks", r, similarity=sim_score))
    return hits


def retrieve_fingerprint_chunks(question: str, top_k: int = 4) -> list:
    if not FINGERPRINT_ENABLED:
        return []
    query_embedding = embed_query(question)
    try:
        rows = vector_rpc_call(FINGERPRINT_VECTOR_RPC, query_embedding, top_k)
    except Exception as e:
        st.warning(f"Fingerprint RPC failed: {e}")
        return []
    hits = []
    for r in rows:
        sim_score = safe_similarity(r.get("similarity", r.get("score", 0.0)))
        if sim_score < MIN_SIMILARITY_THRESHOLD:
            continue
        hits.append(normalize_hit("fingerprint_library_chunks", r, similarity=sim_score))
    return hits


def retrieve_structured_supabase(question: str, top_k: int = 3) -> list:
    q = question.lower().strip()
    sb = get_supabase()
    results = []

    table_specs = [
        ("insights",        "created_at"),
        ("recommendations", "created_at"),
        ("changes",         "created_at"),
        ("snapshots",       "fetched_at"),
    ]

    for table_name, order_col in table_specs:
        try:
            rows = (
                sb.table(table_name)
                .select("*")
                .order(order_col, desc=True)
                .limit(max(top_k * 3, 10))
                .execute()
                .data
            ) or []
        except Exception:
            rows = []

        scored = []
        for r in rows:
            blob = " ".join([
                str(r.get("title", "")),
                str(r.get("summary", "")),
                str(r.get("recommendation_text", "")),
                str(r.get("rationale", "")),
                str(r.get("chunk_text", "")),
                str(r.get("content", "")),
                str(r.get("text", "")),
                str(r.get("category", "")),
                str(r.get("kind", "")),
                str(r.get("source_id", "")),
                str(r.get("file_name", "")),
            ]).lower()

            overlap = sum(1 for token in q.split() if len(token) > 2 and token in blob)
            risk = safe_int(normalize_risk_score(r.get("risk_score", r.get("final_risk_score"))), default=0)

            recency_boost = 0.0
            raw_ts = r.get("created_at") or r.get("fetched_at")
            if raw_ts:
                try:
                    created_at = pd.to_datetime(raw_ts, errors="coerce")
                    if pd.notna(created_at):
                        now = pd.Timestamp.utcnow()
                        if getattr(created_at, "tzinfo", None) is not None:
                            now = now.tz_localize("UTC") if now.tzinfo is None else now
                        else:
                            created_at = created_at.tz_localize(None)
                            now = now.tz_localize(None)
                        age_days = max((now - created_at).days, 0)
                        recency_boost = max(0.0, 1.0 - min(age_days / 30.0, 1.0))
                except Exception:
                    recency_boost = 0.0

            final_score = overlap + (0.25 * risk) + (0.20 * recency_boost)
            scored.append((final_score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        for final_score, row in scored[:top_k]:
            if final_score <= 0:
                continue
            results.append(
                normalize_hit(table_name, row, similarity=min(0.99, 0.35 + (0.1 * final_score)))
            )
    return results


# ── Fingerprint query detector ────────────────────────────────────────────────
# Any question that is clearly about code/files/providers routes fingerprint-first.
FINGERPRINT_TERMS = {
    "fingerprint", "fingerprinting",
    "file", "files", "library", "chunk", "module",
    "provider", "providers",
    "code", "class", "classes", "method", "function",
    "android id", "androidid", "gsf", "gsfid",
    "mediadrm", "drm",
    "deviceid", "device id", "device_id",
    "signal", "identifier", "identifiers",
    "repo", "repository",
    "vulnerable file", "vulnerable class",
    "which file", "which class",
    "sdk fallback", "fallback signal",
    "attestation", "emulator detection",
}


def is_fingerprint_query(question: str) -> bool:
    q = question.lower()
    for term in FINGERPRINT_TERMS:
        if term in q:
            return True
    return False


# ── Ranker (fingerprint_mode hard-suppresses structured sources) ──────────────
def rank_and_dedup_results(
    results: list,
    question: str,
    top_k: int = 8,
    fingerprint_mode: bool = False,
) -> list:
    if not results:
        return []

    q = question.lower()
    deduped = {}

    for hit in results:
        key = (
            f"{hit.get('source_type')}::{hit.get('title')}"
            f"::{(hit.get('chunk_text') or '')[:200]}"
        )

        semantic   = safe_similarity(hit.get("score", 0.0))
        risk       = min(max(hit.get("risk_score", 0), 0), 5) / 5.0
        confidence = min(max(hit.get("confidence", 0.0), 0.0), 1.0)

        recency = 0.0
        if hit.get("has_created_at"):
            try:
                created_at = pd.to_datetime(hit.get("created_at"), errors="coerce")
                if pd.notna(created_at):
                    now_ts = pd.Timestamp.utcnow()
                    if getattr(created_at, "tzinfo", None) is not None:
                        now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts
                    else:
                        created_at = created_at.tz_localize(None)
                        now_ts = now_ts.tz_localize(None)
                    age_seconds = max((now_ts - created_at).total_seconds(), 0)
                    age_days = age_seconds / 86400.0
                    recency = max(0.0, 1.0 - min(age_days / 30.0, 1.0))
            except Exception:
                recency = 0.0

        base_source_boost = {
            "fingerprint_library_chunks": 0.10,
            "vector_chunks":              0.08,
            "recommendations":            0.08,
            "insights":                   0.06,
            "changes":                    0.05,
            "snapshots":                  0.04,
        }.get(hit.get("source_type"), 0.02)

        if fingerprint_mode:
            # Hard-suppress structured risk sources so they cannot outrank fp chunks
            if hit.get("source_type") in ("recommendations", "insights", "changes", "snapshots"):
                base_source_boost = -0.30
            if hit.get("source_type") == "fingerprint_library_chunks":
                base_source_boost = 0.30
        else:
            # Legacy heuristic boosts for general queries
            if is_fingerprint_query(question) and hit.get("source_type") == "fingerprint_library_chunks":
                base_source_boost += 0.10
            if (
                any(t in q for t in ["priority", "triage", "action", "recommendation"])
                and hit.get("source_type") == "recommendations"
            ):
                base_source_boost += 0.06

        final_score = (
            0.50 * semantic
            + 0.15 * risk
            + 0.10 * confidence
            + 0.15 * recency
            + base_source_boost
        )
        hit["final_rank_score"] = round(final_score, 4)

        prev = deduped.get(key)
        if prev is None or hit["final_rank_score"] > prev["final_rank_score"]:
            deduped[key] = hit

    ranked = sorted(deduped.values(), key=lambda x: x.get("final_rank_score", 0.0), reverse=True)
    return ranked[:top_k]


# ── Top-level multisource retrieval  (fingerprint-first routing) ──────────────
def retrieve_multisource_context(question: str, top_k: int = 8) -> list:
    if is_fingerprint_query(question):
        # FINGERPRINT-FIRST PATH: structured risk tables are not called at all
        fp_hits = retrieve_fingerprint_chunks(question, top_k=max(top_k, 10))

        if fp_hits:
            # Optionally supplement with generic vector chunks only
            vec_hits = retrieve_vector_chunks(question, top_k=max(3, top_k // 3))
            combined = fp_hits + vec_hits
            return rank_and_dedup_results(combined, question, top_k=top_k, fingerprint_mode=True)
        else:
            # Nothing in fingerprint library matched - warn and fall back fully
            st.warning(
                "No fingerprint library chunks matched this query. "
                "Falling back to general retrieval."
            )
            results = retrieve_vector_chunks(question, top_k=max(4, top_k))
            results += retrieve_structured_supabase(question, top_k=RAG_STRUCTURED_TOP_K)
            return rank_and_dedup_results(results, question, top_k=top_k)
    else:
        # GENERAL PATH: mixed retrieval, unchanged behaviour
        results = []
        results += retrieve_vector_chunks(question, top_k=max(4, top_k))
        results += retrieve_fingerprint_chunks(question, top_k=max(3, top_k // 2))
        results += retrieve_structured_supabase(question, top_k=RAG_STRUCTURED_TOP_K)
        return rank_and_dedup_results(results, question, top_k=top_k)


# ── Evidence formatting ───────────────────────────────────────────────────────
def build_evidence_context(evidence: list, max_items: int = 5, max_chars: int = 500) -> str:
    context_blocks = []
    for i, ev in enumerate(evidence[:max_items], start=1):
        excerpt = truncate_at_sentence(ev.get("chunk_text", ""), max_chars=max_chars)
        context_blocks.append("\n".join([
            f"Source {i}",
            f"Title: {ev.get('title', 'Untitled')}",
            f"Source Type: {ev.get('source_type', 'unknown')}",
            f"Risk Score: {ev.get('risk_score', 0)}",
            f"Confidence: {ev.get('confidence', 0.0)}",
            f"Excerpt: {excerpt}",
        ]))
    return "\n\n".join(context_blocks)


def classify_question_type(question: str) -> str:
    q = question.lower().strip()
    if any(phrase in q for phrase in ["how many", "count", "number of"]):
        return "count"
    if any(phrase in q for phrase in ["what changed", "new changes", "recent changes", "since the last update"]):
        return "changes"
    if any(phrase in q for phrase in ["what should i do", "action item", "what next", "next step"]):
        return "action"
    if any(phrase in q for phrase in ["why is this a risk", "why does this matter"]):
        return "why"
    if any(term in q for term in ["top", "highest", "priority", "triage", "most important risks"]):
        return "top_risks"
    return "general"


# ── LLM call ──────────────────────────────────────────────────────────────────
from src.llm_client import get_llm_client, chat_json


def call_llm(prompt: str) -> str:
    client = get_llm_client()
    result = chat_json(
        client=client,
        model="google/gemma-4-31b-it",
        system="You are a security risk analyst. Return valid JSON with a single key: answer.",
        user=(
            "Answer the user's question using the provided context. "
            "Be concise, natural, and human-readable.\n\n"
            f"{prompt}\n\n"
            'Return JSON only, like: {"answer": "..."}'
        ),
        temperature=0.3,
        max_tokens=400,
    )
    return (result.get("answer") or "").strip()


def generate_natural_answer(question: str, evidence: list) -> str:
    question_type = classify_question_type(question)
    context = build_evidence_context(evidence, max_items=5, max_chars=500)
    prompt = f"""
You are a sharp security risk analyst.

Question type: {question_type}
User question: "{question}"

Retrieved context:
{context}

Write a clear and decisive answer.

Rules:
- Start with a strong, direct statement (avoid vague phrases like "this focuses on")
- Only include information that directly answers the question
- Do NOT introduce unrelated topics
- Emphasize impact and risk (why this matters)
- Avoid generic phrases like "monitor closely" unless you explain why
- Be concise and specific

Structure:
1. First sentence: clear answer
2. 1-2 sentences: why it matters (impact)
3. Final sentence: "Bottom line: ..."

Do not list items unless necessary.
"""
    return call_llm(prompt).strip()


def count_from_supabase(question: str) -> dict:
    sb = get_supabase()
    q = question.lower().strip()
    count_results = {}

    table_keyword_map = {
        "recommendations": ["recommendation", "action item", "action"],
        "insights":        ["insight", "finding"],
        "changes":         ["change", "diff"],
        "snapshots":       ["snapshot"],
    }

    tables_to_count = [t for t, kws in table_keyword_map.items() if any(kw in q for kw in kws)]
    if not tables_to_count:
        tables_to_count = list(table_keyword_map.keys())

    for table in tables_to_count:
        try:
            resp = sb.table(table).select("id", count="exact").execute()
            count_results[table] = resp.count if resp.count is not None else len(resp.data or [])
        except Exception:
            count_results[table] = "unavailable"

    parts = [f"- {k}: **{v}**" for k, v in count_results.items()]
    total = sum(v for v in count_results.values() if isinstance(v, int))
    answer = "Exact counts from Supabase:\n\n" + "\n".join(parts) + f"\n\n**Total: {total}**"
    return {"answer": answer, "confidence": 0.95}


def answer_from_supabase_rag(question: str, top_k: int = 8) -> dict:
    evidence = retrieve_multisource_context(question, top_k=top_k)

    if not evidence:
        return {
            "answer": "I could not retrieve any relevant evidence from Supabase for this question.",
            "confidence": 0.0,
            "evidence": [],
        }

    if classify_question_type(question) == "count":
        count_result = count_from_supabase(question)
        return {"answer": count_result["answer"], "confidence": count_result["confidence"], "evidence": evidence}

    try:
        answer = generate_natural_answer(question, evidence)
    except Exception as e:
        top = evidence[0]
        fallback_excerpt = truncate_at_sentence(top.get("chunk_text", ""), max_chars=500)
        answer = (
            f"The most relevant item right now is **{top.get('title', 'Untitled')}**.\n\n"
            f"{fallback_excerpt}\n\n"
            f"Bottom line: this appears to be the strongest match for your question, "
            f"but the natural-language summary step failed with: {e}"
        )

    conf = round(min(0.95, max(0.40, evidence[0].get("final_rank_score", 0.0))), 2)
    return {"answer": answer, "confidence": conf, "evidence": evidence}


# ── Chat UI ───────────────────────────────────────────────────────────────────
question = st.text_input("Ask a question about risk insights, changes, or signals")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving from Supabase and ranking evidence..."):
        result = answer_from_supabase_rag(question, top_k=RAG_TOP_K)
        st.session_state.chat_history.append({"question": question, "result": result})

for item in reversed(st.session_state.chat_history):
    st.markdown(f"### Q: {item['question']}")
    st.write(item["result"]["answer"])

    confidence = item["result"].get("confidence")
    if confidence is not None:
        st.caption(f"Confidence: {confidence}")

    with st.expander("Retrieved evidence"):
        evidence = item["result"].get("evidence", [])
        if not evidence:
            st.write("No evidence retrieved.")
        else:
            for idx, ev in enumerate(evidence, start=1):
                st.markdown(
                    f"**Source {idx}: {ev.get('title', 'Untitled')}**  "
                    f"`{ev.get('source_type', 'unknown')}`  "
                    f"score={round(ev.get('final_rank_score', ev.get('score', 0.0)), 3)}"
                )
                st.write(ev.get("chunk_text", "")[:1500])
                st.json({
                    "source_type":      ev.get("source_type"),
                    "source_id":        ev.get("source_id"),
                    "snapshot_id":      ev.get("snapshot_id"),
                    "kind":             ev.get("kind"),
                    "risk_score":       ev.get("risk_score"),
                    "component":        ev.get("component"),
                    "category":         ev.get("category"),
                    "score":            ev.get("score"),
                    "final_rank_score": ev.get("final_rank_score"),
                    "created_at":       ev.get("created_at"),
                    "has_created_at":   ev.get("has_created_at"),
                })