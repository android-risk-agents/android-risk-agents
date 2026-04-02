import os
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Android Risk Executive Dashboard", layout="wide")

st.title("Android Risk Agents – Executive Risk Intelligence Dashboard")
st.markdown("Strategic Risk Posture, Exposure Concentration, Change Intelligence & Actionability")
st.markdown("---")

# ----------------------------
# Env / Supabase
# ----------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
INSIGHTS_TABLE = os.getenv("INSIGHTS_TABLE", "insights")
DEFAULT_FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "500"))

# ---- RAG / embedding env ----
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
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY / SUPABASE_ANON_KEY")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ----------------------------
# Helpers
# ----------------------------
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
        return int(max(1, min(5, val)))
    if val > 5:
        val = val / 20.0
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
        "_source_table": "insights",
    }
    out = df.copy()
    for col, default in needed.items():
        if col not in out.columns:
            out[col] = default
    return out


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
        df_recs = df_recs.rename(columns={
            "final_risk_score": "risk_score"
        })

        if "title" not in df_recs.columns:
            df_recs["title"] = "Recommendation"
        if "recommendation_text" in df_recs.columns:
            df_recs["summary"] = df_recs["recommendation_text"].fillna("")
        elif "summary" not in df_recs.columns:
            df_recs["summary"] = ""
        if "source_id" in df_recs.columns:
            df_recs["component"] = df_recs["source_id"].astype(str).fillna("Recommendation Engine")
        elif "component" not in df_recs.columns:
            df_recs["component"] = "Recommendation Engine"
        if "category" in df_recs.columns:
            df_recs["kind"] = df_recs["category"].fillna("recommendation")
        elif "kind" not in df_recs.columns:
            df_recs["kind"] = "recommendation"
        if "confidence" not in df_recs.columns:
            df_recs["confidence"] = 0.7
        df_recs["confidence"] = df_recs["confidence"].apply(normalize_confidence)
        df_recs["risk_score"] = df_recs["risk_score"].apply(normalize_risk_score)
        df_recs["created_at"] = pd.to_datetime(df_recs["created_at"], errors="coerce")
        df_recs["_source_table"] = "recommendations"

    if not df_insights.empty:
        df_insights = ensure_columns(df_insights)
        df_insights["confidence"] = df_insights["confidence"].apply(normalize_confidence)
        df_insights["risk_score"] = df_insights["risk_score"].apply(normalize_risk_score)
        df_insights["created_at"] = pd.to_datetime(df_insights["created_at"], errors="coerce")
        df_insights["_source_table"] = "insights"

    df_combined = pd.concat([df_insights, df_recs], ignore_index=True)
    df_combined["risk_score"] = df_combined["risk_score"].fillna(0)
    df_combined["confidence"] = df_combined["confidence"].fillna(0.0)
    df_combined = df_combined.dropna(subset=["created_at"])

    return df_combined


def make_mock_data():
    n = 22
    base = pd.DataFrame({
        "title": [f"Insight {i}" for i in range(1, n + 1)],
        "risk_score": np.random.choice([1, 1, 2, 2, 3, 3, 4, 4, 5], size=n, replace=True),
        "confidence": np.round(np.random.uniform(0.5, 1.0, size=n), 2),
        "created_at": pd.date_range(start="2026-03-01", periods=n, freq="12H"),
        "component": np.random.choice(["Auth", "Network", "Storage", "Payments", "UI", "Telemetry"], size=n),
        "kind": np.random.choice(["permission", "api_change", "config", "dependency", "logging"], size=n),
    })

    base["summary"] = base.apply(
        lambda r: f"Detected change in {r['component']} ({r['kind']}). Potential risk requires review.", axis=1
    )
    base["recommended_actions"] = base.apply(
        lambda r: [
            "Validate scope/impact",
            "Add targeted tests",
            "Review permissions/data flows" if r["kind"] == "permission" else "Review change rationale",
        ],
        axis=1,
    )

    base["status"] = "New"
    base["owner"] = ""
    base["due_date"] = (base["created_at"] + pd.to_timedelta(7, unit="D")).dt.date.astype(str)
    base["notes"] = ""

    old = base.copy()
    old["snapshot_id"] = "snapshot_old"
    old["created_at"] = old["created_at"] - pd.to_timedelta(3, unit="D")

    new = base.copy()
    new["snapshot_id"] = "snapshot_new"

    rng = np.random.default_rng(7)
    mod_idx = rng.choice(new.index, size=max(4, n // 5), replace=False)
    new.loc[mod_idx, "risk_score"] = np.clip(new.loc[mod_idx, "risk_score"] + 1, 1, 5)
    new.loc[mod_idx, "confidence"] = np.clip(new.loc[mod_idx, "confidence"] + 0.05, 0.0, 1.0)
    new.loc[mod_idx, "summary"] = new.loc[mod_idx, "summary"].astype(str) + " Updated signals increased risk."

    remove_idx = set(rng.choice(new.index, size=3, replace=False).tolist())
    new = new.drop(index=list(remove_idx)).reset_index(drop=True)

    add_n = 4
    added = pd.DataFrame({
        "title": [f"New Insight {i}" for i in range(1, add_n + 1)],
        "risk_score": rng.choice([3, 4, 5], size=add_n),
        "confidence": np.round(rng.uniform(0.65, 1.0, size=add_n), 2),
        "created_at": pd.date_range(start="2026-03-05", periods=add_n, freq="6H"),
        "component": rng.choice(["Auth", "Network", "Storage", "Payments", "UI", "Telemetry"], size=add_n),
        "kind": rng.choice(["permission", "api_change", "config", "dependency", "logging"], size=add_n),
    })
    added["summary"] = added.apply(
        lambda r: f"Newly detected change in {r['component']} ({r['kind']}). Requires triage.", axis=1
    )
    added["recommended_actions"] = added.apply(
        lambda r: [
            "Triage and assign owner",
            "Confirm exploitability",
            "Add monitoring / logging" if r["kind"] == "logging" else "Patch or mitigate",
        ],
        axis=1,
    )
    added["status"] = "New"
    added["owner"] = ""
    added["due_date"] = (pd.to_datetime(added["created_at"]) + pd.to_timedelta(5, unit="D")).dt.date.astype(str)
    added["notes"] = ""
    added["snapshot_id"] = "snapshot_new"

    new = pd.concat([new, added], ignore_index=True)
    all_df = pd.concat([old, new], ignore_index=True)
    all_df["created_at"] = pd.to_datetime(all_df["created_at"])
    return all_df


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

    changed_df = pd.DataFrame(changed_rows) if changed_rows else pd.DataFrame(columns=list(df_new.columns) + [
        "old_risk_score", "old_confidence", "old_summary", "delta_risk", "delta_conf"
    ])
    persisting_df = pd.DataFrame(persisting_rows) if persisting_rows else pd.DataFrame(columns=df_new.columns)

    return added, removed, changed_df, persisting_df


# ----------------------------
# Load data
# ----------------------------
with st.sidebar:
    st.subheader("Data source")
    use_mock_fallback = st.toggle("Use mock fallback if Supabase fails", value=True)

load_error = None
try:
    all_df = fetch_insights_from_supabase(DEFAULT_FETCH_LIMIT)
    data_mode = "Supabase"
    if all_df.empty:
        raise RuntimeError(f"No rows found in table '{INSIGHTS_TABLE}'.")
except Exception as e:
    load_error = str(e)
    if use_mock_fallback:
        all_df = make_mock_data()
        data_mode = "Mock fallback"
    else:
        st.error(f"Failed to load data from Supabase: {e}")
        st.stop()

st.caption(f"Data source: **{data_mode}**")
if load_error and data_mode == "Mock fallback":
    st.warning(f"Supabase load failed, so the app is showing mock data instead: {load_error}")

if "triage" not in st.session_state:
    st.session_state.triage = {}

# ----------------------------
# Sidebar filters
# ----------------------------
with st.sidebar:
    st.subheader("Filters")

    snapshots = sorted([s for s in all_df["snapshot_id"].dropna().astype(str).unique().tolist() if s])
    if not snapshots:
        snapshots = ["snapshot_unknown"]
        all_df["snapshot_id"] = "snapshot_unknown"

    default_new = snapshots[-1]

    snap_new = st.selectbox("Snapshot", snapshots, index=snapshots.index(default_new))
    snap_old = None
    enable_compare = False

    st.divider()

    max_dt = all_df["created_at"].max()
    days = st.number_input("Recent days", min_value=1, max_value=365, value=14, step=1)
    _max_dt_naive = max_dt.replace(tzinfo=None) if pd.notna(max_dt) and hasattr(max_dt, "tzinfo") else max_dt
    start_dt = _max_dt_naive - timedelta(days=int(days)) if pd.notna(max_dt) else datetime.utcnow() - timedelta(days=14)

    min_risk = st.slider("Min risk score", min_value=1, max_value=5, value=3, step=1)
    min_conf = st.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    comps = sorted(all_df["component"].fillna("Unknown").astype(str).unique().tolist())
    kinds = sorted(all_df["kind"].fillna("unknown").astype(str).unique().tolist())
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


df_new = all_df.copy()

_real_snapshots = [s for s in all_df["snapshot_id"].dropna().astype(str).unique()
                   if s not in ("snapshot_unknown", "", "nan")]
if _real_snapshots:
    df_new = all_df[all_df["snapshot_id"].astype(str) == str(snap_new)].copy()

df_old = None
if snap_old and _real_snapshots:
    df_old = all_df[all_df["snapshot_id"].astype(str) == str(snap_old)].copy()

df = df_new.copy()

_start_dt = pd.to_datetime(start_dt)
if df["created_at"].dt.tz is not None and _start_dt.tzinfo is None:
    _start_dt = _start_dt.replace(tzinfo=timezone.utc)
elif df["created_at"].dt.tz is None and _start_dt.tzinfo is not None:
    _start_dt = _start_dt.replace(tzinfo=None)

df = df[df["created_at"] >= _start_dt]
df = df[df["risk_score"] >= min_risk]
df = df[df["confidence"] >= min_conf]
df = df[df["component"].isin(sel_comps)]
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


def apply_triage(row):
    k = stable_key(row)
    t = st.session_state.triage.get(k, {})
    for col in ["owner", "status", "due_date", "notes"]:
        if col in t:
            row[col] = t[col]
    return row


if len(df) > 0:
    df = df.apply(apply_triage, axis=1)

# ----------------------------
# KPIs
# ----------------------------
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

st.markdown("## 🔎 Executive Risk Summary")
st.markdown(f"""
**Current Portfolio Status:** {risk_status}

- Time window: last **{int(days)}** days (from {pd.to_datetime(start_dt).date()} to {pd.to_datetime(max_dt).date() if pd.notna(max_dt) else 'N/A'})
- {high_risk_pct}% of filtered insights are high severity (risk ≥4)
- Dispersion index (variance): **{risk_variance}**
""")

_rec_df = all_df[all_df["_source_table"] == "recommendations"].copy() if "_source_table" in all_df.columns else pd.DataFrame()
_high_recs = (
    _rec_df[_rec_df["risk_score"] >= 4]
    .sort_values(["risk_score", "confidence"], ascending=[False, False])
    .head(5)
    if not _rec_df.empty else pd.DataFrame()
)

if not _high_recs.empty:
    st.markdown("### 🚨 High Priority Updates")
    for _, rec in _high_recs.iterrows():
        priority_label = rec.get("priority", "")
        title = rec.get("title", "Untitled")
        rationale = rec.get("rationale", "")
        raw_actions = rec.get("recommended_actions", [])
        actions = normalize_actions(raw_actions)
        risk = safe_int(rec.get("risk_score", 0))
        conf = rec.get("confidence", "")

        with st.expander(f"{'🔴' if risk >= 4 else '🟡'} [{priority_label}] {title}  —  Risk {risk} | Conf {conf}", expanded=(risk == 5)):
            if rationale:
                st.markdown(f"**Rationale:** {rationale}")
            if actions:
                st.markdown("**Recommended Actions:**")
                for act in actions:
                    st.markdown(f"- {act}")
else:
    st.info("No high-priority recommendations available under current filters.")

st.markdown("---")

diff_added = diff_removed = diff_changed = diff_persisting = None

st.markdown("---")

# ----------------------------
# Charts
# ----------------------------
st.header("Executive Intelligence Overview")

df_exec = all_df.copy()
df_exec = df_exec[df_exec["risk_score"] > 0]

df_exec["risk_score"] = pd.to_numeric(df_exec["risk_score"], errors="coerce")
df_exec["confidence"] = pd.to_numeric(df_exec["confidence"], errors="coerce")

df_exec["weighted_risk"] = df_exec["risk_score"] * df_exec["confidence"]

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
    x="title",
    y="weighted_risk",
    color="category",
    title="Risk Exposure per Insight (Risk × Confidence)"
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
        mode="markers+text",
        text=df_exec["title"],
        textposition="top center",
        marker=dict(
            size=df_exec["weighted_risk"] * 5,
            color=df_exec["weighted_risk"],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Weighted Risk")
        )
    )
)

fig_quad.add_vline(x=median_conf, line_dash="dash")
fig_quad.add_hline(y=median_risk, line_dash="dash")

fig_quad.update_layout(
    title="High Impact Zone = Top Right Quadrant",
    xaxis_title="Confidence",
    yaxis_title="Risk Score",
    height=500
)

st.plotly_chart(fig_quad, use_container_width=True)

st.subheader("3️⃣ Cumulative Risk Exposure Curve")

sorted_df = df_exec.sort_values("weighted_risk", ascending=False).reset_index(drop=True)
sorted_df["cumulative_exposure"] = sorted_df["weighted_risk"].cumsum()

fig_curve = px.line(
    sorted_df,
    x=sorted_df.index + 1,
    y="cumulative_exposure",
    markers=True,
    title="How Quickly Risk Concentrates Across Insights"
)

fig_curve.update_layout(
    xaxis_title="Top N Insights",
    yaxis_title="Cumulative Weighted Risk"
)

st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("4️⃣ Risk Concentration by Category")

category_risk = (
    df_exec.groupby("category")["weighted_risk"]
    .sum()
    .reset_index()
    .sort_values("weighted_risk", ascending=False)
)

fig_cat = px.bar(
    category_risk,
    x="category",
    y="weighted_risk",
    title="Total Weighted Risk by Category"
)

st.plotly_chart(fig_cat, use_container_width=True)

# ----------------------------
# Triage queue + detail panel
# ----------------------------
st.subheader("🧰 Action Queue (Top items to triage today)")
if total_insights:
    queue = df.copy().sort_values(["risk_score", "confidence", "created_at"], ascending=[False, False, False]).head(10)
    st.caption("Click an item below to see details and assign actions/owner/status.")

    labels = []
    for _, r in queue.iterrows():
        labels.append(f"[{risk_band(int(r['risk_score']))}] {r['title']} • {r['component']} • conf {r['confidence']}")

    selected = st.radio("Top 10", labels, label_visibility="collapsed")
    sel_idx = labels.index(selected)
    sel_row = queue.iloc[sel_idx]

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown(f"### {sel_row['title']}")
        st.write(sel_row.get("summary", ""))

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
                st.write(f"- Risk: {rowc.get('old_risk_score')} → {rowc.get('risk_score')}{badge_delta(rowc.get('risk_score'), rowc.get('old_risk_score'), fmt='{:.0f}')}")
                st.write(f"- Confidence: {rowc.get('old_confidence')} → {rowc.get('confidence')}{badge_delta(rowc.get('confidence'), rowc.get('old_confidence'), fmt='{:.2f}')}")
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
        status = st.selectbox("Status", status_options, index=status_options.index(existing_status) if existing_status in status_options else 0, key=f"status_{k}")
        due_date = st.text_input("Due date (YYYY-MM-DD)", value=existing.get("due_date", sel_row.get("due_date", "")), key=f"due_{k}")
        notes = st.text_area("Notes", value=existing.get("notes", sel_row.get("notes", "")), height=140, key=f"notes_{k}")

        if st.button("Save triage", type="primary"):
            st.session_state.triage[k] = {"owner": owner, "status": status, "due_date": due_date, "notes": notes}
            st.success("Saved.")

        st.markdown("---")
        st.markdown("### Export")
        st.download_button("Download current filtered CSV", data=df_to_csv_bytes(df), file_name="filtered_insights.csv", mime="text/csv")

# ----------------------------
# High-risk table
# ----------------------------
st.subheader("📌 High-Risk Insight Breakdown (filtered)")
top_risks = df[df["risk_score"] >= 4].copy().sort_values(by=["risk_score", "confidence", "created_at"], ascending=[False, False, False])

if len(top_risks) == 0:
    st.info("No high-risk items under current filters.")
else:
    st.dataframe(top_risks[["title", "component", "kind", "risk_score", "confidence", "created_at", "status", "owner", "due_date"]], use_container_width=True)

# ----------------------------
# RAG : Chatbot (Supabase-only, vector search)
# ----------------------------
st.markdown("## Ask the assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def get_embedder():
    from sentence_transformers import SentenceTransformer
    # Use the full HuggingFace model ID, not the shorthand
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


def normalize_hit(source_type: str, row: dict, similarity=None) -> dict:
    """
    Safely normalize a raw RPC or table row into a standard hit dict.
    Handles missing fields gracefully - especially for vector RPC results
    which only return: id, chunk_text, similarity.
    Fingerprint RPC returns: file_id, repo_name, file_name, file_path,
    module_name, category, chunk_index, chunk_title, chunk_summary, chunk_text, similarity.
    """
    # Title fallback chain per source type
    if source_type == "fingerprint_library_chunks":
        title = (
            row.get("chunk_title")
            or row.get("file_name")
            or row.get("file_path")
            or "Fingerprint Evidence"
        )
    elif source_type == "vector_chunks":
        # Basic 2-arg RPC returns no title - derive from chunk_text
        raw_chunk = row.get("chunk_text", "") or ""
        title = raw_chunk[:80].replace("\n", " ").strip() or "Knowledge Chunk"
    else:
        # Structured table rows do have title
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

    risk_score = normalize_risk_score(
        row.get("risk_score", row.get("final_risk_score"))
    )
    if pd.isna(risk_score) if isinstance(risk_score, float) else False:
        risk_score = 0

    confidence = normalize_confidence(row.get("confidence"))
    if isinstance(confidence, float) and pd.isna(confidence):
        confidence = 0.0

    # created_at: NOT returned by vector or fingerprint RPCs - only present on table rows
    raw_created_at = row.get("created_at")
    created_at_str = to_iso_str(raw_created_at) if raw_created_at else ""

    # source_id: not guaranteed from basic 2-arg vector RPC
    source_id = (
        row.get("source_id")
        or row.get("file_id")
        or row.get("id")
        or ""
    )

    return {
        "source_type": source_type,
        "title": title,
        "chunk_text": str(chunk_text or ""),
        "score": safe_similarity(
            similarity if similarity is not None
            else row.get("similarity", row.get("score", 0.0))
        ),
        "risk_score": safe_int(risk_score, default=0),
        "confidence": float(confidence) if confidence else 0.0,
        "source_id": str(source_id),
        "snapshot_id": row.get("snapshot_id", ""),
        "kind": row.get("kind") or row.get("category") or source_type,
        "component": (
            row.get("component")
            or row.get("module_name")
            or row.get("repo_name")
            or row.get("source_id")
            or row.get("file_name")
            or ""
        ),
        "category": row.get("category") or source_type,
        "created_at": created_at_str,
        "has_created_at": bool(created_at_str),  # flag: only use recency if True
        "metadata": row,
    }


def vector_rpc_call(rpc_name: str, query_embedding: list, match_count: int):
    """
    Call a Supabase RPC safely, trying the canonical payload first.
    For match_vector_chunks, the safest 2-arg call is:
      {query_embedding, match_count}
    which returns: id, chunk_text, similarity only.
    """
    sb = get_supabase()
    payload_options = [
        {"query_embedding": query_embedding, "match_count": match_count},
        # 4-arg text overload for metadata-enriched results
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


def retrieve_vector_chunks(question: str, top_k: int = 5) -> list:
    """
    Call match_vector_chunks(query_embedding, match_count).
    Only parse: id, chunk_text, similarity.
    All other fields may be absent - handle safely.
    Source is always Supabase vector_chunks table, never the dashboard df.
    """
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
    """
    Call match_fingerprint_library_chunks(query_embedding, match_count).
    Parse: file_id, repo_name, file_name, file_path, module_name, category,
           chunk_index, chunk_title, chunk_summary, chunk_text, similarity.
    Title fallback: chunk_title -> file_name -> file_path -> 'Fingerprint Evidence'.
    """
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
    """
    Keyword + recency fallback against actual Supabase tables.
    Tables: insights, recommendations, changes, snapshots.
    This is pure Supabase - does NOT touch the dashboard dataframe.
    """
    q = question.lower().strip()
    sb = get_supabase()
    results = []

    table_specs = [
        ("insights", "created_at"),
        ("recommendations", "created_at"),
        ("changes", "created_at"),
        ("snapshots", "fetched_at"),
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

            overlap = sum(
                1 for token in q.split()
                if len(token) > 2 and token in blob
            )

            risk = safe_int(
                normalize_risk_score(r.get("risk_score", r.get("final_risk_score"))),
                default=0
            )

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
                normalize_hit(
                    table_name,
                    row,
                    similarity=min(0.99, 0.35 + (0.1 * final_score))
                )
            )

    return results


def query_needs_fingerprint_boost(question: str) -> bool:
    q = question.lower()
    keywords = [
        "android id", "androidid", "gsf", "mediadrm", "identifier", "device id",
        "signal", "provider", "fingerprint", "sdk", "fallback", "os build",
        "hardware signal", "installed apps", "attestation", "emulator", "tamper",
    ]
    return any(k in q for k in keywords)


def rank_and_dedup_results(results: list, question: str, top_k: int = 8) -> list:
    if not results:
        return []

    q = question.lower()
    deduped = {}
    for hit in results:
        key = (
            f"{hit.get('source_type')}::{hit.get('title')}"
            f"::{(hit.get('chunk_text') or '')[:200]}"
        )

        semantic = safe_similarity(hit.get("score", 0.0))
        risk = min(max(hit.get("risk_score", 0), 0), 5) / 5.0
        confidence = min(max(hit.get("confidence", 0.0), 0.0), 1.0)

        # Only compute recency if the hit actually has created_at
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

        source_boost = {
            "fingerprint_library_chunks": 0.10,
            "vector_chunks": 0.08,
            "recommendations": 0.08,
            "insights": 0.06,
            "changes": 0.05,
            "snapshots": 0.04,
        }.get(hit.get("source_type"), 0.02)

        if query_needs_fingerprint_boost(question) and hit.get("source_type") == "fingerprint_library_chunks":
            source_boost += 0.10
        if any(
            term in q for term in ["priority", "triage", "action", "recommendation"]
        ) and hit.get("source_type") == "recommendations":
            source_boost += 0.06

        final_score = (
            0.50 * semantic
            + 0.15 * risk
            + 0.10 * confidence
            + 0.15 * recency
            + source_boost
        )
        hit["final_rank_score"] = round(final_score, 4)

        prev = deduped.get(key)
        if prev is None or hit["final_rank_score"] > prev["final_rank_score"]:
            deduped[key] = hit

    ranked = sorted(
        deduped.values(),
        key=lambda x: x.get("final_rank_score", 0.0),
        reverse=True
    )
    return ranked[:top_k]


def retrieve_multisource_context(question: str, top_k: int = 8) -> list:
    """
    Pure Supabase retrieval - no dashboard dataframe used.
    Sources: match_vector_chunks RPC, match_fingerprint_library_chunks RPC,
             and direct table queries (insights, recommendations, changes, snapshots).
    """
    results = []
    vector_hits = retrieve_vector_chunks(question, top_k=max(4, top_k))
    fingerprint_hits = retrieve_fingerprint_chunks(question, top_k=max(3, top_k // 2))
    structured_hits = retrieve_structured_supabase(question, top_k=RAG_STRUCTURED_TOP_K)

    results.extend(vector_hits)
    results.extend(fingerprint_hits)
    results.extend(structured_hits)

    return rank_and_dedup_results(results, question, top_k=top_k)


def answer_from_supabase_rag(question: str, top_k: int = 8) -> dict:
    evidence = retrieve_multisource_context(question, top_k=top_k)
    if not evidence:
        return {
            "answer": "I could not retrieve any relevant evidence from Supabase for this question.",
            "confidence": 0.0,
            "evidence": [],
        }

    q = question.lower().strip()

    if any(phrase in q for phrase in ["how many", "count", "number of"]):
        sb = get_supabase()
        count_results = {}

        table_keyword_map = {
            "recommendations": ["recommendation", "action item", "action"],
            "insights": ["insight", "finding"],
            "changes": ["change", "diff"],
            "snapshots": ["snapshot"],
        }

        tables_to_count = []
        for table, keywords in table_keyword_map.items():
            if any(kw in q for kw in keywords):
                tables_to_count.append(table)

        if not tables_to_count:
            tables_to_count = list(table_keyword_map.keys())

        for table in tables_to_count:
            try:
                resp = sb.table(table).select("id", count="exact").execute()
                count_results[table] = resp.count if resp.count is not None else len(resp.data or [])
            except Exception:
                count_results[table] = "unavailable"

        parts = [f"{k}: **{v}**" for k, v in count_results.items()]
        total = sum(v for v in count_results.values() if isinstance(v, int))
        answer = (
            f"Exact counts from Supabase:\n\n"
            + "\n".join(f"- {p}" for p in parts)
            + f"\n\n**Total: {total}**"
        )
        return {"answer": answer, "confidence": 0.95, "evidence": evidence}

    if any(term in q for term in ["top", "highest", "priority", "triage", "action"]):
        lines = []
        for i, ev in enumerate(evidence[:5], start=1):
            lines.append(
                f"{i}. **{ev.get('title', 'Untitled')}** "
                f"[{ev.get('source_type')}] - risk {ev.get('risk_score', 0)}, "
                f"score {round(ev.get('final_rank_score', 0.0), 2)}"
            )
        return {
            "answer": "Top relevant Supabase-backed evidence:\n\n" + "\n".join(lines),
            "confidence": round(
                min(0.95, max(0.55, evidence[0].get("final_rank_score", 0.0))), 2
            ),
            "evidence": evidence,
        }

    top = evidence[0]

    # Build a proper sentence-boundary truncation
    def truncate_at_sentence(text: str, max_chars: int = 400) -> str:
        text = (text or "").replace("\n", " ").strip()
        if len(text) <= max_chars:
            return text
        # Try to cut at last sentence boundary before max_chars
        truncated = text[:max_chars]
        for sep in [". ", "! ", "? "]:
            last = truncated.rfind(sep)
            if last != -1:
                return truncated[:last + 1]
        # No sentence boundary found - cut at last word boundary
        last_space = truncated.rfind(" ")
        if last_space != -1:
            return truncated[:last_space] + "."
        return truncated + "."

    support_lines = []
    for i, ev in enumerate(evidence[:4], start=1):
        excerpt = truncate_at_sentence(ev.get("chunk_text", ""), max_chars=600)
        support_lines.append(
            f"{i}. **[{ev.get('source_type')}] {ev.get('title', 'Untitled')}**\n   {excerpt}"
        )

    answer = (
        f"Based on Supabase retrieval, the strongest match is **{top.get('title', 'Untitled')}** "
        f"from **{top.get('source_type')}**.\n\n"
        f"**Supporting evidence:**\n\n" + "\n\n".join(support_lines)
    )

    conf = round(min(0.95, max(0.40, evidence[0].get("final_rank_score", 0.0))), 2)

    return {
        "answer": answer,
        "confidence": conf,
        "evidence": evidence,
    }


question = st.text_input("Ask a question about risk insights, changes, or signals")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving from Supabase and ranking evidence..."):
        result = answer_from_supabase_rag(question, top_k=RAG_TOP_K)
        st.session_state.chat_history.append({
            "question": question,
            "result": result,
        })

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

                meta = {
                    "source_type": ev.get("source_type"),
                    "source_id": ev.get("source_id"),
                    "snapshot_id": ev.get("snapshot_id"),
                    "kind": ev.get("kind"),
                    "risk_score": ev.get("risk_score"),
                    "component": ev.get("component"),
                    "category": ev.get("category"),
                    "score": ev.get("score"),
                    "final_rank_score": ev.get("final_rank_score"),
                    "created_at": ev.get("created_at"),
                    "has_created_at": ev.get("has_created_at"),
                }
                st.json(meta)
