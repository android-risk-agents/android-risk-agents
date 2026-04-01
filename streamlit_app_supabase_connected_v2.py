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
    # 0-1 fractional scale -> multiply to 1-5
    if val <= 1.0:
        val = round(val * 5)
        return int(max(1, min(5, val)))
    # 0-100 scale (e.g. recommendations.final_risk_score) -> divide to 1-5
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

    # ---- Pull INSIGHTS table ----
    insights_rows = (
        sb.table("insights")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )

    df_insights = pd.DataFrame(insights_rows or [])

    # ---- Pull RECOMMENDATIONS table ----
    rec_rows = (
        sb.table("recommendations")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
    )

    df_recs = pd.DataFrame(rec_rows or [])

    # --- Normalize recommendations table ---
    if not df_recs.empty:
        df_recs = df_recs.rename(columns={
            "final_risk_score": "risk_score"
        })

        # Use column existence checks instead of df.get() which doesn't work for assignment
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
        # confidence may not exist in recommendations table - default to 0.7
        if "confidence" not in df_recs.columns:
            df_recs["confidence"] = 0.7
        df_recs["confidence"] = df_recs["confidence"].apply(normalize_confidence)
        df_recs["risk_score"] = df_recs["risk_score"].apply(normalize_risk_score)
        df_recs["created_at"] = pd.to_datetime(df_recs["created_at"], errors="coerce")
        # Tag source table so snapshot filtering can handle it separately
        df_recs["_source_table"] = "recommendations"

    # --- Normalize insights ---
    if not df_insights.empty:
        df_insights = ensure_columns(df_insights)
        df_insights["confidence"] = df_insights["confidence"].apply(normalize_confidence)
        df_insights["risk_score"] = df_insights["risk_score"].apply(normalize_risk_score)
        df_insights["created_at"] = pd.to_datetime(df_insights["created_at"], errors="coerce")
        df_insights["_source_table"] = "insights"

    # ---- Combine ----
    df_combined = pd.concat([df_insights, df_recs], ignore_index=True)

    # Fill missing scores with defaults instead of dropping rows entirely.
    # Previously this was dropna(subset=[...]) which silently killed all rows
    # whose confidence/risk_score came from tables that don't have those columns.
    df_combined["risk_score"] = df_combined["risk_score"].fillna(0)
    df_combined["confidence"] = df_combined["confidence"].fillna(0.0)
    # Only drop rows with no timestamp - those are truly unusable
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
   # st.write("DEBUG — All rows loaded:", len(all_df))
#st.write(all_df.head())
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

    # Single snapshot selector - old snapshot compare removed as insights table
    # doesn't carry historical snapshot_id per row; compare is not meaningful here.
    snap_new = st.selectbox("Snapshot", snapshots, index=snapshots.index(default_new))
    snap_old = None
    enable_compare = False

    st.divider()

    max_dt = all_df["created_at"].max()
    days = st.number_input("Recent days", min_value=1, max_value=365, value=14, step=1)
    # Strip timezone for timedelta arithmetic to avoid comparison issues downstream
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

# Only apply snapshot filtering when there are real distinct snapshot IDs in the data.
# The insights table does not store snapshot_id directly, so most rows will have
# snapshot_unknown. When that is the case we skip the snapshot filter entirely so
# all records are visible.
_real_snapshots = [s for s in all_df["snapshot_id"].dropna().astype(str).unique()
                   if s not in ("snapshot_unknown", "", "nan")]
if _real_snapshots:
    df_new = all_df[all_df["snapshot_id"].astype(str) == str(snap_new)].copy()

df_old = None
if snap_old and _real_snapshots:
    df_old = all_df[all_df["snapshot_id"].astype(str) == str(snap_old)].copy()

df = df_new.copy()

# Supabase timestamps are timezone-aware (UTC). Normalize start_dt to match.
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

# --- High Priority Updates from recommendations table (rationale + actions) ---
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
# ----------------------------
# ==========================================================
# EXECUTIVE OPERATIONAL INTELLIGENCE
# ==========================================================

st.header("Executive Intelligence Overview")

df_exec = all_df.copy()
df_exec = df_exec[df_exec["risk_score"] > 0]

# Ensure numeric
df_exec["risk_score"] = pd.to_numeric(df_exec["risk_score"], errors="coerce")
df_exec["confidence"] = pd.to_numeric(df_exec["confidence"], errors="coerce")

df_exec["weighted_risk"] = df_exec["risk_score"] * df_exec["confidence"]


# ==========================================================
# 1️⃣ CONFIDENCE-WEIGHTED RISK EXPOSURE
# ==========================================================

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

# ==========================================================
# 2️⃣ RISK–CONFIDENCE PRIORITY QUADRANT
# ==========================================================

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


# ==========================================================
# 3️⃣ CUMULATIVE RISK EXPOSURE CURVE
# ==========================================================

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

# ==========================================================
# 4️⃣ CATEGORY RISK CONCENTRATION
# ==========================================================

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
# RAG : Chatbot (Updated 26.3.29)
# ----------------------------

st.markdown("## Ask the assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def build_dashboard_evidence(df, top_k=5):
    if df is None or df.empty:
        return []

    ranked = (
        df.copy()
        .sort_values(["risk_score", "confidence", "created_at"], ascending=[False, False, False])
        .head(top_k)
    )

    evidence = []
    for _, row in ranked.iterrows():
        chunk_text = (
            f"Title: {row.get('title', '')}\n"
            f"Summary: {row.get('summary', '')}\n"
            f"Risk Score: {row.get('risk_score', '')}\n"
            f"Confidence: {row.get('confidence', '')}\n"
            f"Component: {row.get('component', '')}\n"
            f"Kind: {row.get('kind', '')}\n"
            f"Category: {row.get('category', '')}\n"
            f"Created At: {row.get('created_at', '')}\n"
            f"Status: {row.get('status', '')}\n"
            f"Owner: {row.get('owner', '')}\n"
            f"Due Date: {row.get('due_date', '')}\n"
        )

        evidence.append({
            "title": row.get("title"),
            "chunk_text": chunk_text,
            "source_id": row.get("source_id"),
            "snapshot_id": row.get("snapshot_id"),
            "kind": row.get("kind"),
            "score": float(row.get("confidence", 0.0)) if pd.notna(row.get("confidence")) else None,
            "risk_score": safe_int(row.get("risk_score"), default=0),
            "component": row.get("component"),
            "category": row.get("category"),
            "created_at": str(row.get("created_at")),
        })

    return evidence


def answer_from_dashboard(question, df, top_k=5):
    if df is None or df.empty:
        return {
            "answer": "There is no filtered dashboard data available to answer this question.",
            "confidence": 0.0,
            "evidence": []
        }

    q = question.strip().lower()

    ranked = (
        df.copy()
        .sort_values(["risk_score", "confidence", "created_at"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    evidence = build_dashboard_evidence(df, top_k=top_k)

    # ── Numeric / count questions ──────────────────────────────────────────────
    import re

    # "how many" total
    if re.search(r"how many", q):
        if re.search(r"high.?risk|risk.*(>=?\s*4|score.*[45]|4 or 5)", q) or "high risk" in q:
            count = int((df["risk_score"] >= 4).sum())
            return {"answer": f"There are **{count}** high-risk items (risk score ≥ 4) in the current filtered view.", "confidence": 1.0, "evidence": evidence}
        if re.search(r"total|insight|item|record|row", q):
            return {"answer": f"There are **{len(df)}** items in the current filtered view.", "confidence": 1.0, "evidence": evidence}
        # Generic count fallback
        return {"answer": f"There are **{len(df)}** items in the current filtered view.", "confidence": 1.0, "evidence": evidence}

    # "what is the average / mean risk"
    if re.search(r"average|mean|avg", q) and re.search(r"risk|score", q):
        avg = round(df["risk_score"].mean(), 2)
        return {"answer": f"The average risk score across the **{len(df)}** filtered items is **{avg}**.", "confidence": 1.0, "evidence": evidence}

    # "what is the average confidence"
    if re.search(r"average|mean|avg", q) and "confidence" in q:
        avg = round(df["confidence"].mean(), 2)
        return {"answer": f"The average confidence across the **{len(df)}** filtered items is **{avg}**.", "confidence": 1.0, "evidence": evidence}

    # "max / maximum risk"
    if re.search(r"max(imum)?|highest", q) and re.search(r"risk score|risk$", q):
        mx = safe_int(df["risk_score"].max())
        top_row = ranked.iloc[0]
        return {
            "answer": (
                f"The maximum risk score in the current view is **{mx}**. "
                f"The item with the highest risk is \"{top_row.get('title', 'Untitled')}\" "
                f"(confidence {top_row.get('confidence')}, component {top_row.get('component', 'Unknown')})."
            ),
            "confidence": 1.0,
            "evidence": evidence,
        }

    # "top N" with optional number word
    _top_n_match = re.search(
        r"top\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)", q
    )
    _word_to_n = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                  "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    if _top_n_match:
        raw = _top_n_match.group(1)
        n = _word_to_n.get(raw, None) or int(raw)
        top_n = ranked.head(n)
        lines = []
        for i, (_, row) in enumerate(top_n.iterrows(), start=1):
            lines.append(
                f"{i}. {row.get('title', 'Untitled')} "
                f"(risk={safe_int(row.get('risk_score'))}, "
                f"confidence={row.get('confidence')}, "
                f"component={row.get('component', 'Unknown')})"
            )
        return {
            "answer": f"The top {n} items by risk score currently on the dashboard are:\n\n" + "\n".join(lines),
            "confidence": 0.95,
            "evidence": evidence,
        }

    # ── Category / component breakdown ────────────────────────────────────────
    if re.search(r"break.?down|by category|by component|per category|per component|distribution", q):
        col = "category" if "category" in q else "component"
        breakdown = df.groupby(col)["risk_score"].agg(["count", "mean"]).rename(columns={"count": "items", "mean": "avg_risk"})
        breakdown["avg_risk"] = breakdown["avg_risk"].round(2)
        lines = [f"- {idx}: {row['items']} items, avg risk {row['avg_risk']}" for idx, row in breakdown.iterrows()]
        return {
            "answer": f"Risk breakdown by **{col}**:\n\n" + "\n".join(lines),
            "confidence": 0.95,
            "evidence": evidence,
        }

    # ── Highest risk ──────────────────────────────────────────────────────────
    if "highest risk" in q:
        row = ranked.iloc[0]
        return {
            "answer": (
                f'The highest risk item currently shown is "{row.get("title", "Untitled")}". '
                f'Risk score: {safe_int(row.get("risk_score"))}, '
                f'confidence: {row.get("confidence")}, '
                f'component: {row.get("component", "Unknown")}.'
            ),
            "confidence": 0.95,
            "evidence": evidence,
        }

    # ── What changed ─────────────────────────────────────────────────────────
    if "what changed" in q or "changed since" in q or "last snapshot" in q:
        return {
            "answer": "Snapshot comparison is disabled. All items are shown from the current snapshot.",
            "confidence": 0.5,
            "evidence": evidence,
        }

    # ── What data am I looking at ─────────────────────────────────────────────
    if "what data" in q or "what am i looking at" in q or "which data" in q:
        min_dt = pd.to_datetime(df["created_at"], errors="coerce").min()
        max_dt_local = pd.to_datetime(df["created_at"], errors="coerce").max()
        answer = f"You are viewing **{len(df)}** filtered records."
        if pd.notna(min_dt) and pd.notna(max_dt_local):
            answer += f" Time range: {min_dt.date()} to {max_dt_local.date()}."
        top3_titles = ranked.head(3)["title"].fillna("Untitled").tolist()
        if top3_titles:
            answer += " Highest-priority titles: " + ", ".join(top3_titles) + "."
        return {"answer": answer, "confidence": 0.9, "evidence": evidence}

    # ── Action / triage ───────────────────────────────────────────────────────
    if re.search(r"action|what should|what do i|right now|triage|priorit", q):
        action_rows = ranked.head(3)
        lines = []
        for i, (_, row) in enumerate(action_rows.iterrows(), start=1):
            actions = normalize_actions(row.get("recommended_actions", []))
            action_str = actions[0] if actions else "Review and triage"
            lines.append(
                f"{i}. **{row.get('title', 'Untitled')}** "
                f"(risk={safe_int(row.get('risk_score'))}, conf={row.get('confidence')})\n"
                f"   - Suggested action: {action_str}"
            )
        return {
            "answer": "Top items to triage now:\n\n" + "\n".join(lines),
            "confidence": 0.9,
            "evidence": evidence,
        }

    # ── Rationale for a specific item ─────────────────────────────────────────
    if "rationale" in q or "why" in q or "reason" in q:
        # Try to match a title keyword from the question
        matched = None
        for _, row in ranked.iterrows():
            title_words = set(str(row.get("title", "")).lower().split())
            q_words = set(q.split())
            if len(title_words & q_words) >= 2:
                matched = row
                break
        if matched is None:
            matched = ranked.iloc[0]
        rationale = matched.get("rationale", matched.get("summary", "No rationale available."))
        return {
            "answer": f"**{matched.get('title', 'Untitled')}**\n\nRationale: {rationale}",
            "confidence": 0.85,
            "evidence": evidence,
        }

    # ── Generic fallback: show top 5 with stats ───────────────────────────────
    total = len(df)
    high = int((df["risk_score"] >= 4).sum())
    avg = round(df["risk_score"].mean(), 2)
    top5 = ranked.head(5)
    lines = []
    for i, (_, row) in enumerate(top5.iterrows(), start=1):
        lines.append(
            f"{i}. {row.get('title', 'Untitled')} "
            f"(risk={safe_int(row.get('risk_score'))}, conf={row.get('confidence')}, component={row.get('component', '')})"
        )
    return {
        "answer": (
            f"Current dashboard summary: **{total}** items, **{high}** high-risk (≥4), average risk **{avg}**.\n\n"
            "Top items by risk:\n\n" + "\n".join(lines)
        ),
        "confidence": 0.75,
        "evidence": evidence,
    }


question = st.text_input("Ask a question about risk insights, changes, or signals")

if st.button("Ask") and question.strip():
    with st.spinner("Answering from current dashboard data..."):
        result = answer_from_dashboard(question, df, top_k=5)
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
                st.markdown(f"**Source {idx}: {ev.get('title', 'Untitled')}**")
                st.write(ev.get("chunk_text", "")[:1000])

                meta = {
                    "source_id": ev.get("source_id"),
                    "snapshot_id": ev.get("snapshot_id"),
                    "kind": ev.get("kind"),
                    "risk_score": ev.get("risk_score"),
                    "component": ev.get("component"),
                    "category": ev.get("category"),
                    "score": ev.get("score"),
                    "created_at": ev.get("created_at"),
                }
                st.json(meta)