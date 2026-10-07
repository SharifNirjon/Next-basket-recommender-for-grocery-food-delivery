"""Demo UI: pick a user, see their purchase history next to the API's recommendations.

Run: streamlit run ui/streamlit_app.py  (API must be running; API_URL env var, default :8000)
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Next-basket recommender", layout="wide")
st.title("Next-basket recommender")

try:
    meta = httpx.get(f"{API_URL}/metadata", timeout=5).json()
except httpx.HTTPError as exc:
    st.error(f"API not reachable at {API_URL}: {exc}")
    st.stop()
st.caption(
    f"model `{meta['model_version']}` - trained {meta['trained_at']} - {meta['n_users']:,} users"
)

with st.sidebar:
    user_id = st.number_input("user_id", min_value=1, value=1, step=1)
    k = st.slider("k", 1, 50, 10)
    use_ctx = st.checkbox("set order context", value=False)
    params: dict[str, float] = {"k": k}
    if use_ctx:
        params["hour"] = st.slider("hour of day", 0, 23, 10)
        params["dow"] = st.slider("day of week", 0, 6, 0)
        params["days_since_prior"] = st.slider("days since previous order", 0, 30, 7)

left, right = st.columns(2)
with left:
    st.subheader("Purchase history (top items)")
    h = httpx.get(f"{API_URL}/users/{user_id}/history", params={"n": 20}, timeout=5)
    if h.status_code == 404:
        st.info("Unknown user - no history.")
    else:
        st.dataframe(pd.DataFrame(h.json()["items"]), hide_index=True, use_container_width=True)

with right:
    st.subheader("Recommendations")
    r = httpx.get(f"{API_URL}/recommend/{user_id}", params=params, timeout=5)
    if r.status_code != 200:
        st.error(r.text)
    else:
        body = r.json()
        if body["fallback"]:
            st.warning("Unknown user: popularity fallback")
        elif body["context"]:
            c = body["context"]
            st.caption(
                f"context ({c['source']}): hour={c['hour']}, dow={c['dow']}, "
                f"days_since_prior={c['days_since_prior']:.1f}"
            )
        st.dataframe(pd.DataFrame(body["items"]), hide_index=True, use_container_width=True)
