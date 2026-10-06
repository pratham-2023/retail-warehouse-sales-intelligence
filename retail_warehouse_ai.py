import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Retail & Warehouse Intelligence", page_icon="📊", layout="wide")

DATA_FILE = "Retail and wherehouse Sale.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["SUPPLIER"] = df["SUPPLIER"].fillna("Unknown Supplier")
    numeric_cols = ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["MONTH_NAME"] = pd.to_datetime(df["MONTH"], format="%m").dt.strftime("%b")
    df["TOTAL ACTIVITY"] = (
        df["RETAIL SALES"] + df["RETAIL TRANSFERS"] + df["WAREHOUSE SALES"]
    )
    df["RETAIL_SHARE"] = np.where(
        df["RETAIL SALES"] + df["WAREHOUSE SALES"] > 0,
        df["RETAIL SALES"] / (df["RETAIL SALES"] + df["WAREHOUSE SALES"]),
        0
    )
    return df

@st.cache_data
def detect_anomalies(data):
    x = data[["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES", "TOTAL ACTIVITY"]].copy()
    model = IsolationForest(
        n_estimators=200, contamination=0.02, random_state=42
    )
    labels = model.fit_predict(x)
    out = data.copy()
    out["AI_ANOMALY"] = np.where(labels == -1, "Anomaly", "Normal")
    out["ANOMALY_SCORE"] = -model.decision_function(x)
    return out

if not os.path.exists(DATA_FILE):
    st.error(f"Dataset not found. Put '{DATA_FILE}' in the same folder as this Python file.")
    st.stop()

df = load_data(DATA_FILE)

st.title("Retail & Warehouse Sales Intelligence")
st.caption("AI-assisted analytics for product, supplier and inventory-activity decisions")

with st.sidebar:
    st.header("Filters")
    item_types = st.multiselect(
        "Item Type", sorted(df["ITEM TYPE"].unique()), default=sorted(df["ITEM TYPE"].unique())
    )
    months = st.multiselect(
        "Month", sorted(df["MONTH"].unique()), default=sorted(df["MONTH"].unique())
    )
    suppliers = st.multiselect(
        "Supplier (optional)", sorted(df["SUPPLIER"].unique()), default=[]
    )

filtered = df[df["ITEM TYPE"].isin(item_types) & df["MONTH"].isin(months)].copy()
if suppliers:
    filtered = filtered[filtered["SUPPLIER"].isin(suppliers)]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Retail Sales", f'{filtered["RETAIL SALES"].sum():,.1f}')
c2.metric("Warehouse Sales", f'{filtered["WAREHOUSE SALES"].sum():,.1f}')
c3.metric("Retail Transfers", f'{filtered["RETAIL TRANSFERS"].sum():,.1f}')
c4.metric("Total Activity", f'{filtered["TOTAL ACTIVITY"].sum():,.1f}')
c5.metric("Products", f'{filtered["ITEM DESCRIPTION"].nunique():,}')

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Overview", "Products & Categories", "Suppliers", "AI Insights"]
)

with tab1:
    monthly = filtered.groupby("MONTH", as_index=False)[
        ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]
    ].sum()
    long_month = monthly.melt("MONTH", var_name="Metric", value_name="Value")
    fig = px.bar(long_month, x="MONTH", y="Value", color="Metric", barmode="group",
                 title="Monthly Sales & Transfer Activity")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        cat = filtered.groupby("ITEM TYPE", as_index=False)[
            ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]
        ].sum()
        cat["TOTAL ACTIVITY"] = cat[["RETAIL SALES","RETAIL TRANSFERS","WAREHOUSE SALES"]].sum(axis=1)
        fig = px.bar(cat.sort_values("TOTAL ACTIVITY", ascending=True),
                     x="TOTAL ACTIVITY", y="ITEM TYPE", orientation="h",
                     title="Activity by Item Type")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        scatter_df = filtered.sample(min(5000, len(filtered)), random_state=42).copy()
        # Plotly marker sizes cannot be negative. Some retail-sales records
        # contain negative values, so use a non-negative display-only size.
        scatter_df["RETAIL SALES SIZE"] = scatter_df["RETAIL SALES"].clip(lower=0)
        fig = px.scatter(
            scatter_df,
            x="RETAIL TRANSFERS", y="WAREHOUSE SALES",
            size="RETAIL SALES SIZE", hover_name="ITEM DESCRIPTION",
            title="Transfers vs Warehouse Sales"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    product = filtered.groupby(["ITEM DESCRIPTION", "ITEM TYPE"], as_index=False)[
        ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]
    ].sum()
    product["TOTAL ACTIVITY"] = product[["RETAIL SALES","RETAIL TRANSFERS","WAREHOUSE SALES"]].sum(axis=1)
    top = product.nlargest(15, "TOTAL ACTIVITY")
    fig = px.bar(top.sort_values("TOTAL ACTIVITY"), x="TOTAL ACTIVITY", y="ITEM DESCRIPTION",
                 color="ITEM TYPE", orientation="h", title="Top 15 Products")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top, use_container_width=True, hide_index=True)

with tab3:
    supplier = filtered.groupby("SUPPLIER", as_index=False)[
        ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]
    ].sum()
    supplier["TOTAL ACTIVITY"] = supplier[["RETAIL SALES","RETAIL TRANSFERS","WAREHOUSE SALES"]].sum(axis=1)
    top_s = supplier.nlargest(15, "TOTAL ACTIVITY")
    fig = px.bar(top_s.sort_values("TOTAL ACTIVITY"), x="TOTAL ACTIVITY", y="SUPPLIER",
                 orientation="h", title="Top 15 Suppliers")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_s, use_container_width=True, hide_index=True)

with tab4:
    ai = detect_anomalies(filtered)
    anomalies = ai[ai["AI_ANOMALY"] == "Anomaly"].sort_values("ANOMALY_SCORE", ascending=False)
    st.subheader("AI Anomaly Detection")
    st.write(
        "Isolation Forest identifies unusual combinations of retail sales, transfers and warehouse sales. "
        "These are investigation candidates, not confirmed errors or fraud."
    )
    st.metric("Flagged records", f"{len(anomalies):,}")
    st.dataframe(
        anomalies[[
            "YEAR","MONTH","SUPPLIER","ITEM CODE","ITEM DESCRIPTION","ITEM TYPE",
            "RETAIL SALES","RETAIL TRANSFERS","WAREHOUSE SALES","TOTAL ACTIVITY","ANOMALY_SCORE"
        ]].head(50),
        use_container_width=True, hide_index=True
    )

    st.subheader("Automated Business Recommendations")
    cat = filtered.groupby("ITEM TYPE")[["RETAIL SALES","WAREHOUSE SALES","RETAIL TRANSFERS"]].sum()
    if len(cat):
        warehouse_leader = cat["WAREHOUSE SALES"].idxmax()
        retail_leader = cat["RETAIL SALES"].idxmax()
        st.write(f"• **Retail focus:** {retail_leader} leads the selected categories in retail sales.")
        st.write(f"• **Warehouse focus:** {warehouse_leader} has the highest warehouse sales; inventory planning should prioritize this category.")
    transfer_total = filtered["RETAIL TRANSFERS"].sum()
    warehouse_total = filtered["WAREHOUSE SALES"].sum()
    if warehouse_total > 0:
        ratio = transfer_total / warehouse_total
        st.write(f"• **Transfer intensity:** transfers are {ratio:.1%} of warehouse sales in the current selection; investigate products with unusually high transfer activity.")
    st.write("• **AI review:** inspect flagged anomalies before operational decisions; anomalies can represent legitimate high-volume products.")

st.divider()
st.caption("Dataset: Retail Sales Data with Seasonal Trends & Marketing (Kaggle). The supplied file contains 30,000 records from 2020 and four months (January, March, July and September).")
