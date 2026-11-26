"""ChurnIQ – Customer Churn Prediction System (Streamlit)
Models: Decision Tree · Random Forest · Logistic Regression · KNN · Linear Regression
Validation: hold-out split + Stratified K-Fold cross-validation
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data import CATEGORICAL, DATA_PATH, load_data, load_raw_data, regression_xy
from core.models import (MODEL_NAMES, SCORING, feature_importance, kfold_cv, knn_k_sweep,
                         nearest_neighbours, regression_cv, train_all, train_regression)
from core.ui import GREEN, PALETTE, RED, gauge, hero, inject_css, kpi, risk_of

st.set_page_config(page_title="ChurnIQ – Churn Prediction", page_icon="📉", layout="wide")
inject_css()

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## 📉 ChurnIQ")
PAGES = ["🏠 Dashboard", "🔎 Data Explorer", "🧹 Data Preprocessing", "🎯 Predict Churn", "💲 Predict Charges",
         "📊 Model Performance", "🔁 K-Fold Validation", "🧭 KNN Lab", "📄 Reports"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
with st.sidebar.expander("⚙️ Model settings", expanded=False):
    knn_k = st.slider("KNN neighbours (K)", 1, 51, 11, step=2)
    knn_weights = st.selectbox("KNN weights", ["uniform", "distance"])
    knn_metric = st.selectbox("KNN distance", ["minkowski", "manhattan", "euclidean"])
    test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
    upload = st.file_uploader("Use your own CSV (same columns)", type="csv")

df = load_data(upload.getvalue() if upload else None)
bundle = train_all(df, knn_k, knn_weights, knn_metric, test_size)
M = bundle["models"]
metrics_df = pd.DataFrame({n: M[n]["metrics"] for n in MODEL_NAMES}).T
best_model = metrics_df["F1"].idxmax()


def customer_form(prefix: str, with_charges: bool):
    """Interactive customer input form -> one-row DataFrame."""
    c1, c2, c3 = st.columns(3)
    d = {}
    with c1:
        st.markdown("**👤 Profile**")
        d["gender"] = st.selectbox("Gender", ["Female", "Male"], key=prefix + "g")
        d["SeniorCitizen"] = st.selectbox("Senior citizen", ["No", "Yes"], key=prefix + "s")
        d["Partner"] = st.selectbox("Partner", ["No", "Yes"], key=prefix + "p")
        d["Dependents"] = st.selectbox("Dependents", ["No", "Yes"], key=prefix + "d")
        d["tenure"] = st.slider("Tenure (months)", 0, 72, 12, key=prefix + "t")
    with c2:
        st.markdown("**📡 Services**")
        d["PhoneService"] = st.selectbox("Phone service", ["Yes", "No"], key=prefix + "ph")
        d["MultipleLines"] = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"], key=prefix + "ml")
        d["InternetService"] = st.selectbox("Internet", ["Fiber optic", "DSL", "No"], key=prefix + "i")
        no_net = "No internet service"
        opts = ["No", "Yes", no_net]
        d["OnlineSecurity"] = st.selectbox("Online security", opts, key=prefix + "os")
        d["OnlineBackup"] = st.selectbox("Online backup", opts, key=prefix + "ob")
        d["DeviceProtection"] = st.selectbox("Device protection", opts, key=prefix + "dp")
    with c3:
        st.markdown("**💳 Contract & billing**")
        d["TechSupport"] = st.selectbox("Tech support", opts, key=prefix + "ts")
        d["StreamingTV"] = st.selectbox("Streaming TV", opts, key=prefix + "tv")
        d["StreamingMovies"] = st.selectbox("Streaming movies", opts, key=prefix + "sm")
        d["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key=prefix + "c")
        d["PaperlessBilling"] = st.selectbox("Paperless billing", ["Yes", "No"], key=prefix + "pb")
        d["PaymentMethod"] = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"], key=prefix + "pm")
        if with_charges:
            d["MonthlyCharges"] = st.number_input("Monthly charges ($)", 15.0, 130.0, 70.0, 0.5, key=prefix + "mc")
            d["TotalCharges"] = float(d["tenure"]) * d["MonthlyCharges"]
    return pd.DataFrame([d])


def retention_actions(r: pd.Series):
    a = []
    if r["Contract"] == "Month-to-month": a.append("Offer a discounted 1- or 2-year contract upgrade.")
    if r["InternetService"] == "Fiber optic": a.append("Check fibre service quality and price competitiveness.")
    if r["TechSupport"] == "No" and r["InternetService"] != "No": a.append("Bundle free Tech Support for 3 months.")
    if r["OnlineSecurity"] == "No" and r["InternetService"] != "No": a.append("Offer Online Security as a free trial add-on.")
    if r["PaymentMethod"] == "Electronic check": a.append("Incentivise automatic payment (bank/credit card).")
    if r["tenure"] < 12: a.append("Launch a first-year onboarding & loyalty programme.")
    return a or ["Customer looks stable – maintain regular engagement."]


# ================================================================== DASHBOARD
if page == PAGES[0]:
    hero("Customer Churn Prediction System", "Decision Tree · Random Forest · Logistic Regression · KNN · K-Fold validation")
    c = st.columns(5)
    kpi(c[0], "Customers", f"{len(df):,}", "records in dataset")
    kpi(c[1], "Churn rate", f"{df.ChurnFlag.mean()*100:.1f}%", f"{df.ChurnFlag.sum():,} churned")
    kpi(c[2], "Avg monthly bill", f"${df.MonthlyCharges.mean():.2f}", "per customer")
    kpi(c[3], "Avg tenure", f"{df.tenure.mean():.0f} mo", "customer lifetime")
    kpi(c[4], "Best model", best_model, f"F1 = {metrics_df.loc[best_model,'F1']:.2f}")
    st.write("")
    a, b = st.columns(2)
    with a:
        fig = px.pie(df, names="Churn", hole=.6, title="Churn distribution",
                     color="Churn", color_discrete_map={"No": GREEN, "Yes": RED})
        st.plotly_chart(fig, width="stretch")
    with b:
        g = df.groupby("Contract").ChurnFlag.mean().mul(100).reset_index()
        fig = px.bar(g, x="Contract", y="ChurnFlag", title="Churn rate by contract (%)", text_auto=".1f",
                     color="ChurnFlag", color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"])
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn %")
        st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        fig = px.histogram(df, x="tenure", color="Churn", barmode="overlay", nbins=36, opacity=.75,
                           title="Tenure distribution", color_discrete_map={"No": GREEN, "Yes": RED})
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.box(df, x="InternetService", y="MonthlyCharges", color="Churn", title="Monthly charges by internet service",
                     color_discrete_map={"No": GREEN, "Yes": RED})
        st.plotly_chart(fig, width="stretch")
    st.subheader("💡 Business insights")
    mtm = df[df.Contract == "Month-to-month"].ChurnFlag.mean() * 100
    two = df[df.Contract == "Two year"].ChurnFlag.mean() * 100
    fib = df[df.InternetService == "Fiber optic"].ChurnFlag.mean() * 100
    new = df[df.tenure < 12].ChurnFlag.mean() * 100
    st.info(f"Month-to-month customers churn at **{mtm:.1f}%** vs **{two:.1f}%** on two-year contracts.")
    st.info(f"Fibre-optic customers churn at **{fib:.1f}%**; customers in their first year churn at **{new:.1f}%**.")

# ================================================================== DATA EXPLORER
elif page == PAGES[1]:
    hero("Data Explorer", "Filter, search and slice the customer base")
    f = st.columns(5)
    sq = f[0].text_input("Customer ID contains")
    fc = f[1].multiselect("Contract", df.Contract.unique())
    fi = f[2].multiselect("Internet", df.InternetService.unique())
    fch = f[3].multiselect("Churn", ["Yes", "No"])
    ften = f[4].slider("Tenure range", 0, 72, (0, 72))
    v = df[df.tenure.between(*ften)]
    if sq: v = v[v.customerID.str.contains(sq, case=False)]
    if fc: v = v[v.Contract.isin(fc)]
    if fi: v = v[v.InternetService.isin(fi)]
    if fch: v = v[v.Churn.isin(fch)]
    st.caption(f"{len(v):,} of {len(df):,} customers · churn rate {v.ChurnFlag.mean()*100 if len(v) else 0:.1f}%")
    st.dataframe(v.drop(columns="ChurnFlag"), width="stretch", height=380)
    st.download_button("⬇️ Download filtered CSV", v.drop(columns="ChurnFlag").to_csv(index=False), "filtered_customers.csv")
    st.subheader("Segment analysis")
    by = st.selectbox("Group by", ["Contract", "InternetService", "PaymentMethod", "Dependents", "SeniorCitizen", "TechSupport"])
    seg = df.groupby(by).agg(Customers=("ChurnFlag", "size"), ChurnRate=("ChurnFlag", "mean"),
                             AvgMonthly=("MonthlyCharges", "mean")).reset_index()
    seg["ChurnRate"] = (seg.ChurnRate * 100).round(1)
    a, b = st.columns([3, 2])
    a.plotly_chart(px.bar(seg, x=by, y="ChurnRate", color="ChurnRate", text="ChurnRate",
                          color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"], title=f"Churn % by {by}"), width="stretch")
    b.dataframe(seg.round(2), width="stretch", hide_index=True)

# ================================================================== DATA PREPROCESSING
elif page == PAGES[2]:
    hero("Data Preparation & Pre-processing",
         "Inspect raw data quality, understand every cleaning step and download the prepared dataset")

    raw_df = load_raw_data(upload.getvalue() if upload else None)
    cleaned_df = df.copy()
    missing_table = pd.DataFrame({
        "Missing before cleaning": raw_df.isna().sum(),
        "Blank text values": raw_df.select_dtypes(include="object").apply(
            lambda col: col.astype(str).str.strip().eq("").sum()
        ).reindex(raw_df.columns, fill_value=0),
        "Missing after cleaning": cleaned_df.isna().sum().reindex(raw_df.columns, fill_value=0)
    })
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Raw records", f"{len(raw_df):,}", "rows loaded")
    kpi(c2, "Raw attributes", f"{raw_df.shape[1]}", "columns")
    kpi(c3, "Duplicate rows", f"{raw_df.duplicated().sum():,}", "exact duplicates")
    kpi(c4, "TotalCharges blanks", f"{raw_df['TotalCharges'].astype(str).str.strip().eq('').sum():,}",
        "converted to missing values")

    st.subheader("1. Raw dataset quality")
    st.caption("The table checks both actual null values and empty text cells. Empty TotalCharges entries are not always detected by isnull() until converted to numeric.")
    st.dataframe(missing_table, width="stretch")

    st.subheader("2. Preprocessing operations")
    steps = [
        ("Column and text cleanup", "Trim spaces from column names and text values so categories such as 'Yes ' and 'Yes' are treated consistently."),
        ("Data type conversion", "Convert TotalCharges from text to numeric. Invalid or blank entries become NaN."),
        ("Missing-value treatment", "Fill missing TotalCharges values using the median. Median is less sensitive to extreme values than mean."),
        ("Duplicate check", "Identify and remove exact duplicate rows, if present."),
        ("Identifier removal", "Exclude customerID from model inputs because it identifies a customer but does not represent a useful customer behaviour measurement."),
        ("Categorical encoding", "One-hot encode categories such as Contract, InternetService and PaymentMethod so algorithms can process them."),
        ("Numerical scaling", "Standardize numerical features for scale-sensitive algorithms such as Logistic Regression and KNN."),
        ("Train/test separation", "Split data into training and testing sets. Fit encoders and scalers on training data only to prevent data leakage.")
    ]
    for title, reason in steps:
        with st.expander(title):
            st.write(reason)

    st.subheader("3. Cleaned dataset preview")
    st.write(f"Cleaned shape: **{cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns**")
    st.dataframe(cleaned_df.head(10), width="stretch")
    st.download_button(
        "⬇️ Download cleaned dataset",
        cleaned_df.drop(columns=["ChurnFlag"], errors="ignore").to_csv(index=False),
        "telco_churn_cleaned.csv",
        mime="text/csv"
    )

    st.subheader("4. Descriptive statistics")
    st.write("At least five statistical measures are provided for numerical attributes.")
    st.dataframe(cleaned_df.select_dtypes(include="number").describe().T, width="stretch")

    st.subheader("5. Feature selection used in this project")
    left, right = st.columns(2)
    with left:
        st.markdown("**Churn classification**")
        st.write("Target: `Churn` (Yes / No). Customer profile, service, tenure and billing fields are used as predictors.")
    with right:
        st.markdown("**Monthly charges regression**")
        st.write("Target: `MonthlyCharges`. `customerID`, `TotalCharges` and `Churn` are excluded from predictors.")
    st.info("The model pipelines apply one-hot encoding and numerical scaling after the train/test split. This ensures preprocessing is learned from training data and then applied to unseen test data.")

    st.subheader("6. Run the full preprocessing script")
    st.code("python data_preprocessing.py", language="bash")
    st.caption("The script saves the cleaned CSV and encoded train/test feature and target files in data/processed/.")

# ================================================================== PREDICT CHURN
elif page == PAGES[3]:
    hero("Predict Customer Churn", "Pick an algorithm, describe a customer, get a risk score")
    algo = st.radio("Algorithm", MODEL_NAMES, horizontal=True, index=MODEL_NAMES.index(best_model))
    with st.form("churn_form"):
        row = customer_form("c_", with_charges=True)
        go_btn = st.form_submit_button("⚡ Predict churn")
    if go_btn:
        p = float(M[algo]["pipe"].predict_proba(row)[:, 1][0])
        label, css, _ = risk_of(p)
        a, b = st.columns([2, 3])
        with a:
            st.plotly_chart(gauge(p, f"{algo} – churn probability"), width="stretch")
            st.markdown(f'<span class="pill {css}">{label} risk</span> &nbsp; Predicted: **{"Will churn" if p >= .5 else "Will stay"}**',
                        unsafe_allow_html=True)
        with b:
            cmp = pd.DataFrame({n: float(M[n]["pipe"].predict_proba(row)[:, 1][0]) * 100 for n in MODEL_NAMES}.items(),
                               columns=["Model", "Churn %"])
            fig = px.bar(cmp, x="Churn %", y="Model", orientation="h", color="Model", color_discrete_map=PALETTE,
                         text_auto=".1f", title="All models – side by side", range_x=[0, 100])
            fig.update_layout(showlegend=False, height=290)
            st.plotly_chart(fig, width="stretch")
        t1, t2, t3 = st.tabs(["🛠 Retention actions", "📌 Key drivers", "🧭 Similar customers (KNN)"])
        with t1:
            for x in retention_actions(row.iloc[0]): st.success(x)
        with t2:
            fi_df = feature_importance(M["Random Forest"]["pipe"] if algo in ("KNN", "Decision Tree") else M[algo]["pipe"])
            fi_df = feature_importance(M["Random Forest"]["pipe"]) if algo == "KNN" else fi_df
            st.plotly_chart(px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                   title="Top global drivers of churn"), width="stretch")
        with t3:
            nb = nearest_neighbours(M["KNN"]["pipe"], bundle["X_train"], bundle["y_train"], row, 10)
            share = (nb.Churned == "Yes").mean() * 100
            st.metric("Of the 10 most similar customers, % who churned", f"{share:.0f}%")
            st.dataframe(nb, width="stretch")

# ================================================================== PREDICT CHARGES
elif page == PAGES[4]:
    hero("Predict Monthly Charges", "Linear Regression trained on services, contract & tenure")
    reg = train_regression(df)
    c = st.columns(3)
    kpi(c[0], "R² (test)", f"{reg['r2']:.3f}", "variance explained")
    kpi(c[1], "MAE", f"${reg['mae']:.2f}", "mean absolute error")
    kpi(c[2], "RMSE", f"${reg['rmse']:.2f}", "root mean squared error")
    st.write("")
    with st.form("reg_form"):
        row = customer_form("r_", with_charges=False)
        go_btn = st.form_submit_button("🧮 Predict monthly charges")
    if go_btn:
        amt = float(reg["pipe"].predict(row)[0])
        a, b = st.columns([1, 2])
        a.markdown(f'<div class="kpi"><div class="l">Estimated monthly bill</div><div class="v">${amt:,.2f}</div>'
                   f'<div class="s">≈ ${amt*12:,.0f} per year</div></div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="MonthlyCharges", nbins=40, title="Where this customer sits in the bill distribution")
        fig.add_vline(x=amt, line_color="#ec4899", line_width=3, annotation_text="This customer")
        b.plotly_chart(fig, width="stretch")
    st.subheader("Actual vs predicted (test set)")
    fig = px.scatter(x=reg["y_test"], y=reg["pred"], opacity=.5, labels=dict(x="Actual", y="Predicted"))
    lo, hi = float(reg["y_test"].min()), float(reg["y_test"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#ec4899", dash="dash"), name="Perfect"))
    st.plotly_chart(fig, width="stretch")

# ================================================================== MODEL PERFORMANCE
elif page == PAGES[5]:
    hero("Model Performance", f"Hold-out evaluation on {int(test_size*100)}% unseen customers")
    st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0, color="#4c1d95"), width="stretch")
    st.success(f"🏆 Best F1 score: **{best_model}** ({metrics_df.loc[best_model,'F1']:.3f})")
    sel = st.multiselect("Metrics to chart", list(SCORING), default=list(SCORING))
    long = metrics_df[sel].reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    st.plotly_chart(px.bar(long, x="Metric", y="Score", color="index", barmode="group",
                           color_discrete_map=PALETTE, labels={"index": "Model"}, range_y=[0, 1]), width="stretch")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for n in MODEL_NAMES:
            fig.add_trace(go.Scatter(x=M[n]["fpr"], y=M[n]["tpr"], name=f"{n} (AUC {M[n]['metrics']['ROC-AUC']:.2f})",
                                     line=dict(color=PALETTE[n], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#64748b"), showlegend=False))
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, width="stretch")
    with b:
        cm_model = st.selectbox("Confusion matrix for", MODEL_NAMES, index=MODEL_NAMES.index(best_model))
        cm = M[cm_model]["cm"]
        fig = px.imshow(cm, text_auto=True, x=["Stay", "Churn"], y=["Stay", "Churn"], color_continuous_scale="Purples",
                        labels=dict(x="Predicted", y="Actual"), title=f"{cm_model} – confusion matrix")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Feature importance")
    fm = st.radio("Model", ["Random Forest", "Decision Tree", "Logistic Regression"], horizontal=True)
    st.plotly_chart(px.bar(feature_importance(M[fm]["pipe"], 15), x="Importance", y="Feature", orientation="h"), width="stretch")

# ================================================================== K-FOLD
elif page == PAGES[6]:
    hero("K-Fold Cross-Validation", "A more reliable estimate: every customer is used for both training and testing")
    c = st.columns(3)
    n_splits = c[0].slider("Number of folds (K)", 3, 10, 5)
    shuffle = c[1].toggle("Shuffle before splitting", True)
    metric_pick = c[2].selectbox("Metric to inspect", list(SCORING), index=3)
    cv = kfold_cv(df, n_splits, shuffle, knn_k, knn_weights, knn_metric)
    summ = cv.groupby(["Model", "Metric"]).Score.agg(["mean", "std"]).reset_index()
    pivot = summ.pivot(index="Model", columns="Metric", values="mean")[list(SCORING)]
    pstd = summ.pivot(index="Model", columns="Metric", values="std")[list(SCORING)]
    show = pivot.map(lambda x: f"{x:.3f}") + " ± " + pstd.map(lambda x: f"{x:.3f}")
    st.markdown("**Mean ± std across folds**")
    st.dataframe(show, width="stretch")
    best_cv = pivot[metric_pick].idxmax()
    st.success(f"🏆 Best {metric_pick} under {n_splits}-fold CV: **{best_cv}** ({pivot.loc[best_cv, metric_pick]:.3f})")
    sub = cv[cv.Metric == metric_pick]
    a, b = st.columns(2)
    with a:
        fig = px.line(sub, x="Fold", y="Score", color="Model", markers=True, color_discrete_map=PALETTE,
                      title=f"{metric_pick} per fold")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.plotly_chart(px.box(sub, x="Model", y="Score", color="Model", points="all",
                               color_discrete_map=PALETTE, title=f"{metric_pick} spread").update_layout(showlegend=False),
                        width="stretch")
    with st.expander("📚 How K-Fold works"):
        st.markdown(f"""
1. The dataset is split into **{n_splits} equal folds** (stratified, so each keeps the same churn ratio).
2. For each fold: train on the other {n_splits-1} folds, test on the held-out one.
3. The {n_splits} scores are averaged; a small **±std** means the model is stable.
""")
    st.subheader("Linear Regression – K-Fold R²")
    rcv = regression_cv(df, n_splits)
    st.plotly_chart(px.bar(rcv, x="Fold", y="R²", text_auto=".3f", range_y=[0, 1]), width="stretch")
    st.caption(f"Mean R² = {rcv['R²'].mean():.3f} ± {rcv['R²'].std():.3f}")
    st.download_button("⬇️ Download fold scores (CSV)", cv.to_csv(index=False), "kfold_scores.csv")

# ================================================================== KNN LAB
elif page == PAGES[7]:
    hero("KNN Lab", "K-Nearest Neighbours – find the best K with cross-validation")
    c = st.columns(4)
    kmax = c[0].slider("Max K to test", 11, 61, 41, step=2)
    folds = c[1].slider("CV folds", 3, 10, 5, key="knnfolds")
    sc = c[2].selectbox("Optimise for", list(SCORING), index=3)
    c[3].write("")
    sweep = knn_k_sweep(df, kmax, folds, knn_weights, knn_metric, sc)
    best = sweep.loc[sweep.Mean.idxmax()]
    k1, k2, k3 = st.columns(3)
    kpi(k1, "Best K", f"{int(best.K)}", f"{sc} = {best.Mean:.3f}")
    kpi(k2, "Currently selected K", f"{knn_k}", "change in sidebar ⚙️")
    kpi(k3, "Weights / distance", knn_weights, knn_metric)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep.K, y=sweep.Mean + sweep.Std, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sweep.K, y=sweep.Mean - sweep.Std, fill="tonexty", fillcolor="rgba(236,72,153,.18)",
                             line=dict(width=0), name="± std", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sweep.K, y=sweep.Mean, mode="lines+markers", name=sc, line=dict(color="#ec4899", width=3)))
    fig.add_vline(x=int(best.K), line_dash="dash", line_color=GREEN, annotation_text=f"best K={int(best.K)}")
    fig.update_layout(title=f"{sc} vs K ({folds}-fold CV)", xaxis_title="K (neighbours)", yaxis_title=sc)
    st.plotly_chart(fig, width="stretch")
    st.caption("Small K → flexible but noisy (overfits). Large K → smooth but may miss local patterns (underfits).")
    st.subheader("🔍 Inspect a customer's neighbours")
    idx = st.number_input("Test-set customer #", 0, len(bundle["X_test"]) - 1, 0)
    r = bundle["X_test"].iloc[[int(idx)]]
    nb = nearest_neighbours(M["KNN"]["pipe"], bundle["X_train"], bundle["y_train"], r, knn_k)
    actual = "Yes" if bundle["y_test"].iloc[int(idx)] == 1 else "No"
    st.write(f"Actual churn: **{actual}** · KNN predicted churn probability: **{M['KNN']['proba'][int(idx)]*100:.0f}%** "
             f"({(nb.Churned=='Yes').sum()} of {knn_k} neighbours churned)")
    st.dataframe(nb, width="stretch")

# ================================================================== REPORTS
else:
    hero("Reports & Export", "Download results for presentations and documentation")
    with st.expander("📋 Practical examination rubric coverage (A–I)", expanded=False):
        st.markdown("""
**A. Data Collection:** IBM Telco Customer Churn dataset; dataset dimensions and attribute descriptions are shown in Data Explorer.

**B. Descriptive Statistics:** Numerical summary with count, mean, standard deviation, minimum, quartiles and maximum is available in Data Preprocessing.

**C. Data Exploration & Visualisation:** Dashboard and Data Explorer provide churn distribution, contract churn rate, tenure distribution, monthly-charge boxplot and segment analysis. This Streamlit dashboard also serves as the BI/visualisation tool.

**D. Data Preparation:** Data Preprocessing page explains trimming, numeric conversion, median imputation, duplicate checks, identifier removal, encoding and scaling.

**E. Data Splitting & Cross-Validation:** Hold-out test split and Stratified K-Fold / K-Fold comparison are available in Model Performance and K-Fold Validation.

**F. Model Implementation & Tuning:** Decision Tree, Random Forest, Logistic Regression, KNN and Linear Regression are implemented. KNN K/weights/distance and tree/forest settings are configurable in the application.

**G. Model Evaluation:** Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curves, MAE, RMSE and R² are presented. Models are compared on held-out data.

**H. Deployment:** The Streamlit application can be deployed using Streamlit Community Cloud. Consider privacy, access control, monitoring and data quality before production use.

**I. Presentation & Viva:** Use the dashboard, preprocessing explanations, model comparison and report exports for the live demonstration. Prepare a separate slide deck, poster and article as required by the examination instructions.
""")
    with st.expander("🎤 Viva preparation"):
        st.markdown("""
- **What is data preprocessing?** Cleaning and transforming raw data into a form suitable for machine learning.
- **Why remove customerID?** It is an identifier and does not describe customer behaviour.
- **Why convert TotalCharges to numeric?** It is stored as text, but calculations and ML algorithms require numeric values.
- **Why use median imputation?** Median is less affected by unusually high or low values.
- **What is one-hot encoding?** It converts categories into numeric indicator columns.
- **Why scale numerical data?** It prevents large-scale features from dominating distance- or coefficient-based algorithms.
- **Why split into train and test sets?** To evaluate the model on data it did not learn from.
- **What is cross-validation?** Repeating training and validation over different folds to estimate model stability.
- **What is data leakage?** When information from test data or the target improperly influences model training.
""")
    st.markdown(f"""
<div class="card"><h4>Executive summary</h4>
<p>Dataset: <b>{len(df):,}</b> customers · churn rate <b>{df.ChurnFlag.mean()*100:.1f}%</b>.<br>
Best hold-out model: <b>{best_model}</b> (F1 {metrics_df.loc[best_model,'F1']:.3f}, ROC-AUC {metrics_df.loc[best_model,'ROC-AUC']:.3f}).<br>
KNN configuration: K={knn_k}, weights={knn_weights}, distance={knn_metric}.</p></div>""", unsafe_allow_html=True)
    st.dataframe(metrics_df.style.format("{:.3f}"), width="stretch")
    a, b, c = st.columns(3)
    a.download_button("⬇️ Metrics (CSV)", metrics_df.to_csv(), "model_metrics.csv")
    b.download_button("⬇️ Metrics (JSON)", json.dumps(metrics_df.round(4).to_dict("index"), indent=2), "model_metrics.json")
    scored = df.copy()
    scored["ChurnProbability"] = M[best_model]["pipe"].predict_proba(df[bundle["X_train"].columns])[:, 1].round(4)
    c.download_button("⬇️ Scored customers (CSV)", scored.drop(columns="ChurnFlag").to_csv(index=False), "customers_scored.csv")
    st.subheader("Top 20 at-risk customers (currently active)")
    top = scored[scored.Churn == "No"].nlargest(20, "ChurnProbability")[
        ["customerID", "Contract", "InternetService", "tenure", "MonthlyCharges", "ChurnProbability"]]
    st.dataframe(top, width="stretch", hide_index=True)
