
# app.py
# Streamlit app for polynomial regression and function analysis using World Bank data
# Place this file at the repository root alongside the 'requirements' file and deploy to Streamlit.
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime
import io

st.set_page_config(layout="wide", page_title="Latin Countries Regression & Function Analysis")

st.title("Latin Countries — Regression & Function Analysis")
st.markdown("""
This app fetches historical data (World Bank API) for selected Latin American countries and fits a polynomial regression (degree ≥ 3).
Features:
- Choose up to 3 countries from a curated list.
- Choose category (Population, Life expectancy, Birth rate, Unemployment, Education mapped 0-25, Average income).
- Editable raw data table.
- Fit a polynomial (degree ≥ 3) and display equation.
- Plot data, fitted curve, and optional extrapolation (different color).
- Function analysis: local extrema, monotonic intervals, fastest growth/decline, domain & range (interpreted using real-world units).
- Compare multiple countries or upload your own CSV of U.S. Latin-group series for comparison.
- Download raw data CSV and printer-friendly HTML report.
""")

# ---------------------
# Configuration
# ---------------------
# Up to 3 wealthier Latin American countries with good WDI coverage
COUNTRIES = {
    "Chile": "CL",
    "Uruguay": "UY",
    "Panama": "PA",
    "Costa Rica": "CR",
    "Argentina": "AR"
}

# Map app categories to World Bank indicators (WDI codes)
INDICATORS = {
    "Population": ("SP.POP.TOTL", "people"),
    "Life expectancy": ("SP.DYN.LE00.IN", "years"),
    "Birth rate (per 1000 people)": ("SP.DYN.CBRT.IN", "births per 1000 people"),
    "Average income (GDP per capita, current US$)": ("NY.GDP.PCAP.CD", "current US$"),
    "Murder rate (homicides per 100k)": ("VC.IHR.PSRC.P5", "deaths per 100k"),
    "Unemployment rate (%)": ("SL.UEM.TOTL.ZS", "percent"),
    "Education (mapped 0-25)": ("SE.SCH.LIFE", "years (mapped 0-25)")
}

MIN_YEAR = 1950  # we will warn if data before 1960 may be incomplete; WDI usually from 1960
DEFAULT_START = 1960
NOW_YEAR = datetime.now().year

# ---------------------
# Sidebar controls
# ---------------------
st.sidebar.header("Selection")
selected_countries = st.sidebar.multiselect("Choose 1–3 countries:", list(COUNTRIES.keys()), default=["Chile", "Uruguay"])
if len(selected_countries) == 0:
    st.sidebar.error("Select at least one country.")
    st.stop()
if len(selected_countries) > 3:
    st.sidebar.warning("Limiting to first three countries.")
    selected_countries = selected_countries[:3]

category = st.sidebar.selectbox("Category:", list(INDICATORS.keys()), index=0)
indicator_code, units_desc = INDICATORS[category]

start_year = st.sidebar.number_input("Start year (≥ 1950):", min_value=MIN_YEAR, max_value=NOW_YEAR, value=DEFAULT_START, step=1)
end_year = st.sidebar.number_input("End year:", min_value=MIN_YEAR, max_value=NOW_YEAR, value=NOW_YEAR, step=1)
if start_year >= end_year:
    st.sidebar.error("Start year must be less than end year.")
    st.stop()

year_increment = st.sidebar.slider("Plot resolution (years per point)", 1, 10, 1)
degree = st.sidebar.slider("Polynomial degree (>=3)", 3, 8, 3)
extrap_years = st.sidebar.number_input("Extrapolate forward (years)", min_value=0, max_value=100, value=0, step=1)
show_extrap = st.sidebar.checkbox("Show extrapolated portion in dashed line", value=True)
compare_diaspora = st.sidebar.checkbox("Compare with U.S. Latin-group data (upload CSV)", value=False)
download_report = st.sidebar.checkbox("Show download buttons for report & data", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Data source: World Bank (WDI). Some series begin in 1960; missing values will appear as blank.")

# ---------------------
# Functions to fetch and prepare data
# ---------------------
@st.cache_data(show_spinner=False)
def fetch_wb_series(country_iso2, indicator, start, end):
    url = f"https://api.worldbank.org/v2/country/{country_iso2}/indicator/{indicator}?date={start}:{end}&format=json&per_page=20000"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.Series(dtype=float)
    try:
        data = r.json()
    except Exception:
        return pd.Series(dtype=float)
    if not isinstance(data, list) or len(data) < 2:
        return pd.Series(dtype=float)
    entries = data[1]
    rows = []
    for e in entries:
        y = e.get("date")
        v = e.get("value")
        if v is not None:
            try:
                rows.append((int(y), float(v)))
            except:
                continue
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["year", "value"]).set_index("year").sort_index()
    return df["value"]

def map_education_to_025(series):
    # The indicator SE.SCH.LIFE is expected years of schooling; map to 0-25 scale linearly (25 as max)
    return (series / 25.0) * 25.0

def assemble_data_table(country_series_dict, start, end, category):
    years = list(range(start, end+1))
    df = pd.DataFrame({"year": years})
    for country, series in country_series_dict.items():
        s_full = series.reindex(years)
        if category == "Education (mapped 0-25)":
            s_full = map_education_to_025(s_full)
        df[country] = s_full.values
    return df

# Fetch data
with st.spinner("Fetching data from World Bank..."):
    raw_series = {}
    for c in selected_countries:
        iso = COUNTRIES[c]
        s = fetch_wb_series(iso, indicator_code, start_year, end_year)
        raw_series[c] = s

data_df = assemble_data_table(raw_series, start_year, end_year, category)

st.header("Raw data (editable)")
st.write("Edit values in the table if you want to correct or add numbers. Press the 'Update model' button to re-fit models with edits.")

edited = st.experimental_data_editor(data_df, num_rows="dynamic")
st.write("Edited table preview (you can download CSV below).")

# Download CSV
csv_bytes = edited.to_csv(index=False).encode("utf-8")
if download_report:
    st.download_button("Download edited data (CSV)", csv_bytes, file_name="edited_time_series.csv", mime="text/csv")

# ---------------------
# Modeling helpers
# ---------------------
def prepare_xy(edited_df, country):
    df = edited_df[["year", country]].dropna()
    if df.empty:
        return np.array([]), np.array([])
    x = df["year"].to_numpy(dtype=float)
    y = df[country].to_numpy(dtype=float)
    return x, y

def fit_poly(x, y, deg):
    # center x to improve numeric stability
    xm = x.mean()
    x_center = x - xm
    coeffs = np.polyfit(x_center, y, deg)
    p = np.poly1d(coeffs)
    return {"poly": p, "x_mean": xm, "x": x, "y": y}

def function_analysis(model):
    p = model["poly"]
    xm = model["x_mean"]
    dp = np.polyder(p)
    ddp = np.polyder(dp)
    # critical points: real roots of dp
    crit_roots = np.roots(dp)
    crit_real = crit_roots[np.isclose(crit_roots.imag, 0, atol=1e-6)].real
    crit_info = []
    for r in crit_real:
        year_full = r + xm
        conc = ddp(r)
        typ = "inflection/flat"
        if conc > 0:
            typ = "local minimum"
        elif conc < 0:
            typ = "local maximum"
        crit_info.append({"year_rel": float(r), "year": float(year_full), "second_derivative": float(conc), "type": typ})
    return {"dp": dp, "ddp": ddp, "crit": crit_info}

# Fit models for selected countries
models = {}
for c in selected_countries:
    x, y = prepare_xy(edited, c)
    if x.size == 0:
        st.warning(f"No data points for {c}. Skipping model.")
        continue
    model = fit_poly(x, y, degree)
    models[c] = model

if not models:
    st.stop()

# ---------------------
# Plotting
# ---------------------
st.header("Scatter plot with polynomial fit")
# determine common plotting grid
min_year = min([m["x"].min() for m in models.values()])
max_year = max([m["x"].max() for m in models.values()])
plot_end = max_year + extrap_years
x_plot = np.arange(min_year, plot_end+1, year_increment)

fig = go.Figure()
palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

for i, (country, model) in enumerate(models.items()):
    color = palette[i % len(palette)]
    # scatter
    fig.add_trace(go.Scatter(x=model["x"], y=model["y"], mode="markers", name=f"{country} data", marker=dict(color=color)))
    # fitted curve on observed range
    xm = model["x_mean"]
    x_rel = x_plot - xm
    y_fit = model["poly"](x_rel)
    # separate observed vs extrapolated for this country
    mask_observed = x_plot <= model["x"].max()
    fig.add_trace(go.Scatter(x=x_plot[mask_observed], y=y_fit[mask_observed], mode="lines", name=f"{country} fit (observed)", line=dict(color=color, width=2)))
    # extrapolated portion
    if extrap_years > 0 and show_extrap:
        mask_extra = x_plot > model["x"].max()
        if mask_extra.any():
            fig.add_trace(go.Scatter(x=x_plot[mask_extra], y=y_fit[mask_extra], mode="lines", name=f"{country} extrapolated", line=dict(color=color, width=2, dash="dash")))

fig.update_layout(title=f"{category} — data and polynomial fit (degree {degree})", xaxis_title="Year", yaxis_title=f"{category} ({units_desc})", height=600)
st.plotly_chart(fig, use_container_width=True)

# ---------------------
# Equations and analysis
# ---------------------
st.header("Model equations & function analysis (plain-English interpretation)")

for country, model in models.items():
    st.subheader(country)
    p = model["poly"]
    xm = model["x_mean"]
    # readable equation in variable t = year - xm
    coeffs = p.coeffs
    deg_p = len(coeffs)-1
    terms = []
    for idx, coef in enumerate(coeffs):
        power = deg_p - idx
        if power == 0:
            terms.append(f"{coef:.6g}")
        elif power == 1:
            terms.append(f"{coef:.6g}*(year - {xm:.1f})")
        else:
            terms.append(f"{coef:.6g}*(year - {xm:.1f})^{power}")
    eq = " + ".join(terms)
    st.code(f"y = {eq}", language="text")

    # function analysis
    fa = function_analysis(model)
    crit = fa["crit"]
    if crit:
        st.write("Critical points (from polynomial derivative roots):")
        for item in crit:
            yr = item["year"]
            typ = item["type"]
            dd = item["second_derivative"]
            # format explanatory sentence
            st.write(f"- The {category.lower()} of {country} reached a {typ} around **{yr:.2f}**. (second derivative ≈ {dd:.3g})")
    else:
        st.write("No real critical points found for this model.")

    # increasing / decreasing (sample derivative sign on a grid)
    dp = fa["dp"]
    sample_x = np.linspace(min_year-5, max_year+5+extrap_years, 400)
    dp_vals = dp(sample_x - xm)
    inc_intervals = []
    dec_intervals = []
    sign = None
    start_interval = None
    for xi, val in zip(sample_x, dp_vals):
        sgn = np.sign(val)
        if sign is None:
            sign = sgn
            start_interval = xi
        elif sgn != sign:
            if sign > 0:
                inc_intervals.append((start_interval, xi))
            elif sign < 0:
                dec_intervals.append((start_interval, xi))
            sign = sgn
            start_interval = xi
    # close last
    if sign is not None:
        if sign > 0:
            inc_intervals.append((start_interval, sample_x[-1]))
        elif sign < 0:
            dec_intervals.append((start_interval, sample_x[-1]))

    def fmt(regs):
        if not regs:
            return "None identified (approx)"
        return ", ".join([f"[{r[0]:.1f}, {r[1]:.1f}]" for r in regs])

    st.write(f"- Intervals where model is increasing (approx): {fmt(inc_intervals)}")
    st.write(f("- Intervals where model is decreasing (approx): {fmt(dec_intervals)}")

    # fastest change via second derivative maxima/minima
    ddp = fa["ddp"]
    dd_vals = ddp(sample_x - xm)
    idx_max = np.argmax(dd_vals)
    idx_min = np.argmin(dd_vals)
    st.write(f"- The {category.lower()} was increasing fastest around year ≈ **{sample_x[idx_max]:.2f}**.")
    st.write(f("- The {category.lower()} was decreasing fastest around year ≈ **{sample_x[idx_min]:.2f}**.")

    # domain & range (approx)
    y_sample = p(sample_x - xm)
    st.write(f"- Analysis domain (approx): [{sample_x[0]:.1f}, {sample_x[-1]:.1f}]")
    st.write(f"- Approximate range on analyzed domain: [{np.nanmin(y_sample):.3g}, {np.nanmax(y_sample):.3g}]")

    # conjectures: simple heuristic based on big changes (user must verify historically)
    if len(model["x"]) >= 3:
        obs_df = pd.DataFrame({"year": model["x"], "value": model["y"]}).sort_values("year")
        obs_df["pct_change"] = obs_df["value"].pct_change().abs()
        top = obs_df.sort_values("pct_change", ascending=False).head(3)
        st.write("- Notable observed changes (largest year-to-year percent change):")
        for _, row in top.iterrows():
            if pd.isna(row["pct_change"]):
                continue
            st.write(f"  - Around **{int(row['year'])}** change of {row['pct_change']*100:.1f}% — consider researching events (policy, economy, conflict, pandemic) that may explain this.")
    st.markdown("---")

# ---------------------
# Interpolation / Extrapolation & average rate of change
# ---------------------
st.header("Interpolation / Extrapolation and Average Rate of Change")

pred_country = st.selectbox("Pick country for prediction:", options=list(models.keys()))
pred_model = models[pred_country]
pred_year = st.number_input("Year to evaluate (can be outside data):", value=int(pred_model["x"].max()), min_value=1900, max_value=2100, step=1)
pred_val = pred_model["poly"](pred_year - pred_model["x_mean"])
st.write(f"Model predicts for **{pred_country}** in **{pred_year}**: **{pred_val:.6g} {units_desc}**")
if pred_year > pred_model["x"].max() or pred_year < pred_model["x"].min():
    st.warning("This is an extrapolation (outside observed data). Use caution — uncertainty rising with distance from data.")

# average rate of change between two years
st.write("Average rate of change between two years (model-based):")
yr_a = st.number_input("Year A:", value=int(pred_model["x"].min()), step=1, key="yrA")
yr_b = st.number_input("Year B:", value=int(pred_model["x"].max()), step=1, key="yrB")
if yr_a != yr_b:
    val_a = pred_model["poly"](yr_a - pred_model["x_mean"])
    val_b = pred_model["poly"](yr_b - pred_model["x_mean"])
    avg_rate = (val_b - val_a) / (yr_b - yr_a)
    st.write(f"- Average rate of change from {yr_a} to {yr_b}: **{avg_rate:.6g} {units_desc} per year**")
else:
    st.info("Pick two different years to compute rate.")

# ---------------------
# Compare with uploaded CSV (U.S. Latin groups)
# ---------------------
st.header("Compare with U.S. Latin-group data (optional upload)")
st.write("Upload a CSV with columns: year, group_name1, group_name2, ... Year must be integer years.")
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded is not None:
    try:
        uploaded_df = pd.read_csv(uploaded)
        if "year" not in uploaded_df.columns:
            st.error("CSV must contain 'year' column.")
        else:
            st.write("Preview of uploaded CSV:")
            st.dataframe(uploaded_df.head())
            groups = [c for c in uploaded_df.columns if c != "year"]
            chosen = st.multiselect("Choose groups to plot alongside countries:", options=groups)
            if chosen:
                fig2 = go.Figure()
                for i, (country, model) in enumerate(models.items()):
                    fig2.add_trace(go.Scatter(x=model["x"], y=model["y"], mode="markers", name=f"{country} data"))
                    xm = model["x_mean"]
                    x_rel = x_plot - xm
                    y_fit = model["poly"](x_rel)
                    fig2.add_trace(go.Scatter(x=x_plot, y=y_fit, mode="lines", name=f"{country} fit"))
                for g in chosen:
                    dfg = uploaded_df[["year", g]].dropna()
                    fig2.add_trace(go.Scatter(x=dfg["year"], y=dfg[g], mode="lines+markers", name=g))
                fig2.update_layout(title="Comparison with uploaded U.S. Latin-group data", xaxis_title="Year", yaxis_title=category)
                st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")

# ---------------------
# Printer-friendly HTML report
# ---------------------
st.header("Printer-friendly report")
if st.button("Generate simple HTML report"):
    html_parts = []
    html_parts.append(f"<h1>{category} — Regression & Function Analysis</h1>")
    html_parts.append(f"<p>Countries: {', '.join(selected_countries)}</p>")
    html_parts.append(f"<p>Polynomial degree: {degree}</p>")
    for country, model in models.items():
        p = model["poly"]; xm = model["x_mean"]
        coeffs = p.coeffs; deg_p = len(coeffs)-1
        terms = []
        for idx, coef in enumerate(coeffs):
            power = deg_p - idx
            if power == 0:
                terms.append(f"{coef:.6g}")
            elif power == 1:
                terms.append(f"{coef:.6g}*(year - {xm:.1f})")
            else:
                terms.append(f"{coef:.6g}*(year - {xm:.1f})^{power}")
        eq = " + ".join(terms)
        html_parts.append(f"<h2>{country}</h2><pre>y = {eq}</pre>")
    html = "<html><body>" + "".join(html_parts) + "</body></html>"
    st.download_button("Download report (HTML)", data=html.encode("utf-8"), file_name="report.html", mime="text/html")

st.markdown("---")
st.write("Notes: World Bank API is used. Many indicators start around 1960; complete 70-year historical coverage to 1955 may not be available for all indicators. The app maps 'Education' to a 0-25 scale by using expected years of schooling as a proxy and a linear mapping. Extrapolation is inherently uncertain — treat long-range predictions cautiously.")
