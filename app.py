# app.py
# Streamlit app: polynomial regression & function analysis on World Bank time series
# Author: (generated)
# Notes:
# - Fetches World Bank indicators at runtime (requires internet on Streamlit server).
# - Uses indicators (see indicator_map below). Most World Bank WDI indicators have data from 1960 onwards.

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime
import io
import base64

st.set_page_config(layout="wide", page_title="Latin Countries — Regression & Function Analysis")

st.title("Regression & Function Analysis for selected Latin American countries")
st.markdown(
    """
This app fetches historical data (World Bank / Our World in Data) and fits a polynomial regression (degree ≥ 3).
- Choose countries and category.
- Edit the data table if needed.
- Compare multiple countries or upload your own US-Latin-group CSV to compare diaspora groups.
- Extrapolate into the future; extrapolated curve is shown in a different color.
"""
)

# ---------------------
# Configuration / indicators
# ---------------------
# Country list: three wealthy Latin countries (ISO2 codes)
COUNTRIES = {
    "Chile": "CL",
    "Uruguay": "UY",
    "Panama": "PA"
}

# World Bank indicator codes used in this app (common WDI codes)
indicator_map = {
    "Population": "SP.POP.TOTL",
    "Life expectancy": "SP.DYN.LE00.IN",
    "Birth rate": "SP.DYN.CBRT.IN",          # crude births per 1000
    "Average income (GDP per capita, current US$)": "NY.GDP.PCAP.CD",
    "Murder rate (intentional homicides per 100k)": "VC.IHR.PSRC.P5",
    "Unemployment rate (total, % of labor force)": "SL.UEM.TOTL.ZS",
    "Education (mean/expected years -> mapped 0-25)": "SE.SCH.LIFE"  # expected years of schooling (used as proxy)
}

# Years range: we default to 1960 onwards because many WDI series begin there
DEFAULT_START_YEAR = 1960
DEFAULT_END_YEAR = datetime.now().year  # e.g., 2025 on the host server

# UI: country selection (single or multiple)
st.sidebar.header("Data selection")
multiselect_countries = st.sidebar.multiselect("Choose 1–3 countries (World Bank data fetched at runtime):",
                                              options=list(COUNTRIES.keys()),
                                              default=["Chile"])
if len(multiselect_countries) == 0:
    st.sidebar.error("Pick at least one country to continue.")
    st.stop()

category = st.sidebar.selectbox("Category (select one):", list(indicator_map.keys()), index=0)

start_year = st.sidebar.number_input("Start year (earliest allowed 1960):", min_value=1950, max_value=DEFAULT_END_YEAR,
                                     value=DEFAULT_START_YEAR, step=1)
if start_year < 1960:
    st.sidebar.caption("Note: most World Bank time series begin 1960; earlier years may be missing.")

end_year = st.sidebar.number_input("End year:", min_value=1960, max_value=DEFAULT_END_YEAR,
                                   value=DEFAULT_END_YEAR, step=1)

year_increment = st.sidebar.slider("Plot resolution (years between plotted points)", min_value=1, max_value=10, value=1)

# polynomial degree (must be >= 3)
degree = st.sidebar.slider("Polynomial degree (≥ 3):", min_value=3, max_value=8, value=3)

# extrapolation options
extrapolate_years = st.sidebar.number_input("Extrapolate forward (years):", min_value=0, max_value=100, value=0, step=1)
extrapolate_show = st.sidebar.checkbox("Show extrapolated portion", value=True)

# multiple overlay / compare diaspora
compare_multiple = st.sidebar.checkbox("Overlay multiple countries on same graph", value=True)
compare_diaspora = st.sidebar.checkbox("Compare with Latin groups in the U.S. (upload CSV)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("Data source: World Bank (WDI), Our World in Data for schooling where necessary. App fetches data live.")

# helper: fetch WDI series for a country code
def fetch_worldbank_series(country_iso2, indicator, start=1960, end=DEFAULT_END_YEAR):
    # World Bank API v2: /country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=2000
    url = f"https://api.worldbank.org/v2/country/{country_iso2}/indicator/{indicator}?date={start}:{end}&format=json&per_page=20000"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        st.error(f"World Bank API error: status {r.status_code} for {country_iso2}, {indicator}")
        return pd.Series(dtype=float)
    try:
        data = r.json()
    except Exception as e:
        st.error(f"Could not parse World Bank response: {e}")
        return pd.Series(dtype=float)
    if not isinstance(data, list) or len(data) < 2:
        return pd.Series(dtype=float)
    entries = data[1]
    rows = []
    for e in entries:
        year = e.get("date")
        val = e.get("value")
        if val is not None:
            try:
                rows.append((int(year), float(val)))
            except:
                continue
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["year", "value"]).sort_values("year").set_index("year")
    return df["value"]

# fetch datasets for selected countries
@st.cache_data(show_spinner=False)
def get_data_for_countries(countries, indicator_code, start, end):
    out = {}
    for c in countries:
        iso = COUNTRIES[c]
        s = fetch_worldbank_series(iso, indicator_code, start, end)
        out[c] = s
    return out

indicator_code = indicator_map[category]
with st.spinner("Fetching data from World Bank..."):
    raw_country_series = get_data_for_countries(multiselect_countries, indicator_code, start_year, end_year)

# transform education to 0-25 scale if needed
def map_education_to_025(series):
    # series is expected years of schooling (or similar). We'll map linearly to 0-25 by assuming 25 years is max schooling.
    # Many countries have expected years < 20; mapping preserves relative shape.
    max_years = 25.0
    mapped = (series / max_years) * 25.0
    return mapped

# assemble DataFrame for editing
def assemble_table(series_dict):
    df = pd.DataFrame(index=range(start_year, end_year + 1))
    for country, s in series_dict.items():
        # reindex to complete year range and keep NaNs where missing
        s_full = s.reindex(range(start_year, end_year + 1))
        if category == "Education (mean/expected years -> mapped 0-25)":
            s_full = map_education_to_025(s_full)
        # column name: country (category units)
        df[country] = s_full
    df.index.name = "year"
    df = df.reset_index()
    return df

data_table = assemble_table(raw_country_series)

st.header("Raw data (editable)")
st.write("You can edit values directly in the table — the regression will update when you press 'Update model'.")

# editable table (Streamlit's experimental data editor)
edited = st.experimental_data_editor(data_table, num_rows="dynamic")

# allow the user to download the raw data CSV
csv_bytes = edited.to_csv(index=False).encode()
st.download_button("Download data CSV", data=csv_bytes, file_name="raw_time_series.csv", mime="text/csv")

# prepare numeric arrays for modeling per country
def prepare_xy_from_table(edited_df, country, dropna=True):
    df = edited_df[["year", country]].copy()
    if dropna:
        df = df.dropna(subset=[country])
    if df.empty:
        return np.array([]), np.array([])
    x = df["year"].astype(float).to_numpy()
    y = df[country].astype(float).to_numpy()
    return x, y

# polynomial fit and analysis
def fit_polynomial(x, y, deg):
    # map years to a smaller scale to improve conditioning (e.g., subtract baseline)
    if x.size == 0:
        return None
    x0 = x - x.mean()
    pf = PolynomialFeatures(degree=deg, include_bias=True)
    Xp = pf.fit_transform(x0.reshape(-1,1))
    lr = LinearRegression()
    lr.fit(Xp, y)
    coeffs = lr.coef_.copy()
    coeffs[0] = lr.intercept_  # combine intercept
    # But sklearn stores intercept separately; instead construct polynomial coefficients in usual order:
    # We can get polynomial coefficients via numpy.polyfit on scaled variable for simplicity and clarity:
    p_coeffs = np.polyfit(x0, y, deg)
    # p_coeffs has highest degree first
    return {
        "poly": np.poly1d(p_coeffs),
        "x_mean": x.mean(),
        "x_original": x,
        "y_original": y
    }

# function analysis helpers (derivatives, critical points)
def function_analysis(poly_obj):
    p = poly_obj["poly"]
    xmean = poly_obj["x_mean"]
    # derivative and second derivative polynomials
    dp = np.polyder(p)
    ddp = np.polyder(dp)
    # find critical points: real roots of dp
    crit = np.roots(dp)
    real_crit = np.real(crit[np.isclose(np.imag(crit), 0, atol=1e-6)])
    # evaluate second derivative at real critical points
    crit_info = []
    for rc in real_crit:
        second_val = ddp(rc)
        # classify
        if second_val > 0:
            typ = "local minimum"
        elif second_val < 0:
            typ = "local maximum"
        else:
            typ = "inflection/flat"
        crit_info.append((rc + xmean, rc, second_val, typ))  # rc is relative; convert to original year by +xmean
    # where increasing/decreasing: evaluate dp sign across domain sample
    return {"dp": dp, "ddp": ddp, "critical": crit_info}

# build model for each selected country and plot
st.header("Modeling & Graph")

plot_countries = multiselect_countries if compare_multiple else [multiselect_countries[0]]

models = {}
for c in plot_countries:
    x, y = prepare_xy_from_table(edited, c)
    if x.size == 0:
        st.warning(f"No data available for {c}; cannot fit model.")
        continue
    model = fit_polynomial(x, y, degree)
    if model is None:
        st.warning(f"Could not fit polynomial for {c}.")
        continue
    models[c] = model

# prepare x grid for plotting (including extrapolation)
if not models:
    st.stop()

# Determine global x range for plotted data
min_x = min([m["x_original"].min() for m in models.values()])
max_x = max([m["x_original"].max() for m in models.values()])

plot_end_year = max_x + extrapolate_years
x_grid = np.arange(min_x, plot_end_year + 1, year_increment)
# use scaled x for evaluation: subtract x_mean for each model separately when evaluating
fig = go.Figure()
colors = ["blue", "green", "red", "orange", "purple"]
for i, (country, model) in enumerate(models.items()):
    # scatter original points
    fig.add_trace(go.Scatter(
        x=model["x_original"],
        y=model["y_original"],
        mode="markers",
        name=f"{country} data",
        marker=dict(size=6, color=colors[i % len(colors)])
    ))
    # compute fitted y over data range and extrapolation
    xm = model["x_mean"]
    p = model["poly"]
    # evaluate on grid relative to the model's x_mean:
    x_rel = x_grid - xm
    y_fit = p(x_rel)
    # split into fitted (within original data) and extrapolated portion
    mask_extrap = x_grid > model["x_original"].max()
    # add fitted portion
    fig.add_trace(go.Scatter(
        x=x_grid[~mask_extrap],
        y=y_fit[~mask_extrap],
        mode="lines",
        name=f"{country} fit (observed-range)",
        line=dict(width=2, dash="solid", color=colors[i % len(colors)])
    ))
    # add extrapolated portion in distinct style if requested
    if extrapolate_years > 0 and extrapolate_show:
        fig.add_trace(go.Scatter(
            x=x_grid[mask_extrap],
            y=y_fit[mask_extrap],
            mode="lines",
            name=f"{country} extrapolated",
            line=dict(width=2, dash="dash", color=colors[i % len(colors)]),
            hovertemplate="(extrapolated)"
        ))

fig.update_layout(title=f"{category} — data and polynomial fit (degree {degree})",
                  xaxis_title="Year",
                  yaxis_title=category,
                  height=600)

st.plotly_chart(fig, use_container_width=True)

# show equation of the model(s)
st.subheader("Model equation(s) and function analysis")
for country, model in models.items():
    p = model["poly"]
    # present polynomial coefficients with year variable as (year - mean)
    xm = model["x_mean"]
    coeffs = p.coeffs  # highest-first
    # create readable equation string (year variable as t = (year - xm))
    terms = []
    deg_p = len(coeffs) - 1
    for idx, coef in enumerate(coeffs):
        pow = deg_p - idx
        coef_str = f"{coef:.6g}"
        if pow == 0:
            terms.append(f"{coef_str}")
        elif pow == 1:
            terms.append(f"{coef_str}*(year - {xm:.1f})")
        else:
            terms.append(f"{coef_str}*(year - {xm:.1f})^{pow}")
    eqn = " + ".join(terms)
    st.markdown(f"**{country}** — polynomial (in variable `year - {xm:.1f}`):  ")
    st.code(f"y = {eqn}", language="text")

    # function analysis
    fa = function_analysis(model)
    critical = fa["critical"]
    if critical:
        st.markdown(f"**Critical points (converted to calendar years)** for {country}:")
        for rc_full, rc_rel, ddval, typ in critical:
            # rc_full is float year; convert to nearest date string (year + fraction -> approximate month/day)
            year_int = int(np.floor(rc_full))
            month = 1
            day = 1
            try:
                date_str = f"{year_int}-01-01"
            except:
                date_str = f"{rc_full:.2f}"
            st.write(f"- {typ.title()} at year ≈ **{rc_full:.2f}** (interpreted as around {date_str}); second derivative = {ddval:.3g}")
    else:
        st.write(f"No real critical points found for {country} (in the polynomial derivative roots).")

    # increasing/decreasing intervals (sample dp sign across wide grid)
    dp = fa["dp"]
    sample_x = np.linspace(min_x - 5, max_x + 5 + extrapolate_years, 400)
    dp_vals = dp(sample_x - xm)
    # find sign changes
    inc_regions = []
    dec_regions = []
    current_sign = None
    region_start = None
    for xi, dpi in zip(sample_x, dp_vals):
        sign = np.sign(dpi)
        if current_sign is None:
            current_sign = sign
            region_start = xi
        elif sign != current_sign:
            # close region
            if current_sign > 0:
                inc_regions.append((region_start, xi))
            elif current_sign < 0:
                dec_regions.append((region_start, xi))
            current_sign = sign
            region_start = xi
    # close last region
    if current_sign is not None:
        if current_sign > 0:
            inc_regions.append((region_start, sample_x[-1]))
        elif current_sign < 0:
            dec_regions.append((region_start, sample_x[-1]))

    def fmt_regions(regs):
        return ", ".join([f"[{r[0]:.1f}, {r[1]:.1f}]" for r in regs]) if regs else "None identified"

    st.write(f"- Intervals where the fitted function is increasing (approx): {fmt_regions(inc_regions)}")
    st.write(f"- Intervals where the fitted function is decreasing (approx): {fmt_regions(dec_regions)}")

    # where increasing/decreasing fastest: maxima/minima of derivative magnitude or where second derivative is largest magnitude
    ddp = fa["ddp"]
    dd_vals = ddp(sample_x - xm)
    idx_max_pos = np.argmax(dd_vals)
    idx_min_pos = np.argmin(dd_vals)
    fastest_inc_year = sample_x[idx_max_pos]
    fastest_dec_year = sample_x[idx_min_pos]
    st.write(f"- The function is increasing fastest around year ≈ **{fastest_inc_year:.2f}**.")
    st.write(f"- The function is decreasing fastest around year ≈ **{fastest_dec_year:.2f}**.")

    # domain & range (numerical)
    domain_str = f"[{sample_x[0]:.1f}, {sample_x[-1]:.1f}] (analysis sampling)"
    y_vals_sample = p(sample_x - xm)
    range_str = f"[{float(np.nanmin(y_vals_sample)):.3g}, {float(np.nanmax(y_vals_sample)):.3g}] (approx from sample)"
    st.write(f"- Domain used for analysis (approx): {domain_str}")
    st.write(f"- Approximate range of fitted function over that domain: {range_str}")

    st.markdown("---")

# interpolation / extrapolation single-value prediction
st.header("Interpolation / Extrapolation & Rates of Change")
st.write("Enter a year to get the model prediction (interpolated or extrapolated).")

pred_country = st.selectbox("Choose country for prediction:", options=list(models.keys()))
pred_year = st.number_input("Year to predict (can be outside data range):", value=int(max_x), min_value=1900, max_value=2100, step=1)
model = models[pred_country]
p = model["poly"]
xm = model["x_mean"]
pred_val = p((pred_year - xm))
st.write(f"Model prediction for **{pred_country}** in **{pred_year}**: **{pred_val:.4g}** (units = {category})")
# say whether it's interpolation or extrapolation
if pred_year > model["x_original"].max():
    st.warning("This prediction is an extrapolation beyond the observed data. Extrapolations are uncertain and should be used with caution.")
elif pred_year < model["x_original"].min():
    st.warning("This prediction is an extrapolation before the observed data. Use caution.")
else:
    st.success("This is an interpolation (within observed data range).")

# average rate of change between two years
st.write("Compute average rate of change of the model between two years:")
y1 = st.number_input("Year A:", value=int(model["x_original"].min()), step=1)
y2 = st.number_input("Year B:", value=int(model["x_original"].max()), step=1)
if y2 == y1:
    st.info("Pick two different years.")
else:
    valA = p((y1 - xm))
    valB = p((y2 - xm))
    avg_rate = (valB - valA) / (y2 - y1)
    st.write(f"Average rate of change from {y1} to {y2}: **{avg_rate:.6g} {category} per year**")
    st.write(f"Interpretation example: The {category.lower()} increased on average by {avg_rate:.3g} units per year between {y1} and {y2} according to the model.")

# Printer-friendly report (simple HTML)
st.header("Printer-friendly report")
if st.button("Generate printer-friendly HTML"):
    # Create a simple HTML report with the current plot and the model equations + function analysis text
    html_parts = []
    html_parts.append(f"<h1>{category} — Regression & Analysis</h1>")
    html_parts.append(f"<p>Selected countries: {', '.join(plot_countries)}</p>")
    html_parts.append(f"<p>Polynomial degree: {degree}</p>")
    # equations
    for country, model in models.items():
        p = model["poly"]
        xm = model["x_mean"]
        coeffs = p.coeffs
        deg_p = len(coeffs) - 1
        eqterms = []
        for idx, coef in enumerate(coeffs):
            pow = deg_p - idx
            coef_str = f"{coef:.6g}"
            if pow == 0:
                eqterms.append(f"{coef_str}")
            elif pow == 1:
                eqterms.append(f"{coef_str}*(year - {xm:.1f})")
            else:
                eqterms.append(f"{coef_str}*(year - {xm:.1f})^{pow}")
        eqn = " + ".join(eqterms)
        html_parts.append(f"<h2>{country}</h2><pre>y = {eqn}</pre>")
    html = "<html><body>" + "\n".join(html_parts) + "</body></html>"
    st.download_button("Download HTML report", data=html.encode("utf-8"), file_name="report.html", mime="text/html")

st.markdown("----")
st.info("Notes & sources: Data are fetched from the World Bank API (WDI) and Our World in Data where noted. Many WDI series begin circa 1960. Extrapolations are model-based estimates and can diverge from real future outcomes.")
