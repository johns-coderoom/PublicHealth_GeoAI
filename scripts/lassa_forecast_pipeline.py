"""
Lassa Fever GeoAI Forecasting — Liberia (2016–2026)
End-to-end reproducible pipeline

Author: John Senanu
Description:
    - Load national & county-level Lassa fever data
    - Clean & prepare time series
    - Fit ARIMA + Prophet models (national level)
    - STL decomposition & rolling averages
    - XGBoost GeoAI county-level forecasting
    - Generate figures & spatial hotspot map

.
"""

# ==========================================================
# 0. IMPORTS & GLOBAL CONFIG
# ==========================================================
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL

from prophet import Prophet
from xgboost import XGBRegressor

import geopandas as gpd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parents[1]  # assumes script in scripts/
DATA_DIR = BASE_DIR / "data_template"
SHAPE_DIR = BASE_DIR / "shapefiles"
FIG_DIR = BASE_DIR / "figures"

(FIG_DIR / "national_forecasts").mkdir(parents=True, exist_ok=True)
(FIG_DIR / "spatial_outputs").mkdir(parents=True, exist_ok=True)

# ==========================================================
# 1. LOAD DATA
# ==========================================================
def load_national_data():
    """
    Expects CSV with at least:
        date (YYYY-MM-DD or YYYY-MM)
        cases (integer)
    """
    fn = DATA_DIR / "national_monthly_cases.csv"
    df = pd.read_csv(fn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_county_data():
    """
    Expects CSV with at least:
        date (YYYY-MM-DD or YYYY-MM)
        county (string)
        cases (integer)
    Optional extra covariates can be included (population, rainfall, etc.).
    """
    fn = DATA_DIR / "county_cases.csv"
    df = pd.read_csv(fn)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["county", "date"]).reset_index(drop=True)
    return df


def load_county_shapefile():
    """
    Expects shapefile with a county name field that can be matched
    to the 'county' column in county_cases.csv
    e.g. attribute named 'County' or 'NAME_1'.
    Adjust the join field below if needed.
    """
    shp = SHAPE_DIR / "liberia_counties.shp"
    gdf = gpd.read_file(shp)
    # standardize county name column; adjust as needed
    if "county" in gdf.columns:
        gdf["county_std"] = gdf["county"].str.strip().str.upper()
    elif "County" in gdf.columns:
        gdf["county_std"] = gdf["County"].str.strip().str.upper()
    else:
        raise ValueError("Please update the shapefile join field name.")
    return gdf


# ==========================================================
# 2. NATIONAL TIME-SERIES: PREP & SPLIT
# ==========================================================
def prepare_national_ts(df):
    """
    Returns:
        df_ts: with 'ds', 'y' for Prophet + index on 'ds'
        train_df, test_df: split at end of 2022 by default
    """
    df = df.copy()
    df = df.rename(columns={"date": "ds", "cases": "y"})
    df = df.set_index("ds").asfreq("MS")  # Monthly start freq
    df["y"] = df["y"].fillna(0)

    split_date = "2022-12-01"   # last observed month
    train_df = df.loc[:split_date].copy()
    test_df = df.loc[split_date:].copy()

    return df, train_df, test_df


# ==========================================================
# 3. ARIMA (SARIMAX) NATIONAL FORECAST
# ==========================================================
def fit_arima(train_df, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12)):
    """
    Basic SARIMAX model.
    """
    model = SARIMAX(
        train_df["y"],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return results


def forecast_arima(results, full_index, start_date):
    """
    Extend forecast to match full index (e.g., through 2026-12-01).
    """
    forecast = results.get_forecast(steps=len(full_index))
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int(alpha=0.05)
    df_pred = pd.DataFrame(
        {
            "ds": full_index,
            "arima_mean": pred_mean.values,
            "arima_lower": pred_ci.iloc[:, 0].values,
            "arima_upper": pred_ci.iloc[:, 1].values,
        }
    )
    return df_pred


# ==========================================================
# 4. PROPHET NATIONAL FORECAST
# ==========================================================
def fit_prophet(train_df):
    """
    Prophet expects dataframe with columns 'ds' and 'y'.
    """
    df_prophet = train_df.reset_index().rename(columns={"ds": "ds", "y": "y"})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )
    m.fit(df_prophet)
    return m


def forecast_prophet(m, horizon_months=48, last_date=None):
    future = m.make_future_dataframe(periods=horizon_months, freq="MS")
    fcst = m.predict(future)
    # Keep only needed columns
    out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "yearly"]]
    return out


# ==========================================================
# 5. STL DECOMPOSITION & ROLLING MEANS
# ==========================================================
def stl_decomposition(df_ts):
    series = df_ts["y"]
    stl = STL(series, period=12, robust=True)
    res = stl.fit()
    df_stl = pd.DataFrame(
        {
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid,
        },
        index=df_ts.index,
    )
    return df_stl


def add_rolling_means(df_ts, windows=(3, 6)):
    df = df_ts.copy()
    for w in windows:
        df[f"roll_{w}"] = df["y"].rolling(w, min_periods=1).mean()
    return df


# ==========================================================
# 6. PLOTTING HELPERS (NATIONAL)
# ==========================================================
def plot_rolling(df_roll):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_roll.index, df_roll["y"], marker="o", label="Monthly cases")
    if "roll_3" in df_roll:
        ax.plot(df_roll.index, df_roll["roll_3"], linestyle="--", label="3-month rolling mean")
    if "roll_6" in df_roll:
        ax.plot(df_roll.index, df_roll["roll_6"], linestyle="--", label="6-month rolling mean")
    ax.set_title("Monthly Lassa Fever Cases with Rolling Averages")
    ax.set_xlabel("Date (Month / Year)")
    ax.set_ylabel("Reported Lassa Fever Cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "national_forecasts" / "lassa_ts_with_rolling_means.png", dpi=300)


def plot_stl(df_ts, df_stl):
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df_ts.index, df_ts["y"])
    axes[0].set_title("Observed")

    axes[1].plot(df_stl.index, df_stl["trend"])
    axes[1].set_title("Trend")

    axes[2].plot(df_stl.index, df_stl["seasonal"])
    axes[2].set_title("Seasonal")

    axes[3].scatter(df_stl.index, df_stl["resid"], s=10)
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set_title("Residual")

    fig.suptitle("STL Decomposition of Monthly Lassa Fever Cases")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "national_forecasts" / "lassa_stl_decomposition.png", dpi=300)


def plot_prophet_vs_arima(df_ts, prophet_fc, arima_fc, train_end):
    """
    df_ts      : original series with index
    prophet_fc : df with ds, yhat
    arima_fc   : df with ds, arima_mean
    train_end  : timestamp of last training date
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_ts.index, df_ts["y"], "ko-", label="Observed Cases")

    # Prophet
    ax.plot(prophet_fc["ds"], prophet_fc["yhat"], label="Prophet Forecast")

    # ARIMA
    ax.plot(arima_fc["ds"], arima_fc["arima_mean"], "r--", label="ARIMA Forecast")

    # Vertical line for train/test split
    ax.axvline(pd.to_datetime(train_end), color="gray", linestyle="--", alpha=0.7)

    ax.set_title("ARIMA vs Prophet Forecast Through 2026")
    ax.set_xlabel("Date (Month / Year)")
    ax.set_ylabel("Reported Cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "national_forecasts" / "arima_vs_prophet_2026.png", dpi=300)


def plot_prophet_components(m, prophet_fc):
    """
    Save Prophet's trend + yearly components.
    """
    fig = m.plot_components(prophet_fc)
    fig.set_size_inches(12, 8)
    fig.savefig(FIG_DIR / "national_forecasts" / "lassa_prophet_components.png", dpi=300)


# ==========================================================
# 7. GEOAI: COUNTY-LEVEL XGBOOST FORECASTING
# ==========================================================
def engineer_geoai_features(df_county, max_lag=3, window=3):
    """
    For each county, create:
        - lag1..lagN
        - rolling_mean
        - month, year (seasonality features)
    """
    df = df_county.copy()
    df["county_std"] = df["county"].str.strip().str.upper()
    df = df.set_index("date")

    feature_frames = []
    for county, sub in df.groupby("county_std"):
        sub = sub.sort_index()
        for lag in range(1, max_lag + 1):
            sub[f"lag_{lag}"] = sub["cases"].shift(lag)
        sub["roll_mean"] = sub["cases"].rolling(window, min_periods=1).mean()
        sub["month"] = sub.index.month
        sub["year"] = sub.index.year
        sub["county_std"] = county
        feature_frames.append(sub)

    feat = pd.concat(feature_frames).reset_index().rename(columns={"index": "date"})

    # Drop first rows with NaNs from lags
    feat = feat.dropna(subset=[f"lag_{lag}" for lag in range(1, max_lag + 1)]).reset_index(drop=True)
    return feat


def train_xgboost_geoai(feat_df, train_end="2022-12-01"):
    """
    Train XGBRegressor on historical county-level data.
    """
    feat_df = feat_df.copy()
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    train_mask = feat_df["date"] <= pd.to_datetime(train_end)
    train = feat_df[train_mask]
    test = feat_df[~train_mask]

    feature_cols = [c for c in feat_df.columns if c not in ["cases", "date", "county", "county_std"]]
    X_train = train[feature_cols]
    y_train = train["cases"]
    X_test = test[feature_cols]
    y_test = test["cases"]

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    test["pred"] = model.predict(X_test)

    print("GeoAI XGBoost sample predictions:")
    print(test[["date", "county_std", "cases", "pred"]].head())

    return model, feature_cols


def make_geoai_forecast(feat_df, model, feature_cols, forecast_start="2023-01-01", forecast_end="2026-12-01"):
    """
    Simple iterative forecasting:
        - For each month in forecast horizon, generate prediction using
          previous lags, then append prediction as new 'cases' for next step.
    """
    feat_df = feat_df.copy()
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    counties = feat_df["county_std"].unique()

    # Start from last available month
    current_end = feat_df["date"].max()
    horizon = pd.date_range(start=forecast_start, end=forecast_end, freq="MS")

    forecasts = []

    # We'll maintain a working frame per county
    county_hist = {
        c: feat_df[feat_df["county_std"] == c].set_index("date").sort_index()
        for c in counties
    }

    for dt in horizon:
        for c in counties:
            hist = county_hist[c]

            # build feature row for this county & date
            row = {}
            # lags
            for lag in range(1, 4):
                row[f"lag_{lag}"] = hist["cases"].iloc[-lag] if len(hist) >= lag else np.nan
            # rolling mean of last 3 months
            row["roll_mean"] = hist["cases"].tail(3).mean()
            row["month"] = dt.month
            row["year"] = dt.year
            row["county_std"] = c

            # build DataFrame and predict
            X_row = pd.DataFrame([row])[feature_cols]
            y_pred = float(model.predict(X_row)[0])

            forecasts.append(
                {
                    "date": dt,
                    "county_std": c,
                    "pred_cases": max(y_pred, 0.0),  # non-negative
                }
            )

            # append prediction to history for next iteration
            hist.loc[dt, "cases"] = y_pred
            county_hist[c] = hist

    fc_df = pd.DataFrame(forecasts)
    return fc_df


def plot_geoai_hotspot_map(fc_df, gdf_counties, target_month="2026-12-01"):
    """
    Choropleth of predicted cases for a specific month.
    """
    target_month = pd.to_datetime(target_month)
    month_df = fc_df[fc_df["date"] == target_month].copy()

    month_df["county_std"] = month_df["county_std"].str.upper()
    gdf = gdf_counties.copy()
    gdf_merged = gdf.merge(
        month_df,
        left_on="county_std",
        right_on="county_std",
        how="left",
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    gdf_merged.plot(
        column="pred_cases",
        cmap="Reds",
        linewidth=0.5,
        edgecolor="black",
        legend=True,
        ax=ax,
    )
    ax.set_title("Lassa Fever GeoAI Hotspot Forecast — December 2026")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spatial_outputs" / "GeoAI_hotspot_forecast_2026.png", dpi=300)


# ==========================================================
# 8. MAIN DRIVER
# ==========================================================
def main():
    print("Loading data...")
    national_df = load_national_data()
    county_df = load_county_data()
    gdf_counties = load_county_shapefile()

    print("Preparing national time series...")
    df_ts, train_df, test_df = prepare_national_ts(national_df)

    # ---------- Rolling means ----------
    print("Computing rolling means...")
    df_roll = add_rolling_means(df_ts)
    plot_rolling(df_roll)

    # ---------- STL ----------
    print("Running STL decomposition...")
    df_stl = stl_decomposition(df_ts)
    plot_stl(df_ts, df_stl)

    # ---------- ARIMA ----------
    print("Fitting ARIMA/SARIMAX model...")
    arima_results = fit_arima(train_df)
    full_index = pd.date_range(
        start=df_ts.index.min(),
        end="2026-12-01",
        freq="MS",
    )
    arima_fc = forecast_arima(arima_results, full_index, train_df.index.max())

    # ---------- Prophet ----------
    print("Fitting Prophet model...")
    prophet_model = fit_prophet(train_df)
    horizon_months = (pd.to_datetime("2026-12-01") - train_df.index.max()).days // 30 + 1
    prophet_fc = forecast_prophet(prophet_model, horizon_months=horizon_months)

    # ---------- National comparison plots ----------
    print("Creating national forecast plots...")
    plot_prophet_vs_arima(df_ts, prophet_fc, arima_fc, train_df.index.max())
    plot_prophet_components(prophet_model, prophet_fc)

    # ---------- GeoAI XGBoost ----------
    print("Engineering GeoAI features...")
    feat_df = engineer_geoai_features(county_df)
    print("Training XGBoost GeoAI model...")
    xgb_model, feature_cols = train_xgboost_geoai(feat_df)

    print("Forecasting county-level cases through 2026...")
    geoai_fc = make_geoai_forecast(
        feat_df,
        xgb_model,
        feature_cols,
        forecast_start="2023-01-01",
        forecast_end="2026-12-01",
    )

    print("Creating GeoAI hotspot map for December 2026...")
    plot_geoai_hotspot_map(geoai_fc, gdf_counties, target_month="2026-12-01")

    print("Done. Figures saved in:", FIG_DIR)


if __name__ == "__main__":
    main()
