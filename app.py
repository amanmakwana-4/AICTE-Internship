import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# === Page Setup ===
st.set_page_config(page_title="EV Forecast", layout="wide")

# === Theme Toggle ===
dark_mode = st.toggle("🌗 Dark Mode", value=True)

# === Dynamic Styling ===
dark_bg = "#1c1c1c"
light_bg = "#f0f2f6"

st.markdown(f"""
    <style>
        body {{
            background-color: {'#121212' if dark_mode else '#ffffff'};
            color: {'#ffffff' if dark_mode else '#1b1818'};
        }}
        .stApp {{
            background: {'linear-gradient(to right, #0f2027, #203a43, #2c5364);' if dark_mode else 'linear-gradient(to right, #dfe9f3, #ffffff);'}
        }}
        .title-container {{
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: {'#ffffff' if dark_mode else '#1b1818'};
            margin-top: 20px;
            transition: all 0.3s ease;
        }}
        .subtitle {{
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            padding-top: 10px;
            margin-bottom: 25px;
            color: {'#ffffff' if dark_mode else '#1b1818'};
        }}
        .instruction {{
            text-align: left;
            font-size: 20px;
            padding-top: 10px;
            color: {'#ffffff' if dark_mode else '#1b1818'};
        }}
    </style>
""", unsafe_allow_html=True)

# === Load Model ===
model = joblib.load('forecasting_ev_model.pkl')

# === Title & Image ===
st.markdown("<div class='title-container'>🔮 EV Adoption Forecaster for a County in Washington State</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Welcome to the Electric Vehicle (EV) Adoption Forecast tool.</div>", unsafe_allow_html=True)
st.image("ev-car-factory.jpg", use_container_width=True)
st.markdown("<div class='instruction'>Select a county and see the forecasted EV adoption trend for the next 3 years.</div>", unsafe_allow_html=True)

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === Select County ===
county_list = sorted(df['County'].dropna().unique().tolist())
county = st.selectbox("Select a County", county_list)

if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Forecast Logic ===
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

future_rows = []
forecast_horizon = 36

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    ev_growth_slope = np.polyfit(range(6), cumulative_ev[-6:], 1)[0] if len(cumulative_ev) >= 6 else 0

    row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    historical_ev = historical_ev[-6:]
    cumulative_ev.append(cumulative_ev[-1] + pred)
    cumulative_ev = cumulative_ev[-6:]

# === Plot Data Preparation ===
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# === Plot ===
st.subheader(f"📊 Cumulative EV Forecast for {county} County")
fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')

bg_color = dark_bg if dark_mode else "#ffffff"
ax.set_title(f"Cumulative EV Trend - {county}", fontsize=14, color='white' if dark_mode else 'black')
ax.set_facecolor(bg_color)
fig.patch.set_facecolor(bg_color)
ax.tick_params(colors='white' if dark_mode else 'black')
ax.set_xlabel("Date", color='white' if dark_mode else 'black')
ax.set_ylabel("Cumulative EV Count", color='white' if dark_mode else 'black')
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# === EV Growth Summary ===
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
    trend = "increase 📈" if growth_pct > 0 else "decrease 📉"
    st.success(f"EV adoption in **{county}** is expected to show a **{trend} of {growth_pct:.2f}%** over the next 3 years.")
else:
    st.warning("Cannot calculate forecast percentage due to missing historical data.")

# === Multi-County Comparison ===
st.markdown("---")
st.header("📍 Compare EV Trends for up to 3 Counties")
multi_counties = st.multiselect("Select up to 3 counties", county_list, max_selections=3)

if multi_counties:
    comparison_data = []

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        cty_code = cty_df['county_encoded'].iloc[0]

        hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df['months_since_start'].max()
        last_date = cty_df['Date'].max()

        fc_rows = []
        for _ in range(forecast_horizon):
            forecast_date = last_date + pd.DateOffset(months=1)
            last_date = forecast_date
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            slope = np.polyfit(range(6), cum_ev[-6:], 1)[0] if len(cum_ev) == 6 else 0

            row = {
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct1,
                'ev_total_pct_change_3': pct3,
                'ev_growth_slope': slope
            }

            pred = model.predict(pd.DataFrame([row]))[0]
            fc_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

            hist_ev.append(pred)
            hist_ev = hist_ev[-6:]
            cum_ev.append(cum_ev[-1] + pred)
            cum_ev = cum_ev[-6:]

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(fc_rows)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ], ignore_index=True)
        combined_cty['County'] = cty
        comparison_data.append(combined_cty)

    comp_df = pd.concat(comparison_data, ignore_index=True)

    st.subheader("📈 Multi-County Forecast Comparison")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby('County'):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
    ax.set_title("3-Year EV Adoption Forecast by County", fontsize=16, color='white' if dark_mode else 'black')
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.tick_params(colors='white' if dark_mode else 'black')
    ax.set_xlabel("Date", color='white' if dark_mode else 'black')
    ax.set_ylabel("Cumulative EV Count", color='white' if dark_mode else 'black')
    ax.grid(True, alpha=0.3)
    ax.legend(title="County")
    st.pyplot(fig)

    # Growth Summary
    summaries = []
    for cty in multi_counties:
        cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
        hist_total = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
        forecast_total = cty_df['Cumulative EV'].iloc[-1]
        if hist_total > 0:
            growth = ((forecast_total - hist_total) / hist_total) * 100
            summaries.append(f"{cty}: {growth:.2f}%")
        else:
            summaries.append(f"{cty}: N/A")

    st.success("📊 Growth over next 3 years — " + " | ".join(summaries))

st.success("✅ Forecast Complete")
