# %%
import pandas as pd
import numpy as np 

df = pd.read_csv("/home/rayan/Southwest_stuff/BaggageRevenueModel/combined_bag_revenue.csv")
df.head()

# %%

import requests
import pandas as pd

# --- Requesting monthly CPI data from BLS API ---
series_id = "CUUR0000SA0"  # CPI-U, all items, not seasonally adjusted
payload = {
    "seriesid": [series_id],
    "startyear": "2015",
    "endyear": "2024"
}
url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
response = requests.post(url, json=payload)
response.raise_for_status()
data = response.json()

records = data["Results"]["series"][0]["data"]
df = pd.DataFrame([
    {
        "year": int(r["year"]),
        "month": int(r["period"][1:]),
        "CPI": float(r["value"])
    }
    for r in records if r["period"].startswith("M")
])
df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
df = df.sort_values("date").set_index("date")[["CPI"]]

cpi_quarterly = df.resample("Q").mean()
cpi_quarterly.index = cpi_quarterly.index.to_period("Q")

print(cpi_quarterly.head())
cpi_quarterly.to_csv("cpi_quarterly_2015_2024.csv")


# %%

# Using transformation methods. 

# --- Step 1: Read your CPI dataset ---
cpi = pd.read_csv("cpi_quarterly_2015_2024.csv")

# Convert 'date' (e.g., "2015Q1") into a pandas PeriodIndex
cpi["date"] = pd.PeriodIndex(cpi["date"], freq="Q")
cpi = cpi.set_index("date")

# --- Step 2: Calculate quarter-over-quarter % change (inflation rate) ---
# This represents short-term inflation: change in CPI from one quarter to the next
cpi["cpi_qoq_change"] = cpi["CPI"].pct_change(1) * 100  # in percent

# --- Step 3: Calculate year-over-year % change ---
# This smooths out seasonality and shows annual inflation
cpi["cpi_yoy_change"] = cpi["CPI"].pct_change(4) * 100  # 4 quarters = 1 year

# --- Step 4: Log transformation (often used in econometrics) ---
cpi["log_CPI"] = np.log(cpi["CPI"])

# --- Step 5: Log difference (approximate inflation in log terms) ---
# This is roughly equivalent to the % change, but additive and smoother
cpi["log_diff"] = cpi["log_CPI"].diff() * 100  # *100 to express as % roughly

print(cpi.head(40))


# %%

bts = pd.read_csv("/home/rayan/Southwest_stuff/BaggageRevenueModel/combined_bag_revenue.csv")

print(bts.head(10))
