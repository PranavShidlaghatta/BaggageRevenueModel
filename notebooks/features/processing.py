# %%
'''
NOTE: GENERAL UTILITIES
'''
import pandas as pd

def quarter_standardize(df, date="Date", month="Month", format_used="%B %Y"):
    df[date] = pd.to_datetime(df[month], format=format_used)
    df["Quarter"] = df[date].dt.to_period("Q").astype(str)
    df = df.sort_values("Date").reset_index(drop=True)
    return df

'''
Dataframe must be in long format for this 
Currently assumes cols -> 'unique_id, ds, y'
Returns standardized dataframe that removes the COVID period
'''
def subset_covid(df_long):
    train_end = pd.Timestamp("2019-12-31") # end was 2019-12-31, start 2019-10-01

    covid_end = pd.Timestamp("2021-06-30") # end was 2021-06-30, start 2021-04-01

    train_pre_covid = df_long[df_long['ds'] <= train_end].copy()
    post_covid_actual = df_long[df_long['ds'] > covid_end].copy()

    # Calculate num of missing quarters
    last_train_period = train_pre_covid['ds'].max().to_period('Q')
    covid_end_period = covid_end.to_period('Q')
    gap_quarters = (covid_end_period - last_train_period).n  

    print(f"Num gap quarters offset {gap_quarters}")

    # Shift post-COVID timestamps backward by that many quarters
    post_covid_actual['ds'] = post_covid_actual['ds'] - pd.offsets.QuarterEnd(gap_quarters)

    df_reindexed = (pd.concat([train_pre_covid, post_covid_actual]).sort_values(['ds']).drop_duplicates(subset=['ds']).reset_index(drop=True)) # removed keep = first arg from drop_duplicates()

    return df_reindexed


# %%

'''
NOTE: ORIGINAL DATASET...

- Currently, data reads from 2015Q1 - 2024Q4. 
- May need to shave off up to 2023Q4. 
'''

df = pd.read_csv(r"/home/rayan/Southwest_stuff/BaggageRevenueModel/combined_bag_revenue.csv")
df.head()

bg_long = df.melt(id_vars=['Airline'], var_name='Quarter', value_name="Baggage Revenue")
bg_long['Quarter'] = pd.PeriodIndex(bg_long['Quarter'], freq='Q').to_timestamp(how="start")
bg_long = bg_long[(bg_long["Quarter"] >= "2015-Q1") & (bg_long["Quarter"] <= "2023-Q4")]
bg_long = bg_long.rename(columns={"Airline": "unique_id", "Baggage Revenue": "y"})

bg_long = bg_long[['unique_id', 'Quarter', 'y']].rename(columns={"Quarter": "ds"})

bg_long.head()

# %%

train_end = pd.Timestamp("2019-12-31")

covid_end = pd.Timestamp("2021-06-30")

train_pre_covid = bg_long[bg_long['ds'] <= train_end].copy()
post_covid_actual = bg_long[bg_long['ds'] > covid_end].copy()

# Calculate num of missing quarters
last_train_period = train_pre_covid['ds'].max().to_period('Q')
covid_end_period = covid_end.to_period('Q')
gap_quarters = (covid_end_period - last_train_period).n  

print(f"Num gap quarters offset {gap_quarters}")

# Shift post-COVID timestamps backward by that many quarters
post_covid_actual['ds'] = post_covid_actual['ds'] - pd.offsets.QuarterEnd(gap_quarters)

df_reindexed = (pd.concat([train_pre_covid, post_covid_actual]).sort_values(['unique_id', 'ds']).drop_duplicates(subset=['unique_id', 'ds'], keep='first').reset_index(drop=True))

df_reindexed.head(10)

# %%
'''
NOTE: JET FUEL PORTION. 
-> Jet fuel is measured in cents per gallon. 
'''

# %%
jet_fuel = pd.read_csv(r"/home/rayan/Southwest_stuff/BaggageRevenueModel/data/jet_fuel_prices 2000-2023.csv")

jet_fuel = quarter_standardize(jet_fuel, "Date", "Month of Date", "%B %Y")
jet_fuel = jet_fuel[(jet_fuel["Quarter"] >= "2015Q1") & (jet_fuel["Quarter"] <= "2023Q4")]

# Unsure if we care about Diesel type, *maybe* for ground operations. Not a priority. 
jet_fuel = jet_fuel[jet_fuel['Fuel Type'] == "Jet Fuel"]

jet_fuel["Price (cents per gallon)"] = (jet_fuel["Price (cents per gallon)"]
                                        .str.replace("$", "", regex=False)
                                        .astype(float)
                                        )

jet_fuel_q = (jet_fuel.groupby(["Quarter", "Fuel Type"])["Price (cents per gallon)"].mean().reset_index().sort_values(["Fuel Type", "Quarter"]))
jet_fuel_q['Quarter'] = pd.PeriodIndex(jet_fuel_q['Quarter'], freq='Q').to_timestamp(how="start")

# Odd error when renaming both in the same line. Overall logic is still fine, review later. 
jet_fuel_q = jet_fuel_q.rename(columns={"Quarter": "ds"})
jet_fuel_q = jet_fuel_q.rename(columns={"Price (cents per gallon)": "jetfuel_cost"})
jet_fuel_q = jet_fuel_q[["ds", "jetfuel_cost"]]
jet_fuel_q.head()


# %%

jet_fuel_q = subset_covid(jet_fuel_q)
jet_fuel_q.head()

# %%
'''
NOTE: UNEMPLOYMENT.  
'''

# %%

unemp = pd.read_csv(r"/home/rayan/Southwest_stuff/BaggageRevenueModel/data/US Unemployment Data 2014-2024.csv")
unemp["Month"] = (
    unemp["Month"]
      .str.replace("June", "Jun", regex=False)
      .str.replace("July", "Jul", regex=False)
      .str.replace("Sept", "Sep", regex=False)
)


unemp = quarter_standardize(unemp, format_used="%b %Y")
unemp = unemp[(unemp["Quarter"] >= "2015Q1") & (unemp["Quarter"] <= "2023Q4")]
unemp_q = (unemp.groupby("Quarter").agg({"Total": "mean"}).reset_index())
unemp_q['Quarter'] = pd.PeriodIndex(unemp_q['Quarter'], freq='Q').to_timestamp(how="start")
unemp_q = unemp_q.rename(columns={"Quarter" : "ds", "Total" : "unemployment_rate"})
unemp_q.head()

# %%

unemp_q = subset_covid(unemp_q)
unemp_q.head()

# %%

'''
NOTE: GDP  
'''

gdp_df = pd.read_csv(r"/home/rayan/Southwest_stuff/BaggageRevenueModel/data/GDP_per_capita_1947-2025.csv")
gdp_df = gdp_df.rename(columns={"observation_date" : "Quarter", "A939RX0Q048SBEA" : "GDP"})
gdp_df = gdp_df[(gdp_df["Quarter"] >= "2015-01-01") & (gdp_df["Quarter"] <= "2023-10-01")]
gdp_df['Quarter'] = pd.PeriodIndex(gdp_df['Quarter'], freq='Q').to_timestamp(how="start") # forcing anyway
gdp_df = gdp_df.rename(columns={"Quarter" : "ds"})
gdp_df.head()

# %%

gdp_df = subset_covid(gdp_df)
gdp_df.head()

# %%

'''
NOTE: Merge all dataframes by 'ds' 
-> Merge on 'left'. Merging into bg_long as baggage revenue is the single outcome we care about. 
-> Ie. multiple input linear reg
'''

# %%

bg_merged = (
    bg_long
        .merge(jet_fuel_q[["ds", "jetfuel_cost"]], on="ds", how="left")
        .merge(unemp_q[["ds", "unemployment_rate"]], on="ds", how="left")
        .merge(gdp_df[["ds", "GDP"]], on="ds", how="left")
)


bg_merged.head()
