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

def 

# %%
'''
NOTE: JET FUEL PORTION. 
'''

# %%
jet_fuel = pd.read_csv(r"/home/rayan/Southwest_stuff/BaggageRevenueModel/data/jet_fuel_prices 2000-2023.csv")

jet_fuel = quarter_standardize(jet_fuel, "Date", "Month of Date", "%B %Y")

jet_fuel["Price (cents per gallon)"] = (jet_fuel["Price (cents per gallon)"]
                                        .str.replace("$", "", regex=False)
                                        .astype(float)
                                        )
jet_fuel.head()


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

unemp.head()

# %%

'''
NOTE: GDP 
'''

gdp = pd.read_csv(r"/home/rayan/Southwest_stuff/BaggageRevenueModel/data/GDP_per_capita 1960-2024.csv")

gdp.head()

# %%

# print(gdp["Country Name"].tolist())

print("United States" in gdp["Country Name"].tolist())

us_row = gdp[gdp['Country Name'] == "United States"]

us_row