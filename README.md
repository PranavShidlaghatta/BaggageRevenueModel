
# Baggage Revenue Model

Predicting the Baggage Revenue of Southwest Airlines.


## Description

The objective of this project is forecasting quarterly baggage revenue of Southwest Airlines using BTS data from the last 10 years. Various models such as - Naive Seasonal, ARIMA, SARIMA have been trained on the data to capture trends in the data.

## Datasets

US Jet Fuel Price Data: https://www.bts.gov/browse-statistical-products-and-data/freight-facts-and-figures/diesel-and-jet-fuel-prices 

US Unemployment Data:https://www.bls.gov/charts/employment-situation/civilian-unemployment-rate.htm

GDP per Capita of all countries: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD?locations=US 

## Getting Started

### Dependencies

* Python 3.10.19
* miniconda

### Setup

1. git clone https://github.com/PranavShidlaghatta/BaggageRevenueModel.git

2. cd BaggageRevenueModel

3. conda create --name "your-env" python==3.10.19

4. pip install -r requirements.txt

### models

consists of models in sarimax.py and XBG_reg.py. 
Final chosen model is sarimax.py. 

models/saved models consists of pickled model files. 

#### models/notebooks 

Within each notebook, you may need to change the paths to the datasets. Each dataset will be found in the 
/data folder. 

- bag_rev_eda.ipynb entails exploratory data analysis of the baggage revenue dataset. 
- combined_bag_rev_csv.ipynb created the initial dataset 
- linear_model.ipynb creates an exploratory linear model on NPS 
- bag_rev_forecast.ipynb runs forecast models on the baggage revenue dataset without transformation. 
- imputation_main.ipynb was the original file to create the reindexed dataset. 
- bag_rev_forecast_reindexed.ipynb runs ARIMA/SARIMA models on the reindexed dataset. 
- exog_processing.ipynb creates a reindexed dataset with exogenous factors. 


## Authors

**Pranav Shidlaghatta**

Github: https://github.com/PranavShidlaghatta

Linkedin: https://www.linkedin.com/in/pranav-shidlaghatta

**Rayan Mohammed**

Github: https://github.com/pixelated-explorer

LinkedIn: https://www.linkedin.com/in/rayan-mohammed-5a55bb255/


## License

TBD

## Acknowledgments

* [Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
* [Bureau of Labor Statistics](https://www.bls.gov)
* [StatsForecast Library](https://github.com/Nixtla/statsforecast)
* [World Bank](https://data.worldbank.org/)
