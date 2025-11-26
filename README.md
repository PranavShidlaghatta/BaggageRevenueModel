
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

#### models/saved_models 

These are pickled versions of the models that have been produced by the notebooks. These can be loaded as python dictionaries.   
Models can be used for inference according to their libraries and how they were used in each file. 


`sf_models.pkl` --> From bag_rev_forecast_on_offset.ipynb. Read and load with:   

```
with open("/home/rayan/Southwest_stuff/BaggageRevenueModel/models/saved_models/sf_models.pkl", "wb") as f:
    pickle.dump(sf, f)

# Test load 
with open("/home/rayan/Southwest_stuff/BaggageRevenueModel/models/saved_models/sf_models.pkl", "rb") as f:
    sf_loaded = pickle.load(f)

# Test forecast
preds = sf_loaded.predict(h=horizon)

preds.head()
```


`baggage_rev_xgb_model.pkl` --> From XGB_reg.py. Contains objects  

```
save_objects = {
    "best_model": best_model,
    "grid_search": grid_search,
    "robust_scaler": robust_scaler,
    "standard_scaler": scaler,
    "feature_cols": feature_cols,
    "feature_cols_no_exog": feature_cols_no_exog,
    "feature_cols_jetfuel_only": feature_cols_jetfuel_only,
    "shap_explainer": explainer,
    "shap_values": shap_values
}
```

`sarimax_models_and_results.pkl` --> from sarimax.py. Contains objects  


```
save_pickle({
        "model_no_exog": model_no_exog,
        "forecast_no_exog": forecast_no_exog_df,
        "model_with_exog": sarimax_results_exog,
        "forecast_with_exog": forecast_exog_df,
        "exog_stats": exog_stats,
        "vif_data": vif_data
    }, "sarimax_models_and_results.pkl")
```



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

