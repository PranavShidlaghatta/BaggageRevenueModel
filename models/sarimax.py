import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(file_path):
    return pd.read_csv(file_path)

def train_test_split(y, exog, train_size=0.6):
    split_idx = int(train_size * len(y))
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    exog_train, exog_test = exog.iloc[:split_idx], exog.iloc[split_idx:]
    return y_train, y_test, exog_train, exog_test

def fit_sarimax(y_train, exog_train, order, seasonal_order):
    model = SARIMAX(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order)
    sarimax_results = model.fit(disp=False)
    return sarimax_results

def predict(results, start, end, exog):
    return results.predict(start=start, end=end, exog=exog)

def forecast_future(results, steps, exog):
    return results.get_forecast(steps=steps, exog=exog)

def plot_forecasts(y, train_forecast, test_forecast, future_forecast, forecast_index):
    plt.figure(figsize=(16,6))
    y.plot(label='Actual', linewidth=2)
    train_forecast.plot(label='Train Forecast', linestyle='--')
    test_forecast.plot(label='Test Forecast', linestyle='--')
    pd.Series(future_forecast.predicted_mean.values, index=forecast_index).plot(label='Next Quarters Forecast', linestyle='--')
    plt.xlabel('Quarter')
    plt.ylabel('Revenue')
    plt.title('SARIMAX Quarterly Forecasting with 3 Exogenous Variables')
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # To avoid division by zero, add a small epsilon if zeros may be present
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def main():
    # Load the dataset
    df = load_data(r'C:\Users\Nav\Documents\BaggageRevenueModels\BaggageRevenueModel\data\combined_bag_revenue_exog.csv')

    # Intialize variables from dataframe
    exog_variables = df[['jetfuel_cost']] # 'jetfuel_cost', 'unemployment_rate'
    target_variable = df['y']

    # Train-test split
    y_train, y_test, exog_train, exog_test = train_test_split(target_variable, exog_variables)
    
    # Model parameters - IDK
    order = (1, 1, 2) # parameters for (p, d, q)
    seasonal_order = (1, 0, 1, 4)  # Quarterly seasonality, period=4

    results = fit_sarimax(y_train, exog_train, order, seasonal_order)
    train_forecast = predict(results, start=y_train.index[0], end=y_train.index[-1], exog=exog_train)
    test_forecast = predict(results, start=y_test.index[0], end=y_test.index[-1], exog=exog_test)

    print(results.summary())

    mape_score = get_mean_absolute_percentage_error(y_test, test_forecast)
    print(f"MAPE on test set: {mape_score:.2f}%")

    # Forecast next N quarters 
    steps_ahead = 16 # Change to desired number of quarters
    future_exog = exog_test[-steps_ahead:]  # Adjust or construct real exog for future quarters
    future_forecast = forecast_future(results, steps=steps_ahead, exog=future_exog)
    forecast_index = future_exog.index  # Should be PeriodIndex with freq='Q'

    plot_forecasts(target_variable, train_forecast, test_forecast, future_forecast, forecast_index)

    print("Next quarters forecast values:")
    print(pd.Series(future_forecast.predicted_mean.values, index=forecast_index))
 
    

if __name__ == "__main__":
    main()


