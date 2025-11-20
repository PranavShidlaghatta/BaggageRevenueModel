from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose


def load_and_format(path):
    """
    Load combined baggage revenues along with exogenous factors
    """
    df = pd.read_csv(path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(['unique_id', 'ds'])
    df['y_diff'] = df['y'].diff(periods=4)
    return df

def sarimax_fit(y, exog):
    SARIMAX_model = pm.auto_arima(
        y, exogenous=exog,
        start_p=1, start_q=1,
        test='adf',
        max_p=3, max_q=3, m=4,  # Quarterly seasonality
        start_P=0, seasonal=True,
        max_P=3, max_Q=3,
        d=None, D=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic',
        n_jobs=-1  # Uses all CPU cores
    )
    
    return SARIMAX_model

def sarimax_forecast(df, SARIMAX_model, airline_name, exog_cols, periods):
    n_periods = periods
    last_exog_values = df[exog_cols].iloc[-1].to_dict()
    
    # Create forecast dataframe with exogenous variables
    # Using quarterly frequency 'QS' (quarter start) or 'Q' (quarter end)
    forecast_exog_df = pd.DataFrame(
        {col: [last_exog_values[col]] * n_periods for col in exog_cols},
        index=pd.date_range(
            df.index[-1].to_timestamp() + pd.DateOffset(months=3),
            periods=n_periods,
            freq='QS'  # Quarterly frequency
        )
    )
    
    # Generate forecast with confidence intervals
    fitted, confint = SARIMAX_model.predict(
        n_periods=n_periods,
        return_conf_int=True,
        exogenous=forecast_exog_df[exog_cols]
    )
    
    # Create index for forecast
    index_of_fc = pd.date_range(
        df.index[-1].to_timestamp() + pd.DateOffset(months=3),
        periods=n_periods,
        freq='QS'
    )
    
    # Make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    
    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(df.index.to_timestamp(), df['y'], 
             label='Historical Baggage Revenue', color='#1f76b4', linewidth=2)
    plt.plot(fitted_series, label='Forecast', color='darkgreen', linewidth=2)
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15,
                     label='95% Confidence Interval')
    
    plt.title(f"SARIMAX - Forecast of {airline_name} Baggage Revenue", fontsize=14, fontweight='bold')
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Baggage Revenue ($1000s)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create forecast dataframe for export
    forecast_df = pd.DataFrame({
        'Quarter': index_of_fc,
        'Forecast': fitted,
        'Lower_CI': confint[:, 0],
        'Upper_CI': confint[:, 1]
    })
    
    print(f"\nForecast for {airline_name} (Next {n_periods} Quarters):")
    print(forecast_df.to_string(index=False))
    
    return forecast_df

def calculate_forecast_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE, and MAPE between actual and predicted values.
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    eps = np.finfo(float).eps
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, eps, y_true))) * 100
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return mae, rmse, mape

def main():
    # Load data
    df = load_and_format(r'C:\Users\Nav\Documents\BaggageRevenueModels\BaggageRevenueModel\data\combined_bag_revenue_exog.csv')
    
    # Filter for Southwest Airlines only
    airline = 'Southwest'
    airline_df = df[df['unique_id'] == airline].copy()
    airline_df.set_index('ds', inplace=True)
    airline_df.index = airline_df.index.to_period('Q')
    
    # Exogenous variable columns
    exog_cols = ['GDP', 'jetfuel_cost', 'unemployment_rate']
    exog_variables = airline_df[exog_cols]
    
    # Fit SARIMAX model
    sarimax_model = sarimax_fit(airline_df['y'], exog_variables)
    
    # Print model summary
    print(f"\n{airline} Model Summary:")
    print(sarimax_model.summary())
    
    # Generate forecast for 8 quarters
    forecast_df = sarimax_forecast(airline_df, sarimax_model, airline, exog_cols, periods=8)
    print(f"COMPLETED: Successfully processed {airline}")
    
    return sarimax_model, forecast_df


if __name__ == "__main__":
    model, forecast = main()
