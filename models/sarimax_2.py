from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler


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

def sarimax_no_exog_fit_predict(series, forecast_periods=16, plot_title="SARIMAX (No Exogenous Variables)", test_actuals=None):
    """
    Fit and forecast a SARIMAX model without exogenous variables, optionally print metrics before plotting.
    """

    import pmdarima as pm
    import matplotlib.pyplot as plt
    import pandas as pd

    # Fit SARIMAX with no exogenous factors
    model = pm.auto_arima(
        series, 
        seasonal=True,
        m=4,  # Quarterly
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic',
        n_jobs=-1
    )

    # Forecast
    index_of_fc = pd.date_range(
        series.index[-1].to_timestamp() + pd.DateOffset(months=3),
        periods=forecast_periods,
        freq='QS'
    )
    fitted, conf_int = model.predict(n_periods=forecast_periods, return_conf_int=True)

    # Compute metrics if test_actuals are provided
    if test_actuals is not None:
        y_pred = fitted if hasattr(fitted, 'values') else np.array(fitted)
        y_true = test_actuals
        mae, rmse, mape = calculate_forecast_metrics(y_true, y_pred)

    forecast_df = pd.DataFrame({
        'Quarter': index_of_fc,
        'Forecast': fitted,
        'Lower_CI': conf_int[:, 0],
        'Upper_CI': conf_int[:, 1]
    })
    return model, forecast_df


# def main():
#     # Load data
#     df = load_and_format(r'C:\Users\Nav\Documents\BaggageRevenueModels\BaggageRevenueModel\data\combined_bag_revenue_exog.csv')
    
#     # Filter for Southwest Airlines only
#     airline = 'Southwest'
#     airline_df = df[df['unique_id'] == airline].copy()
#     airline_df.set_index('ds', inplace=True)
#     airline_df.index = airline_df.index.to_period('Q')
    
#     # Exogenous variable columns
#     exog_cols = ['GDP', 'jetfuel_cost'] #, 'unemployment_rate'
#     exog_variables = airline_df[exog_cols]
    
#     # Fit SARIMAX model
#     sarimax_model = sarimax_fit(airline_df['y'], exog_variables)
    
#     # Print model summary
#     print(f"\n{airline} Model Summary:")
#     print(sarimax_model.summary())
    
#     # Generate forecast for 8 quarters
#     forecast_df = sarimax_forecast(airline_df, sarimax_model, airline, exog_cols, periods=8)
#     print(f"COMPLETED: Successfully processed {airline}")
    
#     return sarimax_model, forecast_df


# if __name__ == "__main__":
#     model, forecast = main()

def main():
    # Load data
    df = load_and_format(r'C:\Users\Nav\Documents\BaggageRevenueModels\BaggageRevenueModel\data\combined_bag_revenue_exog.csv')

    airline = 'Southwest'
    airline_df = df[df['unique_id'] == airline].copy()
    if airline_df.empty:
        print(f"Error: {airline} not found in dataset!")
        return None, None

    # Set datetime index and convert to quarterly period
    airline_df = airline_df.set_index('ds')
    airline_df.index = airline_df.index.to_period('Q')
    y = airline_df['y']

    # Use last n quarters for testing
    forecast_periods = 8
    y_train = y.iloc[:-forecast_periods]
    y_test = y.iloc[-forecast_periods:]

    # 1. SARIMAX WITHOUT exogenous factors
    print(f"\nTraining range: {y_train.index.min()} to {y_train.index.max()}")
    print(f"Testing range: {y_test.index.min()} to {y_test.index.max()}")
    model_no_exog, forecast_no_exog_df = sarimax_no_exog_fit_predict(
        y_train,
        forecast_periods=forecast_periods,
        plot_title="SARIMAX (No Exogenous Variables)"
    )
    y_pred_no_exog = forecast_no_exog_df['Forecast'].values
    print(f"\nForecast evaluation for {airline} (no exogenous features):")
    _, _, mape_no_exog = calculate_forecast_metrics(y_test.values, y_pred_no_exog)
    print(f"MAPE without exogenous variables: {mape_no_exog:.2f}%")

    # 2. SARIMAX WITH exogenous factors (MinMax scaled)
    exog_cols = ['GDP', 'jetfuel_cost', 'unemployment_rate']  # or your desired columns
    exog_train = airline_df[exog_cols].iloc[:-forecast_periods].dropna()  # Clean NaNs
    exog_test = airline_df[exog_cols].iloc[-forecast_periods:].dropna()  # Clean NaNs

    # MinMax scale only exogenous factors
    scaler = MinMaxScaler()
    exog_train_scaled = pd.DataFrame(
        scaler.fit_transform(exog_train),
        columns=exog_cols,
        index=exog_train.index  # Preserve index
    )
    exog_test_scaled = pd.DataFrame(
        scaler.transform(exog_test),
        columns=exog_cols,
        index=exog_test.index  # Preserve index
    )

    # Align y_train to scaled exog (if NaN cleaning dropped rows)
    common_idx = exog_train_scaled.index.intersection(y_train.index)
    y_train_aligned = y_train.loc[common_idx]
    exog_train_scaled = exog_train_scaled.loc[common_idx]

    # Quick diagnostics (optional: verify scaling and variation)
    print(f"\nExog train shape after MinMax scaling: {exog_train_scaled.shape} (should match y_train: {len(y_train_aligned)})")
    print(f"Exog test shape after MinMax scaling: {exog_test_scaled.shape} (should be {forecast_periods})")
    print(f"Sample exog_test_scaled (range [0,1]):\n{exog_test_scaled.head()}")
    print(f"Exog test variation (std >0 indicates signal):\n{exog_test_scaled.describe().loc['std']}")

    # Fit SARIMAX with scaled exog
    sarimax_model_exog = sarimax_fit(y_train_aligned, exog_train_scaled)

    # Build test forecast with scaled exogenous variables
    index_of_fc = pd.date_range(
        y_train_aligned.index[-1].to_timestamp() + pd.DateOffset(months=3),
        periods=forecast_periods,
        freq='QS'
    )
    y_pred_exog, confint = sarimax_model_exog.predict(
        n_periods=forecast_periods,
        return_conf_int=True,
        exogenous=exog_test_scaled.values  # Use scaled test exog
    )

    # Align y_test if needed (unlikely, but for completeness)
    y_test_aligned = y_test.loc[exog_test_scaled.index] if len(exog_test_scaled) < len(y_test) else y_test

    print(f"\nForecast evaluation for {airline} (with MinMax scaled exogenous variables):")
    _, _, mape_exog = calculate_forecast_metrics(y_test_aligned.values, y_pred_exog)
    print(f"MAPE with MinMax scaled exogenous variables: {mape_exog:.2f}%")

    # Optional: Compare predictions to baseline
    pred_diff = np.mean(np.abs(y_pred_exog - y_pred_no_exog[:len(y_pred_exog)]))
    print(f"Mean absolute difference in predictions (exog vs. no-exog): {pred_diff:.2f} (higher = exog influencing)")

    return (model_no_exog, forecast_no_exog_df), (sarimax_model_exog, y_pred_exog)

if __name__ == "__main__":
    (_, forecast_no_exog), (_, forecast_exog) = main()
