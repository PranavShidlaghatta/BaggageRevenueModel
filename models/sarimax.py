from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def load_and_format(path):
    """
    Load combined baggage revenues along with exogenous factors
    """
    df = pd.read_csv(path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(['unique_id', 'ds'])
    return df


def sarimax_fit(y, exog=None):
    """
    2-step approach:
      1) Use pmdarima.auto_arima to select (order, seasonal_order)
      2) Fit a statsmodels SARIMAX with those orders and the exogenous vars
    """

    stepwise_model = pm.auto_arima(
        y,
        exogenous=exog,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        m=4,                     # quarterly
        seasonal=True,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None,                  # let auto_arima pick d
        D=None,                  # let auto_arima pick D
        test="adf",
        seasonal_test="ch",      # seasonal differencing test
        information_criterion="aicc",
        trace=False,             # Disabled to reduce output
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        n_jobs=1                 # stepwise search is single-threaded
    )

    print("\nChosen orders from auto_arima:")
    print(f"  order = {stepwise_model.order}")
    print(f"  seasonal_order = {stepwise_model.seasonal_order}")

    order = stepwise_model.order
    seasonal_order = stepwise_model.seasonal_order

    sarimax_model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = sarimax_model.fit(disp=False)

    return results, order, seasonal_order


def calculate_forecast_metrics(y_true, y_pred):
    """
    Compute MAE, RMSE, and MAPE between actual and predicted values.
    """
    y_true = np.expm1(y_true) 
    y_pred = np.expm1(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    eps = np.finfo(float).eps
    mape = np.mean(
        np.abs((y_true - y_pred) / np.where(y_true == 0, eps, y_true))
    ) * 100

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    return mae, rmse, mape


def sarimax_no_exog_fit_predict(series, forecast_periods=16,
                                plot_title="SARIMAX (No Exogenous Variables)",
                                test_actuals=None):
    """
    Fit and forecast a SARIMAX model without exogenous variables.
    Uses pmdarima.auto_arima directly for a quick baseline.
    """

    model = pm.auto_arima(
        series,
        seasonal=True,
        m=4,  # Quarterly
        trace=False,  # Disabled to reduce output
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic',
        n_jobs=1  # stepwise=True → single-threaded
    )

    # Forecast
    index_of_fc = pd.date_range(
        series.index[-1].to_timestamp() + pd.DateOffset(months=3),
        periods=forecast_periods,
        freq='QS'
    )
    fitted, conf_int = model.predict(
        n_periods=forecast_periods,
        return_conf_int=True
    )

    # Compute metrics if test_actuals are provided
    if test_actuals is not None:
        y_pred = np.asarray(fitted)
        y_true = np.asarray(test_actuals)
        mae, rmse, mape = calculate_forecast_metrics(y_true, y_pred)

    forecast_df = pd.DataFrame({
        'Quarter': index_of_fc,
        'Forecast': fitted,
        'Lower_CI': conf_int[:, 0],
        'Upper_CI': conf_int[:, 1]
    })
    return model, forecast_df


def display_exogenous_pvalues(sarimax_results, exog_cols, airline_name="Southwest", alpha=0.05):
    """
    Extract and display p-values for exogenous variables from SARIMAX model.
    
    Parameters:
    -----------
    sarimax_results : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        Fitted SARIMAX model results
    exog_cols : list
        List of exogenous column names
    airline_name : str
        Name of airline for display
    alpha : float
        Significance level (default 0.05 for 95% confidence)
    
    Returns:
    --------
    exog_stats_df : pd.DataFrame
        DataFrame containing coefficients, std errors, z-scores, p-values for exogenous vars
    """
    
    print("\n" + "="*80)
    print(f"Statistical Significance Test for Exogenous Variables - {airline_name}")
    print("="*80)
    
    # Get all parameter names
    param_names = sarimax_results.param_names
    
    # Find indices for exogenous variables
    exog_indices = []
    exog_param_names = []
    for i, param in enumerate(param_names):
        for col in exog_cols:
            if col in param:
                exog_indices.append(i)
                exog_param_names.append(param)
                break
    
    if not exog_indices:
        print("Warning: No exogenous variable parameters found in model!")
        return None
    
    # Extract statistics for exogenous variables
    coefficients = sarimax_results.params[exog_indices]
    std_errors = sarimax_results.bse[exog_indices]
    z_scores = coefficients / std_errors
    pvalues = sarimax_results.pvalues[exog_indices]
    conf_int = sarimax_results.conf_int(alpha=alpha)
    conf_lower = conf_int.iloc[exog_indices, 0]
    conf_upper = conf_int.iloc[exog_indices, 1]
    
    # Create DataFrame with results
    exog_stats_df = pd.DataFrame({
        'Variable': exog_param_names,
        'Coefficient': coefficients.values,
        'Std Error': std_errors.values,
        'z-score': z_scores.values,
        'P-value': pvalues.values,
        f'{int((1-alpha)*100)}% CI Lower': conf_lower.values,
        f'{int((1-alpha)*100)}% CI Upper': conf_upper.values,
        'Significant': ['Yes' if p < alpha else 'No' for p in pvalues.values]
    })
    
    print(f"\nSignificance Level: α = {alpha}")
    print(f"Null Hypothesis (H0): Coefficient = 0 (variable has no effect)")
    print(f"Alternative Hypothesis (H1): Coefficient ≠ 0 (variable has an effect)")
    print("\n" + "-"*80)
    print(exog_stats_df.to_string(index=False))
    print("-"*80)
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    
    significant_vars = exog_stats_df[exog_stats_df['P-value'] < alpha]
    non_significant_vars = exog_stats_df[exog_stats_df['P-value'] >= alpha]
    
    if len(significant_vars) > 0:
        print(f"\n✓ STATISTICALLY SIGNIFICANT variables (p < {alpha}):")
        print("-" * 80)
        for idx, row in significant_vars.iterrows():
            direction = "POSITIVE" if row['Coefficient'] > 0 else "NEGATIVE"
            print(f"\n  • {row['Variable']}")
            print(f"    - Coefficient: {row['Coefficient']:.6f}")
            print(f"    - P-value: {row['P-value']:.6f}")
            print(f"    - Effect: {direction} impact on baggage revenue")
            print(f"    - 95% CI: [{row[f'{int((1-alpha)*100)}% CI Lower']:.6f}, "
                  f"{row[f'{int((1-alpha)*100)}% CI Upper']:.6f}]")
            print(f"    - Interpretation: There is strong statistical evidence that this")
            print(f"      variable affects baggage revenue (we reject H0).")
    
    if len(non_significant_vars) > 0:
        print(f"\n\n✗ NOT STATISTICALLY SIGNIFICANT variables (p >= {alpha}):")
        print("-" * 80)
        for idx, row in non_significant_vars.iterrows():
            print(f"\n  • {row['Variable']}")
            print(f"    - Coefficient: {row['Coefficient']:.6f}")
            print(f"    - P-value: {row['P-value']:.6f}")
            print(f"    - 95% CI: [{row[f'{int((1-alpha)*100)}% CI Lower']:.6f}, "
                  f"{row[f'{int((1-alpha)*100)}% CI Upper']:.6f}]")
            print(f"    - Interpretation: Insufficient evidence to conclude this variable")
            print(f"      has a significant effect (we fail to reject H0).")
            print(f"      Consider removing this variable from the model.")
    
    print("\n" + "="*80)
    print("NOTE ON SCALING:")
    print("="*80)
    print("Since your exogenous variables are MinMax scaled [0,1], the coefficients")
    print("represent the change in log(baggage revenue) for a full range change")
    print("(min to max) of each variable. Larger coefficient magnitudes indicate")
    print("stronger effects, but p-values determine statistical significance.")
    print("="*80 + "\n")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Coefficients with confidence intervals
    colors = ['green' if p < alpha else 'red' for p in exog_stats_df['P-value']]
    y_pos = np.arange(len(exog_stats_df))
    
    ax1.barh(y_pos, exog_stats_df['Coefficient'], color=colors, alpha=0.6)
    ax1.errorbar(exog_stats_df['Coefficient'], y_pos,
                 xerr=[exog_stats_df['Coefficient'] - exog_stats_df[f'{int((1-alpha)*100)}% CI Lower'],
                       exog_stats_df[f'{int((1-alpha)*100)}% CI Upper'] - exog_stats_df['Coefficient']],
                 fmt='none', ecolor='black', capsize=5)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(exog_stats_df['Variable'])
    ax1.set_xlabel('Coefficient Value', fontweight='bold')
    ax1.set_title(f'Exogenous Variable Coefficients\n(Green=Significant, Red=Not Significant)', 
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: P-values
    ax2.barh(y_pos, exog_stats_df['P-value'], color=colors, alpha=0.6)
    ax2.axvline(x=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(exog_stats_df['Variable'])
    ax2.set_xlabel('P-value', fontweight='bold')
    ax2.set_title('Statistical Significance (P-values)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{airline_name} - Exogenous Variables Statistical Test', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{airline_name}_exogenous_pvalues.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return exog_stats_df

def check_multicollinearity(exog_df):
    """
    Calculate VIF for each exogenous variable to detect multicollinearity.
    VIF > 10 indicates high multicollinearity
    VIF > 5 indicates moderate multicollinearity
    """
    print("\n" + "="*80)
    print("MULTICOLLINEARITY CHECK (Variance Inflation Factor)")
    print("="*80)
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = exog_df.columns
    vif_data["VIF"] = [variance_inflation_factor(exog_df.values, i) 
                       for i in range(len(exog_df.columns))]
    
    print("\nVIF Interpretation:")
    print("  VIF = 1: No correlation with other variables")
    print("  VIF = 1-5: Moderate correlation")
    print("  VIF = 5-10: High correlation (consider removing)")
    print("  VIF > 10: Severe multicollinearity (should remove)")
    print("\n" + "-"*80)
    print(vif_data.to_string(index=False))
    print("-"*80)
    
    # Highlight problematic variables
    high_vif = vif_data[vif_data['VIF'] > 5]
    if len(high_vif) > 0:
        print("\n  WARNING: High multicollinearity detected:")
        for idx, row in high_vif.iterrows():
            print(f"  • {row['Variable']}: VIF = {row['VIF']:.2f}")
        print("\nConsider removing variables or using different combinations.")
    else:
        print("\n✓ No severe multicollinearity detected.")
    
    print("="*80 + "\n")
    return vif_data

def main():
    # Load data
    df = load_and_format(
        r'C:\Users\Nav\Documents\BaggageRevenueModels\BaggageRevenueModel\data\combined_bag_revenue_exog.csv'
    )

    airline = 'Southwest'
    airline_df = df[df['unique_id'] == airline].copy()
    if airline_df.empty:
        print(f"Error: {airline} not found in dataset!")
        return None, None

    # Set datetime index and convert to quarterly period
    airline_df = airline_df.set_index('ds')
    airline_df.index = airline_df.index.to_period('Q')
    y = airline_df['y']

    # Apply log transform
    y = np.log1p(y)  # log(1 + y) to handle zero values

    # Use last n quarters for testing
    forecast_periods = 8
    y_train = y.iloc[:-forecast_periods]
    y_test = y.iloc[-forecast_periods:]

    # 1. SARIMAX WITHOUT exogenous factors
    print(f"\nTraining range: {y_train.index.min()} to {y_train.index.max()}")
    print(f"Testing range: {y_test.index.min()} to {y_test.index.max()}")
    
    print("\n" + "="*80)
    print("MODEL 1: SARIMAX WITHOUT Exogenous Variables")
    print("="*80)
    model_no_exog, forecast_no_exog_df = sarimax_no_exog_fit_predict(
        y_train,
        forecast_periods=forecast_periods,
        plot_title="SARIMAX (No Exogenous Variables)"
    )
    y_pred_no_exog = forecast_no_exog_df['Forecast'].values
    print(f"\nForecast evaluation for {airline} (no exogenous features):")
    _, _, mape_no_exog = calculate_forecast_metrics(y_test.values, y_pred_no_exog)

    # 2. SARIMAX WITH exogenous factors (MinMax scaled)
    exog_cols = ['jetfuel_cost', 'unemployment_rate', 'GDP']  # adjust if you add more exogenous vars
    exog_train = airline_df[exog_cols].iloc[:-forecast_periods].dropna()
    exog_test = airline_df[exog_cols].iloc[-forecast_periods:].dropna()

    # MinMax scale only exogenous factors
    scaler = MinMaxScaler()
    exog_train_scaled = pd.DataFrame(
        scaler.fit_transform(exog_train),
        columns=exog_cols,
        index=exog_train.index
    )
    exog_test_scaled = pd.DataFrame(
        scaler.transform(exog_test),
        columns=exog_cols,
        index=exog_test.index
    )

    # Align y_train to scaled exog
    common_idx = exog_train_scaled.index.intersection(y_train.index)
    y_train_aligned = y_train.loc[common_idx]
    exog_train_scaled = exog_train_scaled.loc[common_idx]

    print("\n" + "="*80)
    print("MODEL 2: SARIMAX WITH Exogenous Variables (MinMax Scaled)")
    print("="*80)
    print(f"Exog train shape: {exog_train_scaled.shape} | y_train shape: {len(y_train_aligned)}")
    print(f"Exog test shape: {exog_test_scaled.shape}")

    # Fit SARIMAX with scaled exog via 2-step approach
    sarimax_results_exog, order_exog, seas_order_exog = sarimax_fit(
        y_train_aligned,
        exog_train_scaled
    )

    # Forecast with exogenous variables
    forecast_res = sarimax_results_exog.get_forecast(
        steps=forecast_periods,
        exog=exog_test_scaled
    )
    y_pred_exog = forecast_res.predicted_mean
    confint = forecast_res.conf_int(alpha=0.05)

    # Align y_test if needed
    y_test_aligned = y_test.loc[exog_test_scaled.index] \
        if len(exog_test_scaled) < len(y_test) else y_test

    print(f"\nForecast evaluation for {airline} (with MinMax scaled exogenous variables):")
    _, _, mape_exog = calculate_forecast_metrics(
        y_test_aligned.values,
        y_pred_exog.values
    )

    # Compare predictions to baseline
    pred_diff = np.mean(
        np.abs(y_pred_exog.values - y_pred_no_exog[:len(y_pred_exog)])
    )
    print(f"\nMean absolute difference in predictions (exog vs. no-exog): {pred_diff:.4f}")
    print(f"MAPE improvement: {mape_no_exog - mape_exog:.2f}% (negative = worse with exog)")

    # Build forecast DF for exog case (for symmetry with no-exog)
    forecast_exog_df = pd.DataFrame({
        'Quarter': y_pred_exog.index.to_timestamp(),
        'Forecast': y_pred_exog.values,
        'Lower_CI': confint.iloc[:, 0].values,
        'Upper_CI': confint.iloc[:, 1].values
    })

    # 3. DISPLAY P-VALUES
    exog_stats = display_exogenous_pvalues(
        sarimax_results=sarimax_results_exog,
        exog_cols=exog_cols,
        airline_name=airline,
        alpha=0.05
    )

    # 4. CHECK MULTICOLLINEARITY
    vif_data = check_multicollinearity(exog_train_scaled)

    return (model_no_exog, forecast_no_exog_df), (sarimax_results_exog, forecast_exog_df), exog_stats


if __name__ == "__main__":
    (_, forecast_no_exog), (_, forecast_exog), exog_statistics = main()
