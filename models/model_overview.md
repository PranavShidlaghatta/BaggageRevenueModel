# SARIMAX 


# XGBoost 

This model performs regression using XGBoost (or extreme gradient boosted trees).

The target is a log transform of the baggage revenue using base e. This was to mitigate its strong right skewed distribution. Robust scaling is applied to exogenous columns as well due to a lack of normality. 

XGBoost is an ensemble tree based model that, unlike SARIMA/SARIMAX does not have temporal awareness. 
- Boosting builds sequential trees that learns from previous residuals (reducing error from previous predictions). 
- The gradient with respect to the loss function (and hessian as second derivative) guide tree creation. 

As this is regression, the sum of outputs from all trees are used. 

Lagged values are created from previous bag revenue data points, alongside rolling averages of the target. The same lagged features are also created for exogenous features. 

The data split is sequential to maintain temporal order (rather than random). Training data is before 2020-01-01, validation from 2020-01-01 to 2021-07-01, and testing from 2021-07-01 onwards. 

Hyperparameter tuning is used using negative mean absolute error and grid search. MAE and RMSE are calculated based on the log scale and reverted using `npm.expm1` for actual values. MAPE is also calculated. 

Reproduce training results by changing the path to the file `combined_bag_rev_exog.csv` from the data folder. 

# SARIMAX

This model performs time-series forecasting using SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors), a statistical model that captures both patterns, trends, and seasonality within time-series data.

The target is a log transform of the baggage revenue using base e. This was to mitigate its strong right skewed distribution. The model incorporates exogenous variables (GDP per capita, jet fuel cost, unemployment rate) to enhance predictive accuracy.

SARIMAX is a temporal-aware model that explicitly models time dependencies through autoregressive and moving average components:

Autoregressive (AR) components use past values of the series to predict future values

Differencing (I) makes the model stationary by removing trends

Moving Average (MA) components use past forecast errors to improve predictions

Seasonal components capture quarterly patterns in baggage revenue

Choice of Parameters: Picking the correct SARIMAX parameters is a crucial part of building the model. It can have a significant impact on the model's performance. Over-differencing (making the model too stationary) can cause loss of too much information.

Key Finding: From the bar chart, GDP per capita and jet fuel cost have the most moderate impact, though they are not statistically significant exogenous factors at conventional levels.

The table shows that jet fuel cost and GDP per capita have VIF values indicating multicollinearity (7.01 and 6.11 respectively), suggesting these variables are correlated with each other within the model.

The data split is sequential to maintain temporal order. MAE and RMSE are calculated based on the log scale and reverted using np.expm1 for actual values. MAPE is also calculated.

Reproduce training results by changing the path to the file combined_bag_rev_exog.csv from the data folder.


