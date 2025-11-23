## Baggage-Revenue-by-Quarter.jpg
- Multi-airline comparison of baggage revenue over quarters from 2015 to 2024.
- Shows the sharp drop due to COVID-19.

## logTansformedBaggageRevenue_vs_NPS.jpg
- Scatterplot of log-transformed baggage revenue vs. Net Promoter Score (NPS).
- Negative correlation: higher NPS relates to lower revenue from baggage fees.
- Line fit illustrates the overall trend.

## models_vs_y.jpg
- Actual vs forecasted baggage revenue for several airlines using ARIMA, SARIMA, and Seasonal Naive models.
- Plots indicate model accuracy for recent years across airlines.
- Most models capture recent post-pandemic trend surges.

## baggage_revenue_vs_NPS.jpg
- Scatterplot of absolute baggage revenue vs. NPS for airlines.
- Strong negative relationship: airlines with low NPS generate more baggage revenue.
- Clear outliers can be seen in both directions.

## Southwest_exogenous_pvalues.jpg
- Statistical analysis of exogenous variables affecting Southwest's baggage revenue.
- Shows which predictors (GDP, unemployment rate, jet fuel) are significant.
- Only unemployment rate is statistically significant at alpha=0.05.

## Southwest_no_exog_forecast.jpg
- SARIMAX model forecast for Southwest baggage revenue without exogenous factors.
- Test and predicted values shown with confidence intervals.
- The model captures general quarterly patterns with some variance.

## Baggage-Revenue-By-Quarter-SW.jpg
- Southwest Airlines revenue by quarter from 2015–2024.
- Displays steady growth with a large drop in 2020 and quick recovery afterward.
- Recent quarters show highest-ever revenue.

## mean_shap-1.jpg
- Mean SHAP values ranking feature importance for baggage revenue prediction.
- Top features: quarter, revenue lag (y_lag4), jet fuel cost, unemployment rate.
- Lesser importance for time-related rolling means and standard deviations.

## shap-1.jpg
- SHAP summary plot showing feature impact on model output for baggage revenue forecast.
- Highest impacts: quarter, past revenue, jet fuel cost, and unemployment rate.
- Color bar denotes feature value scale from low to high.
