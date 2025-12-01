# assumptions.md

---

## 1. Data and Business Context Assumptions

- **Historical patterns are predictive of the future.**  
  The models assume that baggage revenue trends and seasonality from past quarters continue to hold going forward, aside from the COVID anomaly.

- **External business factors will not drastically shift.**  
  The project does not explicitly model the May 2025 policy changes or competitive dynamics, assuming historical relationships remain valid.

- **Economic indicators are exogenous and stable.**  
  Jet fuel prices, unemployment rates, and GDP per capita are treated as external drivers that are not influenced by airline baggage revenue.

---

## 2. Preprocessing Assumptions

- **COVID quarters can be removed without distorting the series.**  
  The project assumes that deleting the COVID period and stitching the remaining data preserves the true underlying structure needed for forecasting.

- **Quarterly averages are sufficient.**  
  All exogenous variables are aggregated to quarterly means, assuming intra-quarter variation does not meaningfully impact revenue.

- **Log transforming revenue is appropriate.**  
  A log transform is used to reduce skewness, assuming it improves model behavior and that back-transforming does not introduce large bias.

---

## 3. Statistical Modeling Assumptions

- **Stationarity after differencing (ARIMA/SARIMA).**  
  Time series models assume the differenced series is stationary and that linear relationships adequately describe temporal dynamics.

- **Correct seasonal structure.**  
  Quarter-based seasonality (period 4) is assumed to capture true seasonal patterns.

- **Exogenous effects are linear (SARIMAX).**  
  The contribution of economic features is assumed to be additive and linear.

---

## 4. Machine Learning Assumptions

- **Lag features sufficiently represent temporal structure.**  
  Since XGBoost is not time-aware, the assumption is that selected lag variables capture all relevant dynamics.

- **Dataset size is adequate for XGBoost.**  
  Although the dataset is small, it is assumed that regularization and validation splits prevent overfitting.

---

## 5. Key Risks From These Assumptions

- Structural breaks (like future policies) may cause forecasts to fail.  
- Removing COVID may alter long-term trend estimates.  
- Small sample size may limit the reliability of all model types.  

