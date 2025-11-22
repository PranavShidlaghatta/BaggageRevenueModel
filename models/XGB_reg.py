# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import shap 


df = pd.read_csv(r"/home/rayan/Southwest_stuff/BaggageRevenueModel/data/combined_bag_rev_exog.csv")
df.drop('Unnamed: 0', axis=1, inplace=True)

# Accounts for right skewed distribution in baggage revenue column
df['y'] = np.log1p(df['y'])

df['ds'] = pd.to_datetime(df['ds'])

def create_lag_features(df, group_col, target, lags):
    for lag in lags:
        df[f"{target}_lag{lag}"] = df.groupby(group_col)[target].shift(lag)
    return df

def create_rolling_features(df, group_col, target, windows):
    for w in windows:
        df[f"{target}_roll_mean_{w}"] = (
            df.groupby(group_col)[target].shift(1).rolling(w).mean()
        )
        df[f"{target}_roll_std_{w}"] = (
            df.groupby(group_col)[target].shift(1).rolling(w).std()
        )
    return df

def create_lagged_exogenous(df, group_col, exogenous_cols, lags=[1,4]):
    for col in exogenous_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df

df = df.sort_values(["unique_id", "ds"])

df["year"] = df["ds"].dt.year
df["quarter"] = df["ds"].dt.quarter

df = create_lag_features(df, "unique_id", "y", lags=[1,2,4,8])
df = create_rolling_features(df, "unique_id", "y", windows=[4,8])
df = create_lagged_exogenous(df, "unique_id",
                             ["jetfuel_cost", "unemployment_rate", "GDP"])

df = df.dropna()

train = df[df["ds"] < "2020-01-01"]
valid = df[(df["ds"] >= "2020-01-01") & (df["ds"] < "2021-07-01")]
test  = df[df["ds"] >= "2021-07-01"]

robust_cols = [c for c in df.columns if ("unemployment" in c) or ("GDP" in c)]
robust_scaler = RobustScaler()
train[robust_cols] = robust_scaler.fit_transform(train[robust_cols])
valid[robust_cols] = robust_scaler.transform(valid[robust_cols])
test[robust_cols]  = robust_scaler.transform(test[robust_cols])


scale_cols = [
    c for c in df.columns
    if ("jetfuel" in c) or ("unemployment" in c) or ("GDP" in c)
]
scaler = StandardScaler()
train[scale_cols] = scaler.fit_transform(train[scale_cols])
valid[scale_cols] = scaler.transform(valid[scale_cols])
test[scale_cols]  = scaler.transform(test[scale_cols])

feature_cols = [c for c in df.columns if c not in ["y", "ds", "unique_id"]]

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.5],
    'reg_lambda': [1, 2]
}

xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
    verbose=2
)

grid_search.fit(train[feature_cols], train["y"])

print("Best parameters found:", grid_search.best_params_)

best_model = grid_search.best_estimator_

# Optionally, retrain on all train+valid data for final test
# concat_trainvalid = pd.concat([train, valid])
# best_model.fit(concat_trainvalid[feature_cols], concat_trainvalid["y"])
# y_pred = best_model.predict(test[feature_cols])

# Use best model on your hold-out test set
y_pred = best_model.predict(test[feature_cols])

# --- Calculate MAE and MAPE ---
mae = mean_absolute_error(test["y"], y_pred)
print(f"MAE: {mae:.2f}")

mape = (np.abs((test["y"] - y_pred) / np.where(test["y"] == 0, np.nan, test["y"]))).mean() * 100
print(f"MAPE: {mape:.2f}%")


# %%

print(df)

# %%

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(train[feature_cols])

shap.summary_plot(shap_values, train[feature_cols], plot_type="bar")

# %%
shap.summary_plot(shap_values, train[feature_cols])
