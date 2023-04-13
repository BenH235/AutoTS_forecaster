# AutoTS_forecaster

A python library that can automate a number of time series forecasting processes, including:

- Conducting time series cross validation on a set of time series models (ETS, ARIMA, STL, Prophet and Naive).
- Producing evaluation metrics (MAPE, MAE, MSE) as outputs from cross validation
- Automatically selects model based on minimising chosen performance metric
- Once cross validation is called, you can then create forecasts from selected model (or by specifying other model).

It also includes the ability to include the impact of public holidays (currently only for daily data) and exogenous variables. If using exogenous variables, it is also able to forecast them seperately if the values are not known during the forecasted period (and forecasted seperately in cross validation, if specified).
