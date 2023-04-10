# Automated Time Series class

# Required libraries
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.trend import TrendForecaster
from sktime.forecasting.model_selection import (
   ForecastingGridSearchCV,
   ExpandingWindowSplitter)
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.model_selection import ExpandingWindowSplitter
import numpy as np
import pandas as pd
import holidays
from datetime import date, timedelta


class AutoTS:
    
    """
    
    An automated time series class using the sk-time library.

    This class takes in a time series of varying frequency, evaluates a suite of models
    and returns the best performing model.

    """

    def __init__(self, input_df, target, freq, sp, exog=None, holidays_list=None):
        
        """
        
        Initialize the automated time series class.

        Args:
            input_df (Pandas DataFrame): Dataframe which includes target column, and any
            additional regressors. Must have datetime index.
            
            target (string): The target column included in input_df.
            
            freq (string): i.e., 'd' for daily, 'MS' for month start etc...
            
            sp (int): Seasonal period, i.e., for daily data set to 7.
            
            exog (string/list of strings): exogenous variables included in input_df. 
            
            holidays_list: string or list of strings.
            
            Select from the following - 
            'All', "New Year's Day", 'Christmas Day', 'Good Friday', 'Easter Monday',
            'May Day', 'Spring Bank Holiday', 'Late Summer Bank Holiday',
            'Boxing Day'

        """              
        
        self.input_df = input_df 
        self.target = target 
        self.exog = exog  
        self.freq = freq
        self.sp = sp
        self.holidays_list = holidays_list
        
        # Attributes to check with holidays or regressors are used in model.
        
        self.use_holidays = False 
        self.using_regressors = False
        
        if self.exog is not None:
            self.using_regressors = True
            
        if self.holidays_list is not None:
            self.use_holidays = True 

            
            
    def create_holiday_dataframe(self, dates):
        
            '''
            Args:
            
                dates: datetime - Array of dates. Function will check each date and test whether 
                it is a holiday.
            
            '''
            
            self.use_holidays = True 
            
            # Create a dict-like object for England's public holidays
            uk_holidays = holidays.UK(state = 'England', years=range(dates.min().year, 
                                                                     dates.max().year+1))
            
            if self.holidays_list == 'All':
                holiday_vals = ["New Year's Day", 'Christmas Day', 'Good Friday', 'Easter Monday',
                                'May Day', 'Spring Bank Holiday', 'Late Summer Bank Holiday',
                               'Boxing Day']
            else:
                holiday_vals = self.holidays_list
            
            uk_holidays = pd.DataFrame(uk_holidays.items(), columns = ['ds', 'holiday'])
            prophet_holiday_df = uk_holidays[pd.to_datetime(uk_holidays.ds).isin(dates)]
            prophet_holiday_df = prophet_holiday_df[prophet_holiday_df.holiday.isin(holiday_vals)]
            
            exog_holiday_df = pd.DataFrame()
            exog_holiday_df['holiday_datetime_index'] = dates
            for x in prophet_holiday_df.holiday.unique():
                exog_holiday_df[x] = exog_holiday_df.holiday_datetime_index.isin(
                    prophet_holiday_df[prophet_holiday_df.holiday == x].ds)*1
                  
            
            exog_holiday_df = exog_holiday_df.set_index('holiday_datetime_index', drop=True)

            self.exog_holiday_df = exog_holiday_df.sort_index()
            
        
   
    def ts_cv(self, forecast_horizon, folds, use_regressor_forecast=False, perf_metric='MAPE'):
        
        
        '''
        
        Carries out time series cross validates on dataset.
        
        Args:
        
            forecast_horizon (int): Number of timesteps to forecast on each fold.
            
            folds (int): Number of folds to use in cross validation.
            
            use_regressor_forecast (bool, default: False): Specifies whether
            actual regressor values are used, or forecasted values are used 
            in cross validation.
            
            perf_metric (string, default: 'MAPE'): Performance metric, minimise chosen
            performance metric to select best model during cross valiadtion.
            
        Outputs:
        
            cv_summary (Pandas DataFrame): MAPE values averages for each model, 
            during cross validation period.
            
            best_model (string): Name of model with lowest MAPE.
            
            cv_df (Pandas DataFrame): Forecasted values and actual values during 
            cross validation.
            
            trained_models (dictionary): Saved trainined models.
                                    
        
        '''
        
        self.forecast_horizon = forecast_horizon
        self.folds = folds
        self.use_regressor_forecast = use_regressor_forecast
        self.perf_metric = perf_metric
                
        
        # Defining dataset used for cross validation.
        input_df = self.input_df                   

        # Defining dictionary of models to compare (can easily add more sktime models).
        models_dict = {
                       'Facebook Prophet': Prophet(), 
                       'ETS': AutoETS(auto=True, sp=self.sp, n_jobs=-1), 
                       'ARIMA': StatsForecastAutoARIMA(sp=self.sp) , 
                       'Seasonal Naive': NaiveForecaster(sp=self.sp, strategy='last'),
                       'STL': STLForecaster(sp=self.sp)
                        }   
        
        # Cross validation periods.
        cv = ExpandingWindowSplitter(initial_window = len(input_df) - (self.folds*self.forecast_horizon),
                                     step_length=self.forecast_horizon,
                                     fh = np.arange(1, (self.forecast_horizon+1)))
        
            
        y = input_df[self.target]        
        if self.exog is not None:
            X = input_df[self.exog].astype('float')  
        else:        
            X = None
        if self.use_holidays: 
            self.create_holiday_dataframe(dates = input_df.index)
            X = pd.concat([X, self.exog_holiday_df], axis = 1).astype('float') 
        
        
        trained_models = {}
        model_cv_results= {}
        for m in models_dict:

            # Store testing/prediciton results from cross validation.
            y_test_list = []
            y_pred_list = []
            
            for idx, x in enumerate(list(cv.split(input_df))):
                
                # Pandas series (x[0] refers to training from cross valiadtion. split).
                y_train = y.iloc[x[0]]
                
                # Checking if using regressors, if not, X_train = None.
                if self.exog is not None:
                    X_train = X.iloc[x[0]]
                else:
                    X_train = None
                
                
                # Model from dictionary defined above.
                forecaster = models_dict[m]
                
                # Fit model to data
                model = forecaster.fit(y = y_train,
                                       X = X_train,
                                       fh = np.arange(1, 1+self.forecast_horizon))
                
                # If forecasting regressors in cross valiadtion.
                if use_regressor_forecast:
                    
                    # Current regressor forecast.
                    X_forecaster = NaiveForecaster(sp=self.sp, strategy='last')
                    exog_list = []
                    
                    # Iterate through regressor list.
                    for exog_col in self.exog: 
                        X_forecaster.fit(X_train[exog_col])
                        x_pred = X_forecaster.predict(fh = np.arange(1, 1+self.forecast_horizon))
                        exog_list.append(x_pred.to_frame(name = exog_col))

                        exog_forecasts = pd.concat(exog_list, axis = 1)
                        X_test = X.iloc[x[1]]
                        X_test.loc[:, self.exog] = exog_forecasts

                # Using actual regressor vals. in cross validation:
                elif not self.using_regressors:
                    X_test = None
                else: 
                    X_test = X.iloc[x[1]]    
                
                # Append forecasts list.
                y_pred_list.append(model.predict(fh = np.arange(1, 1 + len(x[1])), X = X_test))
                # Appent actual lists.
                y_test_list.append(input_df[self.target].iloc[x[1]])

                
            # Fit all data (this will be the output models ready for forecasting)
            model.fit(y = y, X = X)
            self.X_train = X
            
            # Output corss validation dataframe.
            df_out = pd.concat(y_test_list).to_frame(name = 'y_test')
            df_out['y_pred'] = np.concatenate(y_pred_list)
            df_out['model'] = m
                          
            # Save models to dictionary
            model_cv_results[m] = df_out
            trained_models[m] = model
           
        # Cross validation attributes.
        self.cv_df = pd.concat(model_cv_results.values())
        self.trained_models = trained_models
        
        # Calculating forecasting error.
        df_find_best = pd.concat(model_cv_results.values())
        df_find_best['MAPE'] = round(abs(df_find_best.y_test - df_find_best.y_pred)/df_find_best.y_test , 3)
        df_find_best['MSE'] = round((df_find_best.y_test - df_find_best.y_pred) ** 2, 3)
        df_find_best['MAE'] = round(abs(df_find_best.y_test - df_find_best.y_pred), 3)
        
        cv_results = df_find_best[['model', 'MAPE', 'MSE', 'MAE']].groupby(by = 'model').mean()
        
        # Cross validation summary dataframe.
        self.cv_summary = cv_results.sort_values(by = self.perf_metric)
        # Best model (based on MAPE)
        self.best_model = cv_results.MAPE.idxmin()
        

        
    def forecast(self, timesteps, pi, model = None, exog_df = None, exog_to_forecast = None):
        
        '''
        
        Create forecasts beyond input dataset.
        
        Requires ts_cv() method to be called.
        
        Args:
        
            timesteps: (int): Forecast horizon (for given timestep frequency)
            
            pi: (float): prediction interval coverage (float or int.)
            
            model (string, default: model with lowest MAPE): Must be one of: 'Facebook Prophet', 
            'ETS', 'ARIMA', 'Seasonal Naive', 'STL'.
            
            exog_df (Pandas Dataframe): Dataframe with regressor values in future (with datetime index of same frequency 
            as input_df). Must have len = timesteps with same column names.
            
            exog_to_forecast (str or list of str): List of column names to forecast.
            
        Outputs:
        
            forecast_df (Pandas dataframe): forecasted dataframe, with length equal to timesteps.
        
        '''   
        
        self.timesteps = timesteps
        self.pi = pi
        self.model = model
        self.exog_df = exog_df
        self.exog_to_forecast = exog_to_forecast
        
        input_df = self.input_df
        
        # Checking whether to use best_model or other model specified by user.
        if self.model is None:
            self.model = self.best_model
        
        # Checking whether input is string, if it is, convert to list of the string.
        if type(exog_to_forecast) == str:
            self.exog_to_forecast = [self.exog_to_forecast]

        # Defining certain warning messages.
        if self.exog is not None:
            if (self.exog_df is None) and (self.exog_to_forecast is None):
                print('Warning: Expected exogenous values as an argument, but got none!')
            elif (self.exog_df is None) and (self.exog_to_forecast is not None):
                print('Forecasting some or all exogenous features in forecast.')
            
        ## Future dataframe.    
        start_dt = input_df.index.max()
        forecast_dates = pd.date_range(start_dt, periods = self.timesteps + 1, freq=self.freq)
        forecast_dates = forecast_dates[1:]
    
        # Defining exog dataframe.
        if self.using_regressors:
            
            if self.exog_to_forecast is not None:
            
                exog_forecaster = NaiveForecaster(strategy="last", sp=self.sp)
                exog_list = []
                for exog in self.exog_to_forecast:
                    exog_forecaster.fit(input_df[exog])
                    y_pred = exog_forecaster.predict(fh = np.arange(1, 1 + self.timesteps))
                    exog_list.append(y_pred.to_frame(name = exog))
                    
                if self.exog_df is None:
                    X = pd.concat(exog_list, axis = 1)
                else:
                    X = pd.concat([self.exog, pd.concat(exog_list, axis = 1)], axis = 1)

            # Using actual regressor vals. in cross validation:
            else: 
                X = self.exog_df     
        else:
            X = None
            
        if self.use_holidays:
            
            self.create_holiday_dataframe(dates = pd.concat([pd.Series(forecast_dates),
                                                             pd.Series(input_df.index)]))
            
            X = pd.concat([X, self.exog_holiday_df.loc[forecast_dates.min():].sort_index()], axis = 1).astype('float')
            
        # Output forecast.  
        forecast = self.trained_models[self.model].predict(fh = np.arange(1, 1 + self.timesteps),
                                                                               X = X)
        output = forecast.to_frame(name = 'yhat')
        
        # Seeing if model has prediction interval capabilities.
        try:
            forecast_interval = self.trained_models[self.model].predict_interval(fh = np.arange(1, 1 + self.timesteps),
                                           X = X, coverage = self.pi)
            output['yhat_lower'] = forecast_interval.iloc[:, 0].values
            output['yhat_upper'] = forecast_interval.iloc[:, 1].values
        except:
            print('Forecaster does not have prediction interval capabilities')

        # Output forecast pandas dataframe.
        self.forecast_df = output
            

   
    