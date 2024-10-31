# Solar output energy Time Series Prediction
On the date the need of renewable energy is rising stream. The world is running towards the sustainable environment for to act against climate change. To attain sustainability there are many renewable energy sources. On the note out of 100 percent of renewable energy 40 percent mainly focused on biomass energy and on second is solar energy at 3.6 percent. While the turning tides are towards renewable energy. It has some disadvantages of its own. Some of it in the solar energy are its unpredictable nature. The solar energy changes vary on the daytime. That depends on the sun and clouds and other ambient factors. Because of this the composing of solar plants to the gird system happens to fail. Unless we have a massive storage system this will continue to exists. 

	To tackle this if we know the energy output of tomorrow, we can reroute the energy system and ready other energy sources to maintain the stability of the gird. For thus we must study the solar output of today and predict the tomorrow output. 
    
	This method of approach is called time series prediction. A steep column of solar output is given to the time series model and the model predict the tomorrow values.
    
This report details a time-series analysis project focused on energy consumption data. The primary objective was to explore trends and seasonal patterns in the data and to apply forecasting techniques using ARIMA models to predict future values. This document outlines the data preparation, visualization, modeling, and evaluation steps undertaken throughout the project.


# Import the required packages



```python
#import the required packages
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

```


```python
#test dataset
twentykwdf= pd.read_csv(r"/kaggle/input/data-of-alice-spring-project-20kw/87-Site_DKA-M9_AC-Phases.csv")
twentykwdf.tail()
```

# Visualization and understanding of dataset


```python
#understanding the data
twentykwdf.describe()
```


```python
#next up is data analysis and data cleaning
type(twentykwdf['timestamp'])
```


```python
#visualization of data 
one_month_data=twentykwdf.head(1000)
plt.figure(figsize=(12, 6))
plt.plot(one_month_data['timestamp'], one_month_data['Active_Power'], label='Time Series Data')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()
# the following graph shows the active power output for three days
```


```python
twentykwdf.index
```


```python
#change the index as timestamp to resample the data for 1 hour
twentykwdf['timestamp'] = pd.to_datetime(twentykwdf['timestamp'])

# Set the 'Date' column as the index
twentykwdf.set_index('timestamp', inplace=True)

# Now, your DataFrame's index is a timestamp
twentykwdf.index
```


```python
# resampling the data for 1 hour
hourly_data=twentykwdf.Active_Power.resample('h').first()
plt.figure(figsize=(12, 6))
plt.plot(hourly_data.index, hourly_data, label='Time Series Data')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()

```


```python
#Using the 10 years of data for training will make the model more unfit for future prediction
#so we are striping the data for 4 year
#Striping last four years of data for the experiment

last_4years=hourly_data.loc['2020-01-01 00:00:00':]
print(last_4years.tail(),last_4years.head())
```


```python
plt.figure(figsize=(12, 6))
plt.plot(last_4years.index, last_4years, label='Time Series Data')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()

```


```python
#removing the nullvalues from the data set
last_4years.dropna()
```


```python
plt.figure(figsize=(12, 6))
plt.plot(last_4years.index, last_4years, label='Time Series Data')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()

```


```python
#to understand the data more clearly we have to look close
#For that ploting one month data in graph
one_month_data = hourly_data.loc['2024-01']

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(one_month_data.index, one_month_data, label='January 2024 Data', linestyle='-')
plt.title('Active Power for January 2024')
plt.xlabel('Date')
plt.ylabel('Active Power (kW)') 
plt.xticks(rotation=45)  
plt.legend()
plt.grid(True)  
plt.show()
```

This graph will help us to understand the seasonality and trend of the dataset


```python
#even though we droped the null values the might be missing values 
#so we are filling the empty data with FFILL()
#to fill the empty data
one_month_data_filled = one_month_data.ffill()

# Decompose the time series (model='additive' or 'multiplicative')
result = seasonal_decompose(one_month_data_filled, model='additive')
plt.figure(figsize=(12, 6))
result.plot()
plt.show()

```

Plots for last four years


```python

#to fill the empty data
last_4yearsfilled = last_4years.ffill()

# Decompose the time series (model='additive' or 'multiplicative')
result = seasonal_decompose(last_4yearsfilled, model='additive')
plt.figure(figsize=(12, 6))
result.plot()
plt.show()

```

on the above seasonal data is compressed as the result of daily ups and down of the data

Splitting the data set for training and testing

# Train test splitting


```python
#train test split
train_size = int(len(last_4yearsfilled) * 0.8)  # Use 80% of the data for training
train, test = last_4yearsfilled[:train_size], last_4yearsfilled[train_size:]

print(f'Training set length: {len(train)}')
print(f'Testing set length: {len(test)}')

# Visualize train and test
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.title('Train-Test Split')
plt.legend()
plt.show()

```

# Time series prediction

ARIMA model is the generalized form of time series prediction model 


```python
# Fit ARIMA model
model = ARIMA(train, order=(5,1,0))  # ARIMA(p,d,q) model
model_fit = model.fit()

# Forecast the next values (same length as test set)
forecast = model_fit.forecast(steps=len(test))

# Visualize the forecast
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()

```


```python
#evaluation of model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Calculate MSE
mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error: {mse}")

# Optionally, calculate RMSE for better interpretability
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
r2=r2_score(test,forecast)
print(f"R2 Square value is ",r2)

```

We have observerd that the model poorly perfoms And we have to imporve the models and parameters for it


```python
last_4yearsfilled

```

We take the last one year to obtain the latest prediction varibles and features


```python
one_yeardata=last_4yearsfilled.loc['2023-10-22':]
```


```python
plt.figure(figsize=(12, 6))
plt.plot(one_yeardata.index, one_yeardata, label='Time Series Data')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()

```


```python
#train test split
train_size = int(len(one_yeardata) * 0.8)  # Use 80% of the data for training
train, test = one_yeardata[:train_size], one_yeardata[train_size:]

print(f'Training set length: {len(train)}')
print(f'Testing set length: {len(test)}')

# Visualize train and test
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.title('Train-Test Split')
plt.legend()
plt.show()

```


```python
# Fit ARIMA model
model = ARIMA(train, order=(1, 1, 1))  # ARIMA(p,d,q) model
model_fit = model.fit()

# Forecast the next values (same length as test set)
forecast = model_fit.forecast(steps=len(test))

# Visualize the forecast
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()

```

To see the result closely plotting the first 5 days of model prediction


```python
plt.figure(figsize=(12, 6))
plt.plot(forecast[:100], label='forecast data')
plt.plot(test[:100],label="actual data")
plt.title('Time Series Data')

plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()

```

The above graphs shows the poor performance of the model by showing that the forecast data and actual data are completely not related

Now we have to optimize the pdq parameters for better result for the model 
# Finding the best paramerter for Arima model


```python
#finding the best p d q values for the arima models
import itertools
import warnings
warnings.filterwarnings("ignore")

# Define range of p, d, q values
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

# Use AIC as a metric for optimization
best_aic, best_pdq, best_model = float("inf"), None, None

for param in pdq:
    try:
        temp_model = ARIMA(train, order=param)
        temp_model_fit = temp_model.fit()
        if temp_model_fit.aic < best_aic:
            best_aic, best_pdq, best_model = temp_model_fit.aic, param, temp_model_fit
    except:
        continue

print("Best ARIMA model order:", best_pdq)
print("Best AIC score:", best_aic)

residuals = best_model.resid
plt.plot(residuals)
plt.title('Residuals of Best ARIMA Model')
plt.show()

```


```python
# Fit ARIMA model
model = ARIMA(train, order=(2, 0, 2))  # ARIMA(p,d,q) model
model_fit = model.fit()

# Forecast the next values (same length as test set)
forecast = model_fit.forecast(steps=len(test))

# Visualize the forecast
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()

```


```python
plt.figure(figsize=(12, 6))
plt.plot(forecast[:100], label='forecast data')
plt.plot(test[:100],label="actual data")
plt.title('Time Series Data')

plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()

```

The model imporved but it did not catch the seasonality of our dataset

# Introducing SARIMA model to understand the seasonality of data


```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
```


```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarima_model = SARIMAX(train, order=best_pdq, seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()

```


```python
# Forecast the same number of steps as the test set
forecast = sarima_fit.forecast(steps=len(test))

# Plot train, test, and forecast data
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('SARIMA Model Forecast')
plt.legend()
plt.show()

```


```python
plt.figure(figsize=(12, 6))
plt.plot(forecast[:100], label='forcast')
plt.plot(test[:100],label="actual data")
plt.title('Time Series Data')

plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()
```


```python
import numpy as np
#evaluation of model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Calculate MSE
mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error: {mse}")

# Optionally, calculate RMSE for better interpretability
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
r2=r2_score(test[:100],forecast[:100])
print(f"R2 Square value is ",r2)
```

# Changing the parameters for SARIMAX


```python
#SARIMA next wave 
from statsmodels.tsa.statespace.sarimax import SARIMAX
sarima_model = SARIMAX(train, order=best_pdq, seasonal_order=(2, 1, 1, 12))
sarima_fit = sarima_model.fit()
# Forecast the same number of steps as the test set
forecast = sarima_fit.forecast(steps=len(test))

# Plot train, test, and forecast data 
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('SARIMA Model Forecast')
plt.legend()
plt.show()
```


```python
plt.figure(figsize=(12, 6))
plt.plot(forecast[:200], label='forcast')
plt.plot(test[:200],label="actual data")
plt.title('Time Series Data')

plt.yticks(rotation=0)
plt.xticks(rotation=0)

plt.legend()
plt.show()
```


```python
# Calculate MSE
mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error: {mse}")

# Optionally, calculate RMSE for better interpretability
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
r2=r2_score(test,forecast)
print(f"R2 Square value is ",r2)
```

# Conclusion
This project provided valuable insights into energy consumption trends and highlighted critical steps in time-series forecasting. Although the ARIMA model showed limited accuracy, it underscored the importance of model selection and tuning in time-series forecasting.But the SARIMAX model performed well and shows the r2 score of .89 that ensure the model performs is promising . Future work could involve refining model parameters, using alternative models, or incorporating additional features to improve prediction reliability.


