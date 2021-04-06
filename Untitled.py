#!/usr/bin/env python
# coding: utf-8

# # ARIMA and SARIMAX Model for Sales Data Forecasting

# In[118]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


df=pd.read_csv('perrin-freres-monthly-champagne-.csv')


# In[70]:


df.head()


# In[71]:


df.tail()


# In[73]:


# Converting Month into Datetime
df['Month']=pd.to_datetime(df['Month'])


# In[74]:


df.head()


# In[75]:


df.set_index('Month',inplace=True)


# In[76]:


df.head()


# # Checking for Stationarity

# In[22]:


df.describe()


# In[78]:


df.plot()


# Looking at the data we can see that data is with some seasonality and might not be stationary

# In[79]:


df.hist()


# Data is not gaussian distributed which shows that the mean and variance of two equal parts of the data would not be close and hence proves data isn't stationary

# In[23]:


df.plot()


# From the data we can see it is seasonal and also from the earlier test it proves that data is not stationary

# In[81]:


### Using Statistical tests(Dicky Fuller Test) for Stationarity

from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df['Sales'])

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[82]:


adfuller_test(df['Sales'])


# ## Differencing

# In[83]:


df.plot()


# Looking at the data, we can see the seasonality in the dataset with a period of 1 year = 12 months. Therefore we are using a differencing of 12.

# In[84]:


df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)


# In[85]:


df['Seasonal First Difference'].plot()


# In[86]:


## Testing new dataset(Seasonal First Difference) with dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())


# In[87]:


df['Seasonal First Difference'].plot()


# In[88]:


df['Seasonal First Difference'].hist()


# ## Auto Regressive Model
# 

# ### Final Thoughts on Autocorrelation and Partial Autocorrelation
# 
# * Identification of an AR model is often best done with the PACF.
#     * For an AR model, the theoretical PACF “shuts off” past the order of the model.  The phrase “shuts off” means that in theory the partial autocorrelations are equal to 0 beyond that point.  Put another way, the number of non-zero partial autocorrelations gives the order of the AR model.  By the “order of the model” we mean the most extreme lag of x that is used as a predictor.
#     
#     
# * Identification of an MA model is often best done with the ACF rather than the PACF.
#     * For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner.  A clearer pattern for an MA model is in the ACF.  The ACF will have non-zero autocorrelations only at lags involved in the model.
#     
#     p,d,q
#     p AR model lags
#     d differencing
#     q MA lags

# In[99]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()


# In[109]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm


# In[110]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)


# In[105]:


# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima_model import ARIMA


# In[106]:


model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()


# In[107]:


model_fit.summary()


# In[108]:


df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# ARIMA model prediction is not correct as the data was seasonal. So we use Seasonal Autoregressive Integrated Moving Average model

# In[111]:


model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()


# In[112]:


df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# In[113]:


from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]


# In[114]:


future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)


# In[115]:


future_datest_df.tail()


# In[116]:


future_df=pd.concat([df,future_datest_df])


# In[117]:


future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 


# # The projected model looks in line with the previous model
