#!/usr/bin/env python
# coding: utf-8

# Final Project:
# 
# -dataset retrieved from yahoo finance 
# -time series analysis on stocks
# - perform a time series analysis on the stock you input
# -dashboard displaying stock information
# 



#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
import datetime
import time
import requests
import io
import os

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, Input, Output , State
from plotly.subplots import make_subplots
import plotly.graph_objects as go




#function to create our arima model
def arima_model(df_train,label,pred_star,pred_end,order,stock_model_name):
    model = ARIMA(df_train, order=order)
    res = model.fit()
    # save model
    res.save(stock_model_name+'.pkl')
    forecast = res.get_forecast(pred_end).summary_frame()
    #return the data frame with all our forecasted values
    return forecast




#testing stationary trends. 
def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput


#lets use differencing method to test.
#differencing will transform the model from non-stationary to stationary
# taking the log removes a lot of the increase in variance - smooths out variance - closer to stationary
#add ing exponential weighting to improve the confidence interal.
def trend_removal(df):
    ts_log = np.log(df['Close'])
    ts_diff = ts_log - ts_log.shift()
    ts_diff_exp = ts_diff  - ts_diff.ewm(halflife = 12).mean()
    ts_diff_exp.dropna(inplace = True)
    return ts_diff_exp




#start and end times for our stock dataframe
start = datetime.datetime(2022,2,1)
end = datetime.datetime(2022,4,18)




#instructions to convert stock market data into dataframe https://towardsdatascience.com/downloading-historical-stock-prices-in-python-93f85f059c1f
url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))






Symbols = companies['Symbol'].tolist()
test_symbols=Symbols[0:3]
print(test_symbols)



# create empty dataframe
stock_final = pd.DataFrame()
# iterate over each symbol
for i in test_symbols:  
    
    try:
        # download the stock price 
        stock = []
        stock = yf.download(i,start=start, end=end, progress=False)
        
        # append the individual stock prices 
        if len(stock) == 0:
            None
        else:
            stock['Name']=i
            stock_final = stock_final.append(stock,sort=False)
    except Exception:
        None



#get rid of volume and adj close dont need them
stock_final2=stock_final[['Open','High','Low','Close','Name']]
stock_final2.tail()



#makingchecklist options for general stock data
checklist_options=stock_final2.columns[0:4]
print(checklist_options)



#getting unique stock names for the dropdownbar
stock_names=stock_final['Name'].unique()



app = Dash(__name__)


app.layout = html.Div([
    html.H4('Stock price market Dashboard'),
    html.P("Select stock:"),
    dcc.Dropdown(
        id="dropdown",
        options=stock_names,
        multi=False,
        value='AAL',
        style={'width':110},
    ),
    dcc.Graph(id="graph"),
    dcc.Graph(id="graph2"),
    dcc.Graph(id="table1"),
    html.Span("Dick Fuller Test:"),
    dcc.Graph(id="table2"),
    html.P(" For the Dick Fuller Test, If we get a p-value of greater than 0.05 and if our test stastictic is more than our critical values."),
    html.P(" Then it is suggest to remove trends by taking log and performing exponential differencing."),
    dcc.Graph(id="graph3"),
    dcc.Graph(id="table3"),
    html.Span("Dick Fuller Test:"),
    dcc.Graph(id="table4"),
    html.P("Usually for differencing the p value should be very close to zero and test statistic just be valid for 99%."),
    html.P("you will want to use the exponential method has it wil tell you a better shape for the trend of which the stock price is going"),
    html.P("perhaps take more data points to reduce the nois of a series. "),
])  

@app.callback(
    Output("graph", "figure"),
    Output("graph2", "figure"),
    Output("table1", "figure"),
    Output("table2", "figure"),
    Output("graph3", "figure"),
    Output("table3", "figure"),
    Output("table4", "figure"),
    Input("dropdown", "value"),
)

def display_graph(value):
    df=stock_final2
    df=df[df['Name']==value]
    #plotting the general stock information
    fig= go.Figure(px.line(df,x=df.index,y='Close',title='Stock Prices'))

    #take the dataframe and resample to fill in weekends
    #plug the open price for our arima model. forrecast the weekely price
    df2=df.resample('D').sum()
    df2.Close=df2.Close.replace(to_replace=0.000000 , method='ffill').values

    #getting our data from open stock prices to train our arima model
    #(1,1,1) first-order autoregressive model
    ar_mod=arima_model(df2.Close,'Forecast','2022-02-01','2022-4-21',(1,1,1),value)

    #adding a sceond figure with the time series forecast
    fig2= go.Figure(px.line(ar_mod,x=ar_mod.index,y='mean',title="Time Serie Analysis"))
    fig2.update_traces(line_color='red')
    
    #array to make into datable
    arr=ar_mod.values
    table1= go.Figure(data=[go.Table(
        header=dict(values=['mean','mean_se','mean_ci_lower','mean_ci_upper'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[arr[:,0],arr[:,1],arr[:,2],arr[:,3]],
                fill_color='lavender',
                align='left'))
    ])

    #performing a dick fuller test to see if we need to remove trends.store values dataframe
    fuller_test1=test_stationarity(df2.Close)
    #make data table of fuller test
    arr_fuller=fuller_test1.values
    arr_fuller=arr_fuller.round(decimals=3)
    table2= go.Figure(data=[go.Table(
        header=dict(values=['Test Statistic','p-value','#Lags Used','Number of Observations Used','Critical Value (1%)','Critical Value (5%)','Critical Value (10%)'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[arr_fuller[0],arr_fuller[1],arr_fuller[2],arr_fuller[3],arr_fuller[4],arr_fuller[5],arr_fuller[6]],
                fill_color='lavender',
                align='left'))
    ])

    result = seasonal_decompose(df2.Close, model='additive')
    result.plot()
    plt.show()

    #if p-value is less than 0.05. implement trend removal
    tr_df=trend_removal(df2)

    #visualizing the partial and auto correlations
    plot_acf(tr_df, lags=20,title=value +' auto correlation')
    plot_pacf(tr_df, lags=20,title=value + ' partial correlation')

    #plotting the series with trend removal
    #differencing,and exponential smoothing: 
    ar_mod2=arima_model(tr_df,'Forecast','2022-02-01','2022-4-21',(0,2,1),'AMZN')
    fig3= go.Figure(px.line(ar_mod2,x=ar_mod2.index,y='mean',title="Trend Removing Analysis of Time Series"))
    fig3.update_traces(line_color='green')

    #making the sceond data table for our arima model with the values of the exponential differencing
    arr2=ar_mod2.values
    table3= go.Figure(data=[go.Table(
        header=dict(values=['mean','mean_se','mean_ci_lower','mean_ci_upper'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[arr2[:,0],arr2[:,1],arr2[:,2],arr2[:,3]],
                fill_color='lavender',
                align='left'))
    ])

    #performing a dick fuller test to see if we need to remove trends.store values dataframe
    fuller_test2=test_stationarity(tr_df)
    #make data table of fuller test
    arr_fuller2=fuller_test2.values
    arr_fuller2=arr_fuller2.round(decimals=3)
    table4= go.Figure(data=[go.Table(
        header=dict(values=['Test Statistic','p-value','#Lags Used','Number of Observations Used','Critical Value (1%)','Critical Value (5%)','Critical Value (10%)'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[arr_fuller2[0],arr_fuller2[1],arr_fuller2[2],arr_fuller2[3],arr_fuller2[4],arr_fuller2[5],arr_fuller2[6]],
                fill_color='lavender',
                align='left'))
    ])

    return [fig,fig2,table1,table2,fig3,table3,table4]


server_port = os.environ.get('PORT') or 9000
app.run_server(port=server_port)



# loading model example
loaded = ARIMAResults.load('AMZN.pkl')
forecast = loaded.get_forecast('2022-4-21').summary_frame()
print(forecast)

