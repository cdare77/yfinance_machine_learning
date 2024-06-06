#%%
########################
# Load Libraries
#######################

import yfinance as yf
import pandas as pd
import numpy as np
import sys

import matplotlib.dates as mdates
from datetime import date, datetime, timedelta
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from plotnine import ggplot, aes, geom_line, scale_x_date





#%%

class StockDataWrapper:

    def __init__(self, data_frame, stock_symbol):
        self.data_frame = data_frame
        self.stock_symbol = stock_symbol
        self.granularity = 0
        self.num_days = 0
        self.lag_length = 0

    def __str__(self):
        my_str = f'Stock Name: {self.stock_symbol}\n'
        if self.granularity == 0:
            my_str += "Granularity of data not set\n"
        else:
            my_str += f"Granularity: {self.granularity}\n"
        
        if self.num_days == 0:
            my_str += "Number of days of stock data not set\n"
        else:
            my_str += f"Number of days of stock data: {self.num_days}\n"
        
        if self.lag_length == 0:
            my_str += "Number of lag variables not set\n"
        else:
            my_str += f"Number of lag variables: {self.lag_length}\n"

        my_str += "\n\n"
        my_str += str(self.data_frame)

        return my_str
    

    def compute_maximal_lag_variables(self):

        if self.granularity == 0 or self.num_days == 0:
            print("""Both the granularity and number of days of stock data
                   must be set prior to computing the maximum number of lag variables""")
            return -1
        
        data_entries = -4
        if self.granularity == '1m':
            data_entries = 389 * self.num_days
        elif self.granularity == '2m':
            data_entries = 195 * self.num_days
        elif self.granularity == '5m':
            data_entries = 78 * self.num_days
        elif self.granularity == '15m':
            data_entries = 26 * self.num_days
        elif self.granularity == '30m':
            data_entries = 13 * self.num_days
        elif self.granularity == '1h':
            data_entries = 7 * self.num_days
        else:
            print("""Granularity is not set properly: must be set to 
                  one of: 1m, 2m, 5m, 15m, 30m, 1h\n""")
            return -1
        
        return int(0.3 * data_entries)
    
    


#%%
#######################
# Import DataFrame
########################


def create_stock_data_from_input():

    user_code = 0

    while user_code == 0:
        num_days_to_build = input("""How many days of intraday stock market
                                data should we use to build our model? Enter
                                a value between 1 and 7:\n
 
                                (Type 'Exit' to quit)\n""")

        if num_days_to_build.isnumeric():
            num_days_to_build = int(num_days_to_build)
            if num_days_to_build < 8 and num_days_to_build > 0:
                user_code = 1
            else:
                print("""The number of days must be between 1 and 7
                       — please retry. \n (Type 'Exit' to quit)\n""")
        elif "exit" in num_days_to_build.lower():
            sys.exit("Exiting program")
        else:
            print("""Non-integer passed as input — please retry.
                  \n
                  (Type 'Exit' to quit)\n""")


    today = date.today()
    num_days_prior = today - timedelta(num_days_to_build)

    granularity_options = ['1m', '2m', '5m', '15m', '30m', '1h']
    user_code = 0

    while user_code == 0:
        granulrity_input = input("""How often should our model look at
                                  stock prices? Choose from 1m, 2m, 5m, 15m, 30m or 1h.\n
                                 
                                 (Type 'Exit' to quit)\n""").lower()
        
        if granulrity_input in granularity_options:
            user_code = 1
        elif "exit" in granulrity_input:
            sys.exit("Exiting program")
        else:
            print("""Input was not among the options 1m, 2m, 5m, 15m,
                   30m, or 1h — please retry.\n
                  
                  (Type 'Exit' to quit)\n""")
            
    
    stock_symbol = input("Please input the stock symbol you would like to examine: (e.g. AAPL)").upper()
        


    df = pd.DataFrame(yf.download(stock_symbol,
                                start=num_days_prior,
                                end=today,
                                interval=granulrity_input)
                            )
    
    stock_data = StockDataWrapper(df, stock_symbol)
    stock_data.num_days = num_days_to_build
    stock_data.granularity = granulrity_input

    return stock_data



# %%
###########################
# Create Lag Variables
###########################

def add_lag_variables_to_df(data_frame, num_lags):

    for i in range(1,num_lags + 1):
        index_str = "Close_L" + str(i)
        data_frame[index_str] = data_frame['Close'].shift(i)

    # Backfill the entries to remove any NaN
    data_frame = data_frame.bfill(axis=0)

    return data_frame


def add_lags_from_input(stock_data):

    user_code = 0
    max_lag_len = stock_data.compute_maximal_lag_variables()
    if max_lag_len == -1:
        print("Cannot set lag variables with current settings")
        return

    # Clean up previous entries
    stock_data.data_frame = stock_data.data_frame[stock_data.data_frame.columns.drop(
        list(stock_data.data_frame.filter(regex='Close_L'))
        )]

    while user_code == 0:
        lag_length = input("How many previous data points should our model look at?\n (Type 'Exit' to quit)")

        if lag_length.isnumeric():
            lag_length = int(lag_length)
            if lag_length <= max_lag_len:
                stock_data.lag_length = lag_length
                stock_data.data_frame = add_lag_variables_to_df(stock_data.data_frame,
                                                                lag_length)
                user_code = 1
            else:
                print("The number of previous data points considered should" +
                      f" not exceed 30%% \nof the total number of data points (in this case, {max_lag_len}) -- please" +
                       " retry. \n\n(Type 'Exit' to quit)")
        elif "exit" in num_days_to_build.lower():
                sys.exit("Exiting program")
        else:
            print("""Non-integer passed as input — please retry.
                  \n
                  (Type 'Exit' to quit)\n""")



# # Predictors
# df_lags = df.filter(regex='Close_L')
# # Outcome
# output_close=df["Close"]





# %%
# regr = linear_model.LinearRegression()

# regr.fit(df_lags, output_close)
# last_datapoint= df.filter(regex='Close_L').iloc[[-1]]


# print("Predicted: " + str(regr.predict(last_datapoint)))
# print(df.iloc[[-1]])




# %%


def simple_moving_average(data_frame, lag_length):
    lag_predictors = []
    for i in range(1, lag_length + 1):
        lag_name = 'Close_L' + str(i)
        lag_predictors.append(lag_name)
    
    return (data_frame[lag_predictors].sum(axis = 1, skipna = True)/lag_length)


def add_simple_moving_average(stock_data, lag_length):
    
    if not isinstance(lag_length, int):
        print("Parameter lag_length must be integer")
        return
    if stock_data.lag_length == 0:
        print("Lag variables must be set prior to adding simple moving average")
        return
    if lag_length > stock_data.lag_length:
        print("Cannot take the average of more lag varaibles than are availible" + 
              f" (currently {stock_data.lag_length})\n")
        return
    
    column_label = "SMA_"  + str(lag_length)
    stock_data.data_frame[column_label] = simple_moving_average(stock_data.data_frame,
                                                                 lag_length)



    


# %%
(trainX, testX, trainY, testY) = train_test_split(df_lags,
	output_close, random_state=43, test_size=0.25)
scaler = preprocessing.StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)


parametersGrid = { "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}
eNet = ElasticNet()


grid = GridSearchCV(eNet, parametersGrid, scoring='r2', cv=10)
grid.fit(trainX, trainY)
predY = grid.predict(testX)
print(predY)

# %%
stonks = create_stock_data_from_input()
p = ggplot(stonks.data_frame, aes(x=stonks.data_frame.index, y='Close')) + geom_line() + scale_x_date(date_labels =  '%m-%d %H:%M') 
p.show()

add_lags_from_input(stonks)

print(stonks)

add_simple_moving_average(stonks, 25)
ggplot(stonks.data_frame, aes(x=stonks.data_frame.index, y='SMA_25')) + geom_line() + scale_x_date(date_labels =  '%m-%d %H:%M') 

# %%
