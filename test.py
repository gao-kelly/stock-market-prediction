import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

path = "/Users/kelly/PycharmProjects/stock_market_data/"


# df = pd.DataFrame()
# for f in files:
    # csv = pd.read_csv(f)
    # df = df.append(csv)

data = pd.read_csv("/Users/kelly/PycharmProjects/stock_market_data/forbes2000/csv/A.csv")
# data.head()
# data.info()
# data.describe()

# split data into x and y, create arrays
x = data[['High', 'Low', 'Open', 'Volume']].values
y = data['Close'].values
# print(x)
# print(y)

# split data into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# create regression model
from sklearn.linear_model import LinearRegression
Model = LinearRegression()

# train model
Model.fit(x_train, y_train)

# print coefficient
print(Model.coef_)

# use model to make predictions
predicted = Model.predict(x_test)
print(predicted)

# actual vs predicted data
data1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted' : predicted.flatten()})
data1.head(20)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted))
print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, predicted)))

graph = data1.head(20)
graph.plot(kind='bar')