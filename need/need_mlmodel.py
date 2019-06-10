# machine learning packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# get precision / RMSE
from sklearn import metrics
from math import sqrt



# filepath to CLEANED need data
filepath = 'https://media.githubusercontent.com/media/LondonEnergyMap/cleandata/master/need/need_ldn.csv'

# read csv into dataframe
df_all = pd.read_csv(filepath)

# select data in year 2012
y = 2012

df = df_all[df_all.year==2012]
df = df.fillna(value=0)

# maximum gas and electricity consumption

maxgcons = 50000
maxecons = 25000

# --------------
# model for predicting gas consumption

x = df[['floorarea_band', 'age', 'proptype', 'epc_band', 'imd_eng', 'mainheatfuel', 'walls', 'cwi', 'loftins', 'boiler']]
y = df.gcons

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

gas_lrmodel = LinearRegression()
gas_lrmodel.fit(x_train, y_train)
predictions = gas_lrmodel.predict(x_test)
predictions_train = gas_lrmodel.predict(x_train)

gas_score = gas_lrmodel.score(x_train, y_train)

rmse = sqrt(metrics.mean_squared_error(y_test, predictions))
mse = metrics.mean_squared_error(y_test, predictions)

rmse_train = sqrt(metrics.mean_squared_error(y_train, predictions_train))

print(gas_score, rmse, mse, rmse_train)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) 

print(df.gcons.describe())

# print(y_test)
# print(predictions)

# plt.figure()
# plt.scatter(y_test, predictions, alpha=0.3)
# plt.ylim([0, maxgcons])
# plt.ylabel('predicted gas consumption')
# plt.xlabel('actual gas consumption')
# plt.title('model score = %.2f%%' %(gas_score*100))

# plt.savefig('need2012gas_lrmodel.png')

# # --------------
# # model for predicting electricity consumption

# y = df.econs

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

# elec_lrmodel = LinearRegression()
# elec_lrmodel.fit(x_train, y_train)
# predictions = elec_lrmodel.predict(x_test)

# elec_score = elec_lrmodel.score(x_train, y_train)

# plt.figure()
# plt.scatter(y_test, predictions, alpha=0.3)
# plt.ylim([0, maxecons])
# plt.ylabel('predicted elec consumption')
# plt.xlabel('actual elec consumption')
# plt.title('model score = %.2f%%' %(elec_score*100))

# plt.savefig('need2012elec_lrmodel.png')

# # --------------
# # model for predicting gas consumption using less parameters

# x = df[['floorarea_band', 'age', 'proptype']]
# y = df.gcons

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

# gas_lrmodel = LinearRegression()
# gas_lrmodel.fit(x_train, y_train)
# predictions = gas_lrmodel.predict(x_test)

# gas_score = gas_lrmodel.score(x_train, y_train)

# plt.figure()
# plt.scatter(y_test, predictions, alpha=0.3)
# plt.ylim([0, maxgcons])
# plt.ylabel('predicted gas consumption')
# plt.xlabel('actual gas consumption')
# plt.title('model score = %.2f%%' %(gas_score*100))

# plt.savefig('need2012smallgas_lrmodel.png')

# # --------------
# # model for predicting electricity consumption using less parameters

# y = df.econs

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

# elec_lrmodel = LinearRegression()
# elec_lrmodel.fit(x_train, y_train)
# predictions = elec_lrmodel.predict(x_test)

# elec_score = elec_lrmodel.score(x_train, y_train)

# plt.figure()
# plt.scatter(y_test, predictions, alpha=0.3)
# plt.ylim([0, maxecons])
# plt.ylabel('predicted elec consumption')
# plt.xlabel('actual elec consumption')
# plt.title('model score = %.2f%%' %(elec_score*100))

# plt.savefig('need2012smallelec_lrmodel.png')
