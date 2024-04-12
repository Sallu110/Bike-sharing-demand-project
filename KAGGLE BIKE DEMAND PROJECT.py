
# kaggle bike project 

import matplotlib.pyplot as plt 
import math
import numpy as np
import pandas as pd

# read the file 

bikes = pd.read_csv('dataset.csv')

# step 2 --prelim Analysis and feature selection.

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index','date','casual','registered'],axis = 1)

# basic checks of missing values 
bikes_prep.isnull().sum()

# visualize the data using histogram 

bikes_prep.hist(rwidth = 0.9)
plt.tight_layout()


# visualooze the continous feature vs demand using scatter plot 
plt.subplot(2,2,1)
plt.title('atemperature vs demand')
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s = 0.1, c = 'b')


plt.subplot(2,2,2)
plt.title('temperature vs demand')
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s = 0.1, c= 'g')


plt.subplot(2,2,3)
plt.title('humidity vs demand')
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s = 0.1, c = 'r')


plt.subplot(2,2,4)
plt.title('windspeed vs demand')
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s = 0.1, c = 'm')

plt.tight_layout()


# plot categorical features vs demand 

plt.subplot(3,3,1)
plt.title('average demand Vs season')

# create the list of unique seasonal values 
cat_list = bikes_prep['season'].unique()

# create average demand per season using groupby 

cat_average = bikes_prep.groupby('season').mean()['demand']

colors = ['m','b','r','g']

plt.bar(cat_list,cat_average,color = colors)



plt.subplot(3,3,2)
plt.title('average demand Vs month')

# create the list of unique seasonal values 

cat_list = bikes_prep['month'].unique()

# create average demand per season using groupby 

cat_average = bikes_prep.groupby('month').mean()['demand']

colors = ['m','b','r','g']

plt.bar(cat_list,cat_average,color = colors)

plt.subplot(3,3,4)
plt.title('average demand Vs year')

# create the list of unique seasonal values 
cat_list = bikes_prep['year'].unique()

# create average demand per season using groupby 

cat_average = bikes_prep.groupby('year').mean()['demand']

colors = ['m','b','r','g']

plt.bar(cat_list,cat_average,color = colors)


plt.subplot(3,3,5)
plt.title('average demand Vs hour')

# create the list of unique seasonal values 
cat_list = bikes_prep['hour'].unique()

# create average demand per season using groupby 

cat_average = bikes_prep.groupby('hour').mean()['demand']

colors = ['m','b','r','g']

plt.bar(cat_list,cat_average,color = colors)


plt.subplot(3,3,6)
plt.title('average demand Vs weekday')

# create the list of unique seasonal values 
cat_list = bikes_prep['weekday'].unique()

# create average demand per season using groupby 

cat_average = bikes_prep.groupby('weekday').mean()['demand']

colors = ['m','b','r','g']

plt.bar(cat_list,cat_average,color = colors)


plt.subplot(3,3,7)
plt.title('average demand Vs workingday')

# create the list of unique seasonal values 
cat_list = bikes_prep['workingday'].unique()

# create average demand per season using groupby 

cat_average = bikes_prep.groupby('workingday').mean()['demand']

colors = ['m','b','r','g']

plt.bar(cat_list,cat_average,color = colors)


plt.subplot(3,3,8)
plt.title('average demand Vs holiday')

# create the list of unique seasonal values 
cat_list = bikes_prep['holiday'].unique()

#create average demand per season using groupby 

cat_average = bikes_prep.groupby('holiday').mean()['demand']

colors = ['m','b','r','g']

plt.bar(cat_list,cat_average,color = colors)

plt.tight_layout()

#----------------------------------------------------------------------------

#assumptions of multiple linear regression 

#----------------------------------------------------------------------------

# check for outliers.

bikes_prep['demand'].describe()

bikes_prep['demand'].quantile([0.05,0.10,0.15,0.90,0.95,0.99])

# srep:4 check multiple linear regression assumptions. 

# linearity with corelation coefficient matrix using corr

correlation = bikes_prep[['atemp','temp','humidity','windspeed','demand']].corr()

bikes_prep = bikes_prep.drop(['weekday','year','workingday','atemp','windspeed'],axis = 1)


# check the autocorelation in demand using accor

df1 = pd.to_numeric(bikes_prep['demand'],downcast = 'float')
plt.acorr(df1, maxlags = 12)

# step 6: create and modify new features 

# log normalize the feature 'demand'

df1 = bikes_prep['demand']
df2 = np.log(df1) 

plt.figure()
df1.hist(rwidth = 0.9,bins = 20)

plt.figure()
df2.hist(rwidth = 0.9,bins = 20)

bikes_prep['demand'] = np.log(bikes_prep['demand'])

# autocorrelation in demand column
t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lags = pd.concat([bikes_prep,t_1,t_2,t_3], axis = 1)

bikes_prep_lags = bikes_prep_lags.dropna()

# step : 7 
# create dummy variables and drop first to avoid dummy variable trap using get_dummies.
# s session, holiday, weather, hour, month 

bikes_prep_lags.dtypes

bikes_prep_lags['season'] = bikes_prep_lags['season'].astype('category')
bikes_prep_lags['holiday'] = bikes_prep_lags['holiday'].astype('category')
bikes_prep_lags['weather'] = bikes_prep_lags['weather'].astype('category')
bikes_prep_lags['hour'] = bikes_prep_lags['hour'].astype('category')
bikes_prep_lags['month'] = bikes_prep_lags['month'].astype('category')

bikes_prep_lags = pd.get_dummies(bikes_prep_lags, drop_first = True)

# create train and test split

# split the X and Y dataset into training and testing set 



Y = bikes_prep_lags[['demand']]
x = bikes_prep_lags.drop(['demand'],axis = 1)

#create the size of 70% of the data 

tr_size  = 0.7*len(x)
tr_size = int(tr_size)

# create the train and test using the tr_size 
x_train = x.values[0:tr_size]
x_test = x.values[tr_size:len(x)]

Y_train = Y.values[0:tr_size]
Y_test = Y.values[tr_size:len(Y)]

from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()


std_reg.fit(x_train,Y_train)

r2_train = std_reg.score(x_train,Y_train)
r2_test = std_reg.score(x_test,Y_test)

# CREATE Y PREDICTION 
y_predict = std_reg.predict(x_test)

from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(Y_test , y_predict))


# final step - calculate RMSLE and campare results 
y_test_e = []
y_predict_e = []

for i  in range(0, len(Y_test)):
    y_test_e.append(math.exp(Y_test[i]))
    y_predict_e.append(math.exp(y_predict[i]))
log_sq_sum = 0.0
for i in range(0,len(y_test_e)):
    log_a = math.log(y_test_e[i]+1)
    log_p = math.log(y_predict_e[i]+1)
    log_diff = (log_a - log_p)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = math.sqrt(log_sq_sum/len(Y_test))


print(rmsle)

# PTOJECT END 


