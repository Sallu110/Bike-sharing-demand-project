
# Bike Demand Prediction Using Machine Learning

## Objective
To predict bike demand based on various features using machine learning models and improve prediction accuracy through feature elimination.

## Dataset
Source: Kaggle

## Features:
1. Weather
2. Temperature
3. Holiday
4. Weekday
5. Working Day
6. Humidity
7. Wind Speed
8. Casual Users
9. Registered Users
10. Year
11. Season
12. Month
13. Hour

## Data Preprocessing
Reading the File
bikes = pd.read_csv('dataset.csv')

## Preliminary Analysis and Feature Selection
```python
bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index', 'date', 'casual', 'registered'], axis=1)
bikes_prep.isnull().sum()
```
# Visualizing the Data

## Histograms for data distribution:
```python
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()
```
![Screenshot 2024-07-15 190044](https://github.com/user-attachments/assets/2f368787-48de-445a-bfc3-a8f8cd2f88a4)


## Scatter plots for continuous features vs. demand:
```python
plt.subplot(2,2,1)
plt.title('Temperature vs Demand')
plt.scatter(bikes_prep['temp'], bikes_prep['demand'], s=0.1, c='g')

plt.subplot(2,2,2)
plt.title('atemperature vs demand')
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s = 0.1, c = 'b')

plt.subplot(2,2,3)
plt.title('Humidity vs Demand')
plt.scatter(bikes_prep['humidity'], bikes_prep['demand'], s=0.1, c='r')

plt.subplot(2,2,4)
plt.title('Windspeed vs Demand')
plt.scatter(bikes_prep['windspeed'], bikes_prep['demand'], s=0.1, c='m')
plt.tight_layout()
```
![Screenshot 2024-07-15 190254](https://github.com/user-attachments/assets/5e7e4636-2e00-435b-974e-75e4c1480b79)



## Categorical Features vs. Demand
```python
plt.subplot(3,3,1)
plt.title('Average Demand Vs Season')

# creat the list of unique seasonal values 
cat_list = bikes_prep['season'].unique()

# create average demand per season using groupby
cat_average = bikes_prep.groupby('season').mean()['demand']
colors = ['m', 'b', 'r', 'g']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,2)
plt.title('average demand Vs month')
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
colors = ['m','b','r','g']
plt.bar(cat_list,cat_average,color = colors)

plt.subplot(3,3,4)
plt.title('average demand Vs year')
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
colors = ['m','b','r','g']
plt.bar(cat_list,cat_average,color = colors)


plt.subplot(3,3,5)
plt.title('average demand Vs hour')
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
colors = ['m','b','r','g']
plt.bar(cat_list,cat_average,color = colors)

plt.subplot(3,3,6)
plt.title('average demand Vs weekday')
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
colors = ['m','b','r','g']
plt.bar(cat_list,cat_average,color = colors)


plt.subplot(3,3,7)
plt.title('average demand Vs workingday') 
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
colors = ['m','b','r','g']
plt.bar(cat_list,cat_average,color = colors)


plt.subplot(3,3,8)
plt.title('average demand Vs holiday')
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
colors = ['m','b','r','g']
plt.bar(cat_list,cat_average,color = colors)
plt.tight_layout()
```

![Screenshot 2024-07-15 191356](https://github.com/user-attachments/assets/5250c8cc-6ecb-4856-908b-81cf1fc6d512)



# Assumptions of Multiple Linear Regression
## Checking for Outliers
```python
bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05, 0.10, 0.15, 0.90, 0.95, 0.99])

# Correlation Analysis

correlation = bikes_prep[['temp', 'humidity', 'windspeed', 'demand']].corr()
bikes_prep = bikes_prep.drop(['weekday', 'year', 'workingday', 'atemp', 'windspeed'], axis=1)

# Autocorrelation in Demand

df1 = pd.to_numeric(bikes_prep['demand'], downcast='float')
plt.acorr(df1, maxlags=12)
```
![Screenshot 2024-07-15 192214](https://github.com/user-attachments/assets/e340272d-0bf6-4c91-823b-d291b52af3ea)

## Feature Engineering
```python
Log Normalization of Demand
bikes_prep['demand'] = np.log(bikes_prep['demand'])
```
![Screenshot 2024-07-15 192515](https://github.com/user-attachments/assets/3a47b9e6-b2c8-44f0-81ac-c0e3fa87618a)

## Lag Features
```python
t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_3 = bikes_prep['demand'].shift(+3).to_frame()
bikes_prep_lags = pd.concat([bikes_prep, t_1, t_2, t_3], axis=1)
bikes_prep_lags = bikes_prep_lags.dropna()

# Creating Dummy Variables

bikes_prep_lags['season'] = bikes_prep_lags['season'].astype('category')
bikes_prep_lags['holiday'] = bikes_prep_lags['holiday'].astype('category')
bikes_prep_lags['weather'] = bikes_prep_lags['weather'].astype('category')
bikes_prep_lags['hour'] = bikes_prep_lags['hour'].astype('category')
bikes_prep_lags['month'] = bikes_prep_lags['month'].astype('category')
bikes_prep_lags = pd.get_dummies(bikes_prep_lags, drop_first=True)

# Model Training and Evaluation
# Train-Test Split

Y = bikes_prep_lags[['demand']]
x = bikes_prep_lags.drop(['demand'], axis=1)

tr_size = int(0.7 * len(x))
x_train = x.values[:tr_size]
x_test = x.values[tr_size:]
Y_train = Y.values[:tr_size]
Y_test = Y.values[tr_size:]

# Linear Regression Model

from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
std_reg.fit(x_train, Y_train)

r2_train = std_reg.score(x_train, Y_train)
r2_test = std_reg.score(x_test, Y_test)

y_predict = std_reg.predict(x_test)

# RMSE AND RMSLE CALCULATION
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(Y_test, y_predict))
```
## RMSLE Calculation
```python
y_test_e = [math.exp(y) for y in Y_test]
y_predict_e = [math.exp(y) for y in y_predict]

log_sq_sum = sum([(math.log(a + 1) - math.log(p + 1))**2 for a, p in zip(y_test_e, y_predict_e)])
rmsle = math.sqrt(log_sq_sum / len(Y_test))
print(rmsle)
```
![Screenshot 2024-07-15 193224](https://github.com/user-attachments/assets/10e0eeba-7768-4e5b-90ab-0161f28bcec6)


## Conclusion
Successfully predicted bike demand using various features with reasonable accuracy.
Feature elimination through graphical analysis improved model performance.
The model can assist in urban planning and optimizing bike availability.






