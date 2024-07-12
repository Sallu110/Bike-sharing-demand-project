
# Bike Demand Prediction Using Machine Learning
# Objective
To predict bike demand based on various features using machine learning models and improve prediction accuracy through feature elimination.

Dataset
Source: Kaggle
Features:
Weather
Temperature
Holiday
Weekday
Working Day
Humidity
Wind Speed
Casual Users
Registered Users
Year
Season
Month
Hour

# Data Preprocessing
Reading the File
bikes = pd.read_csv('dataset.csv')

# Preliminary Analysis and Feature Selection

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index', 'date', 'casual', 'registered'], axis=1)
bikes_prep.isnull().sum()

# Visualizing the Data

# Histograms for data distribution:

bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

# Scatter plots for continuous features vs. demand:

plt.subplot(2,2,1)
plt.title('Temperature vs Demand')
plt.scatter(bikes_prep['temp'], bikes_prep['demand'], s=0.1, c='g')

plt.subplot(2,2,2)
plt.title('Humidity vs Demand')
plt.scatter(bikes_prep['humidity'], bikes_prep['demand'], s=0.1, c='r')

plt.subplot(2,2,3)
plt.title('Windspeed vs Demand')
plt.scatter(bikes_prep['windspeed'], bikes_prep['demand'], s=0.1, c='m')
plt.tight_layout()

# Categorical Features vs. Demand

plt.subplot(3,3,1)
plt.title('Average Demand Vs Season')
cat_list = bikes_prep['season'].unique()
cat_average = bikes_prep.groupby('season').mean()['demand']
colors = ['m', 'b', 'r', 'g']
plt.bar(cat_list, cat_average, color=colors)

# Assumptions of Multiple Linear Regression
# Checking for Outliers

bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05, 0.10, 0.15, 0.90, 0.95, 0.99])

# Correlation Analysis

correlation = bikes_prep[['temp', 'humidity', 'windspeed', 'demand']].corr()
bikes_prep = bikes_prep.drop(['weekday', 'year', 'workingday', 'atemp', 'windspeed'], axis=1)

# Autocorrelation in Demand

df1 = pd.to_numeric(bikes_prep['demand'], downcast='float')
plt.acorr(df1, maxlags=12)

# Feature Engineering
Log Normalization of Demand

bikes_prep['demand'] = np.log(bikes_prep['demand'])

# Lag Features

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

# from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
std_reg.fit(x_train, Y_train)

r2_train = std_reg.score(x_train, Y_train)
r2_test = std_reg.score(x_test, Y_test)

y_predict = std_reg.predict(x_test)

# from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, y_predict))

# RMSLE Calculation

y_test_e = [math.exp(y) for y in Y_test]
y_predict_e = [math.exp(y) for y in y_predict]

log_sq_sum = sum([(math.log(a + 1) - math.log(p + 1))**2 for a, p in zip(y_test_e, y_predict_e)])
rmsle = math.sqrt(log_sq_sum / len(Y_test))
print(rmsle)

# Conclusion
Successfully predicted bike demand using various features with reasonable accuracy.
Feature elimination through graphical analysis improved model performance.
The model can assist in urban planning and optimizing bike availability.






