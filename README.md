# Kaggle bike demand sharing project 

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
Data Preprocessing
Reading the File

# python code
bikes = pd.read_csv('dataset.csv')
Preliminary Analysis and Feature Selection

# python code
bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index', 'date', 'casual', 'registered'], axis=1)
bikes_prep.isnull().sum()
Visualizing the Data Histograms for data distribution:

# python code
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()
Scatter plots for continuous features vs. demand:

# python code
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
Categorical Features vs. Demand

# python code
plt.subplot(3,3,1)
plt.title('Average Demand Vs Season')
cat_list = bikes_prep['season'].unique()
cat_average = bikes_prep.groupby('season').mean()['demand']
colors = ['m', 'b', 'r', 'g']
plt.bar(cat_list, cat_average, color=colors)

# Assumptions of Multiple Linear Regression
# Checking for Outliers

# python code
bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05, 0.10, 0.15, 0.90, 0.95, 0.99])
Correlation Analysis

# python code
correlation = bikes_prep[['temp', 'humidity', 'windspeed', 'demand']].corr

# Insights and Discussion

# Feature Importance:
The graphical analysis showed that features like temperature, humidity, and hour significantly affect bike demand.
Removing features with low correlation, such as weekday and working day, helped improve model performance.
Model Performance:

The linear regression model achieved a reasonable R2 score on both training and test datasets.
The Root Mean Squared Logarithmic Error (RMSLE) provided a robust measure of the model’s predictive accuracy.

# Error Analysis:
Despite reasonable accuracy, some predictions had significant errors, especially for higher demand values.
Autocorrelation analysis and log transformation helped in dealing with skewness in demand data.

# Conclusion
The project successfully predicted bike demand using a variety of features.
Feature elimination based on graphical analysis improved model performance.
The model can be useful for urban planning and optimizing bike availability.




