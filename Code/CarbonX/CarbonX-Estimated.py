import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# Load the historical Bitcoin Greenhouse Gas Emission
df = pd.read_csv('/kaggle/input/emission/GHG Emission.csv')
print('Shape of the data= ', df.shape)
print('Column datatypes= \n',df.dtypes)
df.columns = ["Date", "Hydro-only", "Estimated", "Coal-only"]
df['Date'] = df['Date'].str.replace('T00:00:00', '')

# For Coal-only
df_estimated = df.drop(['Hydro-only', 'Estimated'], axis=1)
df_estimated = df_estimated.set_index('Date')
df_estimated.index = pd.to_datetime(df_estimated.index)

df_estimated.head(5)
df_estimated.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='Coal-only: Greenhouse Gas Emissions')
plt.show()

train = df_estimated.loc[df_estimated.index < "2022-01-01"]
test = df_estimated.loc[df_estimated.index >= "2022-01-01"]

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Coal-only: Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline("2022-01-01", color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

print(df_estimated.loc[(df_estimated.index > '2019-01-01') & (df_estimated.index < '2019-01-08')])
df_estimated.loc[(df_estimated.index > '2019-01-01') & (df_estimated.index < '2019-01-08')] \
    .plot(figsize=(15, 5), title='Coal-only: Week Of Data')
plt.show()


def create_features(df_estimated):
    """
    Create time series features based on time series index.
    """
    df_estimated = df_estimated.copy()
    df_estimated['dayofweek'] = df_estimated.index.dayofweek
    df_estimated['quarter'] = df_estimated.index.quarter
    df_estimated['month'] = df_estimated.index.month
    df_estimated['year'] = df_estimated.index.year
    df_estimated['dayofyear'] = df_estimated.index.dayofyear
    df_estimated['dayofmonth'] = df_estimated.index.day
    df_estimated['weekofyear'] = df_estimated.index.isocalendar().week
    return df_estimated

df_estimated = create_features(df_estimated)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df_estimated, x='year', y='Coal-only')
ax.set_title('Coal-only: Greenhouse Gas Emissions by Year')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df_estimated, x='month', y='Coal-only', palette='Blues')
ax.set_title('Coal-only: Greenhouse Gas Emissions by Month')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df_estimated, x='year', y='Coal-only')
ax.set_title('Coal-only: Greenhouse Gas Emissions by Year')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df_estimated, x='month', y='Coal-only', palette='Blues')
ax.set_title('Coal-only: Greenhouse Gas Emissions by Month')
plt.show()

# group the data by year and month, and calculate the median Estimated value for each group
year_medians = df_estimated.groupby('year')['Coal-only'].median()
month_medians = df_estimated.groupby('month')['Coal-only'].median()

# display the median values as text
print("Median Coal-only Greenhouse Gas Emissions by Year:")
print(year_medians)
print("\nMedian Coal-only Greenhouse Gas Emissions by Month:")
print(month_medians)

train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'Coal-only'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50, learning_rate = 0.01)
reg.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],verbose = 50)

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Coal-only: Feature Importance')
plt.show()

# create a DataFrame of feature importances with column names and sort by ascending importance
fi = pd.DataFrame(data=reg.feature_importances_, index=reg.feature_names_in_, columns=['importance'])
fi_sorted = fi.sort_values('importance', ascending=True)

# display the feature importances as text
print("Feature Importances for Coal-only:")
print(fi_sorted)

test['prediction'] = reg.predict(X_test)
df_estimated = df_estimated.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df_estimated[['Coal-only']].plot(figsize=(15, 5))
df_estimated['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Coal-only: Raw Data and Prediction')
plt.show()

ax = df_estimated.loc[(df_estimated.index > '2022-04-01') & (df_estimated.index < '2022-08-01')]['Coal-only'] \
    .plot(figsize=(15, 5), title='Coal-only: 4 Months of Data')
df_estimated.loc[(df_estimated.index > '2022-04-01') & (df_estimated.index < '2022-08-01')]['prediction'] \
    .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

# display the estimated values and predictions as text
print("Coal-only: Estimated and Predicted Greenhouse Gas Emissions:")
print(df_estimated[['Coal-only', 'prediction']])

# display the 4 months of estimated values and predictions as text
print("\Coal-only: Estimated and Predicted Greenhouse Gas Emissions (April - July 2022):")
print(df_estimated.loc[(df_estimated.index > '2022-04-01') & (df_estimated.index < '2022-08-01'), ['Coal-only', 'prediction']])


score = np.sqrt(mean_squared_error(test['Coal-only'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.4f}')

test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)
test.groupby(['date'])['error'].mean().sort_values(ascending=True).head(10)

# Create a new dataframe for the next year
last_date = df_estimated.index[-1]
next_year = pd.date_range(last_date, periods=365, freq='D')[1:]
df_pred = pd.DataFrame(index=next_year)

# Create features for the next year
df_pred = create_features(df_pred)

# Make predictions for the next year
preds = reg.predict(df_pred[FEATURES])
df_pred['prediction'] = preds

# Plot the results
ax = df_estimated['Coal-only'].plot(figsize=(15, 5), title='Coal-only: Greenhouse Gas Emissions')
df_pred['prediction'].plot(ax=ax, style='.')
plt.legend(['Historical Data', 'Predictions'])
plt.show()

# Display the predicted values
print("Coal-only: Predicted Values:")
print(df_pred['prediction'])