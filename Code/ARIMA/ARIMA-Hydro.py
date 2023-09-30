import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# Load the historical Bitcoin Greenhouse Gas Emission
df = pd.read_csv('/kaggle/input/emission/GHG Emission.csv')
print('Shape of the data= ', df.shape)
print('Column datatypes= \n',df.dtypes)
df.columns = ["Date", "Hydro-only", "Estimated", "Coal-only"]
df['Date'] = df['Date'].str.replace('T00:00:00', '')

train = df.loc[df['Date'] < "2022-01-01"]
test = df.loc[df['Date'] >= "2022-01-01"]

# Convert 'Date' column to datetime type
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

# Set 'Date' column as the index
train.set_index('Date', inplace=True)
test.set_index('Date', inplace=True)

# Fit ARIMA model
model = sm.tsa.ARIMA(train['Hydro-only'], order=(1, 1, 1))
model_fit = model.fit()

# Make predictions for the next one year
predictions_next_year = model_fit.predict(start=test.index[0], end=test.index[-1])

# Plot the results
ax = train['Hydro-only'].plot(figsize=(15, 5), title='Hydro-only: Greenhouse Gas Emissions')
test['Hydro-only'].plot(ax=ax)
predictions_next_year.plot(ax=ax)
plt.legend(['Historical Data', 'Test Set', 'Predictions'])
plt.show()

# Calculate RMSE on the test set
rmse = np.sqrt(mean_squared_error(test['Hydro-only'], predictions_next_year))
print(f'RMSE Score on Test set: {rmse:0.4f}')