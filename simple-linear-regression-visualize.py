import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import matplotlib.pyplot as plt

file_path = 'data/full-year-temp-anomalies.csv'
data = pd.read_csv(file_path, skiprows=5, names=['Year', 'Anomaly'])
data = data[(data['Year'] >= 1970) & (data['Year'] <= 2022)]

# Add a constant to the data
X = sm.add_constant(data['Year'])

# Fit a linear model to the data
model = sm.OLS(data['Anomaly'], X)
results = model.fit()

year_2023 = 2023
prediction_2023 = results.params['const'] + results.params['Year'] * year_2023

st, data2, ss2 = summary_table(results, alpha=0.05)

# Get the standard error and confidence intervals
fittedvalues = data2[:, 2]
predict_mean_se  = data2[:, 3]
predict_mean_ci_low, predict_mean_ci_upp = data2[:, 4:6].T
predict_ci_low, predict_ci_upp = data2[:, 6:8].T

# Calculate the prediction interval for the year 2023
i = len(data2) - 1
predict_ci_low_2023 = prediction_2023 - (predict_ci_upp[i] - fittedvalues[i])
predict_ci_upp_2023 = prediction_2023 + (fittedvalues[i] - predict_ci_low[i])

# Create a new figure
plt.figure(figsize=(10, 6))

# Plot the data
plt.scatter(data['Year'], data['Anomaly'], color='blue', label='Data')

# Plot the fitted linear regression line
plt.plot(data['Year'], fittedvalues, color='red', label='Fitted line')

# Plot the predicted 2023 value with the 95% confidence interval
plt.plot(year_2023, prediction_2023, 'go')
plt.fill_between([year_2023], predict_ci_low_2023, predict_ci_upp_2023, color='g', alpha=0.1)

# Add horizontal lines at the top and bottom of the confidence interval
plt.hlines(predict_ci_low_2023, xmin=year_2023-1, xmax=year_2023+1, colors='g', linestyles='solid')
plt.hlines(predict_ci_upp_2023, xmin=year_2023-1, xmax=year_2023+1, colors='g', linestyles='solid')

# Set the title and labels
plt.title('Anomaly vs Year with Linear Regression and Prediction for 2023')
plt.xlabel('Year')
plt.ylabel('Anomaly')

# Show the legend
plt.legend(loc='upper left')

# Show the plot
plt.show()