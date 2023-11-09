"""
This script calculates the probabillity that 2023 will be the hottest year ever 
based on full year temperature anomalies and a simple linear regression
"""
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt

def plot_scatter(year, anomaly, color, label):
    plt.scatter(year, anomaly, color=color, label=label)

file_path = 'data/full-year-temp-anomalies.csv'

data = pd.read_csv(file_path, skiprows=4)

# cut off data from the 70's because that is when we see a linear relationship
data = data[(data['Year'] >= 1970) & (data['Year'] <= 2022)]

# adding a column of ones to data, which allows the OLS function to calculate an intercept.
# OLS model does not include an intercept by default.
X = sm.add_constant(data['Year'])

# Fit a linear model to the data. First argument is dependendent variable and the second is the independent
model = sm.OLS(data['Anomaly'], X)
results = model.fit()

year_2023 = pd.DataFrame({'const': [1], 'Year': [2023]})
prediction_2023 = results.get_prediction(year_2023)
fitted_values = results.fittedvalues
mean_2023 = prediction_2023.predicted_mean[0]
se_2023 = prediction_2023.se_mean[0]
conf_int_2023 = prediction_2023.conf_int(alpha=0.05)  # Calculate the 95% confidence interval
ymin, ymax = conf_int_2023[0]
print("confidence interval: " + "[" + str(ymax) + ", " + str(ymin) + "]")

highest_temp_2016 = 1.03

# change to see what the probabillity of 2013 temp anomaly will be larger that what you set it to
prob_threshold = 1.03
prob_less_than = norm.cdf(prob_threshold, loc=mean_2023, scale=se_2023)
prob_greater_than = 1 - prob_less_than
print("Probabillity that 2023 temp anomaly will be larger than " + str(prob_threshold) + ": " + str(prob_greater_than))

plt.figure(figsize=(10, 6))

for year, anomaly in zip(data['Year'], data['Anomaly']):
    if year == 2016 or year == 1998:
        plot_scatter(year, anomaly, 'green', 'El Nino Years')
    else:
        plot_scatter(year, anomaly, 'blue', 'Full Year Data')

plt.xlabel('Year')
plt.ylabel('Full Year Anomaly')
plt.title('Full Year Temperature Anomaly')
plt.grid(True)

plt.plot(2023, mean_2023, marker='o', color='red', label='Predicted Value 95%CI')

conf_int_line = plt.vlines(x=2023, ymin=ymin, ymax=ymax, color='red', linestyle='--')

plt.plot(data['Year'], fitted_values, color='orange', linestyle='-', label='Fitted line')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left')

plt.show()
