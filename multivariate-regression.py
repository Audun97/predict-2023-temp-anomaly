""""
This script performs a multivariate linear regression analysis using the 'Year' and 'Anomaly_Jan_Aug' columns as predictors and 'Anomaly_Full_Year' as the response variable. 

The prediction is based on an estimated 'Anomaly_Jan_Aug' value of 1.0475, as the data from NOAA for August 2023 was not available at the time of creating this script. 

The script also calculates the probability that the temperature anomaly for 2023 will be greater than a threshold of 1.03, which is the current record. 

Year 1998 and 2016 are highlighted because they are years which featured an El Nino event.

Finally, the script generates a scatter plot of the data, with the predicted value for 2023 and the fitted line from the OLS model. The plot includes vertical lines representing the 95% confidence interval of the prediction for 2023.
"""
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_scatter(year, anomaly, color, label):
    plt.scatter(year, anomaly, color=color, label=label)

merged_data_path = 'data/merged-full+jan-august.csv'
merged_data = pd.read_csv(merged_data_path)

X = sm.add_constant(merged_data[['Year', 'Anomaly_Jan_Aug']])
model = sm.OLS(merged_data['Anomaly_Full_Year'], X)
results = model.fit()

year_2023 = pd.DataFrame({'const': [1], 'Year': [2023], 'Anomaly_Jan_Aug': [1.0475]})
prediction_2023 = results.get_prediction(year_2023)

mean_2023 = prediction_2023.predicted_mean[0]
se_2023 = prediction_2023.se_mean[0]
conf_int_2023 = prediction_2023.conf_int(alpha=0.05)  # Calculate the 95% confidence interval
ymin, ymax = conf_int_2023[0]
print("Prediction for 2023: " + str(mean_2023))
print("Confidence interval: " + "[" + str(ymax) + ", " + str(ymin) + "]")

# change to see what the probabillity of 2013 temp anomaly will be larger that what you set it to
prob_threshold = 1.03
prob_less_than = norm.cdf(prob_threshold, loc=mean_2023, scale=se_2023)
prob_greater_than = 1 - prob_less_than
print("Probabillity that 2023 temp anomaly will be larger than " + str(prob_threshold) + ": " + str(prob_greater_than))

plt.figure(figsize=(10, 6))

for year, anomaly in zip(merged_data['Year'], merged_data['Anomaly_Full_Year']):
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

extended_years = pd.concat([X, year_2023])
extended_fitted_values = results.predict(extended_years)
plt.plot(extended_years['Year'], extended_fitted_values, color='orange', linestyle='-', label='Fitted line')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left')

plt.show()