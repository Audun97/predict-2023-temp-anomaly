import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

file_path = 'C:\\Users\\Audun\\Downloads\\data (1).csv'

data = pd.read_csv(file_path, skiprows=4)
data = data[(data['Year'] >= 1970) & (data['Year'] <= 2022)]

# Add a constant to the data
X = sm.add_constant(data['Year'])

# Fit a linear model to the data
model = sm.OLS(data['Anomaly'], X)
results = model.fit()

year_2023 = 2023
prediction_2023 = results.params['const'] + results.params['Year'] * year_2023

st, data, ss2 = summary_table(results, alpha=0.05)

# Get the standard error and confidence intervals
fittedvalues = data[:, 2]
predict_mean_se  = data[:, 3]
predict_mean_ci_low, predict_mean_ci_upp = data[:, 4:6].T
predict_ci_low, predict_ci_upp = data[:, 6:8].T

# Calculate the prediction interval for the year 2023
i = len(data) - 1
predict_ci_low_2023 = prediction_2023 - (predict_ci_upp[i] - fittedvalues[i])
predict_ci_upp_2023 = prediction_2023 + (fittedvalues[i] - predict_ci_low[i])

print("Confindance interval: " + "[" + str(predict_ci_low_2023) + "], [" + str(predict_ci_upp_2023) + "]")

highest_temp_2016 = 1.03

# Calculate the probability that the temperature in 2023 will be higher than that in 2016
if highest_temp_2016 < predict_ci_low_2023:
    probability = 1
elif highest_temp_2016 > predict_ci_upp_2023:
    probability = 0
else:
    probability = (predict_ci_upp_2023 - highest_temp_2016) / (predict_ci_upp_2023 - predict_ci_low_2023)

print("Probability temperature anomaly in 2023 highest ever: " + str(probability))
