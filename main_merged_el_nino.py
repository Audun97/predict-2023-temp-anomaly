import pandas as pd
import statsmodels.api as sm

merged_data_path = 'D:\Projects\predict_2023_temp\merged_data_august.csv'

merged_data = pd.read_csv(merged_data_path)

# Exclude the El Nino years 2016 and see how much the actual values diverted from the models prediction
merged_data_excluded = merged_data[(merged_data['Year'] != 2016) & (merged_data['Year'] != 1998)]

# Add a constant to the data
X_excluded = sm.add_constant(merged_data_excluded[['Year', 'Anomaly_Jan_Aug']])

# Fit a multiple linear regression model to the data
model_excluded = sm.OLS(merged_data_excluded['Anomaly_Full_Year'], X_excluded)
results_excluded = model_excluded.fit()

year_1998 = pd.DataFrame({'const': [1], 'Year': [1998], 'Anomaly_Jan_Aug': merged_data.loc[merged_data['Year'] == 1998, 'Anomaly_Jan_Aug'].values})
year_2016 = pd.DataFrame({'const': [1], 'Year': [2016], 'Anomaly_Jan_Aug': merged_data.loc[merged_data['Year'] == 2016, 'Anomaly_Jan_Aug'].values})

# Predict the temperature for 1998 and 2016
prediction_1998 = results_excluded.get_prediction(year_1998)
prediction_2016 = results_excluded.get_prediction(year_2016)

# Get the actual temperatures for 1998 and 2016
actual_1998 = merged_data.loc[merged_data['Year'] == 1998, 'Anomaly_Full_Year'].values[0]
actual_2016 = merged_data.loc[merged_data['Year'] == 2016, 'Anomaly_Full_Year'].values[0]

# Calculate the differences between the predicted and actual temperatures
difference_1998 = prediction_1998.predicted_mean[0] - actual_1998
difference_2016 = prediction_2016.predicted_mean[0] - actual_2016

print("Difference from prediction: 1998 = " + str(difference_1998) + ", 2016 = " + str(difference_2016))