import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib.pyplot as plt

merged_data_path = 'D:\Projects\predict_2023_temp\merged_data_august.csv'

merged_data = pd.read_csv(merged_data_path)

print(merged_data.head())
print(" ")

# Add a constant to the data
X = sm.add_constant(merged_data[['Year', 'Anomaly_Jan_Aug']])

# Fit a multiple linear regression model to the data
model = sm.OLS(merged_data['Anomaly_Full_Year'], X)
results = model.fit()

# Print the summary of the model
print(results.summary())
print(" ")

year_2023 = pd.DataFrame({'const': [1], 'Year': [2023], 'Anomaly_Jan_Aug': [1.0475]})
prediction_2023 = results.get_prediction(year_2023)
print(prediction_2023.summary_frame(alpha=0.05))
print(" ")

# Calculate the mean and standard error of the predictive distribution for 2023
mean_2023 = prediction_2023.predicted_mean[0]
se_2023 = prediction_2023.se_mean[0]

prob_threshold = 1.07

# Calculate the probability that the anomaly is less than or equal to 1.03
prob_less_than = norm.cdf(prob_threshold, loc=mean_2023, scale=se_2023)

# Subtract this probability from 1 to find the probability that the anomaly is larger than 1.03
prob_greater_than = 1 - prob_less_than

print("Probabillity 2023 anonmaly larger than " + str(prob_threshold) + ": " + str(prob_greater_than))
print(" ")

# Plot the full year anomaly data up to 2022
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['Year'], merged_data['Anomaly_Full_Year'], color='blue', label='Data')
plt.xlabel('Year')
plt.ylabel('Full Year Anomaly')
plt.title('Full Year Temperature Anomaly')
plt.grid(True)

# Add a point for the predicted value for 2023
plt.plot(2023, mean_2023, marker='o', color='red')

# Add a vertical line at 2023 to represent the 95% confidence interval for the predicted value
plt.vlines(x=2023, ymin=1.019469, ymax=1.05793, color='red', linestyle='--')

plt.show()