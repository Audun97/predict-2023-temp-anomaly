import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm

merged_data_path = 'D:\\Projects\\predict_2023_temp\\merged_data_august.csv'
merged_data = pd.read_csv(merged_data_path)

X = sm.add_constant(merged_data[['Year', 'Anomaly_Jan_Aug']])
model = sm.OLS(merged_data['Anomaly_Full_Year'], X)
results = model.fit()

year_2023 = pd.DataFrame({'const': [1], 'Year': [2023], 'Anomaly_Jan_Aug': [1.0475]})
prediction_2023 = results.get_prediction(year_2023)

mean_2023 = prediction_2023.predicted_mean[0]
se_2023 = prediction_2023.se_mean[0]
prob_threshold = 1.07
prob_less_than = norm.cdf(prob_threshold, loc=mean_2023, scale=se_2023)
prob_greater_than = 1 - prob_less_than

plt.figure(figsize=(10, 6))

for year, anomaly in zip(merged_data['Year'], merged_data['Anomaly_Full_Year']):
    if year == 2016 or year == 1998:
        plt.scatter(year, anomaly, color='green', label='Data')
    else:
        plt.scatter(year, anomaly, color='blue', label='Data')

plt.xlabel('Year')
plt.ylabel('Full Year Anomaly')
plt.title('Full Year Temperature Anomaly')
plt.grid(True)

plt.plot(2023, mean_2023, marker='o', color='red')

plt.vlines(x=2023, ymin=1.019469, ymax=1.05793, color='red', linestyle='--')

# Draw the fitted line
extended_years = pd.concat([X, year_2023])
extended_fitted_values = results.predict(extended_years)
plt.plot(extended_years['Year'], extended_fitted_values, color='orange', linestyle='-', label='Fitted line')

plt.show()