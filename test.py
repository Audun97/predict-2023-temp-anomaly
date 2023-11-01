import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import t

file_path = 'C:\\Users\\Audun\\Downloads\\data (1).csv'

data = pd.read_csv(file_path, skiprows=4)
data = data[(data['Year'] >= 1970) & (data['Year'] <= 2022)]

X = data['Year'].values.reshape(-1, 1)
y = data['Anomaly'].values

model = LinearRegression()
model.fit(X, y)

year_2023 = np.array([[2023]])
prediction_2023 = model.predict(year_2023)

n = len(X)
mse = mean_squared_error(y, model.predict(X))
alpha = 0.05
t_score = t.ppf(1 - alpha / 2, n - 2)
se = np.sqrt(mse * (1 / n + (year_2023 - X.mean())**2 / np.sum((X - X.mean())**2)))
margin = t_score * se

predict_ci_low_2023 = prediction_2023 - margin
predict_ci_upp_2023 = prediction_2023 + margin

print(str(predict_ci_low_2023) + ", " + str(predict_ci_upp_2023))

plt.figure(figsize=(10, 6))
plt.plot(data['Year'], data['Anomaly'], 'o', label='data')
plt.plot(data['Year'], model.predict(X), 'r-', label='OLS')
plt.plot(2023, prediction_2023, 'ro')
plt.vlines(2023, predict_ci_low_2023, predict_ci_upp_2023, colors='r', linestyles='dashed')
plt.xlabel('Year')
plt.ylabel('Anomaly')
plt.title('Temperature Anomaly Over Time')
plt.legend()
plt.show()