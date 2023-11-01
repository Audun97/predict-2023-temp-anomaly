import pandas as pd

file_path_full_year = 'C:\\Users\\Audun\\Downloads\\data (1).csv'
file_path_Jan_Dec = 'C:\\Users\\Audun\\Downloads\\data_august.csv'

data_full_year = pd.read_csv(file_path_full_year, skiprows=5, names=['Year', 'Anomaly_Full_Year'])
data_Jan_Jul = pd.read_csv(file_path_Jan_Dec, skiprows=5, names=['Year', 'Anomaly_Jan_Jul'])

filtered_data_full_year = data_full_year[(data_full_year['Year'] >= 1970) & (data_full_year['Year'] <= 2022)]
filtered_data_jan_jul = data_Jan_Jul[(data_Jan_Jul['Year'] >= 1970) & (data_Jan_Jul['Year'] <= 2023)]

# Merge the two dataframes on the 'Year' column
merged_data = pd.merge(filtered_data_full_year, filtered_data_jan_jul, on='Year')

print(merged_data.head())

# Save the merged data as a new CSV file
merged_data.to_csv('merged_data_august.csv', index=False)