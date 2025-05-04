import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('sensor_data.csv')

# Load the data
data = pd.read_csv('sensor_data.csv')

#Preview the data
print(data.head())
print(data.info())

# 2.1Visualizing Missing Values with a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Summary table of missing values per column
missing_summary = data.isnull().sum().to_frame(name='Missing Values')
missing_summary['% of Total'] = (missing_summary['Missing Values'] / len(data)) * 100
print(missing_summary)




# 2.2 Handling Missing Values
# Impute numeric columns with median
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Outlier Detection using Z-Score Method
from scipy.stats import zscore

z_scores = np.abs(zscore(data[numeric_cols]))
outliers_z = (z_scores > 3).sum(axis=0)
outlier_counts = pd.Series(outliers_z, index=numeric_cols)
print("Number of outliers per column (Z-Score > 3):")
print(outlier_counts)