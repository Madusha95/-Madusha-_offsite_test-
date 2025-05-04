# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#1.1
# Load dataset
df = pd.read_csv('sensor_data.csv')

# Display first 5 rows
print(df.head())

# Check data types and missing values
print(df.info())

# Summary statistics for each sensor reading
summary_stats = df[['temperature', 'vibration', 'pressure']].describe()
print("Summary Statistics:")
print(summary_stats)

# =============================================
# FIGURE 1: Sensor Distributions (1.2)
# =============================================
plt.figure(figsize=(15, 5))

# Temperature distribution
plt.subplot(1, 3, 1)
sns.histplot(df['temperature'], kde=True, bins=30, color='skyblue')
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')

# Vibration distribution
plt.subplot(1, 3, 2)
sns.histplot(df['vibration'], kde=True, bins=30, color='salmon')
plt.title('Vibration Distribution')
plt.xlabel('Vibration (mm/s)')

# Pressure distribution
plt.subplot(1, 3, 3)
sns.histplot(df['pressure'], kde=True, bins=30, color='lightgreen')
plt.title('Pressure Distribution')
plt.xlabel('Pressure (psi)')

plt.suptitle('Figure 1: Sensor Value Distributions', y=1.05, fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# =============================================
# FIGURE 2: Time Series with Failures (1.3)
# =============================================
plt.figure(figsize=(15, 10))

# Identify failure points
failures = df[df['failure'] == 1]

# Temperature over time
plt.subplot(3, 1, 1)
sns.lineplot(x='timestamp', y='temperature', data=df, color='skyblue', label='Temperature')
plt.scatter(failures['timestamp'], failures['temperature'], color='red', s=50, label='Failure')
plt.title('Temperature Over Time')
plt.legend()

# Vibration over time
plt.subplot(3, 1, 2)
sns.lineplot(x='timestamp', y='vibration', data=df, color='salmon', label='Vibration')
plt.scatter(failures['timestamp'], failures['vibration'], color='red', s=50, label='Failure')
plt.title('Vibration Over Time')
plt.legend()

# Pressure over time
plt.subplot(3, 1, 3)
sns.lineplot(x='timestamp', y='pressure', data=df, color='lightgreen', label='Pressure')
plt.scatter(failures['timestamp'], failures['pressure'], color='red', s=50, label='Failure')
plt.title('Pressure Over Time')
plt.legend()

plt.suptitle('Figure 2: Sensor Measurements Over Time with Failures', y=1.02, fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# =============================================
# FIGURE 3: Correlation Matrix (1.4)
# =============================================
plt.figure(figsize=(8, 6))

corr_matrix = df[['temperature', 'vibration', 'pressure']].corr()
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1,
            linewidths=0.5,
            fmt=".2f",
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title('Figure 3: Sensor Correlation Matrix', pad=20, fontsize=14, weight='bold')
plt.tight_layout()
plt.show()