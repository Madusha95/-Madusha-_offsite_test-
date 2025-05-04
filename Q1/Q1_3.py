import pandas as pd
#1.3
# Load your CSV file
df = pd.read_csv('sensor_data.csv')

# Convert 'timestamp' column to datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract meaningful time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6

print("1.2")  # Display the first few rows of the DataFrame to verify the new features
print(df.head(10))  # Display the first few rows of the DataFrame to verify the new features

# Create lag features (previous 1, 2, and 3 readings)
for col in ['temperature', 'vibration', 'pressure']:
    for lag in [1, 2, 3]:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

print("1.3")  # Display the first few rows of the DataFrame to verify the new features
print(df.head(10)) # Display the first few rows of the DataFrame to verify the new features`
# Create rolling average (3-hour window)
df['temperature_roll_mean3'] = df['temperature'].rolling(window=3).mean()

# Create rolling standard deviation (3-hour window)
df['vibration_roll_std3'] = df['vibration'].rolling(window=3).std()

# Create rolling sum (3-hour window)
df['pressure_roll_sum3'] = df['pressure'].rolling(window=3).sum()

# Display the updated DataFrame
print(df.head(10))


#3.2
# Create lag features: last 1, 2, 3 readings for temperature, vibration, and pressure
for col in ['temperature', 'vibration', 'pressure']:
    for lag in [1, 2, 3]:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# View the first few rows
print("1.3")
print(df.head(10))


#3.3
# Create moving average and moving standard deviation (over 3-hour window)
for col in ['temperature', 'vibration', 'pressure']:
    df[f'{col}_roll_mean3'] = df[col].rolling(window=3).mean()   # Moving Average
    df[f'{col}_roll_std3'] = df[col].rolling(window=3).std()    # Moving Std Deviation

# View the first few rows
print("1.4")
print(df.head(10))