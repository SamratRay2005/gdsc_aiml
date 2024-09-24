# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset
train_data = pd.read_csv('train_data.csv')

# Step 1: Basic Information
print("Dataset Shape:", train_data.shape)
print("\nColumns in the dataset:", train_data.columns)
print("\nData Types:\n", train_data.dtypes)

# Check the first few rows of the dataset
print("\nFirst 5 rows of the dataset:\n", train_data.head())

# Step 2: Check for Missing Values
missing_values = train_data.isnull().sum()
print("\nMissing Values in Each Column:\n", missing_values)

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:\n", train_data.describe())

# Step 4: Data Distribution Visualization
# Histograms for numerical columns
numerical_cols = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD']
train_data[numerical_cols].hist(bins=20, figsize=(12, 8))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# Step 5: Correlation Analysis
correlation_matrix = train_data.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

# Step 6: Time Series Plot (if 'unix' timestamp exists)
if 'unix' in train_data.columns:
    # Convert UNIX timestamp to a datetime forma at
    train_data['date'] = pd.to_datetime(train_data['unix'], unit='s')
    
    # Plot the 'close' price over time
    plt.figure(figsize=(10, 5))
    plt.plot(train_data['date'], train_data['close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close Price Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
else:
    print("\nNo 'unix' column found for time series analysis.")
