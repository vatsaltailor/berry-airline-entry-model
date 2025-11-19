"""Check the data for NaN values"""

import numpy as np
import pandas as pd
import sys

sys.stdout = open('check_data_output.txt', 'w')
sys.stderr = sys.stdout

data = pd.read_csv('Berry_data.csv')

print("Data shape:", data.shape)
print("\nColumn names:", data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

print("\nChecking for NaN values:")
print(data.isnull().sum())

print("\nData types:")
print(data.dtypes)

print("\nBasic statistics:")
print(data.describe())

# Check specific columns
print("\nPopulation1 range:", data['population1'].min(), "to", data['population1'].max())
print("Population2 range:", data['population2'].min(), "to", data['population2'].max())
print("Distance range:", data['distance'].min(), "to", data['distance'].max())
print("Passengers range:", data['passengers'].min(), "to", data['passengers'].max())

# Check for zeros
print("\nZeros in key columns:")
print("Population1 zeros:", (data['population1'] == 0).sum())
print("Population2 zeros:", (data['population2'] == 0).sum())
print("Distance zeros:", (data['distance'] == 0).sum())
print("Passengers zeros:", (data['passengers'] == 0).sum())

sys.stdout.close()

