import pandas as pd

# dataset load
df = pd.read_csv("govdata.csv")

print("Dataset Shape:", df.shape)

print("\nColumns:\n", df.columns)

print("\nFirst 5 rows:\n")
print(df.head())

print("\nMissing Values:\n")
print(df.isnull().sum())