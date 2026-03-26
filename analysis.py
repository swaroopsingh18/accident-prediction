import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("govdata.csv")

# text column remove
df = df.drop(columns=["State/UT"], errors="ignore")
df = df.drop(columns=["S. No."], errors="ignore")

# correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")

plt.title("Road Accident Feature Correlation")
plt.show()
