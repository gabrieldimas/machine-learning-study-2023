import pandas as pd

data = 'wbc.csv'
df = pd.read_csv(data)
df.head()
print(df.head())