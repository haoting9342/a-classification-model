import pandas as pd

df = pd.read_csv("normal-0927.csv")
df = df[df['aiS1TeID'].isin(['393235'])]
df.to_csv("normal-0927-afterclean.csv")