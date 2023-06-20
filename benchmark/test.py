import pandas as pd

print(pd.read_csv("statistics.csv").shape)
print(pd.read_csv("non_nan_statistics.csv").shape)
print(pd.read_csv("non_dup_statistics.csv").shape)