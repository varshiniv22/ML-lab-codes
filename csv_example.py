import pandas as pd
import os
file_path = os.path.join("datasets","housing","housing.csv")
df= pd.read_csv(file_path)
print("\n\n this is excel file", df)
print("info")
print(df.info())
print("head")
print(df.head())
print("tail")
print(df.tail(-3))
print("describe")
print(df.describe())
print("shape")
print(df.shape)
print("categorical values: \n\n",df["ocean_proximity"].value_counts())
