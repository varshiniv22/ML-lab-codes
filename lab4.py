import os
import pandas as pd 

filexl=os.path.join("datasets","housing","housing.xlsx")
data = pd.read_excel(filexl) 
print("________________________________________________________________________________________________________________________________________________")
print("\n \n EXCEL LOAD AND PRINT \n ")
print(data)

file_path=os.path.join("datasets","housing","housing.csv")
df = pd.read_csv(file_path) 
print("________________________________________________________________________________________________________________________________________________")
print("\n \n CSV LOAD AND PRINT \n ")
print(df)

print("________________________________________________________________________________________________________________________________________________")
print("\n \n About excel file \n ")
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using info() to print details about columns along with datatypes \n",data.info())
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using describe() to print summary of a dataframe \n",data.describe())
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using head() to print starting 5 rows by default \n",data.head())
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using tail() to print last 5 rows by default \n",data.tail())
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using value_counts() to count categorical values of a column \n",data["ocean_proximity"].value_counts())
print("________________________________________________________________________________________________________________________________________________")
print("\n \n EXCEL INFO END  \n ")

print("________________________________________________________________________________________________________________________________________________")
print("\n \n About CSV file \n ")
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using info() to print details about columns along with datatypes \n",df.info())
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using describe() to print summary of a dataframe \n",df.describe())
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using head() with - index \n",df.head(-3))
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using tail() with - index \n",df.tail(-3))
print("________________________________________________________________________________________________________________________________________________")
print("\n \n Using value_counts() to count categorical values of a column \n",df["ocean_proximity"].value_counts())
print("________________________________________________________________________________________________________________________________________________")
print("\n \n CSV INFO END  \n ")




