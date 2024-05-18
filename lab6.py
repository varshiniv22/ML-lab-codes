import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

file_path=os.path.join("datasets","housing","housing.csv")
housing=pd.read_csv(file_path)

print("\n\n*************************************************************\n")
print("\n\n***********DROPPING NULL VALUES************\n")
print(housing.dropna(subset=["total_bedrooms"]).describe())

print("\n\n***********DROPPING ENTIRE COLUMN***********\n")
print(housing.drop("total_bedrooms",axis=1))
median=housing["total_bedrooms"].median()
print(median)

print("\n\n***********FILLING NULL VALUES WITH MEDIAN VALUE*************\n")
housing["total_bedrooms"].fillna(median,inplace=True)
print(housing["total_bedrooms"])

print("\n\n **********PRINTING DATASET INFORMATION**************\n")
print(housing.info())


#simple imputer class

imputer=SimpleImputer(strategy="median")
housing_num=housing.drop("ocean_proximity",axis=1)


imputer.fit(housing_num)
print("\n\n*************MEADIAN VALUES FOR NUMERICAL COLUMNS*************\n")
print(housing_num.median().values)
print("*******************************************************************\n")

x=imputer.transform(housing_num)
print("\n\n***************************************************************")
housing_tr=pd.dataframe(x,columns=housing_num.columns,index=housing_num.index)

print("\n**********TRANSFORMED DATA****************\n")
print(housing_tr.info())
print(housing_tr)

print("\n\n HANDLING TEXT AND CATEGORICAL ATTRIBUTES \n")
housing_cat=housing[["ocean_proximity"]]
print(housing_cat.head(10))

print("\n\n ORDINAL ENCODING \n")
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print("CATEGORIES OF THE ENCODER ARE :",ordinal_encoder.categories_)


#Converting to 0 and 1
print("\n\n ONE HOT ENCODING \n")
cat_encoder=OneHotEncoder()
housing_cat_1hot= cat_encoder .fit_transform(housing_cat)
print(housing_cat_1hot.toArray())
print(cat_encoder.categories_)


print("\n\n FEATURE SCALING \n")
new_housing=housing.drop(['ocean_proximity'],axis=1)
print(new_housing.head())
scaler=StandardScaler()
scaled_data=scaler.fit_transform(new_housing)
scaled_df=pd.DataFrame(scaled_data,
                       columns=['longitude','latitude','housing_median-age','total_rooms','total_bedrooms','population','households','median_income','median_house_value'])
print(scaled_df.head())
print(scaled_data)
