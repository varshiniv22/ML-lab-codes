

import os
import pandas as pd
file_path = os.path.join("datasets","housing","housing.xlsx")

df=pd.read_excel(file_path)
print(df)