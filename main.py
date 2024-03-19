import os
import pandas as pd
import time
import numpy as np
import matplotlib as plt
import seaborn as sns

file_name = "Google-Playstore.csv"
file_path = "Data/"
full_path = os.path.join(file_path, file_name)
#full_path = "Data/Google-Playstore.csv"
print("-----------Start CSV Reading----------")
dfGPlayStore = pd.read_csv(full_path)
print("----------CSV Reading Finished---------\n")

print("\n")
print("--------Intial Description--------")
variables_list = dfGPlayStore.columns.tolist()
print("Variables List: " +
      "\t ".join([x + "," if (y + 1) % 3 != 0 else x + ",\n" for y, x in enumerate(variables_list)]))
numeric_columns = dfGPlayStore.select_dtypes(include=['number']).columns.tolist()
print("Numeric Variables: " +
      "\t ".join([x + "," if (y + 1) % 3 != 0 else x + ",\n" for y, x in enumerate(numeric_columns)]))
print("\n")
dataSetDescription = dfGPlayStore.describe()
print(dataSetDescription)