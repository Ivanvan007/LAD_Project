import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

file_name = "Google-Playstore.csv"
file_path = "Data/"
full_path = os.path.join(file_path, file_name)
#full_path = "Data/Google-Playstore.csv"
print("---------------Start CSV Reading---------------")
dfGPlayStore = pd.read_csv(full_path)
print("---------------CSV Reading Finished------------\n")

print("\n")
print("---------------Intial Description--------------")
variables_list = dfGPlayStore.columns.tolist()
print("Variables List: " +
      "\t ".join([x + "," if (y + 1) % 3 != 0 else x + ",\n" for y, x in enumerate(variables_list)]))
numeric_columns = dfGPlayStore.select_dtypes(include=['number']).columns.tolist()
print("Numeric Variables: " +
      "\t ".join([x + "," if (y + 1) % 3 != 0 else x + ",\n" for y, x in enumerate(numeric_columns)]))
print("\n")
dataSetDescription = dfGPlayStore.describe()
print(dataSetDescription)
print("\n")
print("--------Info--------")
dfGPlayStore.info()

covariancia = dfGPlayStore[numeric_columns].cov()
correlacao = dfGPlayStore[numeric_columns].corr()

print("\n")
print("--------Covariance-------")
print(covariancia)
print("--------Correlation--------")
print(correlacao)

print("---------------END Of Description---------------\n")

#dataSetToNumpy = dfGPlayStore.to_numpy()
#numericDataSetToNumpy = dfGPlayStore[numeric_columns].to_numpy()

#sns.histplot(dfGPlayStore)
#plt.figure()
#sns.histplot(dfGPlayStore[numeric_columns])
#plt.figure()

sns.heatmap(correlacao, annot=True)
plt.figure()
sns.heatmap(covariancia, annot=True)
plt.show()