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
dfGPlayStore = pd.read_csv(full_path,nrows=180000)

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

covariance = dfGPlayStore[numeric_columns].cov()
correlation = dfGPlayStore[numeric_columns].corr()

print("\n")
print("--------Covariance-------")
print(covariance)
print("--------Correlation--------")
print(correlation)

print("---------------END Of Description---------------\n")

#dataSetToNumpy = dfGPlayStore.to_numpy()
#numericDataSetToNumpy = dfGPlayStore[numeric_columns].to_numpy()

#sns.histplot(dfGPlayStore)
#plt.figure()
#sns.histplot(dfGPlayStore[numeric_columns])
#plt.figure()



sns.histplot(x='Rating', y='Installs', data=dfGPlayStore)
plt.title('Histograma : Rating vs Installs')
plt.figure()
sns.barplot(x='Rating', y='Installs', data=dfGPlayStore)
plt.title('Gráfico de Barras: Rating vs Installs')
plt.figure()


sns.heatmap(correlation, annot=True)
plt.title('Heatmap Correlation')
plt.figure()
sns.heatmap(covariance, annot=True)
plt.title('Heatmap Covariance')
plt.show()