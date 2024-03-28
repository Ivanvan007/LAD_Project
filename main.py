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
plt.figure()


downloads_per_category = dfGPlayStore.groupby('Category')['Maximum Installs'].sum().reset_index()
plt.figure(figsize=(10,6))
plt.bar(downloads_per_category['Category'], downloads_per_category['Maximum Installs'])
plt.xlabel('Category')
plt.ylabel('Maximum Installs')
plt.title('Installs per Category')
plt.xticks(rotation=90,ha='right',fontsize=10)
plt.tight_layout()
plt.figure()

#downloads_per_category = dfGPlayStore.groupby('Category')['Rating'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Rating', data=dfGPlayStore, ci=None, estimator=lambda x: sum(x) / len(x))
plt.xlabel('Category')
plt.ylabel('Média dos Ratings')
plt.title('Média dos Ratings por Categoria')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.figure()

mean_free_by_category = dfGPlayStore.groupby('Category')['Free'].mean().reset_index()
mean_free_by_category['Free'] *= 100
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Free', data=mean_free_by_category, ci=None)
plt.xlabel('Categoria')
plt.ylabel('Proporção de Apps Gratuitos')
plt.title('Proporção de Apps Gratuitos por Categoria')
plt.xticks(rotation=90, ha='right')
plt.ylim(91, 101)
plt.tight_layout()
plt.figure()

mean_free_by_Rating = dfGPlayStore.groupby('Rating')['Free'].mean().reset_index()
mean_free_by_Rating['Free'] *= 100
plt.figure(figsize=(10, 6))
sns.barplot(x='Rating', y='Free', data=mean_free_by_Rating, ci=None)
plt.xlabel('Rating')
plt.ylabel('Proporção de Apps Gratuitos')
plt.title('Proporção de Apps Gratuitos por Rating')
plt.xticks(rotation=90, ha='right')
plt.ylim(91, 101)
plt.tight_layout()
plt.figure()

dfGPlayStore_pagos = dfGPlayStore[dfGPlayStore['Free'] == 0]
proporcao_pagos_por_rating = (
        dfGPlayStore_pagos.groupby('Rating').size() / dfGPlayStore.groupby('Rating').size())
plt.figure(figsize=(10, 6))
sns.boxplot(x='Rating', y='Free', data=dfGPlayStore_pagos, showfliers=False)
plt.xlabel('Rating')
plt.ylabel('Proporção de Apps Pagos')
plt.title('Proporção de Pagos por Rating')
plt.xticks(rotation=90, ha='right')
#plt.ylim(0, 1)  # Limitando o eixo y entre 0 e 1 (proporção)
plt.tight_layout()
plt.show()

