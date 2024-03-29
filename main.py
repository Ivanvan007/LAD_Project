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





sns.barplot(x='Rating', y='Installs', data=dfGPlayStore)
plt.title('Bar Plot: Rating vs Installs')
plt.figure()

plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.title('Heatmap Correlation')
plt.tight_layout()
plt.figure()

plt.figure(figsize=(10,6))
sns.heatmap(covariance, annot=True)
plt.title('Heatmap Covariance')
plt.tight_layout()
plt.figure()


downloads_per_category = dfGPlayStore.groupby('Category')['Maximum Installs'].sum().reset_index()
downloads_per_category = downloads_per_category.sort_values(by='Maximum Installs', ascending=False)
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
plt.ylabel('Rating Average')
plt.title('Rating Average per Category')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.figure()

mean_free_by_category = dfGPlayStore.groupby('Category')['Free'].mean().reset_index()
mean_free_by_category['Free'] *= 100
mean_free_by_category = mean_free_by_category.sort_values(by='Free', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Free', data=mean_free_by_category, ci=None)
plt.xlabel('Category')
plt.ylabel('Free Apps Proportion')
plt.title('Proportion of Free Apps per Category')
plt.xticks(rotation=90, ha='right')
plt.ylim(91, 101)
plt.tight_layout()
plt.figure()

mean_free_by_Rating = dfGPlayStore.groupby('Rating')['Free'].mean().reset_index()
mean_free_by_Rating['Free'] *= 100
mean_free_by_Rating = mean_free_by_Rating.sort_values(by='Free', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Rating', y='Free', data=mean_free_by_Rating, ci=None)
plt.xlabel('Rating')
plt.ylabel('Free Apps Proportion')
plt.title('Proportion of Free Apps per Rating')
plt.xticks(rotation=90, ha='right')
plt.ylim(91, 101)
plt.tight_layout()
plt.figure()

#dfGPlayStore_pagos = dfGPlayStore[dfGPlayStore['Free'] == 0]
#proporcao_pagos_por_rating = (dfGPlayStore_pagos.groupby('Rating').size() / dfGPlayStore.groupby('Rating').size())
#plt.figure(figsize=(10, 6))
#sns.boxplot(x='Rating', y='Free', data=dfGPlayStore_pagos, showfliers=False)
#plt.xlabel('Rating')
#plt.ylabel('Proporção de Apps Pagos')
#plt.title('Proporção de Pagos por Rating')
#plt.xticks(rotation=90, ha='right')
#plt.ylim(0, 1)  # Limitando o eixo y entre 0 e 1 (proporção)
#plt.tight_layout()
#plt.show()

num_free_apps = dfGPlayStore['Free'].sum()
num_paid_apps = len(dfGPlayStore) - num_free_apps
labels = ['Free Apps', 'Paid Apps']
sizes = [num_free_apps, num_paid_apps]
colors = ['lightblue', 'lightcoral']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Free Apps vs Paid Ones')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.figure()



count_per_category = dfGPlayStore.groupby('Category')['App Name'].count().reset_index()
count_per_category.columns = ['Category', 'Count']
count_per_category = count_per_category.sort_values(by='Count', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(count_per_category['Category'], count_per_category['Count'], color='skyblue')
plt.xticks(rotation=90)
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.title('Number of Apps per Category')
plt.tight_layout()
plt.figure()


count_per_category_free = dfGPlayStore[dfGPlayStore['Free']].groupby('Category')['App Name'].count().reset_index()
count_per_category_free.columns = ['Category', 'Free Apps']
count_per_category_free = count_per_category_free.sort_values(by='Free Apps', ascending=False)
count_per_category_paid = dfGPlayStore[~dfGPlayStore['Free']].groupby('Category')['App Name'].count().reset_index()
count_per_category_paid.columns = ['Category', 'Paid Apps']
count_per_category_paid = count_per_category_paid.sort_values(by='Paid Apps', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(count_per_category_free['Category'], count_per_category_free['Free Apps'], color='skyblue')
plt.xticks(rotation=90)  # Rotar etiquetas del eje x para mejor legibilidad
plt.xlabel('Category')
plt.ylabel('Number of Free Apps')
plt.title('Number of Free Apps per Category')
plt.tight_layout()
plt.figure()

plt.figure(figsize=(10, 6))
plt.bar(count_per_category_paid['Category'], count_per_category_paid['Paid Apps'], color='salmon')
plt.xticks(rotation=90)  # Rotar etiquetas del eje x para mejor legibilidad
plt.xlabel('Category')
plt.ylabel('Number of Paid Apps')
plt.title('Number of Paid Apps per Category')
plt.tight_layout()
plt.figure()




dfGPlayStore['Released'] = pd.to_datetime(dfGPlayStore['Released'])
app_releases_by_month = dfGPlayStore.groupby(dfGPlayStore['Released'].dt.to_period('M')).size().resample('M').mean()
plt.figure(figsize=(10, 6))
plt.plot(app_releases_by_month.index.to_timestamp(), app_releases_by_month.values, marker='o', linestyle='-')
plt.title('Average Apps per Month')
plt.xlabel('Release Date')
plt.ylabel('Average Released Apps')
plt.grid(True)
plt.xticks(rotation=90)  # Rotar etiquetas del eje x para mejor legibilidad
plt.tight_layout()
plt.figure()

mean_rating_free = dfGPlayStore[dfGPlayStore['Free']]['Rating'].mean()
mean_rating_paid = dfGPlayStore[~dfGPlayStore['Free']]['Rating'].mean()
labels = ['Free', 'Paid']
ratings = [mean_rating_free, mean_rating_paid]
colors = ['lightblue', 'lightcoral']
plt.figure(figsize=(8, 8))
plt.pie(ratings, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Average Rating of Free Apps vs Paid Ones')
plt.axis('equal')
plt.show()








