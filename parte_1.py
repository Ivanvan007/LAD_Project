import os
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

file_name = "Google-Playstore.csv"
file_path = "Data/"
full_path = os.path.join(file_path, file_name)
#full_path = "Data/Google-Playstore.csv"

#Start CSV File Reading
#Ler-se-ão apenas as primeiras 180000 linhas do arquivo
dfGPlayStore = pd.read_csv(full_path,nrows=500000)
#dfGPlayStore = pd.read_csv(full_path) #descomentar para usar o dataset enteiro. comentar a linha de cima #dfGPlayStore = dfGPlayStore.dropna()
#CSV File Reading Finished


print("---------------Tarefa 1: Descrição do Dataset--------------")
print("Descrição da tarefa: Fazer uma descrição das características do dataset (e.g., domínio, tamanho, tipos de dados, entidades, etc.")
#Domain
print("Domínio: aplicações mobile disponíveis na Google Play Store.")
#Size
print("Tamanho:")
print("\tNúmero de registos/linhas: para que fosse possível correr o código, limitamos o número de linhas a 180000. No entanto, o dataset original contém 1 048 576 linhas.")
print("\tVolume de dados: 645 MB (o dataset original).")
#data types
print("Tipos de Dados:")
dfGPlayStore.info()
print("Acima apresentam-se os tipos de dados de cada coluna do dataset. De acordo com a penúltima linha, o dataset contém: 4 colunas com dados do tipo bool, 4 do tipo float64, 1 do tipo int64 e 15 do tipo object.")
#Entities
print("Entidades: são as apps (linhas do dataset).")

print("---------------Tarefa 2: Análise Estatística--------------")
print("Descrição da tarefa: desenvolver uma análise estatística utilizando medidas como a média, a variância, covariância, ou correlações.")
#Average
print("Média:")
dataSetDescription = dfGPlayStore.describe()
print(dataSetDescription)
#Variance (só vai dar para calcular a variância das colunas numéricas - colunas do tipo object não têm variância)
'''


Rating
Rating Count
Minimum Installs
Maximum Installs
Price

- Estas são as únicas colunas numéricas do dataset e, portanto, as únicas
para as quais dá para calcular a variância, covariância e os coeficientes de correlação. 


'''
print("Variância:")
print(dataSetDescription.var())
#Covariance
print("Covariância:")
variables_list = dfGPlayStore.columns.tolist()
numeric_columns = dfGPlayStore.select_dtypes(include=['number']).columns.tolist()
covariance = dfGPlayStore[numeric_columns].cov()
correlation = dfGPlayStore[numeric_columns].corr()
print(covariance)
#Correlations
print("--------Correlações (Matriz dos Coeficientes de Correlação)--------")
print(correlation)


print("-----------Tarefa 3: Representação Gráfica dos Resultados-----------")
print("Descrição da tarefa: fazer uma representação gráfica e coerente dos resultados.")

print("-----------1. Histograma do Rating-----------")

# Supondo que dfGPlayStore seja o seu DataFrame

# Análise inicial na consola
print("Descrição da coluna 'Rating':")
print(dfGPlayStore['Rating'].describe())

# Contando os valores nulos
nulos_count = dfGPlayStore['Rating'].isna().sum()
print("\nNúmero de valores nulos na coluna 'Rating':", nulos_count)

# Contagem de valores por rating
rating_counts = dfGPlayStore['Rating'].value_counts().sort_index()
print("\nContagem de apps por rating (excluindo nulos):")
print(rating_counts)

# Criação do histograma
plt.figure(figsize=(10, 6))
# Histograma dos valores não nulos
dfGPlayStore['Rating'].hist(bins=range(0, 6), edgecolor='black')
plt.title('Distribuição de Ratings de Apps')
plt.xlabel('Rating')
plt.ylabel('Número de Apps')
plt.xticks(range(0, 6))

# Incluindo informação sobre valores nulos no gráfico
#plt.text(0.5, nulos_count, f'Valores nulos: {nulos_count}', fontsize=12, color='red')

plt.tight_layout()
plt.savefig('Graficos/Histograma_Rating.png')


print("-----------2. Gráfico de Barras - Categorias de App-----------")
#Código para o gráfico
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
#Save histogram (e não mostrar) as an image file (png) in the folder "Graficos" which is in the same directory as the parte_1.py file
plt.savefig('Graficos/NumberOfAppsPerCategory.png')

print("-----------3. Scatter Plot de Ratings vs. Installs-----------")
# Supondo que dfGPlayStore seja seu DataFrame

# Limpeza e conversão da coluna 'Installs'
# Removendo '+' e substituindo ',' por nada para facilitar a conversão para float
dfGPlayStore['Installs_Clean'] = dfGPlayStore['Installs'].str.replace(r'[+,]', '', regex=True).astype(float)

# Agora, criando o scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(dfGPlayStore['Rating'], dfGPlayStore['Installs_Clean'], alpha=0.5)

# Configurando a escala do eixo y para logarítmica para melhor visualização
plt.yscale('log')

# Ajustando os labels e título do gráfico
plt.title('Scatter Plot de Ratings vs. Installs')
plt.xlabel('Rating')
plt.ylabel('Installs (Log Scale)')
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Ajuste para garantir que o gráfico mostre todas as faixas de valores corretamente
plt.xlim(0, 5)  # Considerando que os ratings vão de 0.0 a 5.0
plt.ylim(1, dfGPlayStore['Installs_Clean'].max())

plt.tight_layout()
plt.savefig('Graficos/ScatterPlot_Ratings_vs_Installs.png')

print("-----------4. Boxplot de Ratings por Categoria-----------")
# Ajustando o tamanho da figura para acomodar melhor as categorias no eixo x
plt.figure(figsize=(14, 8))

# Criação do box plot com categorias no eixo x e ratings no eixo y
sns.boxplot(y='Rating', x='Category', data=dfGPlayStore, orient='v')

# Melhorando a legibilidade do gráfico
plt.xticks(rotation=90)  # Rotacionando os nomes das categorias para melhor visualização
plt.title('Box Plot de Ratings por Categoria')
plt.ylabel('Rating')
plt.xlabel('Categoria')

# Ajustando o layout para evitar cortes nos rótulos
plt.tight_layout()

# Salvando o gráfico
plt.savefig('Graficos/BoxPlot_Ratings_por_Categoria.png')


print("-----------5. Histogram of App Sizes-----------")
def size_to_mb(size):
    # Verifica se o valor já é NaN (float)
    if pd.isna(size):
        return np.nan
    # Garante que a operação só será feita em strings
    if isinstance(size, str):
        if 'M' in size:
            return float(size.replace('M', '').replace(',', ''))
        elif 'k' in size:
            return float(size.replace('k', '').replace(',', '')) / 1024
    # Retorna np.nan para valores que não correspondem aos formatos esperados
    return np.nan

# Aplicando a conversão atualizada e criando o histograma
dfGPlayStore['Size_MB'] = dfGPlayStore['Size'].apply(size_to_mb)

plt.figure(figsize=(10, 6))
plt.hist(dfGPlayStore['Size_MB'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Histograma de App Sizes')
plt.xlabel('Tamanho (MB)')
plt.ylabel('Número de Apps')
plt.grid(axis='y', alpha=0.75)

plt.tight_layout()
plt.savefig('Graficos/Histograma_AppSizes.png')


print("-----------6. Pie Chart of Free vs. Paid Apps-----------")
#Proportion of Free Apps vs Paid Ones
num_free_apps = dfGPlayStore['Free'].sum()
num_paid_apps = len(dfGPlayStore) - num_free_apps
labels = ['Free Apps', 'Paid Apps']
sizes = [num_free_apps, num_paid_apps]
colors = ['lightblue', 'lightcoral']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Free Apps vs Paid Ones')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('Graficos/ProportionOfFreeAppsVsPaidOnes.png')

print("-----------7. Bar Chart of Categories by Maximum Installs-----------")
#InstallsPerCategory
downloads_per_category = dfGPlayStore.groupby('Category')['Maximum Installs'].sum().reset_index()
downloads_per_category = downloads_per_category.sort_values(by='Maximum Installs', ascending=False)
plt.figure(figsize=(10,6))
plt.bar(downloads_per_category['Category'], downloads_per_category['Maximum Installs'])
plt.xlabel('Category')
plt.ylabel('Maximum Installs')
plt.title('Installs per Category')
plt.xticks(rotation=90,ha='right',fontsize=10)
plt.tight_layout()
plt.savefig('Graficos/InstallsPerCategory.png')

print("-----------8. Line Graph of App Releases Over Time-----------")
#AverageAppsPerMonth
# Conversão da coluna 'Released' para datetime
dfGPlayStore['Released'] = pd.to_datetime(dfGPlayStore['Released'])

# Agrupando os lançamentos por mês
app_releases_by_month = dfGPlayStore.groupby(dfGPlayStore['Released'].dt.to_period('M')).size()

# Convertendo o índice de PeriodIndex para DatetimeIndex
app_releases_by_month.index = app_releases_by_month.index.to_timestamp()

# Calculando a média de lançamentos por mês através do resample
app_releases_by_month = app_releases_by_month.resample('M').mean()

# Criando o gráfico
plt.figure(figsize=(10, 6))
plt.plot(app_releases_by_month.index, app_releases_by_month.values, marker='o', linestyle='-')
plt.title('Average Apps per Month')
plt.xlabel('Release Date')
plt.ylabel('Average Released Apps')
plt.grid(True)
plt.xticks(rotation=90)  # Rotar etiquetas del eje x para mejor legibilidad
plt.tight_layout()
plt.savefig('Graficos/AverageAppsPerMonth.png')

print("-----------9. Rating Average Per Category-----------")
#RatingAveragePerCategory
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Rating', data=dfGPlayStore, ci=None, estimator=lambda x: sum(x) / len(x))
plt.xlabel('Category')
plt.ylabel('Rating Average')
plt.title('Rating Average per Category')
plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.savefig('Graficos/RatingAveragePerCategory.png')

print("-----------10. Heatmap de Correlação entre Variáveis Numéricas-----------")
#HeatmapCorrelation
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.title('Heatmap Correlation')
plt.tight_layout()
plt.savefig('Graficos/HeatmapCorrelation.png')


print("-----------Gráficos Adicionais-----------")

print("----1. Gráfico de Barras para a coluna Installs----")

#Confirmação com prints na consola

# Contando o número de ocorrências de cada nível de rating, excluindo nulos
contagem_categorias = dfGPlayStore['Installs'].value_counts()

# Contando os valores nulos na coluna 'Rating'
nulos_count = dfGPlayStore['Installs'].isna().sum()

# Criando um novo Series para "Nulos"
nulos_series = pd.Series({'Nulos': nulos_count})

# Concatenando contagem_categorias com nulos_series para incluir os nulos
contagem_total = pd.concat([contagem_categorias, nulos_series])

# Imprimindo os valores de contagem para cada categoria, incluindo "Nulos"
print("Contagem de Apps por Número de Installs (incluindo nulos):")
print(contagem_total)

# O total agora pode ser simplesmente a soma de contagem_total, já que inclui os nulos
total_apps = contagem_total.sum()

# Imprimindo o total de apps, agora incluindo os nulos automaticamente
print("\nTotal de Apps (incluindo nulos):", total_apps)


#Código para o gráfico
# Passo 1: Limpeza e conversão
dfGPlayStore['Installs_numeric'] = dfGPlayStore['Installs'].str.replace(r'\D', '', regex=True).astype(float)
# Criando um mapeamento reverso de números limpos para strings originais
installs_mapping = dfGPlayStore.dropna(subset=['Installs_numeric']).drop_duplicates('Installs_numeric').set_index('Installs_numeric')['Installs']

# Passo 2: Ordenação
# Já temos 'Installs_numeric' para ordenação e contagem. Agora, precisamos preparar o plot_df
# Incluindo a contagem de nulos
nulos_count = dfGPlayStore['Installs_numeric'].isna().sum()
contagem_installs = dfGPlayStore['Installs_numeric'].value_counts().sort_index()

# Passo 3: Mapeamento de volta para strings originais
# Usamos o mapeamento reverso para substituir os valores numéricos pelas strings originais
contagem_installs.index = contagem_installs.index.map(installs_mapping)

# Preparando o DataFrame para plotagem
contagem_df = contagem_installs.reset_index()
contagem_df.columns = ['Installs', 'Count']
nulos_df = pd.DataFrame({'Installs': ['Nulos'], 'Count': [nulos_count]})
plot_df = pd.concat([nulos_df, contagem_df], ignore_index=True)

# Plotagem
plt.figure(figsize=(10,10))
ax = sns.barplot(data=plot_df, y='Installs', x='Count', order=['Nulos'] + list(contagem_installs.index))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.xlabel('Frequência', labelpad=10)
plt.title('Número de Apps por Número de Instalações', pad=30, fontweight='bold', loc='center', fontsize=20)

# Ajustando o posicionamento do texto para barras horizontais
for p in ax.patches:
    width = p.get_width()
    y = p.get_y() + p.get_height() / 2
    x = width + ax.get_xlim()[1] * 0.01
    ax.annotate(f'{int(width)}', (x, y), va='center')
plt.tight_layout()
plt.savefig('Graficos/Bar_chart_Installs_Column.png')


print("----2. Gráfico de Barras para a coluna Minimum Installs")

#Confirmação com prints na consola

# Contando o número de ocorrências de cada nível de rating, excluindo nulos
contagem_categorias = dfGPlayStore['Minimum Installs'].value_counts()

# Contando os valores nulos na coluna 'Rating'
nulos_count = dfGPlayStore['Minimum Installs'].isna().sum()

# Criando um novo Series para "Nulos"
nulos_series = pd.Series({'Nulos': nulos_count})

# Concatenando contagem_categorias com nulos_series para incluir os nulos
contagem_total = pd.concat([contagem_categorias, nulos_series])

# Imprimindo os valores de contagem para cada categoria, incluindo "Nulos"
print("Contagem de Apps por Número de Minimum Installs (incluindo nulos):")
print(contagem_total)

# O total agora pode ser simplesmente a soma de contagem_total, já que inclui os nulos
total_apps = contagem_total.sum()

# Imprimindo o total de apps, agora incluindo os nulos automaticamente
print("\nTotal de Apps (incluindo nulos):", total_apps)


#Código para o gráfico
# Passo 1: Limpeza e conversão
dfGPlayStore['Installs_numeric'] = dfGPlayStore['Minimum Installs'].astype(float)

# Criando um mapeamento reverso de números limpos para strings originais
installs_mapping = dfGPlayStore.dropna(subset=['Installs_numeric']).drop_duplicates('Installs_numeric').set_index('Installs_numeric')['Minimum Installs']

# Passo 2: Ordenação
# Já temos 'Installs_numeric' para ordenação e contagem. Agora, precisamos preparar o plot_df
# Incluindo a contagem de nulos
nulos_count = dfGPlayStore['Installs_numeric'].isna().sum()
contagem_installs = dfGPlayStore['Installs_numeric'].value_counts().sort_index()

# Passo 3: Mapeamento de volta para strings originais
# Usamos o mapeamento reverso para substituir os valores numéricos pelas strings originais
contagem_installs.index = contagem_installs.index.map(installs_mapping)

# Preparando o DataFrame para plotagem
contagem_df = contagem_installs.reset_index()
contagem_df.columns = ['Minimum Installs', 'Count']
nulos_df = pd.DataFrame({'Minimum Installs': ['Nulos'], 'Count': [nulos_count]})
plot_df = pd.concat([nulos_df, contagem_df], ignore_index=True)

# Plotagem
plt.figure(figsize=(10,10))
ax = sns.barplot(data=plot_df, y='Minimum Installs', x='Count', order=['Nulos'] + list(contagem_installs.index))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.xlabel('Frequência', labelpad=10)
plt.title('Número de Apps por Número de Minimum Installs', pad=30, fontweight='bold', loc='center', fontsize=20)

# Ajustando o posicionamento do texto para barras horizontais
for p in ax.patches:
    width = p.get_width()
    y = p.get_y() + p.get_height() / 2
    x = width + ax.get_xlim()[1] * 0.01
    ax.annotate(f'{int(width)}', (x, y), va='center')
plt.tight_layout()
plt.savefig('Graficos/Bar_Chart_Minimum_Installs_Column.png')


print("----3. Gráfico de Barras para a coluna Free")

#Confirmação com prints na consola

# Contando o número de ocorrências de cada nível de rating, excluindo nulos
contagem_categorias = dfGPlayStore['Free'].value_counts()

# Contando os valores nulos na coluna 'Rating'
nulos_count = dfGPlayStore['Free'].isna().sum()

# Criando um novo Series para "Nulos"
nulos_series = pd.Series({'Nulos': nulos_count})

# Concatenando contagem_categorias com nulos_series para incluir os nulos
contagem_total = pd.concat([contagem_categorias, nulos_series])

# Imprimindo os valores de contagem para cada categoria, incluindo "Nulos"
print("Contagem de Apps por Número de Modalidades de Preço (incluindo nulos):")
print(contagem_total)

# O total agora pode ser simplesmente a soma de contagem_total, já que inclui os nulos
total_apps = contagem_total.sum()

# Imprimindo o total de apps, agora incluindo os nulos automaticamente
print("\nTotal de Apps (incluindo nulos):", total_apps)

#Gráfico

# Mapeamento de True/False para Free/Paid, tratando nulos separadamente
dfGPlayStore['Free_Paid'] = dfGPlayStore['Free'].map({True: 'Gratuitas', False: 'Pagas'}).fillna('Nulos')

# Contagem, agora incluindo explicitamente "Nulos"
nulos_count = dfGPlayStore['Free'].isna().sum()
contagem_apps = dfGPlayStore['Free_Paid'].value_counts()

# Caso "Nulos" não esteja já presente
if 'Nulos' not in contagem_apps:
    contagem_apps['Nulos'] = 0

# Preparação do DataFrame para plotagem
contagem_df = contagem_apps.reset_index()
contagem_df.columns = ['Free_Paid', 'Count']

# Ordenando manualmente para garantir a ordem desejada
contagem_df['Free_Paid'] = pd.Categorical(contagem_df['Free_Paid'], ["Gratuitas", "Pagas", "Nulos"])
contagem_df.sort_values('Free_Paid', inplace=True)

# Plotagem para barras verticais
plt.figure(figsize=(10,10))
ax = sns.barplot(data=contagem_df, x='Free_Paid', y='Count')
plt.xticks(rotation=45)  # Ajuste se necessário
plt.ylabel('Frequência', labelpad=10)
plt.title('Número de Apps por Modalidade de Preço', pad=30, fontweight='bold', loc='center', fontsize=20)

# Ajustando o posicionamento do texto para barras verticais
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.xlabel('Modalidade de Preço', labelpad=10)
plt.savefig('Graficos/Bar_Chart_Free_Column.png')



print("----4. Gráfico de Barras para a coluna Price----")


#Confirmação com prints na consola

# Contando o número de ocorrências de cada nível de rating, excluindo nulos
contagem_categorias = dfGPlayStore['Price'].value_counts()

# Contando os valores nulos na coluna 'Rating'
nulos_count = dfGPlayStore['Price'].isna().sum()

# Criando um novo Series para "Nulos"
nulos_series = pd.Series({'Nulos': nulos_count})

# Concatenando contagem_categorias com nulos_series para incluir os nulos
contagem_total = pd.concat([contagem_categorias, nulos_series])

# Imprimindo os valores de contagem para cada categoria, incluindo "Nulos"
print("Contagem de Apps por Preço (incluindo nulos):")
#Ordenar por ordem decrescente de Preço
print(contagem_total.sort_values(ascending=False))

# O total agora pode ser simplesmente a soma de contagem_total, já que inclui os nulos
total_apps = contagem_total.sum()

# Imprimindo o total de apps, agora incluindo os nulos automaticamente
print("\nTotal de Apps (incluindo nulos):", total_apps)

#Gráfico


# Definindo os intervalos de preço e as labels atualizados
max_price = dfGPlayStore['Price'].max()
bins = [0, 0.01, 1, 5, 10, 20, 50] + list(np.arange(100, max_price + 50, 50))
labels = ['0.00 (Gratuitas)', '0.01-1.00', '1.01-5.00', '5.01-10.00', '10.01-20.00', '20.01-50.00'] + [f'{i}-{i+49.99}' for i in np.arange(50, max_price, 50)]
labels.append('Nulos')  # Incluindo "Nulos" nas labels

# Categorizando os preços e ajustando para incluir "Nulos" como uma categoria
dfGPlayStore['Price_Range'] = pd.cut(dfGPlayStore['Price'], bins=bins, labels=labels[:-1], right=False, include_lowest=True)

# Ajustando as categorias para incluir 'Nulos'
dfGPlayStore['Price_Range'] = dfGPlayStore['Price_Range'].cat.add_categories(['Nulos'])
dfGPlayStore['Price_Range'] = dfGPlayStore['Price_Range'].fillna('Nulos')

# Contagem de apps por intervalo de preço
price_range_count = dfGPlayStore['Price_Range'].value_counts()

# Preparando o DataFrame para plotagem
price_range_df = price_range_count.reset_index()
price_range_df.columns = ['Price_Range', 'Count']
price_range_df = price_range_df.sort_values(by='Price_Range')

# Plotagem das barras horizontais
plt.figure(figsize=(10, 12))
ax = sns.barplot(data=price_range_df, y='Price_Range', x='Count')
plt.xlabel('Frequência', labelpad=10)
plt.ylabel('Intervalo de Preço', labelpad=10)
plt.title('Número de Apps por Intervalo de Preço', pad=20, fontweight='bold', loc='center', fontsize=20)

# Ajustando o posicionamento do texto para barras horizontais
for p in ax.patches:
    width = p.get_width()
    ax.annotate(f'{int(width)}', (width, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', xytext=(5, 0), textcoords='offset points')

plt.tight_layout()
plt.savefig('Graficos/Bar_Chart_Price_Column.png')


print("----5. Gráfico de Barras para a coluna Currency----")


# Contando o número de ocorrências para cada moeda, excluindo nulos
contagem_categorias = dfGPlayStore['Currency'].value_counts()

# Contando os valores nulos na coluna 'Currency'
nulos_count = dfGPlayStore['Currency'].isna().sum()

# Criando um novo Series para "Nulos"
nulos_series = pd.Series({'Nulos': nulos_count})

# Concatenando contagem_categorias com nulos_series para incluir os nulos
contagem_total = pd.concat([contagem_categorias, nulos_series])

# Imprimindo os valores de contagem para cada categoria, incluindo "Nulos"
print("Contagem de Apps por Moeda (incluindo nulos):")
print(contagem_total.sort_values(ascending=False))

# O total agora pode ser simplesmente a soma de contagem_total, já que inclui os nulos
total_apps = contagem_total.sum()

# Imprimindo o total de apps, agora incluindo os nulos automaticamente
print("\nTotal de Apps (incluindo nulos):", total_apps)

# Preparação do DataFrame para plotagem
currency_df = contagem_total.reset_index()
currency_df.columns = ['Currency', 'Count']
currency_df = currency_df.sort_values(by='Count', ascending=False)

# Plotagem das barras verticais
plt.figure(figsize=(10, 8))
ax = sns.barplot(data=currency_df, x='Currency', y='Count')
plt.xticks(rotation=45)
plt.xlabel('Moeda', labelpad=10)
plt.ylabel('Frequência', labelpad=10)
plt.title('Número de Apps por Moeda', pad=20, fontweight='bold', loc='center', fontsize=20)

# Ajustando o posicionamento do texto para barras verticais
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.savefig('Graficos/Bar_Chart_Currency_Column.png')


print("----6. Average Rating Of Free Apps Vs Paid Ones----")
#AverageRatingOfFreeAppsVsPaidOnes
mean_rating_free = dfGPlayStore[dfGPlayStore['Free']]['Rating'].mean()
mean_rating_paid = dfGPlayStore[~dfGPlayStore['Free']]['Rating'].mean()
labels = ['Free', 'Paid']
ratings = [mean_rating_free, mean_rating_paid]
colors = ['lightblue', 'lightcoral']
plt.figure(figsize=(8, 8))
plt.pie(ratings, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Average Rating of Free Apps vs Paid Ones')
plt.axis('equal')
plt.savefig('Graficos/AverageRatingOfFreeAppsVsPaidOnes.png')



count_per_category_free = dfGPlayStore[dfGPlayStore['Free']].groupby('Category')['App Name'].count().reset_index()
count_per_category_free.columns = ['Category', 'Free Apps']
count_per_category_free = count_per_category_free.sort_values(by='Free Apps', ascending=False)
count_per_category_paid = dfGPlayStore[~dfGPlayStore['Free']].groupby('Category')['App Name'].count().reset_index()
count_per_category_paid.columns = ['Category', 'Paid Apps']
count_per_category_paid = count_per_category_paid.sort_values(by='Paid Apps', ascending=False)

print("----7. Number Of Paid Apps Per Category----")
#NumberOfPaidAppsPerCategory
plt.figure(figsize=(10, 6))
plt.bar(count_per_category_paid['Category'], count_per_category_paid['Paid Apps'], color='salmon')
plt.xticks(rotation=90)  # Rotar etiquetas del eje x para mejor legibilidad
plt.xlabel('Category')
plt.ylabel('Number of Paid Apps')
plt.title('Number of Paid Apps per Category')
plt.tight_layout()
plt.savefig('Graficos/NumberOfPaidAppsPerCategory.png')

print("----8. Number Of Free Apps Per Category")
#NumberOfFreeAppsPerCategory
plt.figure(figsize=(10, 6))
plt.bar(count_per_category_free['Category'], count_per_category_free['Free Apps'], color='skyblue')
plt.xticks(rotation=90)  # Rotar etiquetas del eje x para mejor legibilidad
plt.xlabel('Category')
plt.ylabel('Number of Free Apps')
plt.title('Number of Free Apps per Category')
plt.tight_layout()
plt.savefig('Graficos/NumberOfFreeAppsPerCategory.png')


print("----9. Proportion Of Free Apps Per Rating")
#ProportionOfFreeAppsPerRating
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
plt.savefig('Graficos/ProportionOfFreeAppsPerRating.png')


print("----10. Proportion Of Free Apps Per Category")
#ProportionOfFreeAppsPerCategory
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
plt.savefig('Graficos/ProportionOfFreeAppsPerCategory.png')

print("------------------Tarefa 4: Análise Crítica------------------")
print("Nesta secção, os dados e gráficos produzidos nas tarefas anteriores serão analisados criticamente.")