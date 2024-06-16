import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# Carregar o dataset
data = pd.read_csv('Data/Google-Playstore.csv', nrows=8000)

# Remover colunas indesejadas
data = data.drop(['App Name', 'App Id', 'Minimum Installs', 'Maximum Installs', 'Currency',
                  'Developer Id', 'Developer Website', 'Developer Email', 'Released',
                  'Last Updated', 'Privacy Policy', 'Scraped Time'], axis=1)

# Remover linhas onde a variável alvo 'Installs' é nula
data = data.dropna(subset=['Installs'])

# Tratar a coluna 'Installs'
data['Installs'] = data['Installs'].str.replace('[+,]', '', regex=True)
data['Installs'] = pd.to_numeric(data['Installs'], errors='coerce').astype('Int64')

# Funções para tratamento de dados
def handle_missing_values(df):
    df['Category'].fillna(df['Category'].mode()[0], inplace=True)
    df['Rating'].fillna(df['Rating'].mean(), inplace=True)
    df['Rating Count'].fillna(df['Rating Count'].mean(), inplace=True)
    df['Free'].fillna(True, inplace=True)
    df['Price'].fillna(0.0, inplace=True)
    df['Size'] = df['Size'].apply(size_to_mb)
    df['Size'].fillna(df['Size'].mean(), inplace=True)
    df['Minimum Android'].fillna(df['Minimum Android'].mode()[0], inplace=True)
    df['Content Rating'].fillna(df['Content Rating'].mode()[0], inplace=True)
    df['Ad Supported'].fillna(True, inplace=True)
    df['In App Purchases'].fillna(False, inplace=True)
    df['Editors Choice'].fillna(False, inplace=True)

def size_to_mb(size):
    if pd.isna(size):
        return np.nan
    if isinstance(size, str):
        if 'M' in size or 'm' in size:
            return float(size.replace('M', '').replace('m', '').replace(',', '.'))
        elif 'K' in size or 'k' in size:
            return float(size.replace('K', '').replace('k', '').replace(',', '.')) / 1024
        elif 'G' in size or 'g' in size:
            return float(size.replace('G', '').replace('g', '').replace(',', '.')) * 1024
    return np.nan

def parse_android_version(version):
    if pd.isna(version):
        return np.nan
    if 'Varies with device' in version:
        return np.nan
    if 'and up' in version:
        version = version.replace('and up', '').strip()
    if '-' in version:
        version = version.split('-')[0].strip()
    version = version.replace('W', '').strip()
    try:
        return float(version)
    except ValueError:
        return np.nan

handle_missing_values(data)
data['Minimum Android'] = data['Minimum Android'].apply(parse_android_version)
mean_android_version = data['Minimum Android'].mean()
data['Minimum Android'].fillna(mean_android_version, inplace=True)

data['Rating Count'] = data['Rating Count'].astype(int)
data['Minimum Android'] = [int(x * 10) / 10 for x in data['Minimum Android']]

label_encoder_category = LabelEncoder()
label_encoder_content_rating = LabelEncoder()
data['Category'] = label_encoder_category.fit_transform(data['Category'])
category_mapping = dict(zip(label_encoder_category.classes_, label_encoder_category.transform(label_encoder_category.classes_)))
data['Content Rating'] = label_encoder_content_rating.fit_transform(data['Content Rating'])
content_mapping = dict(zip(label_encoder_content_rating.classes_, label_encoder_content_rating.transform(label_encoder_content_rating.classes_)))

# Preparação dos dados
X = data[['Category', 'Rating','Rating Count', 'Free','Price','Size','Minimum Android','Content Rating', 'Ad Supported','In App Purchases','Editors Choice']]
y = data['Installs']

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para avaliar o modelo
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return name, rmse

# Avaliar modelos
models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge(alpha=1.0)),
    ('Lasso Regression', Lasso(alpha=1.0)),
    ('SVR (linear kernel)', SVR(kernel='linear')),
    ('SVR (rbf kernel)', SVR(kernel='rbf')),
    ('K-NN', KNeighborsRegressor(n_neighbors=2)),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor(n_estimators=200)),
    ('Neural Network (single layer)', MLPRegressor(hidden_layer_sizes=(10000,), max_iter=10000)),
    ('Neural Network (multi layer)', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=10000))
]

# Armazenar RMSE de cada modelo
model_rmse = [evaluate_model(name, model, X_train, y_train, X_test, y_test) for name, model in models]

# Criar gráfico de comparação de RMSE
model_names = [name for name, _ in model_rmse]
rmse_values = [rmse for _, rmse in model_rmse]

plt.figure(figsize=(12, 8))
plt.barh(model_names, rmse_values, color='skyblue')
plt.xlabel('RMSE')
plt.title('Comparison of RMSE for Different Models')
plt.grid(True)
plt.show()
