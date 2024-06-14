import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import graphviz
import os
from IPython.display import Image

# Carregar o dataset
data = pd.read_csv('Data/Google-Playstore.csv', nrows=8000)

# Remover colunas indesejadas
data = data.drop(['App Name', 'App Id', 'Minimum Installs', 'Maximum Installs', 'Currency',
                  'Developer Id', 'Developer Website', 'Developer Email', 'Released',
                  'Last Updated', 'Privacy Policy', 'Scraped Time'], axis=1)

# Inspecionar os primeiros registros e verificar valores ausentes
print("Dados iniciais:")
print(data.head())
print(data.info())
print("Valores ausentes antes da imputação:")
print(data.isnull().sum())

# Remover linhas onde a variável alvo 'Installs' é nula
data = data.dropna(subset=['Installs'])

# Tratar a coluna 'Installs'
data['Installs'] = data['Installs'].str.replace('[+,]', '', regex=True)
data['Installs'] = pd.to_numeric(data['Installs'], errors='coerce').astype('Int64')

print("Exemplos de valores na coluna 'Installs' após a conversão:")
print(data['Installs'].head(10))

print("Valores ausentes após remoção de linhas com 'Installs' nulos:")
print(data.isnull().sum())

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

print("Content Mapping ", content_mapping, "\nCategory Mapping ", category_mapping)

# Preparação dos dados
X = data[['Category', 'Rating','Rating Count', 'Free','Price','Size','Minimum Android','Content Rating', 'Ad Supported','In App Purchases','Editors Choice']]
y = data['Installs']

# Aplicar PCA para redução de dimensionalidade
pca = PCA(n_components=5, svd_solver='auto')
X_reduced = pca.fit_transform(X)

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Função para avaliar o modelo
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model: {name}")
    print(f"R^2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}\n")
    return model

print("Tamanho atual do dataframe:", data.shape)

models = {}

# Avaliar modelos e armazenar no dicionário
models['Linear Regression'] = evaluate_model('Linear Regression', LinearRegression(), X_train, y_train, X_test, y_test)
models['Ridge Regression'] = evaluate_model('Ridge Regression', Ridge(alpha=1.0), X_train, y_train, X_test, y_test)
models['Lasso Regression'] = evaluate_model('Lasso Regression', Lasso(alpha=1.0), X_train, y_train, X_test, y_test)

models['SVR (linear kernel)'] = evaluate_model('SVR (linear kernel)', SVR(kernel='linear'), X_train, y_train, X_test, y_test)
models['SVR (rbf kernel)'] = evaluate_model('SVR (rbf kernel)', SVR(kernel='rbf'), X_train, y_train, X_test, y_test)
models['K-NN'] = evaluate_model('K-NN', KNeighborsRegressor(n_neighbors=2), X_train, y_train, X_test, y_test)
models['Decision Tree'] = evaluate_model('Decision Tree', DecisionTreeRegressor(), X_train, y_train, X_test, y_test)

models['Random Forest'] = evaluate_model('Random Forest', RandomForestRegressor(n_estimators=200), X_train, y_train, X_test, y_test)
models['Neural Network (single layer)'] = evaluate_model('Neural Network (single layer)', MLPRegressor(hidden_layer_sizes=(10000,), max_iter=10000), X_train, y_train, X_test, y_test)
models['Neural Network (multi layer)'] = evaluate_model('Neural Network (multi layer)', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=10000), X_train, y_train, X_test, y_test)


def preprocess_new_app(new_app):
    # Verificar se new_app é um DataFrame, se não, converter para DataFrame
    if not isinstance(new_app, pd.DataFrame):
        new_app = pd.DataFrame([new_app], columns=X.columns)

    # Preencher valores ausentes na nova aplicação
    handle_missing_values(new_app)

    # Tratar a coluna 'Size'
    new_app['Size'] = new_app['Size'].apply(size_to_mb)

    # Tratar a coluna 'Minimum Android'
    new_app['Minimum Android'] = new_app['Minimum Android'].apply(parse_android_version)
    new_app['Minimum Android'] = [int(x * 10) / 10 for x in new_app['Minimum Android']]

    # Codificar
def preprocess_new_app(new_app):
    # Verificar se new_app é um DataFrame, se não, converter para DataFrame
    if not isinstance(new_app, pd.DataFrame):
        new_app = pd.DataFrame([new_app], columns=X.columns)

    # Preencher valores ausentes na nova aplicação
    handle_missing_values(new_app)

    # Tratar a coluna 'Size'
    new_app['Size'] = new_app['Size'].apply(size_to_mb)

    # Tratar a coluna 'Minimum Android'
    new_app['Minimum Android'] = new_app['Minimum Android'].apply(parse_android_version)
    new_app['Minimum Android'] = [int(x * 10) / 10 for x in new_app['Minimum Android']]

    # Codificar variáveis categóricas
    new_app['Category'] = new_app['Category'].apply(lambda x: category_mapping.get(x, -1))
    new_app['Content Rating'] = new_app['Content Rating'].apply(lambda x: content_mapping.get(x, -1))

    # Verificar se há valores não reconhecidos e substituí-los por uma categoria válida (opcional)
    new_app['Category'] = new_app['Category'].replace(-1, category_mapping['Tools'])
    new_app['Content Rating'] = new_app['Content Rating'].replace(-1, content_mapping['Everyone'])

    # Aplicar PCA para redução de dimensionalidade
    new_app_reduced = pca.transform(new_app)

    return new_app_reduced


def predict_installs(new_app_data):
    # Pré-processar os dados da nova aplicação
    new_app_processed = preprocess_new_app(new_app_data)

    # Previsões para cada modelo
    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(new_app_processed)
        predictions[model_name] = prediction[0]

    # Imprimir previsões
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction:.0f} installs")


# Exemplo de dados de uma nova aplicação (formato CSV)
new_app_csv = 'new_app_example.csv'  # Substitua pelo caminho para o seu arquivo CSV

# Carregar exemplo de nova aplicação a partir de um arquivo CSV
new_app_df = pd.read_csv(new_app_csv)

# Prever o número de instalações para a nova aplicação
predict_installs(new_app_df)
