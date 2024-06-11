import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Carregar o dataset
data = pd.read_csv('Data/Google-Playstore.csv') #data é um DataFrame

# Limitar o dataset a 50.000 linhas
data = data.head(5000) #data = data.sample(n=500000, random_state=42) #para uma amostra aleatória #se for para as PRIMEIRAS 500 000 linhas: data = data.head(500000)


# Inspecionar os primeiros registros e verificar valores ausentes
print("Dados iniciais:")
print(data.head())
print(data.info())
print("Valores ausentes antes da imputação:")
print(data.isnull().sum())

# Remover linhas onde a variável alvo 'Installs' é nula
data = data.dropna(subset=['Installs'])

# Converter a coluna 'Installs' para um formato numérico (caso seja necessária)
#data['Installs'] = pd.to_numeric(data['Installs'], errors='coerce')
# data = data.dropna(subset=['Installs'])

# Ver alguns exemplos dos valores na coluna 'Installs' antes da conversão
print("Exemplos de valores na coluna 'Installs' antes da conversão:")
print(data['Installs'].head(10))



# Vamos tratar os dados de 'Installs' para que possamos utilizá-los como variável alvo
# Remover caracteres especiais
data['Installs'] = data['Installs'].str.replace('[+,]', '', regex=True) #retira os caracteres + e , e substitui por '', ou seja, retira

# Verificar se a conversão foi bem sucedida e converter para int
data['Installs'] = pd.to_numeric(data['Installs'], errors='coerce').astype('Int64')

# Verificar novamente alguns exemplos após a conversão
print("Exemplos de valores na coluna 'Installs' após a conversão:")
print(data['Installs'].head(10))

# Atualizar os valores ausentes após a remoção de linhas
print("Valores ausentes após remoção de linhas com 'Installs' nulos:")
print(data.isnull().sum())

# Preparação dos dados
# Suponha que a coluna 'Installs' é a variável alvo e as outras são features
X = data.drop(['Installs', 'App Name', 'App Id', 'Developer Website', 'Developer Email', 'Released',	'Last Updated', 'Content Rating',	'Privacy Policy', 'Scraped Time'], axis=1)
y = data['Installs']

# Verificar se ainda existem valores ausentes em y
print(f"Valores ausentes em y: {y.isnull().sum()}")


# Identificar variáveis numéricas e categóricas
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessamento
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputação de valores ausentes
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputação de valores ausentes
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para avaliar o modelo e retornar o pipeline treinado
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    '''
    As linhas dentro desta função relacionadas com o pipeline, criam e treinam um pipeline
    que aplica pré-processamento aos dados (`preprocessor`) e então ajusta o modelo (`model`).
    Verificam valores ausentes após a imputação, fazem previsões e avaliam o desempenho do modelo
    usando métricas de erro.
    '''
    # Verificar o tipo de 'y' antes do treinamento
    print(f"Tipo de 'y' (Installs) antes do treino: {y_train.dtype}")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])  # Aqui é onde o modelo é adicionado ao pipeline e o que o pipeline faz é aplicar o preprocessor e depois o modelo

    pipeline.fit(X_train, y_train)
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    print(f"Valores ausentes após imputação (treino) para o modelo {name}:")
    print(pd.DataFrame(X_train_transformed).isnull().sum())
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model: {name}")
    print(f"R^2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}\n")
    return pipeline
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)
#Saber informações sobre o dataframe
print("data.info():", data.info())
#Ter o describe do dataset
print("data.describe():", data.describe())
#Qual o tipo de dados da coluna 'Installs'
print("Tipo de dados da coluna 'Installs':", data['Installs'].dtype)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)
# Avaliar modelo e obter o pipeline treinado
pipeline = evaluate_model('Linear Regression', LinearRegression(), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)

# Carregar o ficheiro CSV da nova app
new_app_df = pd.read_csv('new_app.csv')  # Substitua 'path/to/' pelo caminho correto do seu ficheiro CSV

# Remover colunas não presentes nos dados de treino
new_app_df = new_app_df[X.columns]

# Função para prever a variável alvo para uma nova app
def predict_new_app(pipeline, new_app):
    # Preprocessar a nova app
    new_app_transformed = pipeline.named_steps['preprocessor'].transform(new_app)
    # Prever a variável alvo
    prediction = pipeline.named_steps['model'].predict(new_app_transformed)
    return prediction

# Fazer a previsão utilizando o pipeline treinado anteriormente
prediction = predict_new_app(pipeline, new_app_df)
print(f"Previsão de 'Installs' para a nova app 'Gakondo': {prediction[0]:.0f}")
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)

# Avaliar modelo
pipeline = evaluate_model('Ridge Regression', Ridge(alpha=1.0), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)

# Avaliar modelo
pipeline = evaluate_model('Lasso Regression', Lasso(alpha=1.0), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)

# Avaliar modelo
pipeline = evaluate_model('SVR (linear kernel)', SVR(kernel='linear'), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)

# Avaliar modelo
pipeline = evaluate_model('SVR (rbf kernel)', SVR(kernel='rbf'), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)

# Avaliar modelo
pipeline = evaluate_model('K-NN', KNeighborsRegressor(n_neighbors=5), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)

# Avaliar modelo
pipeline = evaluate_model('Decision Tree', DecisionTreeRegressor(), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)

# Visualizar a árvore de decisão
# Importar funções para plotagem de árvores de decisão
from sklearn.tree import plot_tree

# Extrair o modelo treinado do pipeline
decision_tree_model = pipeline.named_steps['model']

# Plotar a árvore de decisão
plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Árvore de Decisão")
plt.show()


# Visualizar uma árvore do Random Forest

# Avaliar modelo
pipeline = evaluate_model('Random Forest', RandomForestRegressor(), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)
# Extrair o modelo Random Forest treinado do pipeline
random_forest_model = pipeline.named_steps['model']

# Selecionar uma árvore do Random Forest
estimator = random_forest_model.estimators_[0]

# Plotar a árvore do Random Forest
plt.figure(figsize=(20, 10))
plot_tree(estimator, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Uma Árvore do Random Forest")
plt.show()
# Avaliar modelo
pipeline = evaluate_model('Neural Network (single layer)', MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)
# Avaliar modelo
pipeline = evaluate_model('Neural Network (multi layer)', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000), X_train, y_train, X_test, y_test)
#Imprimir tamanho do dataframe
print("Tamanho atual do dataframe:", data.shape)