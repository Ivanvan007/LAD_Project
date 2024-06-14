import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import joblib
import traceback
import numpy as np

# Carregar modelos e mapeamentos
models = joblib.load('models.pkl')
label_encoder_category = joblib.load('label_encoder_category.pkl')
label_encoder_content_rating = joblib.load('label_encoder_content_rating.pkl')
category_mapping = joblib.load('category_mapping.pkl')
content_mapping = joblib.load('content_mapping.pkl')

# Função para logar mensagens em CSV
def log_message(message, data=None):
    print(message)
    if data is not None:
        message += f"\n{data.to_csv(index=False)}"
    with open('log.csv', 'a') as log_file:
        log_file.write(message + '\n')

def handle_missing_values(df):
    log_message("Handling missing values...", df)
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
    log_message("After handling missing values:", df)

def size_to_mb(size):
    log_message(f"Converting size: {size}")
    if pd.isna(size): # Se o tamanho for NaN, retornar NaN
        print("Entrou no pd.isna")
        return np.nan
    if isinstance(size, str): # Se o tamanho for uma string
        print("Entrou no isinstance")
        if 'M' in size or 'm' in size:
            return float(size.replace('M', '').replace('m', '').replace(',', '.'))
        elif 'K' in size or 'k' in size:
            return float(size.replace('K', '').replace('k', '').replace(',', '.')) / 1024
        elif 'G' in size or 'g' in size:
            return float(size.replace('G', '').replace('g', '').replace(',', '.')) * 1024

    #print("Não entrou no isinstance")
    #return np.nan
    return float(size)
def parse_android_version(version):
    log_message(f"Parsing Android version: {version}")
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

def preprocess_new_app(new_app):
    log_message("Preprocessing new app...", new_app)
    if not isinstance(new_app, pd.DataFrame):
        new_app = pd.DataFrame([new_app])

    log_message("Initial new app data:", new_app)
    handle_missing_values(new_app)

    new_app['Size'] = new_app['Size'].apply(size_to_mb)
    log_message("After converting size:", new_app)

    new_app['Minimum Android'] = new_app['Minimum Android'].apply(parse_android_version)
    new_app['Minimum Android'] = [int(x * 10) / 10 for x in new_app['Minimum Android']]
    log_message("After parsing Android version:", new_app)

    new_app['Category'] = new_app['Category'].apply(lambda x: category_mapping.get(x, -1))
    new_app['Content Rating'] = new_app['Content Rating'].apply(lambda x: content_mapping.get(x, -1))
    new_app['Category'] = new_app['Category'].replace(-1, category_mapping['Tools'])
    new_app['Content Rating'] = new_app['Content Rating'].replace(-1, content_mapping['Everyone'])
    log_message("After encoding categories:", new_app)

    return new_app

def predict_installs(new_app_data):
    new_app_processed = preprocess_new_app(new_app_data)
    if new_app_processed.isnull().values.any():
        log_message("Data after preprocessing contains NaN:", new_app_processed)
        raise ValueError("Input contains NaN after preprocessing.")

    predictions = {}
    for model_name, model in models.items():
        prediction = model.predict(new_app_processed)
        predictions[model_name] = prediction[0]
        log_message(f"Prediction with {model_name}: {prediction[0]}")

    return predictions

# Criar interface Tkinter
def on_submit():
    try:
        new_app_data = {
            'Category': category_entry.get(),
            'Rating': float(rating_entry.get()),
            'Rating Count': int(rating_count_entry.get()),
            'Free': free_var.get(),
            'Price': float(price_entry.get()),
            'Size': size_entry.get(),
            'Minimum Android': minimum_android_entry.get(),
            'Content Rating': content_rating_entry.get(),
            'Ad Supported': ad_supported_var.get(),
            'In App Purchases': in_app_purchases_var.get(),
            'Editors Choice': editors_choice_var.get()
        }

        log_message(f"New app data: {new_app_data}")

        # Verificar se as categorias e classificações de conteúdo são válidas
        if new_app_data['Category'] not in category_mapping:
            raise ValueError(f"Categoria inválida: {new_app_data['Category']}")
        if new_app_data['Content Rating'] not in content_mapping:
            raise ValueError(f"Classificação de conteúdo inválida: {new_app_data['Content Rating']}")

        new_app_df = pd.DataFrame([new_app_data])
        predictions = predict_installs(new_app_df)

        prediction_message = "\n".join([f"{model}: {int(pred)} installs" for model, pred in predictions.items()])
        messagebox.showinfo("Previsão", prediction_message)
    except ValueError as e:
        log_message(f"ValueError: {str(e)}")
        messagebox.showerror("Erro", str(e))
    except Exception as e:
        log_message(f"Exception: {str(e)}\n{traceback.format_exc()}")
        messagebox.showerror("Erro", f"Ocorreu um erro: {e}")

def create_interface():
    root = tk.Tk()
    root.title("App Information")

    labels = [
        "Category:", "Rating:", "Rating Count:", "Free:", "Price:",
        "Size:", "Minimum Android:", "Content Rating:", "Ad Supported:",
        "In App Purchases:", "Editors Choice:"
    ]

    entries = []

    for i, label in enumerate(labels):
        ttk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
        if label in ["Free:", "Ad Supported:", "In App Purchases:", "Editors Choice:"]:
            var = tk.BooleanVar()
            checkbox = ttk.Checkbutton(root, variable=var)
            checkbox.grid(row=i, column=1, padx=10, pady=5, sticky=tk.W)
            entries.append(var)
        else:
            entry = ttk.Entry(root)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky=tk.EW)
            entries.append(entry)

    global category_entry, rating_entry, rating_count_entry, free_var, price_entry
    global size_entry, minimum_android_entry, content_rating_entry, ad_supported_var
    global in_app_purchases_var, editors_choice_var

    (category_entry, rating_entry, rating_count_entry, free_var, price_entry,
     size_entry, minimum_android_entry, content_rating_entry, ad_supported_var,
     in_app_purchases_var, editors_choice_var) = entries

    submit_button = ttk.Button(root, text="Submit", command=on_submit)
    submit_button.grid(row=len(labels), columnspan=2, pady=10)

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=2)
    root.mainloop()

if __name__ == "__main__":
    create_interface()
