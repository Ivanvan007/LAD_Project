import os
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import pandas as pd
import pickle
import sklearn



# Carregar o modelo treinado (substitua 'models['Linear Regression']' pelo modelo desejado)
#import joblib

with open('knn.pkl', 'rb') as f:
    model = pickle.load(f)

# Mapeamentos de LabelEncoders usados durante o treinamento
category_mapping = {'Action': 0, 'Adventure': 1, 'Arcade': 2, 'Art & Design': 3,
                    'Auto & Vehicles': 4, 'Beauty': 5, 'Board': 6, 'Books & Reference': 7,
                    'Business': 8, 'Card': 9, 'Casino': 10, 'Casual': 11, 'Comics': 12,
                    'Communication': 13, 'Dating': 14, 'Education': 15, 'Educational': 16,
                    'Entertainment': 17, 'Events': 18, 'Finance': 19, 'Food & Drink': 20,
                    'Health & Fitness': 21, 'House & Home': 22, 'Libraries & Demo': 23,
                    'Lifestyle': 24, 'Maps & Navigation': 25, 'Medical': 26, 'Music': 27,
                    'Music & Audio': 28, 'News & Magazines': 29, 'Parenting': 30,
                    'Personalization': 31, 'Photography': 32, 'Productivity': 33,
                    'Puzzle': 34, 'Racing': 35, 'Role Playing': 36, 'Shopping': 37,
                    'Simulation': 38, 'Social': 39, 'Sports': 40, 'Strategy': 41, 'Tools': 42,
                    'Travel & Local': 43, 'Trivia': 44, 'Video Players & Editors': 45,
                    'Weather': 46, 'Word': 47}

content_mapping = {'Everyone': 0, 'Everyone 10+': 1, 'Mature 17+': 2, 'Teen': 3}  # Atualize com seu mapeamento real


# Função para converter o tamanho para MB
def size_to_mb(size):
    try:
        if 'M' in size:
            return float(size.replace('M', ''))
        elif 'K' in size:
            return float(size.replace('K', '')) / 1024
        elif 'G' in size:
            return float(size.replace('G', '')) * 1024
    except ValueError:
        return np.nan
    return size


# Função para processar a nova entrada
def preprocess_new_app(new_app):
    new_app_df = pd.DataFrame([new_app],
                              columns=['Category', 'Rating', 'Rating Count', 'Free', 'Price', 'Size', 'Minimum Android',
                                       'Content Rating', 'Ad Supported', 'In App Purchases', 'Editors Choice'])
    new_app_df['Size'] = new_app_df['Size'].apply(size_to_mb)
    new_app_df['Category'] = new_app_df['Category'].apply(lambda x: category_mapping.get(x, -1))
    new_app_df['Content Rating'] = new_app_df['Content Rating'].apply(lambda x: content_mapping.get(x, -1))
    return new_app_df


# Função para submeter os dados e fazer a previsão
def submit():
    try:
        category = entry_category.get()
        rating = float(entry_rating.get())
        rating_count = int(entry_rating_count.get())
        free = var_free.get()
        price = float(entry_price.get())
        size_mb = float(entry_size_mb.get())
        min_android = float(entry_min_android.get())
        content_rating = content_rating_var.get()
        ad_supported = var_ad_supported.get()
        in_app_purchases = var_in_app_purchases.get()
        editors_choice = var_editors_choice.get()

        new_app_data = [category, rating, rating_count, free, price, size_mb, min_android, content_rating, ad_supported,
                        in_app_purchases, editors_choice]
        new_app_processed = preprocess_new_app(new_app_data)

        # Fazer a previsão com o modelo treinado
        prediction = model.predict(new_app_processed)
        installs_prediction = prediction[0]

        messagebox.showinfo("Prediction", f"Predicted number of installs: {installs_prediction:.0f}")
    except ValueError as e:
        messagebox.showerror("Input Error", f"Please enter valid data for all fields. {str(e)}")


# Criar a janela principal
root = tk.Tk()
root.title("Data Entry Form")
root.geometry("640x480")
root.minsize(288, 162)
root.maxsize(1152, 818)

# Criar labels e entry fields
fields = ['Category', 'Rating', 'Rating Count', 'Price', 'Size_MB', 'Minimum Android']
entries = {}

for field in fields:
    row = tk.Frame(root)
    label = tk.Label(row, width=20, text=field, anchor='w')
    entry = tk.Entry(row)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label.pack(side=tk.LEFT)
    entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries[field] = entry

entry_category = entries['Category']
entry_rating = entries['Rating']
entry_rating_count = entries['Rating Count']
entry_price = entries['Price']
entry_size_mb = entries['Size_MB']
entry_min_android = entries['Minimum Android']

# Criar checkbuttons para campos booleanos
boolean_fields = ['Free', 'Ad Supported', 'In App Purchases', 'Editors Choice']
variables = {}

for field in boolean_fields:
    var = tk.IntVar()
    chk = tk.Checkbutton(root, text=field, variable=var)
    chk.pack(anchor='w')
    variables[field] = var

var_free = variables['Free']
var_ad_supported = variables['Ad Supported']
var_in_app_purchases = variables['In App Purchases']
var_editors_choice = variables['Editors Choice']

# Criar menu dropdown para Content Rating
content_rating_var = tk.StringVar()
content_rating_label = tk.Label(root, text="Content Rating", width=20, anchor='w')
content_rating_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
content_rating_menu = ttk.Combobox(root, textvariable=content_rating_var)
content_rating_menu['values'] = list(content_mapping.keys())
content_rating_menu.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
content_rating_menu.current(0)  # Definir valor padrão

# Criar botão de submit
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack(pady=20)

# Executar a aplicação
root.mainloop()
