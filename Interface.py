import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np

def submit():
    try:
        category = entry_category.get()
        rating = float(entry_rating.get())
        rating_count = float(entry_rating_count.get())
        free = var_free.get()
        price = float(entry_price.get())
        size_mb = float(entry_size_mb.get())
        min_android = float(entry_min_android.get())
        content_rating = content_rating_var.get()
        ad_supported = var_ad_supported.get()
        in_app_purchases = var_in_app_purchases.get()
        editors_choice = var_editors_choice.get()

        data = np.array([[category, rating, rating_count, free, price, size_mb, min_android, content_rating, ad_supported, in_app_purchases, editors_choice]])
        print(data)
        messagebox.showinfo("Success", "Data has been collected successfully!")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid data for all fields.")

# Create main window
root = tk.Tk()
root.title("Data Entry Form")
root.geometry("640x480")
root.minsize(288, 162)
root.maxsize(1152, 818)

# Create labels and entry fields
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

# Create checkbuttons for boolean fields
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

# Create dropdown menu for Content Rating
content_rating_var = tk.StringVar()
content_rating_label = tk.Label(root, text="Content Rating", width=20, anchor='w')
content_rating_label.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
content_rating_menu = ttk.Combobox(root, textvariable=content_rating_var)
content_rating_menu['values'] = ('Everyone', 'Teen', 'Mature')
content_rating_menu.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
content_rating_menu.current(0)  # Set default value

# Create submit button
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack(pady=20)

# Run the application
root.mainloop()




