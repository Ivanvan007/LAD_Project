import tkinter as tk
from tkinter import messagebox
import numpy as np


def submit():
    try:
        rating = float(entry_rating.get())
        rating_count = int(entry_rating_count.get())
        free = var_free.get()
        price = float(entry_price.get())
        ad_supported = var_ad_supported.get()
        in_app_purchases = var_in_app_purchases.get()
        editors_choice = var_editors_choice.get()
        size_mb = float(entry_size_mb.get())

        data = np.array([[rating, rating_count, free, price, ad_supported, in_app_purchases, editors_choice, size_mb]])
        print(data)
        messagebox.showinfo("Success", "Data has been collected successfully!")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid data for all fields.")


# Create main window
root = tk.Tk()
root.title("Data Entry Form")

root.geometry("640x480")  # width: 400 pixels, height: 300 pixels

# Set the minimum and maximum size of the window
root.minsize(288, 162)  # minimum size: width 200, height 150
root.maxsize(1152, 818)  # maximum size: width 800, height 600

# Create labels and entry fields
fields = ['Rating', 'Rating Count', 'Price', 'Size_MB']
entries = {}

for field in fields:
    row = tk.Frame(root)
    label = tk.Label(row, width=20, text=field, anchor='w')
    entry = tk.Entry(row)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label.pack(side=tk.LEFT)
    entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries[field] = entry

entry_rating = entries['Rating']
entry_rating_count = entries['Rating Count']
entry_price = entries['Price']
entry_size_mb = entries['Size_MB']

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

# Create submit button
submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack(pady=20)

# Run the application
root.mainloop()




