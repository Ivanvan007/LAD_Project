import os
import pandas as pd
import time
import numpy as np
import matplotlib as plt
import seaborn as sns

file_name = "Google-Playstore.csv"
file_path = "Data/"
full_path = os.path.join(file_path, file_name)
#full_path = "Data/Google-Playstore.csv"
print("-----------Start CSV Reading----------")
dfGPlayStore = pd.read_csv(full_path)
print("----------CSV Reading Finished---------\n")

print("\n")

variables_list = dfGPlayStore.columns.tolist()
print(variables_list)