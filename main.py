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

t0 = time.time()
print(t0)
variables_list = [x for x in dfGPlayStore.columns]
print(variables_list)
t1 = time.time()
print(t1-t0)

print("\n---------------------------------\n")

t2 = time.time()
print(t2)
variables_list2 = dfGPlayStore.columns.tolist()
print(variables_list2)
t3 = time.time()
print(t3-t2)

print("\n")
print(t3-t0)