#%%

# read json from the result folder
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# read json file 
def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


path = '/Users/alessiogandelli/dev/internship/BERTopic_evaluation/myresults/Basic/Climate/'

# read all the json files in the folder
json_files = [read_json(path+pos_json) for pos_json in os.listdir(path) if pos_json.endswith('.json')]


#%%

for model in json_files:
    for version in model:
        print(version['Params'])




# %%
