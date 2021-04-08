import pickle
import numpy as np
import pandas as pd

boost = False
f = open("config.txt","r")
config = []
for x in f:
    config.append(x.split('"')[1])
ip_file = config[0]
model = config[1]
parameters = config[2]
data_format = config[3]
if config[4] == "1":
    boost = True
with open(ip_file, 'rb') as f:
    data_set = pickle.load(f)

print("hello")