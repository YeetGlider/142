import pandas as pd 
import numpy as np
import csv

df = pd.read_csv('final.csv')
df = df[df['soup'].notna()]

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])