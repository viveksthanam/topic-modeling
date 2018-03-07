
# coding: utf-8

# In[2]:


import glob
import os

file_list = glob.glob(os.path.join(os.getcwd(), "C:\datasets", "ABronte_Agnes.txt"))

corpus = []

for file_path in file_list:
    with open(file_path) as f_input:
        corpus.append(f_input.read())

print (corpus)  

