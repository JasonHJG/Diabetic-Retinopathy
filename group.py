#/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:45:56 2017

@author: jingang
"""
"""
to move files into corresponding dir

dir name start with 0/1/2/3/4

from each row of the dataframe, find the 

"""


import pandas as pd
import os
import shutil

# create sub folders
# group_data/0
src=os.path.dirname(os.path.abspath(__file__))

if os.path.exists(src+'/Joseph_data'):
    shutil.rmtree(src+'/Joseph_data')
if not os.path.exists(src+'/Joseph_data'):        
    os.mkdir(src+'/Joseph_data')
   
path=src+'/DR/Joseph_data'

for i in range(5):
    os.mkdir(path+'/'+str(i))



training_dir='/mnt/dfs/jason/DR_data/DR/all_800/trainLabels.csv'
#testing_dir='/mnt/dfs/jason/DR_data/DR/all_800/testLabels.csv'
#train_val_dir='/mnt/dfs/jason/DR_data/DR/all_800/valLabels.csv'
image_dir='/mnt/dfs/jason/DR_data/DR/all_800/all'







training_set = pd.read_csv(training_dir)
#testing_set = pd.read_csv(testing_dir)
#train_val_set = pd.read_csv(train_val_dir)



train_length=training_set["label"].count()
#test_length=testing_set["label"].count()
#train_val_length=train_val_set["label"].count()
"""
for i in range(train_val_length):
    label = train_val_set["label"][i]
    name = train_val_set["name"][i]
    filefrom=image_dir+'/'+name+'.jpg'
    fileto=path+'/'+str(label)
    print filefrom
    print fileto
    try:
	shutil.copy2(filefrom,fileto)
    except:
        print "cannot find the corresponding picture"
"""

for i in range(train_length):
    label = training_set["label"][i]
    name = training_set["name"][i]
    filefrom=image_dir+'/'+name+'.jpg'
    fileto=path+'/'+str(label)
    print filefrom
    print fileto
    try:
        shutil.copy2(filefrom,fileto)
    except:
        print "cannot find the correspoding picture"
     

        
"""
for i in range(test_length):
    label = testing_set["label"][i]
    name = testing_set["name"][i]
    filefrom=image_dir+'/'+name+'.jpg'
    fileto=path+'/'+str(label)
    print filefrom
    print fileto
    try:
        shutil.copy2(filefrom,fileto)
    except:
        print "cannot find the correspoding picture"
  
"""
