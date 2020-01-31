# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:58:14 2020

@author: shanr
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# loading data
train = pd.read_csv("data_mnist.csv")
test = pd.read_csv("test_mnist.csv")

# 1. Label distribution

g = sns.countplot(train.label)
# 2. missing value
train.isnull().any().value_counts()
test.isnull().any().value_counts()

#%%
def accurcy(y_predict, y_test):
    # reform y_test
    predict_list = y_predict.tolist()
    test_list = y_test.tolist()
    n = 0
    for i in range(len(y_predict)):
        if predict_list[i] == test_list[i]:
            n += 1
    acc = n / len(y_predict)
    
    return acc

#%% use k-fold cross validation to find best k value
"""
1. train:test = 4:1
2. 5 folds
3. for each fold, calculate k from 1 to 20
4. find the best k
"""
result = {}
k_fold = KFold(n_splits=5,random_state=42)

fold_number = 1
for fold in k_fold.split(train):
    train_index = fold[0]
    test_index = fold[1]
    
    train_fold = train.iloc[train_index]
    test_fold = train.iloc[test_index]
    
    x_train = train_fold.iloc[:,1:]
    y_train = train_fold.iloc[:,0]
    x_test = test_fold.iloc[:,1:]
    y_test = test_fold.iloc[:,0]

    for k in range(1,21):
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(x_train,y_train)
        test_y_predict = knn.predict(x_test)
        acc = accurcy(test_y_predict, y_test)
        
        if fold_number not in result:
            result[fold_number] = {}
            result[fold_number][k] = acc
        else:
            result[fold_number][k] = acc
    
    fold_number += 1
    print("k_fold round" + str(fold_number))
#%%Fold 1

plt.xticks(np.arange(1,21,1))
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.title("Fold 1")
plt.plot(list(result[1].keys()), list(result[1].values()))
plt.show()
#%%Fold 2

plt.xticks(np.arange(1,21,1))
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.title("Fold 2")
plt.plot(list(result[2].keys()), list(result[2].values()))
plt.show()
#%%Fold 3

plt.xticks(np.arange(1,21,1))
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.title("Fold 3")
plt.plot(list(result[3].keys()), list(result[3].values()))
plt.show()
#%%Fold 4

plt.xticks(np.arange(1,21,1))
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.title("Fold 4")
plt.plot(list(result[4].keys()), list(result[4].values()))
plt.show()
#%%Fold 5

plt.xticks(np.arange(1,21,1))
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.title("Fold 5")
plt.plot(list(result[5].keys()), list(result[5].values()))
plt.show()
#%%
highest_k = {}
for r in result:
    print("Fold", r)
    highest_k[r] = (0,0)
    for k in result[r]:
        print("k=%d, acc=%f" %(k, result[r][k]))
        if result[r][k] > highest_k[r][1]:
            highest_k[r] = (k, result[r][k])
        
"""
result shows that when k = 3, accurcy has a highest value
"""
#%% train with whole data

X = train.iloc[:,1:]
Y = train.iloc[:,0]
X = X / 255

model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
model.fit(X,Y)
#%%
predict_test = model.predict(test)
#%%
result_1 = pd.DataFrame()
result_1["Label"] = predict_test
result_1["ImageId"] = list(range(1,10001))
result_1.to_csv("Ruocheng_Shan_hw1_submission.csv", index=False)












