import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import nibabel as nib
import glob

from collections import Counter

from imblearn.over_sampling import SMOTE 
from sklearn.datasets import make_classification

import copy
# Python 3.8

# The number of 0 class is 0.
# def fn(x):
#     if 33<=x<77 :
#         return 0
#     elif 77<=x < 121:
#         return 1
#     elif 121<=x < 165:
#         return 2


# SSEIT is higher than 121 or not.
def fn(x):
    if x< 121:
        return 0
    elif 121<=x :
        return 1
    
# def fn(x):
#     if x >99 :
#         return 1
#     else :
#         return 0
# Part1 data processing
# First experiment: just use resized gray scale image as dataset
# Second experiment: use Canny edge dectection image as dataset

# First experiment: just use resized gray scale image as dataset

# dir = "Dataset"
dir = r"C:\Users\User\Desktop\python_work\AffCompResearch\ClassifierAffectiveComputing\Resnet101Finetuned\Resnet101Finetuned"


categories = ["\Resnet101Finetuned"]
# categories = ["fouled_scaled", "clean"]
# categories = [sys.argv[1], sys.argv[2]]



data =[]

path=dir

all_files = glob. glob(path + "/*.gz")
all_files.sort()
# all_files=set(all_files)

# except_files=set(except_files)

# all_files.difference(except_files)

# data_img=[]

except_Ss=["S06",  "S26"]
data_label=[]
df = pd.DataFrame()
i=0
for file in all_files:

    if "affint" in file: 
        if  "S06" in file or "S13" in file or "S24" in file or "S26" in file or "S29" in file:
            continue
        
        # all_files=list(set(all_files)-set(except_files))
        filename = os.path.join("", file)
        print(filename)

        # all_files.remove(except_files[0])
        # filename.remove(except_files[1])
        # filename.remove(except_files[2])
        img=nib.load(filename)
        # print(np.array(img.dataobj))
        # df.append(pd.DataFrame(np.array(img.dataobj).flatten()))

        # df = pd.concat([df, pd.DataFrame(np.array(img.dataobj).flatten()[:14]).T],axis=0, ignore_index=True)
        # i+=1


        image = np.array(img.dataobj).flatten()
        label=0
        data.append(image)



dir=r"C:\Users\User\Desktop\python_work\AffCompResearch\ClassifierAffectiveComputing"

data_label_cost=pd.read_excel(dir+'/BehavioralDataForSearchlights.xlsx', index_col=0)
# data_label_cost["demeaned_SSEIT"]
# data_label_cost["SSEIT total"]

data_label_cost['SSEIT total']= data_label_cost['SSEIT total'].map(fn)

# "S06" in file or "S13" in file or "S24" in file or "S26" in file or "S29" in file:
# data_label_cost.iloc[1]
data_label_cost=data_label_cost.drop(data_label_cost.index[[22]])
# labels=data_label_cost['SSEIT total']
labels=data_label_cost.pop('SSEIT total')

labels=list(labels)

# for i in range(len(labels)):
#     data[i][1]=labels[i]



counter = Counter(labels)
print(counter)


# Select one test subject

data_ori=copy.deepcopy(data)
data_label_ori=copy.deepcopy(labels)

tp,tn,fp,fn=0,0,0,0

prediction_list=[]
ytest_list=[]
for i in range(len(labels)):
    xtest=data[i]
    ytest=labels[i]

    data.pop(i)
    # data.remove([data[i]])
    labels.pop(i)
    # labels.remove(labels[i])
    
    weight = len(labels) / (2 * np.bincount(labels))
    # class_weight = {0: weight[0], 1: weight[1]}
    # class_weight = {0: 0.6, 1: 0.4}
    class_weight = {0: 0.5, 1: 0.5}

    import random
    n = random.randint(0,100)
    sm = SMOTE(random_state=n)
    X_res, y_res = sm.fit_resample(np.array(data).reshape(36,902629), np.array(labels))

    xtest = np.array(xtest).reshape(1,902629)

    # for i in range(len(labels)):
    #     X_res[i][1]=y_res[i]

    from sklearn import preprocessing
    X_res = preprocessing.normalize(X_res)
    xtest = preprocessing.normalize(xtest)

    # xtest_one=xtest
    # ytest_one=ytest
    # print("check")
    
    # print(X_res)

    # print("Hello")
    # pick_in = open('data1.pickle','wb')
    # pickle.dump(data,pick_in)
    # pick_in.close()








    # X_res, xtest_giveup, y_res, ytest_giveup = train_test_split(X_res, y_res, test_size = 0.3)

    # Part2 
    # resized image dataset
    # pick_in = open('data1.pickle','rb')


    # data = pickle.load(pick_in)
    # pick_in.close()

    # random.shuffle(data)
    # features = []
    # labels = []


    # for feature, label in data:
    #     features.append(feature)
    #     labels.append(label)


    # X_res, xtest, y_res, ytest = train_test_split(X_res, y_res, test_size = 0.5, random_state=n)


    
    import random
    n = random.randint(0,10000)

    print(n)
    model = SVC(random_state=n)
    # model = SVC(C=0.01,kernel="poly",gamma="auto",class_weight=class_weight)
    # model = SVC(C=1,kernel="rbf",gamma="auto",class_weight=class_weight)
    # model = SVC(C=1,kernel="linear",gamma="auto",class_weight=class_weight)
    # model = SVC(C=1,kernel="sigmoid",gamma="auto",class_weight=class_weight)
    # model = SVC(C=1,kernel="precomputed",gamma="auto",class_weight=class_weight)

    model.fit(X_res, y_res)

    # pick = open('SVMmodel.sav','wb')
    # pick = open('SVMmodelwithEdgedata.sav','wb')
    # pickle.dump(model,pick)
    # pick.close()

    prediction=model.predict(xtest)

    
    prediction_list.append(prediction[0])
    ytest_list.append(ytest)

    # prediction_list=prediction
    # ytest_list=ytest

    # print(sum(xtest[0]))

    if prediction[0]==1 and ytest==1:
        tp+=1
    elif prediction[0]==1 and ytest==0:
        fp+=1
    elif prediction[0]==0 and ytest==1:
        fn+=1
    elif prediction[0]==0 and ytest==0:
        tn+=1

    # data_ori=copy.deepcopy(data)
    data=copy.deepcopy(data_ori)
    # data_label_ori=copy.deepcopy(labels)
    labels=copy.deepcopy(data_label_ori)


# accuracy = model.score(xtest,ytest)

# categories = ["clean", "fouled_scaled"]


# print("Accuracy:",accuracy)

f1score=f1_score(ytest_list, prediction_list, average='macro')
print("F1 score : ",f1score)

print(confusion_matrix(ytest_list, prediction_list))
# 
tn, fp, fn, tp =confusion_matrix(ytest_list, prediction_list).ravel()

Sensitivity = tp / (tp + fn)
Specificity = tn/ (tn + fp)
Precision = tp / (tp + fp)
FNR = fn/( fn+tp)
FPR = fp/(fp+tp)

F1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)

print("Sensitivity", Sensitivity)
print("Specificity", Specificity)
print("Precision", Precision)
print("False-negative rate", FNR)
print("False discovery rate", FPR)



# print("Prediction is : ",categories[prediction[0]])

# mypet=xtest[0].reshape(50,50)

# plt.imshow(mypet,cmap='gray')
# plt.show()