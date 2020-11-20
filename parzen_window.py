import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

#Gaussian kernal function
def gaussian_kernal(u):
    return (1/(math.sqrt(2*math.pi)))*math.e**(-0.5*(u[0,0]**2+u[0,1]**2+u[0,2]**2+u[0,3]**2))

#calculate conditional probability
def cal_pcb(cv,train,cal_pcb_h):
    sum=0
    for i in range(len(train)):
        sum+=(1/h**4)*gaussian_kernal((cv-train[i])/cal_pcb_h)
    p_cp=(1/len(train))*sum
    #print(p_cp)
    return p_cp

def def_h(data1,data2,def_h_h):
    #separate the training dataset into 3 groups by their labels
    data_train_1 = []
    data_train_2 = []
    data_train_3 = []
    for x in range(len(data1)):
        if data1[x, 4] == 0:
            data_train_1.append(data1[x])
        elif data1[x, 4] == 1:
            data_train_2.append(data1[x])
        else:
            data_train_3.append(data1[x])
    #print(data_train_1,data_train_2,data_train_3)
    # calculate prior class probability
    pw1 = len(data_train_1) / len(data1)#P(wk)1
    pw2 = len(data_train_2) / len(data1)#P(wk)2
    pw3 = len(data_train_3) / len(data1)#P(wk)3
    #print(pw1,pw2,pw3)
    right = 0
    for j in range(len(data2)):
        p_list = []
        pcp_label1 = cal_pcb(data2[j], data_train_1, def_h_h)#P(x|wk)1
        pcp_label2 = cal_pcb(data2[j], data_train_2, def_h_h)#P(x|wk)2
        pcp_label3 = cal_pcb(data2[j], data_train_3, def_h_h)#P(x|wk)3
        p_list.append(pw1 * pcp_label1)#P(wk|x)1
        p_list.append(pw2 * pcp_label2)#P(wk|x)2
        p_list.append(pw3 * pcp_label3)#P(wk|x)3
        #record the times of right prediction
        if p_list.index(max(p_list)) == data2[j, 4]:
            right = right + 1
        #print(p_list)
    #print(right)
    return right

if __name__=='__main__':
	#read data
    iris_data = pd.read_csv("iris.data", header=None)
    labels_codes = pd.Categorical(iris_data[4]).codes
    #print(iris_data)
    for i in range(150):
        iris_data.loc[i, 4] = labels_codes[i]
    datalist = iris_data.values.tolist()
    #print(datalist)
    #shuffle data
    random.seed(17)
    random.shuffle(datalist)
    #print(datalist)
    date_set = np.mat(datalist)
    # print(dateset)

    #split data
    data_set = np.vsplit(date_set, 5)
    #print("number of subset:",len(data_set))
    #print(data_set)
    sub_set1 = data_set[0].copy()
    sub_set2 = data_set[1].copy()
    sub_set3 = data_set[2].copy()
    sub_set4 = data_set[3].copy()
    sub_set5 = data_set[4].copy()

    h = 1
    step=0.05
    average_acc=[]
    h_list_cv=[]
    while(h>0):
        cv_acc = []
        for i in range(5):
            data_cv = data_set[4 - i].copy()
            if i == 0:
                data_train = np.vstack((sub_set1, sub_set2, sub_set3, sub_set4))
            elif i == 1:
                data_train = np.vstack((sub_set1, sub_set2, sub_set3, sub_set5))
            elif i == 2:
                data_train = np.vstack((sub_set1, sub_set2, sub_set4, sub_set5))
            elif i == 3:
                data_train = np.vstack((sub_set1, sub_set3, sub_set4, sub_set5))
            else:
                data_train = np.vstack((sub_set2, sub_set3, sub_set4, sub_set5))
            cv_acc.append(def_h(data_train,data_cv,h) / len(data_cv))
        cv_acc=np.mat(cv_acc)
        average_acc.append(np.mean(cv_acc))
        h_list_cv.append(h)
        h=h-step
    #print(average_acc)
    #print(h_list_cv)
    plt.plot(h_list_cv,average_acc)
    plt.xlabel("hyperparameter h")
    plt.ylabel("average accuracy")
    plt.title("Parzen window")
    plt.show()
    h_list=[]
    for x in range(len(average_acc)):
        if average_acc[x]==max(average_acc):
            h_list.append(h_list_cv[x])
    print("highest average accuracy:",round(max(average_acc),3))
    for z in range(len(h_list)):
        h_list[z]=round(h_list[z],2)
    print("corresponding h:",h_list)
