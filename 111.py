import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

def gaussian_kernal(u):
    return (1/(math.sqrt(2*math.pi)))*math.e**(-0.5*(u[0,0]**2+u[0,1]**2+u[0,2]**2+u[0,3]**2))

def cal_pcb(cv,train,cal_pcb_h):
    sum=0
    for i in range(len(train)):
        sum+=(1/h**4)*gaussian_kernal((cv-train[i])/cal_pcb_h)
    p_cp=(1/len(train))*sum
    return p_cp

def def_h(data1,data2,def_h_h):
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
    pw1 = len(data_train_1) / len(data1)
    pw2 = len(data_train_2) / len(data1)
    pw3 = len(data_train_3) / len(data1)
    right = 0
    for j in range(len(data2)):
        p_list = []
        pcp_label1 = cal_pcb(data2[j], data_train_1, def_h_h)
        pcp_label2 = cal_pcb(data2[j], data_train_2, def_h_h)
        pcp_label3 = cal_pcb(data2[j], data_train_3, def_h_h)
        p_list.append(pw1 * pcp_label1)
        p_list.append(pw2 * pcp_label2)
        p_list.append(pw3 * pcp_label3)
        if p_list.index(max(p_list)) == data2[j, 4]:
            right = right + 1
    return right

if __name__=='__main__':
    iris_data = pd.read_csv("iris.data", header=None)
    labels_codes = pd.Categorical(iris_data[4]).codes
    # print(iris_data)
    for i in range(150):
        iris_data.loc[i, 4] = labels_codes[i]
    datalist = iris_data.values.tolist()
    # print(datalist)
    random.seed(17)
    random.shuffle(datalist)
    # print(datalist)
    date_set = np.mat(datalist)
    # print(dateset)

    data_set = np.vsplit(date_set, 5)

    sub_set1 = data_set[0].copy()
    sub_set2 = data_set[1].copy()
    sub_set3 = data_set[2].copy()
    sub_set4 = data_set[3].copy()
    sub_set5 = data_set[4].copy()

    h = 1
    step=0.05
    binary=90
    average_acc=[]
    h_list_cv=[]
    while(h>0):
        cv_acc = []
        for i in range(5):
            if i == 0:
                data = np.vstack((sub_set1, sub_set2, sub_set3, sub_set4))
            elif i == 1:
                data = np.vstack((sub_set1, sub_set2, sub_set3, sub_set5))
            elif i == 2:
                data = np.vstack((sub_set1, sub_set2, sub_set4, sub_set5))
            elif i == 3:
                data = np.vstack((sub_set1, sub_set3, sub_set4, sub_set5))
            else:
                data = np.vstack((sub_set2, sub_set3, sub_set4, sub_set5))
            data_train = data[0:binary]
            data_cv = data[binary:120]
            cv_acc.append(def_h(data_train,data_cv,h) / len(data_cv))
        cv_acc=np.mat(cv_acc)
        average_acc.append(np.mean(cv_acc))
        h_list_cv.append(h)
        h=h-step
    print(average_acc)
    print(h_list_cv)
    plt.plot(h_list_cv,average_acc)
    plt.xlabel("hyperparameter h")
    plt.ylabel("average accuracy")
    plt.title("Parzen window")
    plt.show()
    h_list=[]
    for x in range(len(average_acc)):
        if average_acc[x]==max(average_acc):
            h_list.append(h_list_cv[x])
    print(h_list)

    average_test_acc=[]
    for i in range(len(h_list)):
        test_acc=[]
        for j in range(5):
            test_data=data_set[4-j].copy()
            if j == 0:
                cal_data = np.vstack((sub_set1, sub_set2, sub_set3, sub_set4))
            elif j == 1:
                cal_data = np.vstack((sub_set1, sub_set2, sub_set3, sub_set5))
            elif j == 2:
                cal_data = np.vstack((sub_set1, sub_set2, sub_set4, sub_set5))
            elif j == 3:
                cal_data = np.vstack((sub_set1, sub_set3, sub_set4, sub_set5))
            else:
                cal_data = np.vstack((sub_set2, sub_set3, sub_set4, sub_set5))
            test_acc.append(def_h(cal_data,test_data,h_list[i])/len(test_data))
        test_acc=np.mat(test_acc)
        print(test_acc)
        average_test_acc.append(np.mean(test_acc))
    print(average_test_acc)