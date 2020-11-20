import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

#calculate distance function
def cal_distance(test,cal):
    distance_list=[]#list of distance
    for i in range(len(cal)):
        distance=[]
        distance.append(cal[i, 4])#append corresponding label
        d=test-cal[i,:]#test sample value-training sample value
        #calculate distance
        distance.append(math.sqrt(d[0,0]**2+d[0,1]**2+d[0,2]**2+d[0,3]**2))
        distance_list.append(distance)
    #print(distance_list)
    return distance_list

if __name__=='__main__':
    iris_data=pd.read_csv("iris.data",header=None)
    labels_codes=pd.Categorical(iris_data[4]).codes
    for i in range(150):
        iris_data.loc[i,4]=labels_codes[i]
    datalist=iris_data.values.tolist()
    random.seed(17)
    random.shuffle(datalist)
    data_set=np.mat(datalist)
    data_set=np.vsplit(data_set,5)

    sub_set1=data_set[0].copy()
    sub_set2=data_set[1].copy()
    sub_set3=data_set[2].copy()
    sub_set4=data_set[3].copy()
    sub_set5=data_set[4].copy()

    average_acc=[]
    k_list=[]
    for K in range(1,120):
        if K%3!=0:
            k_list.append(K)
            accuracy=[]
            for i in range(5):
                test_data = data_set[4 - i].copy()
                if i == 0:
                    cal_data = np.vstack((sub_set1, sub_set2, sub_set3, sub_set4))
                elif i == 1:
                    cal_data = np.vstack((sub_set1, sub_set2, sub_set3, sub_set5))
                elif i == 2:
                    cal_data = np.vstack((sub_set1, sub_set2, sub_set4, sub_set5))
                elif i == 3:
                    cal_data = np.vstack((sub_set1, sub_set3, sub_set4, sub_set5))
                else:
                    cal_data = np.vstack((sub_set2, sub_set3, sub_set4, sub_set5))
                right = 0
                for x in range(len(test_data)):
                    d_set = np.mat(cal_distance(test_data[x], cal_data))#list of distance
                    d_set = (d_set[np.lexsort(d_set.T)])[0, :, :]#sort distance list
                    p_wk = [0, 0, 0]#be used to record P(wk|x)
                    #calculate P(wk|x) for each test sample
                    for y in range(K):
                        if d_set[y, 0] == 0:
                            p_wk[0] = p_wk[0] + 1 / K
                        elif d_set[y, 0] == 1:
                            p_wk[1] = p_wk[1] + 1 / K
                        else:
                            p_wk[2] = p_wk[2] + 1 / K
                    #print(p_wk)
                    #calculate accuracy
                    if p_wk.index(max(p_wk)) == test_data[x, 4]:
                        right = right + 1
                accuracy.append(right / len(test_data))
            accuracy=np.mat(accuracy)
            print(accuracy)
            average_acc.append(np.mean(accuracy))
    plt.scatter(k_list,average_acc)
    plt.title('KNN')
    plt.xlabel('hyperparameter K')
    plt.ylabel('average accuracy')
    plt.show()
    k_max_list=[]
    for z in range(len(average_acc)):
        if average_acc[z]==max(average_acc):
            k_max_list.append(k_list[z])
    print("highest average accuracy:",round(max(average_acc),3))
    print("corresponding K:",k_max_list)
