import numpy as np
import pandas as pd
import random
import xlrd
import xlwt
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,precision_score,accuracy_score,recall_score,f1_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import  ExtraTreesClassifier,GradientBoostingClassifier##极限树
from sklearn.feature_selection import  SelectFromModel


def LDA_dimensionality(X, y, k):
    '''
    X为数据集，y为label，k为目标维数
    '''
    label_ = list(set(y))

    X_classify = {}

    for label in label_:
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == label])
        X_classify[label] = X1

    mju = np.mean(X, axis=0)
    mju_classify = {}

    for label in label_:
        mju1 = np.mean(X_classify[label], axis=0)
        mju_classify[label] = mju1

    #St = np.dot((X - mju).T, X - mju)

    Sw = np.zeros((len(mju), len(mju)))  # 计算类内散度矩阵
    for i in label_:
        Sw += np.dot((X_classify[i] - mju_classify[i]).T,
                     X_classify[i] - mju_classify[i])

    # Sb=St-Sw

    Sb = np.zeros((len(mju), len(mju)))  # 计算类内散度矩阵
    for i in label_:
        Sb += len(X_classify[i]) * np.dot((mju_classify[i] - mju).reshape(
            (len(mju), 1)), (mju_classify[i] - mju).reshape((1, len(mju))))

    eig_vals, eig_vecs = np.linalg.eig(
        np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵

    sorted_indices = np.argsort(eig_vals)
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]  # 提取前k个特征向量
    return topk_eig_vecs

data_train=[]
data_test=[]
LDA_score = []
QDA_score = []
LDA_label = []
LDA_label_5 = []
acc = []
LDA_acc = []
if '__main__' == __name__:

    data_1_4 = pd.read_excel('..\Features\PCA_Skin.xlsx',header=None).values
    data_4_5 =  pd.read_excel('..\Features\PCA_Flesh.xlsx',header=None).values
    index_hxf = [x for x in range(0,59)]
    random.shuffle(index_hxf)
    index_hxf = np.array(index_hxf)

    index_mnt = [x for x in range(59, 119)]
    random.shuffle(index_mnt)
    index_mnt = np.array(index_mnt)

    index_mng = [x for x in range(119,179)]
    random.shuffle(index_mng)
    index_mng = np.array(index_mng)

    hxf_shuffle_1_4 = data_1_4[index_hxf,:]
    hxf_train_1_4 = hxf_shuffle_1_4[0:41,0:5]
    hxf_test_1_4 = hxf_shuffle_1_4[41:,0:5]

    hxf_shuffle_4_5 = data_4_5[index_hxf,:]
    hxf_train_4_5 = hxf_shuffle_4_5[0:41,0:5]
    hxf_test_4_5 = hxf_shuffle_4_5[41:,0:5]

    hxf_train = np.hstack((hxf_train_1_4,hxf_train_4_5[:,1:]))
    hxf_test = np.hstack((hxf_test_1_4,hxf_test_4_5[:,1:]))

    ####马奶提的数据随机划分
    mnt_shuffle_1_4 = data_1_4[index_mnt, :]
    mnt_train_1_4 = mnt_shuffle_1_4[0:42, 0:5]
    mnt_test_1_4 = mnt_shuffle_1_4[42:, 0:5]

    mnt_shuffle_4_5 = data_4_5[index_mnt, :]
    mnt_train_4_5 = mnt_shuffle_4_5[0:42, 0:5]
    mnt_test_4_5 = mnt_shuffle_4_5[42:, 0:5]

    mnt_train = np.hstack((mnt_train_1_4, mnt_train_4_5[:, 1:]))
    mnt_test = np.hstack((mnt_test_1_4, mnt_test_4_5[:, 1:]))

    ####木纳格的数据随机划分
    mng_shuffle_1_4 = data_1_4[index_mng, :]
    mng_train_1_4 = mng_shuffle_1_4[0:42, 0:5]
    mng_test_1_4 = mng_shuffle_1_4[42:, 0:5]

    mng_shuffle_4_5 = data_4_5[index_mng, :]
    mng_train_4_5 = mng_shuffle_4_5[0:42, 0:5]
    mng_test_4_5 = mng_shuffle_4_5[42:, 0:5]

    mng_train = np.hstack((mng_train_1_4, mng_train_4_5[:, 1:]))
    mng_test = np.hstack((mng_test_1_4, mng_test_4_5[:, 1:]))

    train = np.vstack((hxf_train,mnt_train,mng_train))
    test = np.vstack((hxf_test,mnt_test,mng_test))
    data_train = train[:,1:]
    y_train = train[:,0]

    data_test = test[:,1:]
    y_test = test[:,0]

    for i in range(1,data_train.shape[1]+1):
        x_train = data_train[:,0:i]
        x_test = data_test[:,0:i]

        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        label = lda.predict(x_test)
        score = lda.predict_proba(x_test)
        LDA_score.append(accuracy_score(y_test,label))
        LDA_label.append(label)

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(x_train, y_train)
        qda_label = qda.predict(x_test)

        QDA_score.append(accuracy_score(y_test, qda_label))
    print('lda_score:',LDA_score)

    LDA_score_1 = pd.DataFrame(LDA_score)
    LDA_score_1.to_excel('./LDA_skin_fusion_accurcy.xlsx', index=False, header=False)

    LDA_label_1 = pd.DataFrame(LDA_label)
    LDA_label_1.to_excel('./LDA_skin_fusion_label.xlsx', index=False, header=False)

    data_ave = pd.read_excel('..\Features\PCA_ave_skin_flesh.xlsx',header=None).values
    hxf_shuffle_ave = data_ave[index_hxf,:]
    hxf_train_ave= hxf_shuffle_ave[0:41,0:5]
    hxf_test_ave = hxf_shuffle_ave[41:,0:5]


    hxf_train = hxf_train_ave
    hxf_test = hxf_test_ave

    ####马奶提的数据随机划分
    mnt_shuffle_ave = data_ave[index_mnt, :]
    mnt_train_ave= mnt_shuffle_ave[0:42, 0:5]
    mnt_test_ave = mnt_shuffle_ave[42:, 0:5]


    mnt_train = mnt_train_ave
    mnt_test = mnt_test_ave

    ####木纳格的数据随机划分
    mng_shuffle_ave= data_ave[index_mng, :]
    mng_train_ave = mng_shuffle_ave[0:42, 0:5]
    mng_test_ave = mng_shuffle_ave[42:, 0:5]

    # mng_shuffle_4_5 = data_4_5[index_mng, :]
    # mng_train_4_5 = mng_shuffle_4_5[0:42, 0:4]
    # mng_test_4_5 = mng_shuffle_4_5[42:, 0:4]

    mng_train = mng_train_ave
    mng_test = mng_test_ave

    train = np.vstack((hxf_train,mnt_train,mng_train))
    test = np.vstack((hxf_test,mnt_test,mng_test))
    data_train = train[:,1:]
    y_train = train[:,0]

    data_test = test[:,1:]
    y_test = test[:,0]
    for i in range(1,data_train.shape[1]+1):
        x_train = data_train[:,0:i]
        x_test = data_test[:,0:i]

        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train, y_train)
        label = lda.predict(x_test)
        score = lda.predict_proba(x_test)
        LDA_acc.append(accuracy_score(y_test,label))
        LDA_label.append(label)

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(x_train, y_train)
        qda_label = qda.predict(x_test)
        # qda_score = qda.predict_proba(x_test)
        # print('QDA:',accuracy_score(y_test, qda_label))
        QDA_score.append(accuracy_score(y_test, qda_label))
    print('lda_acc:',LDA_acc)
    # print('qda_score:',QDA_score)
    # print(acc)
    LDA_acc = pd.DataFrame(LDA_acc)
    LDA_acc.to_excel('./LDA_Average_accuracy.xlsx', index=False, header=False)


LDA_label_1 = pd.DataFrame(LDA_label)
LDA_label_1.to_excel('./LDA_Average_label.xlsx', index=False, header=False)