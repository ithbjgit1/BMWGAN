import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as Accuracy
from sklearn.metrics import precision_score as Precision
from sklearn.metrics import recall_score as Recall
from sklearn.metrics import f1_score as F1_measure
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as AUC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

data = pd.DataFrame(pd.read_csv(
        r'D:\PycharmProjects\pythonProject2\BWGAN\数据\last.csv',header=None,skiprows=1))
print(data)

# num1=46  #生成数据的数量
# num2=105  #少数类数据的数量
# plt.figure(figsize=(8, 5))
# plt.scatter(data.iloc[num2+num1:,0],data.iloc[num2+num1:,1],c='orange',label='majority class') #多数类
# plt.scatter(data.iloc[num1:num2+num1,0],data.iloc[num1:num2+num1,1],c='olive',label='minority class')  #少数类
# plt.scatter(data.iloc[:num1,0],data.iloc[:num1,1],c='r',marker='x',label='synthetic data')
# plt.title('LGS')
# plt.legend()
# plt.savefig('D:\PycharmProjects\pythonProject2\局部分布\评价程序/LGS.eps',
#             format='eps',dpi=1000,bbox_inches='tight')
# plt.show()

# 转化标签数据
y_label = data.iloc[:, -1]
le = LabelEncoder()
le = le.fit(y_label)
data.loc[:, 'label'] = le.transform(y_label)

# print(data.head())
# 转化标签数据
y_label = data.iloc[:, -1]
le = LabelEncoder()
le = le.fit(y_label)
data.loc[:, 'label'] = le.transform(y_label)
# print(data)

# #'''-----------高斯贝叶斯-----------'''
# classify = GaussianNB()

# # # '''-----------决策树-----------'''
classify = tree.DecisionTreeClassifier(random_state=42)
# # #
# # # '''-----------支持向量机-----------'''
# # #
# classify = SVC(kernel='linear', probability=True,random_state=42)  # 此时用的是支持向量机模型
# # #
# # # '''-----------随机森林-----------'''
# classify = RandomForestClassifier(random_state=42)
# # #
# # # '''-----------Adaboost-----------'''
# classify = AdaBoostClassifier(DecisionTreeClassifier(),
#                         algorithm="SAMME",n_estimators=100,random_state=42)   #这里的基分类器用的是决策树，还可以使用常见的ID3/C4.5/朴素贝叶斯
# # #
# # # '''-----------神经网络-----------'''
# classify = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                      random_state=42) #需要不断调试隐含层个数
# classify = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(10,50,10), random_state=42) #需要不断调试隐含层个数
##进行交叉验证

columns=data.shape[1]-2
X=data.iloc[:,0:columns]
# print(X)
Y=data.iloc[:,-1]
kf=StratifiedKFold(n_splits=10)
# kf=KFold(n_splits=5)
Newdata_X=np.array(X)
Newdata_Y=np.array(Y)

list1_ACC=[]   #存储准确率
list2_Pre=[]   #存储精准率
list3_Recall=[]   #存储召回率
list4_f1=[]   #存储f1_measure
list5_G_means=[]   #存储G—means
list6_FPR=[]   #存储假正率
list7_spe=[]   #存储特效性
list8_AUC=[]   ##存储AUC面积


for i ,(train ,test) in enumerate(kf.split(Newdata_X, Newdata_Y)):
    classify=classify.fit(Newdata_X[train], Newdata_Y[train])
    Newdata_y_predict = classify.predict(Newdata_X[test])    ## 运用smote算法得到测试集预测之后的分类
    Newdata_score = classify.predict_proba(Newdata_X[test])
    cm = CM(Newdata_Y[test], Newdata_y_predict, labels=[1, 0])  # 输出混淆矩阵
    Newdata_ACC = Accuracy(Newdata_Y[test], Newdata_y_predict)  # 输出准确率
    Newdata_precision = Precision(Newdata_Y[test], Newdata_y_predict)  # 精准率
    Newdata_recall = Recall(Newdata_Y[test], Newdata_y_predict)  # 召回率
    Newdata_F1measure = F1_measure(Newdata_Y[test], Newdata_y_predict)  # f1_measure 越接近于1越好
    Newdata_TP = cm[0, 0]  # 原本正类，预测后也是正类
    Newdata_TN = cm[1, 1]  # 原本是负类，预测后也是负类
    Newdata_FP = cm[1, 0]  # 原本是负类，预测后是正类
    Newdata_FN = cm[0, 1]  # 原本是正类，预测后成为负类
    Newdata_G_means = (Newdata_recall * ((Newdata_TN / (Newdata_TN + Newdata_FP)))) ** 0.5  # G_means计算方法
    Newdata_FPR = Newdata_FP / (np.sum(cm[1, :]))  # 假正率 负类中被错误分类的比例
    Newdata_Spe = Newdata_TN / np.sum(cm[1, :])  # 特效性 负类中被正确分类的比例
    list1_ACC.append(Newdata_ACC)
    list2_Pre.append(Newdata_precision)
    list3_Recall.append(Newdata_recall)
    list4_f1.append(Newdata_F1measure)
    list5_G_means.append(Newdata_G_means)
    list6_FPR.append(Newdata_FPR)
    list7_spe.append(Newdata_Spe)
    #Roc曲线 与 AUC面积
    Newdata_rocfpr, Newadata_rocrecallr, Newdata_threshold = roc_curve(Newdata_Y[test], Newdata_score[:, 1], pos_label=1)
    Newdata_AUC_area = AUC(Newdata_Y[test], Newdata_score[:, 1])  #smote之后的AUC面积
    list8_AUC.append(Newdata_AUC_area)
    plt.rcParams['axes.facecolor'] = 'whitesmoke'  #背景
    plt.plot([0, 1], [0, 1], c='red', linestyle='--')
    plt.plot(Newdata_rocfpr, Newadata_rocrecallr, color=plt.cm.tab10(i + 1),
             label=f'ROC(%d) curve (AUC=%0.3f)' % (i + 1, Newdata_AUC_area), linewidth=3)
    # plt.rcParams['axes.facecolor'] = 'silver'
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', fontsize=8.5)  # loc显示图例在右下方
plt.grid(linestyle='-.')  #设置网格
plt.savefig('D:\PycharmProjects\pythonProject2\sci3\数据/DFGANROC.png',
            format='png',dpi=1000,bbox_inches='tight')
plt.show()


print('bordline1之后Accuracy为：%0.3f+%0.3f'%(np.mean(list1_ACC),np.std(list1_ACC)))
print('bordline1之后Precision为：%0.3f+%0.3f'%(np.mean(list2_Pre),np.std(list2_Pre)))
print('bordline1之后Recall为：%0.3f+%0.3f'%(np.mean(list3_Recall),np.std(list3_Recall)))
print('bordline1之后F1_measure为：%0.3f+%0.3f'%(np.mean(list4_f1),np.std(list4_f1)))
print('bordline1之后G_means为：%0.3f+%0.3f'%(np.mean(list5_G_means),np.std(list5_G_means)))
print('bordline1之后FPR：%0.3f+%0.3f'%(np.mean(list6_FPR),np.std(list6_FPR)))
print('bordline1之后Spe为：%0.3f+%0.3f'%(np.mean(list7_spe),np.std(list7_spe)))
print('bordline1之后AUC面积为为：%0.3f+%0.3f'%(np.mean(list8_AUC),np.std(list8_AUC)))


##数据储存
df1=pd.DataFrame([np.round(np.mean(list8_AUC)*100,1),
                  np.round(np.mean(list1_ACC)*100,1),
                  np.round(np.mean(list3_Recall)*100,1),
                  np.round(np.mean(list2_Pre)*100,1),
                 np.round(np.mean(list5_G_means)*100,1),
                  np.round(np.mean(list4_f1)*100,1)],
                   index=['AUC','Accuracy','Recall','Precision','G','F1_measure'],
                  )
# print(df1)
# print(df1.values)
Store_data=pd.DataFrame(pd.read_excel(
    r'D:\PycharmProjects\pythonProject2\高斯蒙特卡洛论文程序\高斯蒙特卡洛数据集\数据存储.xlsx',
    sheet_name='sheet1'))
Store_data.loc[:,'values_new']=df1.values
print(Store_data)
Store_data.to_excel(r'D:\PycharmProjects\pythonProject2\高斯蒙特卡洛论文程序\高斯蒙特卡洛数据集\数据存储.xlsx',
                    sheet_name='sheet1',index=False)
#