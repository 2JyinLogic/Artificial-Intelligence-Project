import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
# 读取数据
df = pd.read_csv('./CW_Data.csv', sep=',', header=0)
q1 = list(df['Q1'])
q2 = list(df['Q2'])
q3 = list(df['Q3'])
q4 = list(df['Q4'])
q5 = list(df['Q5'])
cls = list(df['Programme'])

X = []
y = []
for i in range(len(q1)):
    X.append([q1[i], q2[i], q3[i], q4[i], q5[i]])
    y.append(cls[i])

X = np.array(X)
y = np.array(y)
# print(X.shape, y.shape)

# 数据标准化
X_std = StandardScaler().fit_transform(X)

# 训练测试数据分离
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=666)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)






# 加载数据，vectorized数据表示已将文本变为向量表示(这里使用tf-idf作为特征)
# newsgroups_train = fetch_20newsgroups_vectorized('train')
# X_train = newsgroups_train['data']
# y_train = newsgroups_train['target']
#
# newsgroups_test = fetch_20newsgroups_vectorized('test')
# X_test = newsgroups_test['data']
# y_test = newsgroups_test['target']
#
# y_train = np.array(y_train)  # 变为numpy数组
# y_test = np.array(y_test)  # 变为numpy数组
#
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# (11314, 130107) (7532, 130107) (11314,) (7532,)

nb = MultinomialNB()  # 定义多项式朴素贝叶斯分类器
nb.fit(X_train, y_train)

print(accuracy_score(y_test, nb.predict(X_test)))  # 打印分类准确率
print(classification_report(y_test, nb.predict(X_test)))  # 分类报告中包含precision/recall/f1-score