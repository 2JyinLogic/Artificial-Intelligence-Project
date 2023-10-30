from pasta.augment import inline
from sklearn.naive_bayes import MultinomialNB
from tensorflow import keras
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname="C:\Windows\Fonts\SimHei.ttf", size=14)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import metrics
from sklearn.model_selection import train_test_split


def evaluate(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    pre = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return acc, recall, pre, f1



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

# 定义分类器
# knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
# lr = LogisticRegression(penalty='l2')

# 模型训练和测试
# knn.fit(X_train, y_train)
# lr.fit(X_train, y_train)
#
# knn_result = knn.predict(X_test)
# lr_result = lr.predict(X_test)

nb = MultinomialNB()  # 定义多项式朴素贝叶斯分类器
nb.fit(X_train, y_train)

nb_result = nb.predict(X_test)

nb_acc, nb_recall, nb_pre, nb_f1 = evaluate(y_test, nb_result)
with open('result.txt', 'a', encoding='utf-8') as f:
    f.write('kNN  acc:' + str(nb_acc) + ', recall:' + str(nb_recall) + ', precision:' + str(nb_pre) + ', f1:' + str(nb_f1))
    f.write('\n')

