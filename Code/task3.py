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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


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
print(X.shape, y.shape)

# 数据标准化
X_std = StandardScaler().fit_transform(X)

# 训练测试数据分离交叉验证写法（运行5次）

X_train = np.concatenate((X_std[:100, :], X_std[200:500, :]), axis=0)
X_test = X_std[100:200, :]
y_train = np.concatenate((y[0:100], y[200:500]))
y_test = y[100:200]

X_train = X_std[100:500, :]
X_test = X_std[:100, :]
y_train = y[100:500]
y_test = y[:100]

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 定义分类器
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
lr = LogisticRegression(penalty='l2')

# 模型训练和测试
knn.fit(X_train, y_train)
lr.fit(X_train, y_train)

knn_result = knn.predict(X_test)
lr_result = lr.predict(X_test)

knn_acc, knn_recall, knn_pre, knn_f1 = evaluate(y_test, knn_result)
lr_acc, lr_recall, lr_pre, lr_f1 = evaluate(y_test, lr_result)
with open('result.txt', 'a', encoding='utf-8') as f:
    f.write('kNN  acc:' + str(knn_acc) + ', recall:' + str(knn_recall) + ', precision:' + str(knn_pre) + ', f1:' + str(knn_f1))
    f.write('\n')
    f.write('LR  acc:' + str(lr_acc) + ', recall:' + str(lr_recall) + ', precision:' + str(lr_pre) + ', f1:' + str(lr_f1))
    f.write('\n')

kmeans = KMeans(n_clusters=5)

kmeans_result = kmeans.fit_predict(X_test)

kmeans_acc, kmeans_recall, kmeans_pre, kmeans_f1 = evaluate(y_test, kmeans_result)
with open('result.txt', 'a', encoding='utf-8') as f:
    f.write('K-means  acc:' + str(kmeans_acc) + ', recall:' + str(kmeans_recall) + ', precision:' + str(kmeans_pre) + ', f1:' + str(kmeans_f1))
    f.write('\n')

# 神经网络(tensorflow搭建)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
nn_result = model.predict(X_test)
nn_result = np.argmax(nn_result, axis=1)

nn_acc, nn_recall, nn_pre, nn_f1 = evaluate(y_test, nn_result)
with open('result.txt', 'a', encoding='utf-8') as f:
    f.write('NN  acc:' + str(nn_acc) + ', recall:' + str(nn_recall) + ', precision:' + str(nn_pre) + ', f1:' + str(nn_f1))
    f.write('\n')

# 用PCA降维，通过调参验证降到4维效果最佳
pca = PCA(n_components=4)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# # 数据集最后一行是空值，要去掉，否则会Nan
# plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='y=1')
# plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='y=2')
# plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], label='y=3')
# plt.scatter(X_pca[y == 3, 0], X_pca[y == 3, 1], label='y=4')
# plt.scatter(X_pca[y == 4, 0], X_pca[y == 4, 1], label='y=5')
# plt.legend()
# plt.show()

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
lr = LogisticRegression(penalty='l2')

knn.fit(X_train, y_train)
lr.fit(X_train, y_train)

knn_result = knn.predict(X_test)
lr_result = lr.predict(X_test)

knn_acc, knn_recall, knn_pre, knn_f1 = evaluate(y_test, knn_result)
lr_acc, lr_recall, lr_pre, lr_f1 = evaluate(y_test, lr_result)
with open('result.txt', 'a', encoding='utf-8') as f:
    f.write('kNN+PCA  acc:' + str(knn_acc) + ', recall:' + str(knn_recall) + ', precision:' + str(knn_pre) + ', f1:' + str(knn_f1))
    f.write('\n')
    f.write('LR+PCA  acc:' + str(lr_acc) + ', recall:' + str(lr_recall) + ', precision:' + str(lr_pre) + ', f1:' + str(lr_f1))
    f.write('\n')

kmeans = KMeans(n_clusters=5)

kmeans_result = kmeans.fit_predict(X_test)

kmeans_acc, kmeans_recall, kmeans_pre, kmeans_f1 = evaluate(y_test, kmeans_result)
with open('result.txt', 'a', encoding='utf-8') as f:
    f.write('K-means+PCA  acc:' + str(kmeans_acc) + ', recall:' + str(kmeans_recall) + ', precision:' + str(kmeans_pre) + ', f1:' + str(kmeans_f1))
    f.write('\n')

# 神经网络(tensorflow搭建)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)
nn_result = model.predict(X_test)
nn_result = np.argmax(nn_result, axis=1)

nn_acc, nn_recall, nn_pre, nn_f1 = evaluate(y_test, nn_result)
with open('result.txt', 'a', encoding='utf-8') as f:
   f.write('NN+PCA  acc:' + str(nn_acc) + ', recall:' + str(nn_recall) + ', precision:' + str(nn_pre) + ', f1:' + str(nn_f1))
   f.write('\n')
