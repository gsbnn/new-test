import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('classificationdata.csv')
print(data)
# 将类别进行独热编码
y = pd.get_dummies(data['index'])
X = data.drop('index', axis=1)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=[len(X.columns)]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 定义损失函数和优化器
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=0)

# 绘制网络训练损失曲线图和准确度曲线图
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 在测试集上进行预测
y_pred = np.argmax(model.predict(X_test), axis=-1)
y_true = np.argmax(y_test.values, axis=-1)

# 显示评价指标
print("准确率：", accuracy_score(y_true, y_pred))
print("混淆矩阵：\n", confusion_matrix(y_true, y_pred))


cm = confusion_matrix(y_true, y_pred)
labels = ['class 1', 'class 2', 'class 3']

plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(np.arange(len(labels)), labels)
plt.show()



""""

在这段代码中，我们首先导入了matplotlib库来绘制网络训练损失曲线图和准确度曲线图，并导入了accuracy_score()和confusion_matrix()函数来计算评价指标。

然后，我们将类别进行独热编码，并将数据集分为训练集和测试集。

接着，我们构建了一个包含两个隐藏层的BP神经网络模型，并在compile()函数中设置了损失函数、优化器和评价指标，其中损失函数使用了交叉熵函数，评价指标为准确度。

然后，我们训练模型，并使用subplot()函数将网络训练损失曲线图和准确度曲线图绘制在同一张图上。

接着，我们在测试集上进行预测，并将预测结果和真实结果进行比较，使用accuracy_score()函数计算准确率，使用confusion_matrix()函数计算混淆矩阵，并在屏幕上显示这些评价指标。
"""