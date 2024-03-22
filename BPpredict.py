import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 读取数据集
data = pd.read_csv('wine.csv')

# 去除缺失值
data = data.dropna()

# 将数据集分为特征和目标变量
X = data.drop('quality', axis=1)
y = data['pH']

# 对特征进行标准化
X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / X_std

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建神经网络模型
model = Sequential([
    Dense(64, activation='relu', input_shape=[len(X.columns)]),
    Dense(64, activation='relu'),
    Dense(1)
])

# 定义损失函数和优化器
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# 训练模型
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 输出预测结果
print("预测结果：", y_pred)
""""
在这段代码中，我们首先使用pandas库读取一个汽车燃油效率数据集，然后去除缺失值。接下来，我们将数据集分为特征和目标变量，并使用StandardScaler()函数对特征进行标准化。然后，我们使用train_test_split()函数将数据集分割为训练集和测试集。

接着，我们使用Sequential()函数创建一个序列模型，并使用Dense()函数添加三个全连接层。第一个和第二个层都有64个神经元，并使用ReLU激活函数。最后一层只有一个神经元，用于输出预测结果。

然后，我们使用compile()函数定义损失函数和优化器。在本例中，我们使用均方误差作为损失函数，并使用Adam优化器。接下来，我们使用fit()函数在训练集上训练模型，并使用validation_split参数将一部分训练数据用于验证。

最后，我们使用predict()函数在测试集上进行预测，并输出预测结果。

神经网络的结构如下：

Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                576       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 4,801
Trainable params: 4,801
Non-trainable params: 0
_________________________________________________________________
损失函数传播过程如下：

前向传播：神经网络计算每个样本的预测值。

计算损失：使用均方误差计算预测值与真实值之间的误差。

反向传播：使用链式法则计算每个参数的梯度。

更新参数：使用优化器更新每个参数的值，以最小化损失函数。

重复上述步骤，直到达到指定的训练次数或达到收敛条件。
"""""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 绘制预测结果与真实结果对比图
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=3)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# 显示评价指标
print("平均绝对误差：", mean_absolute_error(y_test, y_pred))
print("均方根误差：", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2得分：", r2_score(y_test, y_pred))

"""
在这段代码中，我们首先导入了matplotlib库来绘制预测结果与真实结果对比图，并导入了mean_absolute_error()、mean_squared_error()和r2_score()函数来计算评价指标。

然后，我们在模型训练和预测的基础上，使用scatter()函数绘制预测结果与真实结果对比图，并使用plot()函数添加一条直线，以表示理论上的完美预测结果。

最后，我们使用mean_absolute_error()、mean_squared_error()和r2_score()函数计算预测结果的平均绝对误差、均方根误差和R2得分，并在屏幕上显示这些评价指标。
"""
