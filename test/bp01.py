# -*- coding: utf-8 -*-
__author__ = "cs-pc/2018-03-20"



'''基于TensorFlow的神经网络测试程序
参考网址：https://developers.google.cn/machine-learning/crash-course/introduction-to-neural-networks/programming-exercise

神经网络简介
学习目标：
    .使用 TensorFlow DNNRegressor 类定义神经网络 (NN) 及其隐藏层
    .训练神经网络学习数据集中的非线性规律，并实现比线性回归模型更好的效果
'''



import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
#------------------------------------



'''设置'''
tf.logging.set_verbosity(tf.logging.ERROR)  #日志级别设置成 ERROR，避免干扰
pd.options.display.max_rows = 10
'''显示的最大行数和列数，如果超额就显示省略号，这个指的是多少个dataFrame的列。如果比较多又不允许换行，就会显得很乱。'''
pd.options.display.float_format = '{:.1f}'.format  #限制输出1位小数

'''首先加载和准备数据
手工编辑的数据共计20行：
    .第1、2维为供四舍五入计算得出目标数据的数据，
    .第3维是干扰数据（不变化），
    .第4维是干扰数据（随机变化）。'''
train_dataframe = pd.read_csv("train.csv", sep=",")
# print('读取的csv文件：')
# print(train_dataframe)

train_dataframe = train_dataframe.reindex(
    np.random.permutation(train_dataframe.index))  # 随机给数据进行排序
#------------------------------------


def preprocess_features(train_dataframe):
    """准备来自手工准备的简单数据集的输入特征。
    参数：
        train_dataframe: 一个符合Pandas DataFrame格式的简单手工数据集。
    返回：
        processed_features: 一个包含用于模型的特征（含合成特征）的DataFrame。
    """
    selected_features = train_dataframe[["x1", "x2", "x3", "x4"]]
    processed_features = selected_features.copy()
    # 创建一个合成特征
    processed_features["x12"] = (train_dataframe["x1"] * train_dataframe["x2"])
    return(processed_features)

def preprocess_targets(train_dataframe):
    """准备来自train_dataframe的目标特征(即标签)。
    参数：
        train_dataframe: 一个符合Pandas DataFrame格式的简单手工数据集
    返回：
        output_targets: 一个包含目标特征的DataFrame。
    """
    output_targets = pd.DataFrame()
    # 创建目标特征
    output_targets["yy"] = train_dataframe["yy"]
    return(output_targets)

print("1.数据准备完成……")
#------------------------------------



# 选择前10行数据进行训练。
training_examples = preprocess_features(train_dataframe.head(10))
training_targets = preprocess_targets(train_dataframe.head(10))

# 选择最后10行数据进行验证。
validation_examples = preprocess_features(train_dataframe.tail(10))
validation_targets = preprocess_targets(train_dataframe.tail(10))

# 仔细检查一下我们做的对不对。
print("Training examples summary:")  # 训练数据综述
display.display(training_examples.describe())
print("Validation examples summary:")  # 验证数据综述
display.display(validation_examples.describe())

print("Training targets summary:")  # 训练目标综述
display.display(training_targets.describe())
print("Validation targets summary:")  # 验证目标综述
display.display(validation_targets.describe())

print("2.数据分析完成……")
#------------------------------------



'''构建神经网络
神经网络由 DNNRegressor 类定义。

使用 hidden_units 定义神经网络的结构。
hidden_units 参数会创建一个整数列表，其中每个整数对应一个隐藏层，表示其中的节点数。以下面的赋值为例：
    hidden_units=[3,10]
上述赋值为神经网络指定了两个隐藏层：
    .第一个隐藏层包含 3 个节点。
    .第二个隐藏层包含 10 个节点。
如果我们想要添加更多层，可以向该列表添加更多整数。
例如，hidden_units=[10,20,30,40] 会创建 4 个分别包含 10、20、30 和 40 个单元的隐藏层。

默认情况下，所有隐藏层都会使用 ReLu 激活函数，且是全连接层。'''

def construct_feature_columns(input_features):
    """构建TensorFlow的特征列。
    参数：
        input_features：关于数字输入特征的名称。
    返回：
        一组特征列。
    """
    return(set([
        tf.feature_column.numeric_column(my_feature)
        for my_feature in input_features
    ]))

def my_input_fn(features, targets, batch_size=1, shuffle=True,
                num_epochs=None):
    """训练一个输出特征的线性回归模型
    参数：
      features: pandas 的特征DataFrame
      targets: pandas 的目标DataFrame
      batch_size: 被传递到模型的 Size of batches
      shuffle: True or False. 是否 shuffle 数据.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    返回：
      Tuple of (features, labels) for next data batch
    """

    # 将 pandas 数据转换成 np 数组的字典
    features = {key: np.array(value) for key, value in dict(features).items()}

    # 构建数据集，并配置批处理/重复
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # 判断是否要打乱数据集
    if shuffle:
        ds = ds.shuffle(10)

    # 返回下一批数据
    features, labels = ds.make_one_shot_iterator().get_next()
    return(features, labels)


def train_nn_regression_model(learning_rate, steps, batch_size, hidden_units,
                              training_examples, training_targets,
                              validation_examples, validation_targets):
    """训练神经网络回归模型
    除了训练，还有输出训练过程进度信息的功能，以及一段时间的训练和验证损失的plot图。
    参数：
        learning_rate: `float`数值, 学习速率
        steps: 非零 `int`数值, 训练次数的总数。一个训练次数包括使用单一批次数据的前向和后向的过程。
        batch_size: 非零 `int`数值, 数据批量大小。
        hidden_units: 整形 `list` 数值, 指定在每层神经元的数目。
        training_examples: `DataFrame` 包含从CSV数据中得到的一个或多个列，用于训练的输入特征。
        training_targets: `DataFrame` 包含从CSV数据中得到的一个列，用于训练的目标特征。
        validation_examples: DataFrame` 包含从CSV数据中得到的一个或多个列，用于验证的输入特征。
        validation_targets: `DataFrame` 包含从CSV数据中得到的一个列，用于验证的目标特征。
    返回：
        一个基于训练数据训练出的线性回归模型。
  """

    periods = 10
    steps_per_period = steps / periods

    # 创建一个线性回归对象
    my_optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units)

    # 创建输入函数（训练输入、训练预测输入、预测验证输入）
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["yy"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["yy"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["yy"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # 训练模型，但在循环中这样做，我们就可以周期性地评估损失
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # 从先验状态开始训练模型
        dnn_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        # 休息一下，计算预测
        training_predictions = dnn_regressor.predict(
            input_fn=predict_training_input_fn)
        training_predictions = np.array(
            [item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(
            input_fn=predict_validation_input_fn)
        validation_predictions = np.array(
            [item['predictions'][0] for item in validation_predictions])

        # 计算训练和验证损失
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions,
                                       validation_targets))
        # 偶尔输出当前的损失。
        print("  period %02d : %0.2f" % (period,
                                         training_root_mean_squared_error))
        # 从这个周期添加损失指标到我们的列表中
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # 在周期内输出损失度量图。
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    print("Final RMSE (on training data):   %0.2f" %
          training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" %
          validation_root_mean_squared_error)

    return(dnn_regressor)

print("3.神经网络构建完成……")
#------------------------------------



'''训练神经网络模型
调整超参数，目标是将 RMSE 降到 110 以下。

运行以下代码块来训练神经网络模型。

我们已经知道，在使用了很多特征的线性回归练习中，110 左右的 RMSE 已经是相当不错的结果。
我们将得到比它更好的结果。

在此练习中，您的任务是修改各种学习设置，以提高在验证数据上的准确率。

对于神经网络而言，过拟合是一种真正的潜在危险。
您可以查看训练数据损失与验证数据损失之间的差值，以帮助判断模型是否有过拟合的趋势。
如果差值开始变大，则通常可以肯定存在过拟合。

由于存在很多不同的可能设置，强烈建议您记录每次试验，以在开发流程中进行参考。

此外，获得效果出色的设置后，尝试多次运行该设置，看看结果的重复程度。
由于神经网络权重通常会初始化为较小的随机值，因此每次运行结果应该存在差异。
'''

dnn_regressor = train_nn_regression_model(
    learning_rate=0.01,
    steps=500,
    batch_size=10,
    hidden_units=[10, 2],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

print("4.训练神经网络模型1次完成……")
#------------------------------------



dnn_regressor = train_nn_regression_model(
    learning_rate=0.001, #修改学习速率
    steps=2000, #修改训练次数
    batch_size=100, #修改数据批量大小
    hidden_units=[10, 10], #修改隐藏层的神经元数量
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

print("5.训练神经网络模型2次完成……")
#------------------------------------



'''用测试数据进行评估
确认您的验证效果结果经受得住测试数据的检验。

获得满意的模型后，用测试数据评估该模型，以与验证效果进行比较。'''
test_data = pd.read_csv("test.csv", sep=",")

test_examples = preprocess_features(test_data) #验证用输入数据
test_targets = preprocess_targets(test_data) #验证用目标数据

predict_testing_input_fn = lambda: my_input_fn(test_examples,
                                               test_targets["yy"],
                                               num_epochs=1,
                                               shuffle=False)

test_predictions = dnn_regressor.predict(input_fn=predict_testing_input_fn)
test_predictions = np.array(
    [item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

print("6.用测试数据进行评估完成……")
#------------------------------------
