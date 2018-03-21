# -*- coding: utf-8 -*-
__author__ = "cs-pc/2018-03-20"



'''对bp01.py程序进行以下改进：
    1.增加主函数，并对其余代码部分进行封装后调用，部分变量定义为公共变量；
    2.增加单层循环、三层循环的模型训练模式，并合并多次图形输出的内容进行集中展现；
    3.因训练量巨大，故中间部分详细输出内容被屏蔽（保留结论性内容），同时增加变化中的循环参数输出。'''



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



'''设置'''
tf.logging.set_verbosity(tf.logging.ERROR)  #日志级别设置成 ERROR，避免干扰
pd.options.display.max_rows = 10
'''显示的最大行数和列数，如果超额就显示省略号，这个指的是多少个dataFrame的列。如果比较多又不允许换行，就会显得很乱。'''
pd.options.display.float_format = '{:.1f}'.format  #限制输出1位小数



def ImportData():
    '''首先加载和准备数据
    手工编辑的数据共计20行：
        .第1、2维为供四舍五入计算得出目标数据的数据，
        .第3维是干扰数据（不变化），
        .第4维是干扰数据（随机变化）。'''
    dataframe = pd.read_csv("train.csv", sep=",")
    # print('读取的csv文件：')
    # print(train_dataframe)
    
    dataframe = dataframe.reindex(
        np.random.permutation(dataframe.index))  # 随机给数据进行排序
    return dataframe

def preprocess_features(dataframe):
    """准备来自手工准备的简单数据集的输入特征。
    参数：
        dataframe: 一个符合Pandas DataFrame格式的简单手工数据集。
    返回：
        processed_features: 一个包含用于模型的特征（含合成特征）的DataFrame。
    """
    selected_features = dataframe[
        ["x1",
         "x2",
         "x3",
         "x4"]]
    processed_features = selected_features.copy()
    # 创建一个合成特征（非线性）
    processed_features["x12"] = dataframe["x1"] * dataframe["x2"]
    return(processed_features)

def preprocess_targets(dataframe):
    """准备来自train_dataframe的目标特征(即标签)。
    参数：
        dataframe: 一个符合Pandas DataFrame格式的简单手工数据集
    返回：
        output_targets: 一个包含目标特征的DataFrame。
    """
    output_targets = pd.DataFrame()
    # 创建目标特征
    output_targets["yy"] = dataframe["yy"]
    return output_targets

def PrepareData(dataframe):
    global training_examples, training_targets, validation_examples, validation_targets
    # 选择前10行数据进行训练。
    training_examples = preprocess_features(dataframe.head(10))
    training_targets = preprocess_targets(dataframe.head(10))

    # 选择最后10行数据进行验证。
    validation_examples = preprocess_features(dataframe.tail(10))
    validation_targets = preprocess_targets(dataframe.tail(10))

    # 仔细检查一下我们做的对不对。
    print("Training examples summary:")  # 训练数据综述
    display.display(training_examples.describe())
    print("Validation examples summary:")  # 验证数据综述
    display.display(validation_examples.describe())

    print("Training targets summary:")  # 训练目标综述
    display.display(training_targets.describe())
    print("Validation targets summary:")  # 验证目标综述
    display.display(validation_targets.describe())

def construct_feature_columns(input_features):
    """构建TensorFlow的特征列。
    参数：
        input_features：关于数字输入特征的名称。
    返回：
        一组特征列。
    """
    return set([
        tf.feature_column.numeric_column(my_feature)
        for my_feature in input_features
    ])

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

    '''参考：
    https://baijiahao.baidu.com/s?id=1583657817436843385&wfr=spider&for=pc
    在初学时，我们只需要关注两个最重要的基础类：Dataset和Iterator。

Dataset可以看作是相同类型“元素”的有序列表。
在实际使用时，单个“元素”可以是向量，也可以是字符串、图片，甚至是tuple或者dict。

先以最简单的，Dataset的每一个元素是一个数字为例，如何将这个dataset中的元素取出呢？
方法是从Dataset中示例化一个Iterator，然后对Iterator进行迭代
在非Eager模式下，读取上述dataset中元素的方法为：………………………………
语句iterator = dataset.make_one_shot_iterator()从dataset中实例化了一个Iterator，
这个Iterator是一个“one shot iterator”，即只能从头到尾读取一次。
one_element = iterator.get_next()表示从iterator里取出一个元素。

由于这是非Eager模式，所以one_element只是一个Tensor，并不是一个实际的值。
调用sess.run(one_element)后，才能真正地取出一个值

如果一个dataset中元素被读取完了，再尝试sess.run(one_element)的话，
就会抛出tf.errors.OutOfRangeError异常，这个行为与使用队列方式读取数据的行为是一致的
'''

    # 判断是否要打乱数据集
    if shuffle:
        ds = ds.shuffle(10000)
        # shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，表示打乱时使用的buffer的大小

    # 返回下一批数据
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_nn_regression_model(learning_rate, steps, batch_size, hidden_units,
                              training_examples, training_targets,
                              validation_examples, validation_targets, times):
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
        times: 训练次数
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
    # print("RMSE (on training data):")
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
        # print("  period %02d : %0.2f" % (period,
        #                                  training_root_mean_squared_error))
        # 从这个周期添加损失指标到我们的列表中
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    # print("Model training finished.")

    # 在周期内输出损失度量图。
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    # print(training_rmse)
    # print("================")
    # print(validation_rmse)
    plt.plot(training_rmse, label="training-" + str(times))
    plt.plot(validation_rmse, label="validation-" + str(times))
    plt.legend()
    #plt.show()

    print("Final RMSE (on training data):   %0.2f" %
          training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" %
          validation_root_mean_squared_error)

    return dnn_regressor

def TestData():
    '''用测试数据进行评估。'''
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

#------------------------------------

def train_1():
    '''对学习速率进行多次循环计算'''
    global dnn_regressor
    global plt
    i = 0
    for i_learning_rate in range(1, 11, 2): # [(0.001, 0.004, 0.008)]
        i = i + 1
        print("------[2, 6]--@"+str(i_learning_rate/1000))
        dnn_regressor = train_nn_regression_model(
            learning_rate = i_learning_rate / 1000, # 0.001
            steps = 100, # 500,2000
            batch_size = 10, # 100
            hidden_units = [2, 6], # [10, 10], [10, 2]
            training_examples = training_examples,
            training_targets = training_targets,
            validation_examples = validation_examples,
            validation_targets = validation_targets,
            times = i)
        TestData()
        print("3.--训练神经网络模型" + str(i) + "次，及评估完成++++++++++++++++")
    plt.show()

def train_2():
    '''对学习速率、隐藏层单元数量进行多次循环计算'''
    global dnn_regressor
    global plt
    i = 0
    for i_learning_rate in range(1, 11, 2): # [(0.001, 0.004, 0.008)]
        for i_unit1 in range(2, 7, 2):
            for i_unit2 in range(2, 7, 2):
                i = i + 1
                print("------["+str(i_unit1)+", "+str(i_unit2)+"]--@"+str(i_learning_rate/1000))
                dnn_regressor = train_nn_regression_model(
                    learning_rate = i_learning_rate / 1000, # 0.001
                    steps=100, # 500,2000
                    batch_size=10, # 100
                    hidden_units=[i_unit1, i_unit2], # [10, 10],[10, 2]
                    training_examples=training_examples,
                    training_targets=training_targets,
                    validation_examples=validation_examples,
                    validation_targets=validation_targets,
                    times = i_learning_rate)
                TestData()
        print("3.--训练神经网络模型" + str(i) + "次，及评估完成++++++++++++++++")
    plt.show()

#------------------------------------

if __name__ == '__main__':
    print("0.数据初始化完成……………………………………………………………………………………………………………………………………")
    train_dataframe = ImportData()
    print("1.数据准备完成……………………………………………………………………………………………………………………………………")
    PrepareData(train_dataframe)
    print("2.数据分析完成……………………………………………………………………………………………………………………………………")
    
    train_1() # 单层循环训练模型及评估
    #train_2() # 3层循环训练模型及评估
    print("4.训练神经网络模型全部完成------------------------------------------------")
