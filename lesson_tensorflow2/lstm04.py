# -*- coding: utf-8 -*-

__author__ = "cs-pc/2018-04-03"

'''Tensorflow实例：利用LSTM预测股票每日最高价（二）
参考：
https://blog.csdn.net/mylove0414/article/details/56969181
https://github.com/lyshello123/stock_predict_with_LSTM/blob/master/stock_predict_2.py
https://blog.csdn.net/wlzzhcsdn/article/details/78207293

根据股票历史数据中的最低价、最高价、开盘价、收盘价、交易量、交易额、跌涨幅等因素，
对下一日股票最高价进行预测。
实验用到的数据长这个样子： 
label是标签y，也就是下一日的最高价。列C到I为输入特征。 
本实例共6100多行，用前5800个数据做训练数据。

单因素输入特征及RNN、LSTM的介绍请戳上一篇lstm02.py
'''



# 导入包及声明常量===================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

rnn_unit = 10         #隐含层单元数量
input_size = 7        #输入（训练）层维度
output_size = 1       #输出（预测）层维度
lr = 0.0006           #学习率

# 导入数据===================================================
f = open('dataset_2.csv')
df = pd.read_csv(f)           #读入股票数据
data = df.iloc[:,2:10].values #取第3-10列，跳过前两列是股票索引、日期，目标是最后列，即明日最高价

#以折线图展示最后预测列数据
# plt.figure()
# plt.plot(data[:, 7])
# plt.show()

# 生成训练集、测试集===================================================
'''考虑到真实的训练环境，这里把每批次训练样本数（batch_size）、时间步（time_step）、
训练集的数量（train_begin,train_end）设定为参数，使得训练更加机动。
读者注：
batch_size的意思是抽样间隔，放进去的大小是time_step决定的'''
#——————————获取训练集——————————
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=5800):
    batch_index = []
    data_train = data[train_begin:train_end]
    mean = np.mean(data_train, axis=0)
    std = np.std(data_train, axis=0)
    normalized_train_data = (data_train - mean) / std  #标准化
    print('训练集长度：', len(normalized_train_data), '；time_step：', time_step)
    train_x, train_y = [], []   #训练集x和y初定义
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :7]
        y = normalized_train_data[i:i + time_step, 7, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y

#——————————获取测试集——————————
def get_test_data(time_step=20, test_begin=5800):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  #标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  #有size个sample
    print('测试集长度：', len(normalized_test_data), '；time_step：', time_step)
    '''确实是需要减1，使得测试集长度整除不下time_step的时候，size可以取上限。
    也就是说有可能最后一个测试样本的时间步小于设定的time_step。
    在真正测试的时候，当时为了方便，就直接舍弃掉了后面一部分的测试数据。'''
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :7]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :7]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, 7]).tolist())
    '''读者注：
    大家画图的问题是因为拆分测试集的时候把y的多余部分放进去了，
    而不满step_size的没有在最后被调用，因此少了一段，
    把 test_x.append((data_test[(i+1)*time_step:,:7]).tolist())
    test_y.extend((data_test[(i+1)*time_step:,7]).tolist())
    注释掉就行，就一样长了。'''
    return mean, std, test_x, test_y

# 构建神经网络===================================================
#输入层、输出层权重、偏置
weights = {
         'in' :tf.Variable(tf.random_normal([input_size, rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit, 1]))
         }
biases = {
        'in' :tf.Variable(tf.constant(0.1, shape = [rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1, shape = [1, ]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X,[-1, input_size]) #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit]) #将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size,dtype = tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, 
                                                 input_rnn,
                                                 initial_state = init_state, 
                                                 dtype = tf.float32)
    #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit]) #作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

# 训练模型===================================================
def train_lstm(batch_size=60, time_step=20, train_begin=0, train_end=5800):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    #损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #重复训练2000次
        for i in range(20):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op,loss],
                                    feed_dict = {X:train_x[batch_index[step]:batch_index[step + 1]],
                                                 Y:train_y[batch_index[step]:batch_index[step + 1]]})
            print("Number of iterations:", i, " loss:", loss_)
        if i % 20 == 0:
            print("model_save: ", saver.save(sess, 'model_save2\\modle.ckpt'))
        #我是在window下跑的，这个地址是存放模型的地方，模型参数文件名为modle.ckpt
        #在Linux下面用 'model_save2/modle.ckpt'
        print("The train has finished")
'''嗯，这里说明一下，这里的参数是基于已有模型恢复的参数，
意思就是说之前训练过模型，保存过神经网络的参数，现在再取出来作为初始化参数接着训练。
如果是第一次训练，就用sess.run(tf.global_variables_initializer())，
也就不要用到 module_file = tf.train.latest_checkpoint() 和saver.store(sess, module_file)了。
'''

# 模型预测及测试===================================================
def prediction(time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x,test_y = get_test_data(time_step)
    with tf.variable_scope("sec_lstm", reuse=True):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred,feed_dict={X:[test_x[step]]})
            predict = prob.reshape((-1)) #numpy矩阵变维，变化为一维
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[7] + mean[7]
        test_predict = np.array(test_predict) * std[7] + mean[7]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / 
                         test_y[:len(test_predict)])
        #acc为测试集偏差
        print("The accuracy of this predict:",acc)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='r',)
        plt.plot(list(range(len(test_y))), test_y, color='b')
        plt.show()
'''蓝色折线是真实值，红色折线是预测值
偏差大概在1.36%

读者：
做prediction的时候，
您是把1-20天内的日期/开盘价/收盘价等特征作为输入得到第2-21天内的最高价预测值，这与实际做预测的方式不符合。
我们实际预测的做法应该是拿1-20天内的日期/开盘价/收盘价等特征作为输入得到第21天内的最高价预测值。
我根据这两点修改了代码，最后的acc为19.37%，折线图里两端基本吻合，但中间部分的预测值比实际值要低得多。
'''



if __name__ == "__main__":
    train_lstm()
    prediction()


