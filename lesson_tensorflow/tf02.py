# -*- coding: utf-8 -*-

__author__ = "cs-pc/2018-03-26"

'''参考：
https://blog.csdn.net/jerr__y/article/details/57084077
TensorFlow入门（二）简单前馈网络实现 mnist 分类
两层FC层做分类：MNIST
在本教程中，我们来实现一个非常简单的两层全连接网络来完成MNIST数据的分类问题。 
输入[-1,28*28], FC1 有 1024 个neurons， FC2 有 10 个neurons。
这么简单的一个全连接网络，结果测试准确率达到了 0.98。还是非常棒的！！！
'''

import numpy as np
import tensorflow as tf

# 设置按需使用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)



'''==========================================================================================
1. 导入数据'''
# 用tensorflow 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print('training data shape ', mnist.train.images.shape)
print('training label shape ', mnist.train.labels.shape)
print("-------------------")



'''==========================================================================================
2. 构建网络'''
# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# input_layer
X_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# FC1
W_fc1 = weight_variable([784, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(X_, W_fc1) + b_fc1)

# FC2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_pre = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
print("--------------------------------------")



'''==========================================================================================
3. 训练和评估'''
# 1.损失函数：cross_entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_pre))
# y_ 中只有标签所在的那一类是 1， 其余全部都是0.
# 2.优化函数：AdamOptimizer, 优化速度要比 GradientOptimizer 快很多
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 3.预测结果评估
#　预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。
# argmax()取最大值所在的下标
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.arg_max(y_, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始运行
sess.run(tf.global_variables_initializer())
# 这大概迭代了不到 10 个 epoch， 训练准确率已经达到了0.98
for i in range(5000):
    X_batch, y_batch = mnist.train.next_batch(batch_size=100)
    train_step.run(feed_dict={X_: X_batch, y_: y_batch})
    if (i+1) % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={X_: mnist.train.images, y_: mnist.train.labels})
        print("step %d, training acc %g" % (i+1, train_accuracy))
    if (i+1) % 1000 == 0:
        test_accuracy = accuracy.eval(feed_dict={X_: mnist.test.images, y_: mnist.test.labels})
        print("= " * 10, "step %d, testing acc %g" % (i+1, test_accuracy))
print("-------------------")
