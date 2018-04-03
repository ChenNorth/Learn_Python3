# -*- coding: utf-8 -*-

__author__ = "cs-pc/2018-04-02"

'''参考：
https://blog.csdn.net/jerr__y/article/details/70809528
TensorFlow入门（七） 充分理解 name / variable_scope

前言: 本例子主要介绍 name_scope 和 variable_scope 的正确使用方式，
学习并理解本例之后，你就能够真正读懂 TensorFlow 的很多代码并能够清晰地理解模型结构了。

之前写过一个例子了： TensorFlow入门（四） name / variable_scope 的使用。
但是当时其实还对 name / variable_scope 不是非常理解。
所以又学习了一番，攒了这篇博客。学习本例子不需要看上一篇，但是咱们还是从上一篇说起：

* 起因：在运行 RNN LSTM 实例代码的时候出现 ValueError。 * 
在 TensorFlow 中，经常会看到这 name_scope 和 variable_scope 两个东东出现，
这到底是什么鬼，到底系做咩噶!!! 在做 LSTM 的时候遇到了下面的错误：
ValueError: Variable rnn/basic_lstm_cell/weights already exists, disallowed.

然后谷歌百度都查了一遍，结果也不知是咋回事。
我是在 jupyter notebook 运行的示例程序，
第一次运行的时候没错，然后就总是出现上面的错误。
后来才知道是 get_variable() 和 variable_scope() 搞的鬼。 

=========================================================

1. 先说结论
要理解 name_scope 和 variable_scope， 首先必须明确二者的使用目的。
我们都知道，和普通模型相比神经网络的节点非常多，节点节点之间的连接（权值矩阵）也非常多。
所以我们费尽心思，准备搭建一个网络，然后有了图1的网络，WTF! 
因为变量太多，我们构造完网络之后，一看，什么鬼，这个变量到底是哪层的？？

为了解决这个问题，我们引入了 name_scope 和 variable_scope， 
二者又分别承担着不同的责任：
    * name_scope: * 为了更好地管理变量的命名空间而提出的。
        比如在 tensorboard 中，因为引入了 name_scope， 我们的 Graph 看起来才井然有序。
    * variable_scope: * 大大大部分情况下，跟 tf.get_variable() 配合使用，
        实现变量共享的功能。

下面通过两组实验来探索 TensorFlow 的命名机制。
'''



'''==========================================================================================
2. （实验一）三种方式创建变量： tf.placeholder, tf.Variable, tf.get_variable
2.1 实验目的：探索三种方式定义的变量之间的区别'''
import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 1.placeholder 
v1 = tf.placeholder(tf.float32, shape=[2,3,4])
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[2,3,4], name='ph')
print(v1.name)
v1 = tf.placeholder(tf.float32, shape=[2,3,4], name='ph')
print(v1.name)
print(type(v1))
print(v1)
print("-------------------")

# 2. tf.Variable()
v2 = tf.Variable([1,2], dtype=tf.float32)
print(v2.name)
v2 = tf.Variable([1,2], dtype=tf.float32, name='V')
print(v2.name)
v2 = tf.Variable([1,2], dtype=tf.float32, name='V')
print(v2.name)
print(type(v2))
print(v2)
print("--------------------------------------")

# 3.tf.get_variable() 创建变量的时候必须要提供 name
v3 = tf.get_variable(name='gv', shape=[])  
print(v3.name)
# v4 = tf.get_variable(name='gv', shape=[2]) # 此处报错
# print(v4.name)

print(type(v3))
print(v3)
print("-------------------")

'''还记得有这么个函数吗？ tf.trainable_variables(), 
它能够将我们定义的所有的 trainable=True 的所有变量以一个list的形式返回。 
very good, 现在要派上用场了。'''
vs = tf.trainable_variables()
print(len(vs))
for v in vs:
    print(v)
print("--------------------------------------")

'''2.2 实验1结论
从上面的实验结果来看，这三种方式所定义的变量具有相同的类型。
而且只有 tf.get_variable() 创建的变量之间会发生命名冲突。
在实际使用中，三种创建变量方式的用途也是分工非常明确的。其中
    .tf.placeholder() 占位符。* trainable==False *
    .tf.Variable() 一般变量用这种方式定义。 * 可以选择 trainable 类型 *
    .tf.get_variable() 一般都是和 tf.variable_scope() 配合使用，
        从而实现变量共享的功能。 * 可以选择 trainable 类型 *'''



'''==========================================================================================
3. （实验二） 探索 name_scope 和 variable_scope
3.1 实验二目的：熟悉两种命名空间的应用情景。'''
import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with tf.name_scope('nsc1'):
    v1 = tf.Variable([1], name='v1')
    with tf.variable_scope('vsc1'):
        v2 = tf.Variable([1], name='v2')
        v3 = tf.get_variable(name='v3', shape=[])
print('v1.name: ', v1.name)
print('v2.name: ', v2.name)
print('v3.name: ', v3.name)
print("-------------------")

with tf.name_scope('nsc1'):
    v4 = tf.Variable([1], name='v4')
print('v4.name: ', v4.name)
print("--------------------------------------")

'''==========================================================================================
tf.name_scope() 并不会对 tf.get_variable() 创建的变量有任何影响。 
tf.name_scope() 主要是用来管理命名空间的，这样子让我们的整个模型更加有条理。
而 tf.variable_scope() 的作用是为了实现变量共享，
它和 tf.get_variable() 来完成变量共享的功能。

1.第一组，用 tf.Variable() 的方式来定义。'''
import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 拿官方的例子改动一下
def my_image_filter():
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    return None

# First call creates one set of 4 variables.
result1 = my_image_filter()
# Another set of 4 variables is created in the second call.
result2 = my_image_filter()
# 获取所有的可训练变量
vs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(vs))
for v in vs:
    print(v)
print("-------------------")



'''==========================================================================================
2.第二种方式，用 tf.get_variable() 的方式'''
import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 下面是定义一个卷积层的通用方式
def conv_relu(kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    return None


def my_image_filter():
    # 按照下面的方式定义卷积层，非常直观，而且富有层次感
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu([5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu( [5, 5, 32, 32], [32])


with tf.variable_scope("image_filters") as scope:
    # 下面我们两次调用 my_image_filter 函数，但是由于引入了 变量共享机制
    # 可以看到我们只是创建了一遍网络结构。
    result1 = my_image_filter()
    scope.reuse_variables()
    result2 = my_image_filter()

# 看看下面，完美地实现了变量共享！！！
vs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(vs))
for v in vs:
    print(v)
print("--------------------------------------")

'''3.2 实验 2 结论
首先我们要确立一种 Graph 的思想。
在 TensorFlow 中，我们定义一个变量，相当于往 Graph 中添加了一个节点。
和普通的 python 函数不一样，在一般的函数中，我们对输入进行处理，然后返回一个结果，
而函数里边定义的一些局部变量我们就不管了。
但是在 TensorFlow 中，我们在函数里边创建了一个变量，就是往 Graph 中添加了一个节点。
出了这个函数后，这个节点还是存在于 Graph 中的。'''




'''==========================================================================================
4. 优雅示例
在深度学习中，通常说到 变量共享 我们都会想到 RNN 。下面我找了两个源码中非常漂亮的例子，可以参考学习学习。'''
# 例1：MultiRNNCell(RNNCell) 中这样来创建 各层 Cell
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
# for i, cell in enumerate(self._cells):
#     with vs.variable_scope("cell_%d" % i):
#         if self._state_is_tuple:
#             ...

# 例2：tf.contrib.rnn.static_bidirectional_rnn 双端 LSTM 
with vs.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with vs.variable_scope("fw") as fw_scope:
        output_fw, output_state_fw = static_rnn(
        cell_fw, inputs, initial_state_fw, dtype,
        sequence_length, scope=fw_scope)

    # Backward direction
    with vs.variable_scope("bw") as bw_scope:
        reversed_inputs = _reverse_seq(inputs, sequence_length)
        tmp, output_state_bw = static_rnn(
              cell_bw, reversed_inputs, initial_state_bw,
              dtype, sequence_length, scope=bw_scope)

print("-------------------")

