# -*- coding: utf-8 -*-

__author__ = "cs-pc/2018-03-26"

'''参考：
https://blog.csdn.net/jerr__y/article/details/60877873
TensorFlow入门（四） name / variable_scope 的使用

name/variable_scope 的作用
name / variable_scope 详细理解请看：
TensorFlow入门（七） 充分理解 name / variable_scope

* 起因：在运行 RNN LSTM 实例代码的时候出现 ValueError。 * 
在 TensorFlow 中，经常会看到这两个东东出现，这到底是什么鬼，是用来干嘛的。
在做 LSTM 的时候遇到了下面的错误：
ValueError: Variable rnn/basic_lstm_cell/weights already exists, disallowed.

然后谷歌百度都查了一遍，结果也不知是咋回事。
我是在 jupyter notebook 运行的示例程序，
第一次运行的时候没错，然后就总是出现上面的错误。
后来才知道是 get_variable() 和 variable_scope() 搞的鬼。 

下面就来分析一下 TensorFlow 中到底用这来干啥。
'''

import tensorflow as tf

# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)



with tf.variable_scope('vs1') as vs:
    a = tf.get_variable('var1', shape=[2,4], dtype=tf.float32)
    b = tf.get_variable('var2', shape=[3,4], dtype=tf.float32)
#     c = tf.get_variable('var2', shape=[3,4], dtype=tf.float32)
# 报错，tf.get_variable() 是根据 name 来管理变量名的，所以 name 一定不能重复。
    c = tf.Variable([[0,0,0,0]], name='var3')
    print('vs.name=%s' % vs.name)
    
print('a: %s' % a)
print('b: %s' % b)
print('c: %s' % c)
print("-------------------")



with tf.variable_scope('vs1', reuse=True) as vs:
#     d = tf.get_variable('var1', shape=[3,4], dtype=tf.float32)    # 报错， d 和 a 共用变量，所以shape一定是相同的
    d = tf.get_variable('var1', shape=[2,4], dtype=tf.float32)      # 正确, d 和 a 共用一个变量
#     e = tf.get_variable('var3', shape=[1,5], dtype=tf.float32)    # 报错，在 vs1/var3 还没有存在
    e = tf.Variable([[1,2,3,4,5]], dtype=tf.float32, name='var4')   # 正确，tf.variable_scope 不影响使用 Variable 方式定义变量     
    print('vs.name=%s' % vs.name)
    
print('d: %s' % d)
print('e: %s' % e)
'''从上面可以看出，无论是 tf.get_variable 还是 tf.Variable, 
创建的都是 tf.Variable 变量。
把 a,b,c,d 成为变量名，把 'vs1/var1:0','vs1/var2:0' 称为内存变量名称
（注意：这样的称呼只是我自己便于理解，不是专业的叫法）。
不同变量名可以指向同一个内存变量，这就是所谓的变量共享。

可以看出 tf.variable_scope() 功能是要比 name_scope 更加强大的，
下面看看 tf.name_scope 就没这么强大了。它主要就是提供命名管理罢了。'''
print("--------------------------------------")



with tf.name_scope('ns1') as ns:
    print(type(ns), ns)
#     print('ns.name=%s' % ns.name) # 报错

# with tf.name_scope('ns1') as ns:
#     print('ns.name=%s' % ns.name)
print("-------------------")



'''==========================================================================================
1. 首先看看比较简单的 tf.name_scope(‘scope_name’).
tf.name_scope 主要结合 tf.`Variable() 来使用，方便参数命名管理。'''
'''
Signature: tf.name_scope(*args, **kwds)
Docstring:
Returns a context manager for use when defining a Python op.
'''
# 也就是说，它的主要目的是为了更加方便地管理参数命名。
# 与 tf.Variable() 结合使用。简化了命名
with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

# 下面是在另外一个命名空间来定义变量的
with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

# 所以，实际上weights1 和 weights2 这两个引用名指向了不同的空间，不会冲突
print(weights1.name)
print(weights2.name)
print("-------------------")

# 注意，这里的 with 和 python 中其他的 with 是不一样的
# 执行完 with 里边的语句之后，这个 conv1/ 和 conv2/ 空间还是在内存中的。
# 这时候如果再次执行上面的代码，就会再生成其他命名空间
with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

print(weights1.name)
print(weights2.name)
print("--------------------------------------")



'''==========================================================================================
2.下面来看看 tf.variable_scope(‘scope_name’)
tf.variable_scope() 主要结合 tf.get_variable() 来使用，实现变量共享。'''
# 这里是正确的打开方式~~~可以看出，name 参数才是对象的唯一标识
import tensorflow as tf
with tf.variable_scope('v_scope1') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2,3])
    bias1 = tf.get_variable('bias', shape=[3])

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope1', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')

print(Weights1.name)
print(Weights2.name)
# 可以看到这两个引用名称指向的是同一个内存对象
print("-------------------")

# 也可以结合 tf.Variable() 一块使用。
# 注意， bias1 的定义方式
with tf.variable_scope('v_scope2') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2,3])
    #bias1 = tf.Variable([0.52], name='bias') # 注释会在下方正确输出v_scope1

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope2', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')
    bias2 = tf.Variable([0.52], name='bias')

print(Weights1.name)
print(bias1.name)
print(Weights2.name)
print(bias2.name)
print("--------------------------------------")

# 如果 reuse=True 的scope中的变量没有已经定义，会报错！！
# 注意， bias1 的定义方式
with tf.variable_scope('v_scope3') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2,3])
    bias1 = tf.Variable([0.52], name='bias')

print(Weights1.name)
print(bias1.name)

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope3', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')
    #bias2 = tf.get_variable('bias', [1])  # 报错，v_scope3/bias does not exist, or was not created with tf.get_variable().

print(Weights2.name)
print(bias2.name) #上2行注释会输出“v_scope2_1/bias:0”

# 这样子的话就会报错
# Variable v_scope/bias does not exist, or was not created with tf.get_variable()

