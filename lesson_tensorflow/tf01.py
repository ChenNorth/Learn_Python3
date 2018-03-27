# -*- coding: utf-8 -*-

__author__ = "cs-pc/2018-03-26"

'''参考：
https://blog.csdn.net/jerr__y/article/details/57084008
TensorFlow入门（一）基本用法
本例子主要是按照 tensorflow的中文文档来学习 tensorflow 的基本用法。
按照文档说明，主要存在的一些问题：
    1.就是 Session() 和 InteractiveSession() 的用法。
        后者用 Tensor.eval() 和 Operation.run() 来替代了 Session.run(). 
        其中更多的是用 Tensor.eval()，所有的表达式都可以看作是 Tensor.
    2.另外，tf的表达式中所有的变量或者是常量都应该是 tf 的类型。
    3.只要是声明了变量，就得用 sess.run(tf.global_variables_initializer()) 
        或者 x.initializer.run() 方法来初始化才能用。
'''

import tensorflow as tf
import numpy as np



'''==========================================================================================
例一：平面拟合
通过本例可以看到机器学习的一个通用过程：1.准备数据 -> 2.构造模型（设置求解目标函数） -> 3.求解模型'''
# 1.准备数据：使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 2.构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 3.求解模型
# 设置损失函数：误差的均方差
loss = tf.reduce_mean(tf.square(y - y_data))
# 选择梯度下降的方法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 迭代的目标：最小化损失函数
train = optimizer.minimize(loss)

############################################################
# 以下是用 tf 来解决上面的任务
# 1.初始化变量：tf 的必备步骤，主要声明了变量，就必须初始化才能用
init = tf.global_variables_initializer()

# 设置tensorflow对GPU的使用按需分配
config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
# 2.启动图 (graph)
sess = tf.Session(config=config)
sess.run(init)

# 3.迭代，反复执行上面的最小化损失函数这一操作（train op）,拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
print("-------------------")
# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]



'''==========================================================================================
例二：两个数求和'''
input1 = tf.constant(2.0)
input2 = tf.constant(3.0)
input3 = tf.constant(5.0)

intermd = tf.add(input1, input2)
mul = tf.multiply(input2, input3)

with tf.Session() as sess:
    result = sess.run([mul, intermd])  # 一次执行多个op
    print(result)
    print(type(result))
    print(type(result[0]))
print("--------------------------------------")



'''==========================================================================================
1.变量，常量
1.1 用 tensorflow 实现计数器，主要是设计了在循环中调用加法实现计数'''
# 创建变量，初始化为0
state = tf.Variable(0, name="counter")

# 创建一个 op , 其作用是时 state 增加 1
one = tf.constant(1) # 直接用 1 也就行了
new_value = tf.add(state, 1)
update = tf.assign(state, new_value)

# 启动图之后， 运行 update op
with tf.Session() as sess:
    # 创建好图之后，变量必须经过‘初始化’ 
    sess.run(tf.global_variables_initializer())
    # 查看state的初始化值
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)  # 这样子每一次运行state 都还是1
        print(sess.run(state))
print("-------------------")



'''==========================================================================================
1.2 用 tf 来实现对一组数求和，再计算平均'''
h_sum = tf.Variable(0.0, dtype=tf.float32)
# h_vec = tf.random_normal(shape=([10]))
h_vec = tf.constant([1.0,2.0,3.0,4.0])
# 把 h_vec 的每个元素加到 h_sum 中，然后再除以 10 来计算平均值
# 待添加的数
h_add = tf.placeholder(tf.float32)
# 添加之后的值
h_new = tf.add(h_sum, h_add)
# 更新 h_new 的 op
update = tf.assign(h_sum, h_new)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 查看原始值
    print('s_sum =', sess.run(h_sum))
    print("vec = ", sess.run(h_vec))
    
    # 循环添加
    for _ in range(4):
        sess.run(update, feed_dict={h_add: sess.run(h_vec[_])})
        print('h_sum =', sess.run(h_sum))
    
#     print 'the mean is ', sess.run(sess.run(h_sum) / 4)
# 这样写4是错误的，必须转为 tf变量或者常量
    print('the mean is ', sess.run(sess.run(h_sum) / tf.constant(4.0)))
print("--------------------------------------")



'''==========================================================================================
1.3 只用一个变量来实现计数器
上面的计数器是 TensorFlow 官方文档的例子，但是让人觉得好臃肿，所以下面我写了个更加简单的，
只需要定义一个变量和一个加1的操作（op）。通过for循环就能够实现了。'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# 如果不是 assign() 重新赋值的话，每一次 sess.run()都会把 state再次初始化为 0.0
state = tf.Variable(0.0, tf.float32)
# 通过 assign 操作来改变state的值。
add_op = tf.assign(state, state+1)

sess.run(tf.global_variables_initializer())
print('init state ', sess.run(state))
for _ in range(3):
    sess.run(add_op)
    print(sess.run(state))
'''这样子和我们平时实现计数器的方法基本上就 一致了。
我们要重点理解的是，
TensorFlow 中通过 tf.assign(ref, value) 的方式来把 value 值赋给 ref 变量。
这样子，每一次循环的时候，ref 变量才不会再做定义时候的初始化操作。'''
print("-------------------")

# 在函数内部用 assign 不会改变外边的值呀
def chang_W(W1):
    tf.assign(W1, [1.1, 1.2,1.3])
    
W = tf.get_variable('Weights', initializer=[0.2, 0.3, 0.4])
sess.run(tf.global_variables_initializer())
print('THE INIT W IS ', sess.run(W))
chang_W(W)
print('AFTER RUNNING THE FUNC ', sess.run(W))



'''==========================================================================================
2. InteractiveSession() 的用法
InteractiveSession() 主要是避免 Session（会话）被一个变量持有'''
a = tf.constant(1.0)
b = tf.constant(2.0)
c = a + b

# 下面的两种情况是等价的
with tf.Session():
    print(c.eval())

sess = tf.InteractiveSession()
print(c.eval())
sess.close()
print("--------------------------------------")



a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.Variable(3.0)
d = a + b

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

###################
# 这样写是错误的
# print a.run()  
# print d.run()

####################

# 这样才是正确的
print(a.eval())  
print(d.eval())

# run() 方法主要用来
x = tf.Variable(1.2)
# print x.eval()  # 还没初始化，不能用
x.initializer.run()  # x.initializer 就是一个初始化的 op， op才调用run() 方法
print(x.eval())

sess.close()
print("-------------------")



'''==========================================================================================
2.1 怎样使用 tf.InteractiveSession()来完成上面1.2中求和、平均的操作呢?'''
h_sum = tf.Variable(0.0, dtype=tf.float32)
# h_vec = tf.random_normal(shape=([10]))
h_vec = tf.constant([1.0,2.0,3.0,4.0])
# 把 h_vec 的每个元素加到 h_sum 中，然后再除以 10 来计算平均值
# 待添加的数
h_add = tf.placeholder(tf.float32)
# 添加之后的值
h_new = tf.add(h_sum, h_add)
# 更新 h_new 的 op
update = tf.assign(h_sum, h_new)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print('s_sum =', h_sum.eval())
print("vec = ", h_vec.eval())
print("vec = ", h_vec[0].eval())


for _ in range(4):
    update.eval(feed_dict={h_add: h_vec[_].eval()})
    print('h_sum =', h_sum.eval())
sess.close()
print("--------------------------------------")



'''==========================================================================================
3.使用feed来对变量赋值
这些需要用到feed来赋值的操作可以通过tf.placeholder()说明，以创建占位符。 
下面的例子中可以看出 session.run([output], …) 和 session.run(output, …) 的区别。
前者输出了 output 的类型等详细信息，后者只输出简单结果。'''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.0], input2:[2.0]}))

print("-------------------")

with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1:[7.0], input2:[2.0]})
    print(type(result))
    print(result)
print("--------------------------------------")

with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1:7.0, input2:2.0})
    print(type(result))
    print(result)
print("-------------------")

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.0, 3.0], input2:[2.0, 1.0]}))
print("--------------------------------------")

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.0, 3.0], input2:[2.0, 1.0]}))
print("-------------------")

