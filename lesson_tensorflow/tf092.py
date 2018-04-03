# -*- coding: utf-8 -*-

__author__ = "cs-pc/2018-03-26"

'''参考：
https://blog.csdn.net/jerr__y/article/details/78594494
TensorFlow入门（九）使用 tf.train.Saver()保存模型
'''



'''==========================================================================================
先运行tf09.py，再运行tf092.py
注意，在上面保存完了模型之后。应该把 kernel restart 之后才能使用下面的模型导入。
否则会因为两次命名 “v1” 而导致名字错误。'''
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Create some variables.
v1 = tf.Variable([11.0, 16.3], name="v1")
v2 = tf.Variable(33.5, name="v2")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
# Restore variables from disk.
ckpt_path = './ckpt/test-model.ckpt'
saver.restore(sess, ckpt_path + '-'+ str(1))
print("Model restored.")

print(sess.run(v1))
print(sess.run(v2))
print("-------------------")



'''==========================================================================================
导入模型之前，必须重新再定义一遍变量。

但是并不需要全部变量都重新进行定义，只定义我们需要的变量就行了。

也就是说，你所定义的变量一定要在 checkpoint 中存在；
但不是所有在checkpoint中的变量，你都要重新定义。'''

