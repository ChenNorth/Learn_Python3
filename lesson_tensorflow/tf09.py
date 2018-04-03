# -*- coding: utf-8 -*-

__author__ = "cs-pc/2018-03-26"

'''参考：
https://blog.csdn.net/jerr__y/article/details/78594494
TensorFlow入门（九）使用 tf.train.Saver()保存模型
'''



'''==========================================================================================
关于模型保存的一点心得
    saver = tf.train.Saver(max_to_keep=3)
在定义 saver 的时候一般会定义最多保存模型的数量，
一般来说，
    *如果模型本身很大，我们需要考虑到硬盘大小。
    *如果你需要在当前训练好的模型的基础上进行 fine-tune，那么尽可能多的保存模型，
        后继 fine-tune 不一定从最好的 ckpt 进行，因为有可能一下子就过拟合了。
        但是如果保存太多，硬盘也有压力呀。
如果只想保留最好的模型，方法就是每次迭代到一定步数就在验证集上计算一次 accuracy 或者 f1 值，
如果本次结果比上次好才保存新的模型，否则没必要保存。

如果你想用不同 epoch 保存下来的模型进行融合的话，3到5 个模型已经足够了，
假设这各融合的模型成为 M，而最好的一个单模型称为 m_best, 
这样融合的话对于M 确实可以比 m_best 更好。
但是如果拿这个模型和其他结构的模型再做融合的话，M 的效果并没有 m_best 好，
因为M 相当于做了平均操作，减少了该模型的“特性”。

但是又有一种新的融合方式，就是利用调整学习率来获取多个局部最优点，
就是当 loss 降不下了，保存一个 ckpt， 然后开大学习率继续寻找下一个局部最优点，
然后用这些 ckpt 来做融合，还没试过，
单模型肯定是有提高的，就是不知道还会不会出现上面再与其他模型融合就没提高的情况。'''


'''==========================================================================================
如何使用 tf.train.Saver() 来保存模型
之前一直出错，主要是因为坑爹的编码问题。所以要注意文件的路径绝对不不要出现什么中文呀。'''
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Create some variables.
v1 = tf.Variable([1.0, 2.3], name="v1")
v2 = tf.Variable(55.5, name="v2")

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

ckpt_path = './ckpt/test-model.ckpt'
# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
sess.run(init_op)
save_path = saver.save(sess, ckpt_path, global_step=1)
print("Model saved in file: %s" % save_path)
print("-------------------")



'''==========================================================================================
注意，在上面保存完了模型之后。应该把 kernel restart 之后才能使用下面的模型导入。
否则会因为两次命名 “v1” 而导致名字错误。
先运行tf09.py，再运行tf092.py'''

