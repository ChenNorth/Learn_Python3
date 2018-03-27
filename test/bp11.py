# -*- coding: utf-8 -*-



from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

from PIL import Image



__author__ = "cs-pc/2018-03-26"

'''参考：
如何优雅地用TensorFlow预测时间序列：TFTS库详细教程
https://www.leiphone.com/news/201708/4b1oCSXThGVwyVZg.html
之
从Numpy数组中读入时间序列数据：
改进：显示PNG格式图片
'''



x = np.array(range(1000))
noise = np.random.uniform(-0.2, 0.2, 1000)
y = np.sin(np.pi * x / 100) + x / 200. + noise
plt.ylabel("yyy")
plt.xlabel("xxx")
plt.title("xixihaha")
plt.tight_layout()
plt.plot(x, y)
plt.legend()
plt.savefig('timeseries_y.png')
# plt.show() 因是agg模式无法绘图

image = Image.open("timeseries_y.png")
image.show()

data = {
    tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
    tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
}

reader = NumpyReader(data)

with tf.Session() as sess:
    full_data = reader.read_full()
    # 调用read_full方法会生成读取队列
    # 要用tf.train.start_queue_runners启动队列才能正常进行读取
    # 主线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #print(sess.run(full_data))
    coord.request_stop()

'''随机选取窗口长度为window_size的序列，并包装成batch_size大小的batch数据。
换句话说，一个batch内共有batch_size个序列，每个序列的长度为window_size。'''
train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
    reader, batch_size=2, window_size=10)

with tf.Session() as sess:
    batch_data = train_input_fn.create_batch()
    # 主线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    one_batch = sess.run(batch_data[0])
    coord.request_stop()

print('one_batch_data:', one_batch)


