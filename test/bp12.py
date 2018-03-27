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
使用AR模型预测时间序列
改进：显示PNG格式图片
'''



def main(_):
    x = np.array(range(1000))
    noise = np.random.uniform(-0.2, 0.2, 1000)
    y = np.sin(np.pi * x / 100) + x / 200. + noise
    plt.plot(x, y)
    plt.savefig('timeseries_y.jpg')

    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
    }

    reader = NumpyReader(data)

    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader, batch_size=16, window_size=40)

    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=200, input_window_size=30, output_window_size=10,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

    ar.train(input_fn=train_input_fn, steps=6000)

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=250)))

    plt.figure(figsize=(15, 5))
    plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
    plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')
    plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    plt.savefig('predict_result.png')

    image = Image.open("predict_result.png")
    image.show()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
    '''A。如果当前的python 文件中没有 main 函数，则可以通过 tf.app.run(待执行的函数名称)来实现
    B。大多数情况下python 文件中包含了 main 函数，则直接可以通过 tf.app.run() 实现即可！'''
    print("hhhhhhhhhhhhhhh")







