# -*- coding: utf-8 -*-



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

import numpy as np
import tensorflow as tf

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from PIL import Image



__author__ = "cs-pc/2018-03-26"

'''参考：
如何优雅地用TensorFlow预测时间序列：TFTS库详细教程
https://www.leiphone.com/news/201708/4b1oCSXThGVwyVZg.html
之
使用LSTM预测单变量时间序列
.改进：显示PNG格式图片
.注意：以下LSTM模型的例子必须使用TensorFlow最新的开发版的源码。
    具体来说，要保证
    “from tensorflow.contrib.timeseries.python.timeseries.estimators
     import TimeSeriesRegressor”可以成功执行。

.给出两个用LSTM预测时间序列模型的例子，分别是:
    .train_lstm.py：在LSTM中进行单变量的时间序列预测（bp13.py）；
    .train_lstm_multivariate.py：使用LSTM进行多变量时间序列预测。
'''



class _LSTMModel(ts_model.SequentialTimeSeriesModel):
    """一个使用单一RNN神经元创建的时间序列建模的例子"""  

    def __init__(self, num_units, num_features, dtype=tf.float32):
        """初始化/配置模型对象。
        注意，我们不在这里开始建图。相反，这个对象是TensorFlow图的配置工厂，
        它依靠一个Estimator运行。
        Args:
            num_units: 在模型中的lstm神经单元数量。
            num_features: 时间序列的维度（单位时长的特征）。
            dtype: 浮点值点的数据类型使用。
        """
        super(_LSTMModel, self).__init__(
            # 预先注册我们将输出的度量值（这里只是一个平均值）
            train_output_names=["mean"],
            predict_output_names=["mean"],
            num_features=num_features,
            dtype=dtype)
        self._num_units = num_units
        # 由 initialize_graph() 填充数据
        self._lstm_cell = None
        self._lstm_cell_run = None
        self._predict_from_lstm_output = None

    def initialize_graph(self, input_statistics):
        """Save templates for components, which can then be used repeatedly.
        This method is called every time a new graph is created. It's safe to start
        adding ops to the current default graph here, but the graph should be
        constructed from scratch.
        Args:
        input_statistics: A math_utils.InputStatistics object.
        """
        super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)
        self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
        # Create templates so we don't have to worry about variable reuse.
        self._lstm_cell_run = tf.make_template(
            name_="lstm_cell",
            func_=self._lstm_cell,
            create_scope_now_=True)
        # Transforms LSTM output into mean predictions.
        self._predict_from_lstm_output = tf.make_template(
            name_="predict_from_lstm_output",
            func_=lambda inputs: tf.layers.dense(inputs=inputs, units=self.num_features),
            create_scope_now_=True)

    def get_start_state(self):
        """Return initial state for the time series model."""
        return (
            # Keeps track of the time associated with this state for error checking.
            tf.zeros([], dtype=tf.int64),
            # The previous observation or prediction.
            tf.zeros([self.num_features], dtype=self.dtype),
            # The state of the RNNCell (batch dimension removed since this parent
            # class will broadcast).
            [tf.squeeze(state_element, axis=0)
            for state_element
            in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])

    def _transform(self, data):
        """Normalize data based on input statistics to encourage stable training."""
        mean, variance = self._input_statistics.overall_feature_moments
        return (data - mean) / variance

    def _de_transform(self, data):
        """Transform data back to the input scale."""
        mean, variance = self._input_statistics.overall_feature_moments
        return data * variance + mean

    def _filtering_step(self, current_times, current_values, state, predictions):
        """Update model state based on observations.
        Note that we don't do much here aside from computing a loss. In this case
        it's easier to update the RNN state in _prediction_step, since that covers
        running the RNN both on observations (from this method) and our own
        predictions. This distinction can be important for probabilistic models,
        where repeatedly predicting without filtering should lead to low-confidence
        predictions.
        Args:
        current_times: A [batch size] integer Tensor.
        current_values: A [batch size, self.num_features] floating point Tensor
            with new observations.
        state: The model's state tuple.
        predictions: The output of the previous `_prediction_step`.
        Returns:
        A tuple of new state and a predictions dictionary updated to include a
        loss (note that we could also return other measures of goodness of fit,
        although only "loss" will be optimized).
        """
        state_from_time, prediction, lstm_state = state
        with tf.control_dependencies(
                [tf.assert_equal(current_times, state_from_time)]):
            transformed_values = self._transform(current_values)
            # Use mean squared error across features for the loss.
            predictions["loss"] = tf.reduce_mean(
                (prediction - transformed_values) ** 2, axis=-1)
            # Keep track of the new observation in model state. It won't be run
            # through the LSTM until the next _imputation_step.
            new_state_tuple = (current_times, transformed_values, lstm_state)
        return (new_state_tuple, predictions)

    def _prediction_step(self, current_times, state):
        """Advance the RNN state using a previous observation or prediction."""
        _, previous_observation_or_prediction, lstm_state = state
        lstm_output, new_lstm_state = self._lstm_cell_run(
            inputs=previous_observation_or_prediction, state=lstm_state)
        next_prediction = self._predict_from_lstm_output(lstm_output)
        new_state_tuple = (current_times, next_prediction, new_lstm_state)
        return new_state_tuple, {"mean": self._de_transform(next_prediction)}

    def _imputation_step(self, current_times, state):
        """Advance model state across a gap."""
        # Does not do anything special if we're jumping across a gap. More advanced
        # models, especially probabilistic ones, would want a special case that
        # depends on the gap size.
        return state

    def _exogenous_input_step(
            self, current_times, current_exogenous_regressors, state):
        """Update model state based on exogenous regressors."""
        raise NotImplementedError(
            "Exogenous inputs are not implemented for this example.")


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    '''用函数加噪声的方法生成一个模拟的时间序列数据,
    此处y对x的函数关系比之前复杂，因此更适合用LSTM这样的模型找出其中的规律。'''
    x = np.array(range(1000))
    noise = np.random.uniform(-0.2, 0.2, 1000)
    y = np.sin(np.pi * x / 50 ) + np.cos(np.pi * x / 50) + np.sin(np.pi * x / 25) + noise

    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
    }

    reader = NumpyReader(data)

    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader, batch_size=4, window_size=100)

    # 定义一个LSTM模型
    estimator = ts_estimators.TimeSeriesRegressor(
        model=_LSTMModel(num_features=1, num_units=128),
        optimizer=tf.train.AdamOptimizer(0.001))
    '''num_features = 1表示单变量时间序列，即每个时间点上观察到的量只是一个单独的数值。
    num_units=128表示使用隐层为128大小的LSTM模型。'''

    estimator.train(input_fn=train_input_fn, steps=2000)
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
    # 评估后开始预测：在训练时，我们在已有的1000步的观察量的基础上向后预测200步：
    (predictions,) = tuple(estimator.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=200)))

    observed_times = evaluation["times"][0]
    observed = evaluation["observed"][0, :, :]
    evaluated_times = evaluation["times"][0]
    evaluated = evaluation["mean"][0]
    predicted_times = predictions['times']
    predicted = predictions["mean"]

    plt.figure(figsize=(15, 5))
    plt.axvline(999, linestyle="dotted", linewidth=4, color='r')
    observed_lines = plt.plot(observed_times, observed, label="observation", color="k")
    evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color="g")
    predicted_lines = plt.plot(predicted_times, predicted, label="prediction", color="r")
    plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],
                loc="upper left")
    plt.savefig('predict_result2.png')

    image = Image.open("predict_result2.png")
    image.show()