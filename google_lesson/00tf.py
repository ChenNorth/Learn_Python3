# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#https://developers.google.cn/machine-learning/crash-course/exercises

'''预热
* 编程练习：(TensorFlow) Hello World
* 编程练习：TensorFlow 编程概念
* 编程练习：创建和操控张量
* 编程练习：Pandas 简介'''

# 准备工作：Hello World==================================================================

print("1111111111111111")
#学习目标：在浏览器中运行 TensorFlow 程序。
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的TF警告

c = tf.constant('Hello, world!')
with tf.Session() as sess:
    print(sess.run(c))

'''TensorFlow 编程概念
学习目标：
    * 学习 TensorFlow 编程模型的基础知识，重点了解以下概念：
        张量
        指令
        图
        会话
    * 构建一个简单的 TensorFlow 程序，使用该程序绘制一个默认图并创建一个运行该图的会话'''

'''概念概览
TensorFlow 的名称源自张量，张量是任意维度的数组。
借助 TensorFlow，您可以操控具有大量维度的张量。
即便如此，在大多数情况下，您会使用以下一个或多个低维张量：
    * 标量是零维数组（零阶张量）。例如，\'Howdy\' 或 5
    * 矢量是一维数组（一阶张量）。例如，[2, 3, 5, 7, 11] 或 [5]
    * 矩阵是二维数组（二阶张量）。例如，[[3.1, 8.2, 5.9][4.3, -2.7, 6.5]]

TensorFlow 指令会创建、销毁和操控张量。典型 TensorFlow 程序中的大多数代码行都是指令。

TensorFlow 图（也称为计算图或数据流图）是一种图数据结构。
很多 TensorFlow 程序由单个图构成，但是 TensorFlow 程序可以选择创建多个图。
图的节点是指令；图的边是张量。张量流经图，在每个节点由一个指令操控。
一个指令的输出张量通常会变成后续指令的输入张量。
TensorFlow 会实现延迟执行模型，意味着系统仅会根据相关节点的需求在需要时计算节点。

张量可以作为常量或变量存储在图中。
您可能已经猜到，常量存储的是值不会发生更改的张量，而变量存储的是值会发生更改的张量。
不过，您可能没有猜到的是，常量和变量都只是图中的一种指令。
常量是始终会返回同一张量值的指令。变量是会返回分配给它的任何张量的指令。

要定义常量，请使用 tf.constant 指令，并传入它的值。例如：
  x = tf.constant([5.2])

同样，您可以创建如下变量：
  y = tf.Variable([5])

或者，您也可以先创建变量，然后再如下所示地分配一个值（注意：您始终需要指定一个默认值）：
  y = tf.Variable([0])
  y = y.assign([5])

定义一些常量或变量后，您可以将它们与其他指令（如 tf.add）结合使用。
在评估 tf.add 指令时，它会调用您的 tf.constant 或 tf.Variable 指令，
以获取它们的值，然后返回一个包含这些值之和的新张量。

图必须在 TensorFlow 会话中运行，会话存储了它所运行的图的状态：
将 tf.Session() 作为会话：
  initialization = tf.global_variables_initializer()
  print y.eval()

在使用 tf.Variable 时，您必须在会话开始时调用 tf.global_variables_initializer，
以明确初始化这些变量，如上所示。

注意：会话可以将图分发到多个机器上执行（假设程序在某个分布式计算框架上运行）。

总结
TensorFlow 编程本质上是一个两步流程：
    * 将常量、变量和指令整合到一个图中。
    * 在一个会话中评估这些常量、变量和指令。'''

print("2222222222222222")
# 创建一个简单的 TensorFlow 程序===================================
import tensorflow as tf

import matplotlib.pyplot as plt # 数据集可视化。
import numpy as np              # 低级数字 Python 库。
import pandas as pd             # 较高级别的数字 Python 库。

#TensorFlow 提供了一个默认图。
# 不过，我们建议您明确创建自己的 Graph，以便跟踪状态
# （例如，您可能希望在每个单元格中使用一个不同的 Graph）。
# Create a graph.
g = tf.Graph()

# # Establish the graph as the "default" graph.
# with g.as_default():
#   # Assemble a graph consisting of the following three operations:
#   #   * Two tf.constant operations to create the operands.
#   #   * One tf.add operation to add the two operands.
#   x = tf.constant(8, name="x_const")
#   y = tf.constant(5, name="y_const")
#   sum = tf.add(x, y, name="x_y_sum")


#   # Now create a session.
#   # The session will run the default graph.
#   with tf.Session() as sess:
#     print(sum.eval())

print("3333333333333333")
'''练习：引入第三个运算数
修改上面的代码列表，以将三个整数（而不是两个）相加：
1.定义第三个标量整数常量 z，并为其分配一个值 4。
2.将 sum 与 z 相加，以得出一个新的和。
    提示：请参阅有关 tf.add() 的 API 文档，了解有关其函数签名的更多详细信息。
3.重新运行修改后的代码块。该程序是否生成了正确的总和？'''
# Establish our graph as the "default" graph.
with g.as_default():
  # Assemble a graph consisting of three operations. 
  # (Creating a tensor is an operation.)
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  sum = tf.add(x, y, name="x_y_sum")
  
  # Task 1: Define a third scalar integer constant z.
  z = tf.constant(4, name="z_const")
  # Task 2: Add z to `sum` to yield a new sum.
  new_sum = tf.add(sum, z, name="x_y_z_sum")

  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    # Task 3: Ensure the program yields the correct grand total.
    print(new_sum.eval())
