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

'''https://developers.google.cn/machine-learning/crash-course/
first-steps-with-tensorflow/programming-exercises'''

'''三项练习：
1.Pandas 简介。 Pandas 是用于进行数据分析和建模的重要库，广泛应用于 TensorFlow 编码。
该教程提供了您学习本课程所需的全部 Pandas 信息。如果您已了解 Pandas，则可以跳过此练习。
2.使用 TensorFlow 的起始步骤。此练习介绍了线性回归。
3.合成特征和离群值。此练习介绍了合成特征，以及输入离群值会造成的影响。'''

'''Pandas 简介
学习目标：
1.大致了解 pandas 库的 DataFrame 和 Series 数据结构
2.存取和处理 DataFrame 和 Series 中的数据
3.将 CSV 数据导入 pandas 库的 DataFrame
4.对 DataFrame 重建索引来随机打乱数据
pandas 是一种列存数据分析 API。它是用于处理和分析输入数据的强大工具，
很多机器学习框架都支持将 pandas 数据结构作为输入。
虽然全方位介绍 pandas API 会占据很长篇幅，但它的核心概念非常简单，
我们会在下文中进行说明。
有关更完整的参考，请访问 pandas 文档网站，其中包含丰富的文档和教程资源。'''

import pandas as pd

print(pd.__version__)

# 基本概念====================================================================

#pandas 中的主要数据结构被实现为以下两类：
#DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
#Series，它是单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。
print("1111111111111111")
print(pd.Series(['San Francisco', 'San Jose', 'Sacramento']))

print("2222222222222222")
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
print(pd.DataFrame({ 'City name': city_names, 'Population': population }))
#可以将映射 string 列名称的 dict 传递到它们各自的 Series，从而创建DataFrame对象。
#如果 Series 在长度上不一致，系统会用特殊的 NA/NaN 值填充缺失的值

print("3333333333333333")
#california_housing_dataframe = pd.read_csv(
#    "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv",
#    sep=",")
california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())
#使用 DataFrame.describe 来显示关于 DataFrame 的有趣统计信息。

#另一个实用函数是 DataFrame.head，它显示 DataFrame 的前几个记录：
print("4444444444444444")
print(california_housing_dataframe.head())

#pandas 的另一个强大功能是绘制图表。
#例如，借助 DataFrame.hist，您可以快速了解一个列中值的分布
print("5555555555555555")
california_housing_dataframe.hist('housing_median_age') #可输出柱状图

# 访问数据====================================================================

#可以使用熟悉的 Python dict/list 指令访问 DataFrame 数据：
print("6666666666666666")
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(type(cities['City name']))
print(cities['City name'])
print("-------")
print(type(cities['City name'][1]))
print(cities['City name'][1])
print("-------")
print(type(cities[0:2]))
print(cities[0:2])

# 操控数据====================================================================

print("7777777777777777")
print(population / 1000.)

#NumPy 是一种用于进行科学计算的常用工具包。
#pandas Series 可用作大多数 NumPy 函数的参数：
print("8888888888888888")
import numpy as np
print(np.log(population))

#对于更复杂的单列转换，您可以使用 Series.apply。
#像 Python 映射函数一样，Series.apply 将以参数形式接受 lambda 函数，
#而该函数会应用于每个值。
#下面的示例创建了一个指明 population 是否超过 100 万的新 Series：
print("9999999999999999")
print(population.apply(lambda val: val > 1000000))

#DataFrames 的修改方式也非常简单。
#例如，以下代码向现有 DataFrame 添加了两个 Series：
print("aaaaaaaaaaaaaaaa")
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)

# 练习 1====================================================================

#通过添加一个新的布尔值列（当且仅当以下两项均为 True 时为 True）修改 cities 表格：
#   城市以圣人命名。
#   城市面积大于 50 平方英里。
print("bbbbbbbbbbbbbbbb")
cities['Is wide and has saint name'] = \
    (cities['Area square miles'] > 50) & \
    cities['City name'].apply(lambda name: name.startswith('San'))
print(cities)

# 索引====================================================================

#Series 和 DataFrame 对象也定义了 index 属性，
#该属性会向每个 Series 项或 DataFrame 行赋一个标识符值。
#默认情况下，在构造时，pandas 会赋可反映源数据顺序的索引值。
#索引值在创建后是稳定的；也就是说，它们不会因为数据重新排序而发生改变。
print("cccccccccccccccc")
print(city_names.index)
print(cities.index)
#调用 DataFrame.reindex 以手动重新排列各行的顺序。
#例如，以下方式与按城市名称排序具有相同的效果：
print(cities.reindex([2, 0, 1]))

print("dddddddddddddddd")
#重建索引是一种随机排列 DataFrame 的绝佳方式。
#在下面的示例中，我们会取用类似数组的索引，
#然后将其传递至 NumPy 的 random.permutation 函数，该函数会随机排列其值的位置。
#如果使用此重新随机排列的数组调用 reindex，会导致 DataFrame 行以同样的方式随机排列。
#尝试多次运行以下单元格！
print(cities.reindex(np.random.permutation(cities.index)))
print(cities.reindex(np.random.permutation(cities.index)))
print(cities.reindex(np.random.permutation(cities.index)))

# 练习 2====================================================================

#reindex 方法允许使用未包含在原始 DataFrame 索引值中的索引值。请试一下
print("eeeeeeeeeeeeeeee")
#如果您的 reindex 输入数组包含原始 DataFrame 索引值中没有的值，
#reindex 会为此类“丢失的”索引添加新行，并在所有对应列中填充 NaN 值：
print(cities.reindex([0, 4, 5, 2]))
#这种行为是可取的，因为索引通常是从实际数据中提取的字符串
#在这种情况下，如果允许出现“丢失的”索引，您将可以轻松使用外部列表重建索引，
#因为您不必担心会将输入清理掉。

