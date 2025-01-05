import os
import pandas as pd
import numpy
import tensorflow as tf
# print(os.path.join('.', 'data'))
# os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
print('data_file:' + data_file)
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
# pandsa 读取文件测试
data = pd.read_csv(data_file)
# print(data)
#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000

# 通过平均值处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# print(inputs)
# numeric_only=True 仅处理数值类型的数据
inputs = inputs.fillna(inputs.mean(numeric_only=True))
# print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
X = tf.constant(inputs.to_numpy(dtype=float))
Y = tf.constant(outputs.to_numpy(dtype=float))
print(X)
print('========')
print(Y)


print('++++++++++++++++++++++++++++++++========')
# 练习 删除缺失值最多的列
data = pd.read_csv(data_file)
print(data)
column_name = data.isna().sum().idxmax()
print(column_name)
# 删除缺失值最多的列
data = data.drop(column_name, axis=1)
print(data)
tensor = tf.constant(data.to_numpy(dtype=int))
print(tensor)
