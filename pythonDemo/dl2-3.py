import tensorflow as tf

X = tf.reshape(tf.range(20), (5, 4))
print(X)
# 降维
print(tf.reduce_sum(X))
# 沿0轴降维
print(tf.reduce_sum(X, axis=0))
# 沿1轴降维
print(tf.reduce_sum(X, axis=1))
# 保留维度
print(tf.reduce_sum(X, axis=1, keepdims=True))
# 求平均值
print(tf.reduce_mean(X))
# 等价于直接reduce
print(tf.reduce_sum(X, axis=[0,1]))

Y = tf.reshape(tf.range(60), (3,4,5))
print(Y)
print(tf.reduce_sum(Y, axis=2))

# 按照维度求平均值
print(Y)
print(tf.reduce_sum(Y, axis=0))
print(tf.reduce_mean(Y, axis=0))
print(tf.reduce_mean(tf.reduce_mean(Y, axis=0), axis=0))

# 手动求平均值
print(X)
print(tf.reduce_sum(X, axis=0) / X.shape[0])
print(X.shape[0])

# 保留维度，再求每个标量和平均值的差
print(X)
X_x = tf.reduce_mean(X, axis=1, keepdims=True)
print(X_x)
print(X - X_x)

# 累积总和
print(X)
print(tf.cumsum(X, axis=0))

# 点积
print('点积==============================>')
x = tf.range(4, dtype=tf.float32)
y = tf.ones(4, dtype=tf.float32)
print(x)
print(y)
print(tf.tensordot(x, y, axes=1))
# 等效于乘积相加
print(tf.reduce_sum(x * y))
# 矩阵向量积 (相当于0轴向量和目标向量做点积)
print('矩阵向量积==============================>')
A = tf.reshape(tf.range(20), (5, 4))
x = tf.range(4)
print(A)
print(x)
print(tf.linalg.matvec(A, x))

