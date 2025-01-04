import numpy as np
import pandas as pd

# loc iloc测试
data = pd.DataFrame(np.arange(25).reshape(5,5), index=list('abcde'), columns=list('ABCDE'))
# print(data)
#     A   B   C   D   E
# a   0   1   2   3   4
# b   5   6   7   8   9
# c  10  11  12  13  14
# d  15  16  17  18  19
# e  20  21  22  23  24
# print(data.loc['e'])
# print(data.iloc[1])
# print(data.iloc[:1])
# print(data.iloc[:])
# print(data.loc[:,['A']])
print(data.loc[:, ['A', 'B']])