from ledoit_wolf import shrinkage
from SmoteCovPy import SmoteCov
import numpy as np
a = SmoteCov("iris.csv")
mc = a.get_minority_class_group()
matrix = np.array(mc)
s = shrinkage(matrix)
for i in range(len(s)):
    print(i)
    print(s[i])