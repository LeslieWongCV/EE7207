# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 12:41 下午
# @Author  : Yushuo Wang
# @FileName: print_gaussian.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.mplot3d

x, y = np.mgrid[-1:1:200j, -1:1:200j]
z=(1/2*math.pi*3**2)*np.exp(-(x**2+y**2)/2*3**2)
ax = plt.subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)#绘面

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('weight')
plt.show()