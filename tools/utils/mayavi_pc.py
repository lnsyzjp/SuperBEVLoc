# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:55:54 2023

@author: ZJP
"""

import numpy as np
from mayavi import mlab

from read_bin import *

bin_file1 = r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\kitti\000000.bin"
bin_file2 = r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\nclt\1328454074543495.bin"
bin_file3 = r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\mulran\1564719512892604519.bin"

# pc1 = load_kitti_bin(bin_file1)
pc1 = load_nclt_bin(bin_file2)
# pc1 = load_mulran_bin(bin_file3)


x = pc1[:, 0]  # x position of point
y = pc1[:, 1]  # y position of point
z = pc1[:, 2]  # z position of point
r = pc1[:, 3]  # reflectance value of point

d = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # Map Distance from sensor


 
fig = mlab.figure(bgcolor=(1, 1, 1), size=(700, 500))
mlab.points3d(x, y, z,
              d,  # Values used for Color
              mode="point",
              colormap='spectral',  # 'bone', 'copper', 'gnuplot', 'spectral', 'summer'
              # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
              figure=fig)
mlab.show()