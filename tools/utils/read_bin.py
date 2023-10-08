
# !/usr/bin/python


import numpy as np
'''
hits = load_nclt_bin(nclt_bin_file)
hits = load_kitti_bin(kitti_bin_file)
hits = load_mulran_bin(mulran_bin_file)
'''

velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodatasize = 8

def data2xyzi(data, flip=True):
    xyzil = data.view(velodatatype)
    xyz = np.hstack(
        [xyzil[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
    xyz = xyz * 0.005 - 100.0

    if flip:
        R = np.eye(3)
        R[2, 2] = -1
        xyz = np.matmul(xyz, R)
    # print(np.shape(xyz))
    xyzi = np.column_stack((xyz,xyzil['i']))
    return xyzi

def load_nclt_bin(velofile):
    return data2xyzi(np.fromfile(velofile))
def load_kitti_bin(kitti_bin_file):
    data = np.fromfile(kitti_bin_file, dtype=np.float32, count=-1).reshape([-1, 4])
    return data
def load_mulran_bin(mulran_bin_file):
    data = np.fromfile(mulran_bin_file, dtype=np.float32, count=-1).reshape([-1, 4])
    return data
if __name__ == '__main__':
    kitti_bin_file = r'../../data/kitti/000000.bin'
    nclt_bin_file = r'../../data/nclt/1326030975726043.bin'
    mulran_bin_file = r'../../data/mulran/1561000444390857630.bin'
    hits = load_nclt_bin(mulran_bin_file)
    # hits2 = load_kitti_bin(kitti_bin_file)
    # print(hits[1991][3])
    # print(hits)
    print(np.shape(hits))
