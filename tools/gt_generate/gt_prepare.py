"""
读取各数据集的pose或gps文件，构建训练集
['Query':'filename','Positives':[...],'Negatives':[...]]
Oxford RobotCar: 10m内为pos，50m外为negs
Kitti:
                11            3
0    -4.440892e-16 -5.551115e-17
1     8.586941e-01  4.690294e-02
2     1.716275e+00  9.374345e-02

Nclt
2012-01-08， 2812
2012-02-05， 2823
2012-06-15，
2013-02-23，
2013-04-05
"""
import os
import numpy as np
import pickle
import random
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
kitti_pos_root = 'E:\Datasets\\1_KITTI\dataset\poses\\'
nclt_pos_root = 'E:\Datasets\\2_NCLT\\'
mulran_pos_root = 'E:\Datasets\\3_MulRan\\'

def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['x','y']])
    ind_nn = tree.query_radius(df_centroids[['x','y']],r=10)
    ind_r = tree.query_radius(df_centroids[['x','y']], r=50)
    queries = {}
    for i in range(len(ind_nn)):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i],[i]).tolist()
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(),ind_r[i]).tolist()
        random.shuffle(negatives)
        queries[i] = {"query":query,
                      "positives":positives,"negatives":negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)

def trans_kittipos_to_gt(bin_path,filename):
    bins = os.listdir(bin_path)
    pos = pd.read_csv(kitti_pos_root+ filename,header=None,sep=' ')

    pos.iloc[:, [3]] = pos.iloc[:, [3]] * -1#坐标系转换
    pos_t = pos.iloc[:, [11,3]]#取x,y

    # pos_t.insert(0, 'file', range(1, 1 + len(pos_t)))
    pos_t.insert(0,'file',bins)
    pos_t.rename(columns = {11:'x',3:'y'},inplace=True)
    return pos_t
def trans_ncltpos_to_gt(bin_path,filename,num):
    bins = os.listdir(bin_path)

    new_bins = []
    pos = pd.read_csv(nclt_pos_root + filename, header=None,sep=',')
    pos_t = pos.iloc[:, [1, 2]]  # 取x,y

    pos_t.rename(columns={1: 'x', 2: 'y'}, inplace=True)
    scan_index = []
    num_scan = int(num/10)
    scale_pos_scan = round(len(pos_t)/num_scan,5)

    for i in range(num_scan):
        print(i)
        # print(int(i * scale_pos_scan),scale_pos_scan,i)
        scan_index.append(int(i * scale_pos_scan))
        new_bins.append(bins[int(i * 10)])


    #

    pos_t = pos_t.iloc[scan_index, :]  # 取x,y
    pos_t.insert(0, 'file', new_bins)
    # print(pos_t)
    return pos_t


def trans_mulranpos_to_gt(bin_path,filename,num):
    bins = os.listdir(bin_path)
    pos = pd.read_csv(mulran_pos_root + filename, header=None, sep=',')
    pos_t = pos.iloc[:, [4, 8]]  # 取x,y

    pos_t.rename(columns={4: 'x', 8: 'y'}, inplace=True)
    print('pos的数量：',len(pos_t))
    scan_index = []
    scale = round(len(pos_t)/num,5)
    print('缩放尺度：',scale)
    # num_scan = round(num/10)
    for i in range(0,num):
        scan_index.append(int(i * scale+1))

    # print(scan_index)
    print(len(scan_index))
    pos_t = pos_t.iloc[scan_index, :]  # 取x,y
    pos_t.insert(0, 'file', bins)
    # print(len(pos_t))
    return pos_t
if __name__ == "__main__":
    bin_path = "E:\\Datasets\\1_KITTI\\dataset\\sequences\\08\\velodyne\\"
    # bin_path = "E:\Datasets\\2_NCLT\\2013-04-05\\velodyne_sync\\"
    # bin_path = "E:\Datasets\\3_MulRan\DCC02\Ouster\\"
    pos_t = trans_kittipos_to_gt(bin_path,"08.txt")#00，02，05，06,08
    plt.figure()
    plt.plot(pos_t['x'][:2000],pos_t['y'][:2000])
    plt.show()
    # 2012-01-08，2012-02-05，2012-06-15，2013-02-23，2013-04-05
    #28127,28239,16545,24022,20901
    # pos_t = trans_ncltpos_to_gt(bin_path,"2013-04-05/groundtruth_2013-04-05.csv", num = 20901)

    #78672
    #kaist01,8226,kaist02,8941,    DCC01,5542,DCC02,7561
    # pos_t = trans_mulranpos_to_gt(bin_path,"DCC02/global_pose.csv",7561)
    # print(pos_t)
    # construct_query_dict(pos_t, 'DCC02.pickle')