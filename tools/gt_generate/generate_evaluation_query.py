
"""
Database，Query
NCLT：
Database：2012-01-08，
Query：2012-02-05.pickle,2012-06-15.pickle,2013-02-23.pickle,2013-04-05.pickle

KITTI：取半
1。00训练，其余测试
Database和Query取半
00，02，05，06,08

MulRan:
Database:DCC01.pickle,kaist01.pickle
Query:DCC02.pickle,kaist02.pickle
"""
import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from gt_prepare import *

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def construct_query_sets(database_locations, query_locations):
    database_tree = KDTree(database_locations[['x', 'y']])
    database={}
    query = {}
    pos_num = 0
    for k in range(len(database_locations)):
        database[k] = {'file': database_locations.iloc[k]['file'],
                      'x': database_locations.iloc[k]['x'],
                      'y': database_locations.iloc[k]['y'],}


    for key in range(len(query_locations)):
    #     # print(query_locations.iloc[key])
        coor = np.array([[query_locations.iloc[key]["x"], query_locations.iloc[key]["y"]]])
        index = database_tree.query_radius(coor, r=10)
        query[key] = {'file': query_locations.iloc[key]['file'],
                      'x': query_locations.iloc[key]['x'],
                      'y': query_locations.iloc[key]['y'],
                      'gt':index[0].tolist()}
        if query[key]['gt'] != []:
            pos_num+=1
    return database,query




if __name__ == "__main__":
    kitti_pos_root = 'E:\Datasets\\1_KITTI\dataset\poses\\'
    nclt_pos_root = 'E:\Datasets\\2_NCLT\\'
    mulran_pos_root = 'E:\Datasets\\3_MulRan\\'

    '''
    Nclt
    2012-01-08，2012-02-05，2012-06-15，2013-02-23，2013-04-05
    28127,      28239,      16545,     24022,      20901
    '''
    # bin_path = "E:\Datasets\\2_NCLT\\2012-01-08\\velodyne_sync\\"
    # database_locations = trans_ncltpos_to_gt(bin_path,"2012-01-08/groundtruth_2012-01-08.csv", num=28127)
    # bin_path2 = "E:\Datasets\\2_NCLT\\2013-04-05\\velodyne_sync\\"
    # query_locations = trans_ncltpos_to_gt(bin_path2,"2013-04-05/groundtruth_2013-04-05.csv", num=20901)
    # database,query = construct_query_sets(database_locations, query_locations)
    # # output_to_file(database, '2012-01-08_evaluation_database.pickle')
    # output_to_file(query, '2013-04-05_evaluation_query.pickle')


    '''
    MulRan
    #kaist01,8226,kaist02,8941,    DCC01,5542,DCC02,7561
    '''
    bin_path = "E:\Datasets\\3_MulRan\DCC01\Ouster\\"
    bin_path2 = "E:\Datasets\\3_MulRan\DCC02\Ouster\\"
    database_locations = trans_mulranpos_to_gt(bin_path,"DCC01/global_pose.csv", 5542)
    query_locations = trans_mulranpos_to_gt(bin_path2,"DCC02/global_pose.csv", 7561)
    database,query = construct_query_sets(database_locations, query_locations)
    output_to_file(database, 'DCC01_evaluation_database.pickle')
    output_to_file(query, 'DCC02_evaluation_query.pickle')


    '''
    KITTI 
    00，02，05，06,08
    4541         4661         2761        1101         4071
    '''
    # rate = 0.65
    # bin_path = "E:\\Datasets\\1_KITTI\\dataset\\sequences\\08\\velodyne\\"
    #  # 00，02，05，06,08
    # kitti_data = trans_kittipos_to_gt(bin_path,"08.txt")
    # database_locations = kitti_data.iloc[:int(len(kitti_data)*rate), :]
    # query_locations = kitti_data.iloc[int(len(kitti_data)*rate):, :]
    # # print(len(database_locations))
    # database,query = construct_query_sets(database_locations, query_locations)
    # output_to_file(database, 'kitti08_evaluation_database.pickle')
    # output_to_file(query, 'kitti08_evaluation_query.pickle')
