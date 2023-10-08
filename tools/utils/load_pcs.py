
import os
import sys
import pickle
import numpy as np
from tools.utils.read_bin import *
import random
import config as cfg
from tools.utils.bev_image import *
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
config_filename = '../config//config.yaml'
config = yaml.safe_load(open(config_filename))
DATASET_FOLDER = config["data_root"]["DATASET_FOLDER"]
def get_queries_dict(filename):
    #gt
    # key:{'query':file,'positives':[files],'negatives:[files]}
    with open(filename, 'rb') as handle:
        queries = pickle.load(handle)
        print("Queries Loaded.")
        return queries
def get_sets_dict(filename):
    #database,query
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        print("Trajectories Loaded.")
        return trajectories

def load_pcbev_file(data_folder,filename):
    '''
    加载单个点云 bev
    '''
    # returns Nx3 matrix
    #os.path.join(cfg.DATASET_FOLDER, filename)
    filename = os.path.join(data_folder,filename)
    # print(filename)
    pc = load_kitti_bin(filename)
    im = gen_bev_img(pc)
    # pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return im

def load_pcbev_files(data_folder,filenames):
    '''
    加载列表中所有点云
    '''
    ims = []
    for filename in filenames:
        # print(filename)
        im = load_pcbev_file(data_folder,filename)
        ims.append(im)
    ims = np.array(ims)
    return ims

def rotate_point_cloud(batch_data):
    """ 旋转数据增强
    Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi) - np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data



def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):

    # get query tuple for dictionary entry
    # return list [query,positives,negatives]
    #获取指定的bev图
    query = load_pcbev_file(DATASET_FOLDER,dict_value["query"])  # Nx3
    #打乱了pos的顺序
    random.shuffle(dict_value["positives"])
    #加载pos的若干个点云--------------------------------------------------------------------
    pos_files = []
    for i in range(num_pos):
        pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
    positives = load_pcbev_files(DATASET_FOLDER,pos_files)
    #------------------------------------------------------------------------------------
    # hard_neg=[]加载negs的若干个点云-------------------------------------------------------
    neg_files = []
    neg_indices = []
    if (len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    # ------------------------------------------------------------------------------------
    else:
    # hard_neg != []加载negs的若干个点云-------------------------------------------------------
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while (len(neg_files) < num_neg):

            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(
                    QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    # ------------------------------------------------------------------------------------
    negatives = load_pcbev_files(DATASET_FOLDER,neg_files)

    if other_neg is False:
        return [query, positives, negatives]
    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys()) - set(neighbors))
        random.shuffle(possible_negs)

        if (len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]

        neg2 = load_pcbev_file(DATASET_FOLDER,QUERY_DICT[possible_negs[0]]["query"])

        return [query, positives, negatives, neg2]

if __name__ == "__main__":
    filename1 = '../../data/kitti/kitti00.pickle'
    filename2 = '../../data/kitti/kitti00_evaluation_database.pickle'
    queries = get_queries_dict(filename1)
    trajectories = get_sets_dict(filename2)

    print(queries[0])
    print(trajectories[0])

    print(os.path.join(DATASET_FOLDER, queries[0]['query']))
    im = load_pcbev_file(DATASET_FOLDER,queries[0]['query'])
    # print(np.shape(im) )

    #get_query_tuple

    res = get_query_tuple(queries[0], 2, 18,
                    queries, hard_neg=[], other_neg=True)

    print(np.shape(res[2]))