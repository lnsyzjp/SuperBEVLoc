import os
import argparse
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

from tools.utils.load_pcs import *
from tools.utils.read_bin import *
import modules.lpr_loss as lpr_loss
import modules.superbevnet as sbev

from tensorboardX import SummaryWriter
import yaml

cudnn.enabled = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_filename = '../config//config.yaml'
config = yaml.safe_load(open(config_filename))
BATCH_NUM_QUERIES = config["TRAIN"]["BATCH_NUM_QUERIES"]
TRAIN_POSITIVES_PER_QUERY = config["TRAIN"]["TRAIN_POSITIVES_PER_QUERY"]
TRAIN_NEGATIVES_PER_QUERY = config["TRAIN"]["TRAIN_NEGATIVES_PER_QUERY"]
MAX_EPOCH = config["TRAIN"]["MAX_EPOCH"]
BASE_LEARNING_RATE = config["TRAIN"]["BASE_LEARNING_RATE"]
MOMENTUM = config["TRAIN"]["MOMENTUM"]
OPTIMIZER = config["TRAIN"]["OPTIMIZER"]
DECAY_STEP = config["TRAIN"]["DECAY_STEP"]
DECAY_RATE = config["TRAIN"]["DECAY_RATE"]
MARGIN_1 = config["TRAIN"]["MARGIN_1"]
MARGIN_2 = config["TRAIN"]["MARGIN_2"]
FEATURE_OUTPUT_DIM = config["global"]["FEATURE_OUTPUT_DIM"]

LOSS_FUNCTION = config["LOSS"]["LOSS_FUNCTION"]
TRIPLET_USE_BEST_POSITIVES = config["LOSS"]["TRIPLET_USE_BEST_POSITIVES"]
LOSS_LAZY = config["LOSS"]["LOSS_LAZY"]
LOSS_IGNORE_ZERO_BATCH = config["LOSS"]["LOSS_IGNORE_ZERO_BATCH"]
DATASET_FOLDER = config["data_root"]["DATASET_FOLDER"]
DATASET_FOLDER2 = config["data_root"]["DATASET_FOLDER2"]
TRAIN_FILE = config["data_root"]["TRAIN_FILE"]
# TEST_FILE = config["data_root"]["TEST_FILE"]
MODEL_FILENAME = config["global"]["MODEL_FILENAME"]
LOG_DIR = config["global"]["LOG_DIR"]
RESULTS_FOLDER = config["global"]["RESULTS_FOLDER"]

BN_INIT_DECAY = config["TRAIN"]["BN_INIT_DECAY"]
BN_DECAY_DECAY_RATE = config["TRAIN"]["BN_DECAY_DECAY_RATE"]
BN_DECAY_DECAY_STEP = config["TRAIN"]["BN_DECAY_DECAY_STEP"]
BN_DECAY_CLIP = config["TRAIN"]["BN_DECAY_CLIP"]
RESUME = config["TRAIN"]["RESUME"]

EVAL_POSITIVES_PER_QUERY = config["EVAL"]["EVAL_POSITIVES_PER_QUERY"]
EVAL_NEGATIVES_PER_QUERY = config["EVAL"]["EVAL_NEGATIVES_PER_QUERY"]
EVAL_BATCH_SIZE = config["EVAL"]["EVAL_BATCH_SIZE"]

EVAL_DATABASE_FILE = config["EVAL"]["EVAL_DATABASE_FILE"]
EVAL_QUERY_FILE = config["EVAL"]["EVAL_QUERY_FILE"]

OUTPUT_FILE = RESULTS_FOLDER + 'results.txt'
def evaluate():
    model = sbev.SuperBEV()
    model = model.to(device)

    resume_filename = LOG_DIR + "model.ckpt"  # "checkpoint.pth.tar"
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)

    print(evaluate_model(model))


def evaluate_model(model):
    DATABASE_SETS = get_sets_dict(EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(EVAL_QUERY_FILE)

    # print(DATABASE_SETS[0])
    # print(QUERY_SETS[0])

    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    DATABASE_VECTORS = []
    QUERY_VECTORS = []
#---------------------------------------------------------------------------------------
    # for i in range(len(DATABASE_SETS)):
    #     DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS[i]))
    DATABASE_VECTORS.append(get_latent_vectors(model, DATABASE_SETS,DATASET_FOLDER))
    # for j in range(len(QUERY_SETS)):
    #     QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS[j]))
    QUERY_VECTORS.append(get_latent_vectors(model, QUERY_SETS,DATASET_FOLDER2))
    # ---------------------------------------------------------------------------------------


    pair_recall, pair_similarity, pair_opr = get_recall(DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
    recall += np.array(pair_recall)
    count += 1
    one_percent_recall.append(pair_opr)
    for x in pair_similarity:
        similarity.append(x)

    print()
    ave_recall = recall / count
    # print(ave_recall)

    # print(similarity)
    average_similarity = np.mean(similarity)
    # print(average_similarity)

    ave_one_percent_recall = np.mean(one_percent_recall)
    # print(ave_one_percent_recall)

    with open(OUTPUT_FILE, "a") as output:
        output.write("Results:\n")
        output.write("Average Recall @N:\n")
        output.write(str(ave_recall))
        output.write("\n\n")
        output.write("Average Similarity:\n")
        output.write(str(average_similarity))
        output.write("\n\n")
        output.write("Average Top 1% Recall:\n")
        output.write(str(ave_one_percent_recall))
        output.write("\n\n")

    return ave_one_percent_recall


def get_latent_vectors(model, dict_to_process,data_folder):
    # print('dict_to_process',dict_to_process)#{1:{file,x,y},2:...}
    model.eval()
    is_training = False
    # print('len(dict_to_process.keys())',len(dict_to_process.keys()))
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = EVAL_BATCH_SIZE * \
                (1 + EVAL_POSITIVES_PER_QUERY + EVAL_NEGATIVES_PER_QUERY)
    q_output = []
    for q_index in range(len(train_file_idxs) // batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index + 1) * (batch_num)]
        file_names = []
        for index in file_indices:
            # print('dict_to_process[index]', dict_to_process)
            file_names.append(dict_to_process[index]["file"])
        queries = load_pcbev_files(data_folder,file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)

            # print("tensor shape", feed_tensor.size())
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        # out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if (len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        # print('file_indices',file_indices)#[0 1 2]
        # print('dict_to_process[index]', dict_to_process)
        for index in file_indices:
            # print(index)

            file_names.append(dict_to_process[index]["file"])
        queries = load_pcbev_files(data_folder,file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    # print(q_output.shape)
    return q_output


def get_recall(DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS):
    database_output = DATABASE_VECTORS[0]
    queries_output = QUERY_VECTORS[0]

    # print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # print('#################################',QUERY_SETS[i]['gt'])
        true_neighbors = QUERY_SETS[i]['gt']
        if (len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]), k=num_neighbors)
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if (j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall

def plotdescriptor(res):
    a = np.random.random((20,len(res), 1))
    # print(np.shape(a))

    for j in range(len(res)):
        for i in range(20):
            # print(res[j])
            a[i][j] = res[j].detach().numpy()

    # print(a)
    plt.tight_layout()
    plt.imshow(a, cmap="nipy_spectral", vmin=0, vmax=1)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # params
    folder = 'E:\Datasets\\1_KITTI\dataset\sequences\\00\\velodyne\\'
    im = load_pcbev_file(folder,'000000.bin')

    data = torch.tensor(im)
    data = data.unsqueeze(dim=0)
    data = data.float()

    model = sbev.SuperBEV()

    resume_filename = LOG_DIR + "model.ckpt"  # "checkpoint.pth.tar"
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    res = model(data)
    print('descriptor',np.shape(res))
    # print(res[0][0])
    plotdescriptor(res[0][0])



