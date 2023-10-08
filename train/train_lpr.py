import sys
import os
sys.path.append("../")
import torch
import torch.nn as nn
from sklearn.neighbors import KDTree, NearestNeighbors
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from tools.utils.load_pcs import *
from tools.utils.read_bin import *
import modules.lpr_loss as lpr_loss
import modules.superbevnet as sbev

import eval.evaluate as evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import yaml
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_bn_decay(batch):
    bn_momentum = BN_INIT_DECAY * \
        (BN_DECAY_DECAY_RATE **
         (batch * BATCH_NUM_QUERIES // BN_DECAY_DECAY_STEP))
    return min(BN_DECAY_CLIP, 1 - bn_momentum)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# learning rate halfed every 5 epoch


def get_learning_rate(epoch):
    learning_rate = BASE_LEARNING_RATE * ((0.9) ** (epoch // 5))
    learning_rate = max(learning_rate, 0.000007)  # CLIP THE LEARNING RATE!
    print('learning rate',learning_rate)
    return learning_rate

def get_feature_representation(filename, model):
    model.eval()
    queries = load_pcbev_files(DATASET_FOLDER,[filename])
    queries = np.expand_dims(queries, axis=1)
    # if(BATCH_NUM_QUERIES-1>0):
    #    fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
    #    q=np.vstack((queries,fake_queries))
    # else:
    #    q=queries
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output


def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])

    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=num_to_take)
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs


def get_latent_vectors(model, dict_to_process):
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = BATCH_NUM_QUERIES * \
        (1 + TRAIN_POSITIVES_PER_QUERY + TRAIN_NEGATIVES_PER_QUERY + 1)
    q_output = []

    model.eval()

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pcbev_files(DATASET_FOLDER,file_names)

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_pcbev_files(DATASET_FOLDER,[dict_to_process[index]["query"]])
        queries = np.expand_dims(queries, axis=1)

        # if (BATCH_NUM_QUERIES - 1 > 0):
        #    fake_queries = np.zeros((BATCH_NUM_QUERIES - 1, 1, NUM_POINTS, 3))
        #    q = np.vstack((queries, fake_queries))
        # else:
        #    q = queries

        #fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 3))
        #fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 3))
        #fake_other_neg = np.zeros((BATCH_NUM_QUERIES, 1, NUM_POINTS, 3))
        #o1, o2, o3, o4 = run_model(model, q, fake_pos, fake_neg, fake_other_neg)
        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    # print(q_output.shape)
    return q_output

def train():
    global HARD_NEGATIVES, TOTAL_ITERATIONS
    bn_decay = get_bn_decay(0)
    #tf.summary.scalar('bn_decay', bn_decay)

    #loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
    if LOSS_FUNCTION == 'quadruplet':
        loss_function = lpr_loss.quadruplet_loss
    else:
        loss_function = lpr_loss.triplet_loss_wrapper
    learning_rate = get_learning_rate(0)

    print(LOSS_FUNCTION)
    train_writer = SummaryWriter(os.path.join(LOG_DIR, 'train'))
    #test_writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'test'))

    model = sbev.SuperBEV()
    model = model.to(device)
    print('define model')
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if OPTIMIZER == 'momentum':
        optimizer = torch.optim.SGD(
            parameters, learning_rate, momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)

    if RESUME:
        resume_filename = LOG_DIR + "checkpoint.pth.tar"
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch']
        TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)

        model.load_state_dict(saved_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    model = nn.DataParallel(model)

    LOG_FOUT.write("\n")
    LOG_FOUT.flush()
    print('start')
    for epoch in range(starting_epoch, MAX_EPOCH):
        # torch.cuda.empty_cache()
        print('epoch',epoch)
        print()
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(model, optimizer, train_writer, loss_function, epoch)

        log_string('EVALUATING...')
        OUTPUT_FILE = RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'
        eval_recall = evaluate.evaluate_model(model)
        log_string('EVAL RECALL: %s' % str(eval_recall))

        train_writer.add_scalar("Val Recall", eval_recall, epoch)


def train_one_epoch(model, optimizer, train_writer, loss_function, epoch):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    is_training = True
    sampled_neg = 4000
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    num_to_take = 10

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    for i in range(len(train_file_idxs)//BATCH_NUM_QUERIES):
        # for i in range (5):
        batch_keys = train_file_idxs[i *
                                     BATCH_NUM_QUERIES:(i+1)*BATCH_NUM_QUERIES]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        for j in range(BATCH_NUM_QUERIES):
            # print('pos num',len(TRAINING_QUERIES[batch_keys[j]]["positives"]))
            # print('TRAIN_POSITIVES_PER_QUERY',TRAIN_POSITIVES_PER_QUERY)
            if (len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < TRAIN_POSITIVES_PER_QUERY):
                faulty_tuple = True
                break

            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], TRAIN_POSITIVES_PER_QUERY, TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))

            elif (len(HARD_NEGATIVES.keys()) == 0):
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                # print(hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], TRAIN_POSITIVES_PER_QUERY, TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                hard_negs = list(set().union(
                    HARD_NEGATIVES[batch_keys[j]], hard_negs))
                # print('hard', hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], TRAIN_POSITIVES_PER_QUERY, TRAIN_NEGATIVES_PER_QUERY,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_kleys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

            if (q_tuples[j][3].shape[0] != 2):
                no_other_neg = True
                break

        if(faulty_tuple):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'FAULTY TUPLE' + '-----')
            continue

        if(no_other_neg):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'NO OTHER NEG' + '-----')
            continue

        queries = []
        positives = []
        negatives = []
        other_neg = []
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries, dtype=np.float32)#(2, 1, 2, 401, 401)
        queries = np.expand_dims(queries, axis=1)
        other_neg = np.array(other_neg, dtype=np.float32)#(2, 1, 2, 401, 401)
        other_neg = np.expand_dims(other_neg, axis=1)
        positives = np.array(positives, dtype=np.float32)#(2, 2, 2, 401, 401)
        negatives = np.array(negatives, dtype=np.float32)#(2, 18, 2, 401, 401)
        log_string('----' + str(i) + '-----')
        # print(queries.shape)
        # print(other_neg.shape)
        # print(positives.shape)
        # print(negatives.shape)
        if (len(queries.shape) != 5):

            log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        model.train()
        optimizer.zero_grad()

        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg)
        loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, MARGIN_1, MARGIN_2, use_min=TRIPLET_USE_BEST_POSITIVES, lazy=LOSS_LAZY, ignore_zero_loss=LOSS_IGNORE_ZERO_BATCH)
        loss.backward()
        optimizer.step()

        log_string('batch loss: %f' % loss)
        train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
        TOTAL_ITERATIONS += BATCH_NUM_QUERIES

        # EVALLLL

        if (epoch > 5 and i % (1400 // BATCH_NUM_QUERIES) == 29):
            TRAINING_LATENT_VECTORS = get_latent_vectors(
                model, TRAINING_QUERIES)
            print("Updated cached feature vectors")

        if (i % (6000 // BATCH_NUM_QUERIES) == 101):
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            save_name = LOG_DIR + MODEL_FILENAME
            torch.save({
                'epoch': epoch,
                'iter': TOTAL_ITERATIONS,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
                save_name)
            print("Model Saved As " + save_name)

def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
    feed_tensor = feed_tensor.view((-1, 1, 2, 401, 401))
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    output = output.view(BATCH_NUM_QUERIES, -1, FEATURE_OUTPUT_DIM)
    o1, o2, o3, o4 = torch.split(
        output, [1, TRAIN_POSITIVES_PER_QUERY, TRAIN_NEGATIVES_PER_QUERY, 1], dim=1)

    return o1, o2, o3, o4

if __name__ == "__main__":

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
    TRAIN_FILE = config["data_root"]["TRAIN_FILE"]
    # TEST_FILE = config["data_root"]["TEST_FILE"]
    MODEL_FILENAME = config["global"]["MODEL_FILENAME"]
    LOG_DIR = config["global"]["LOG_DIR"]

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write('\n')

    RESULTS_FOLDER = config["global"]["RESULTS_FOLDER"]
    DATASET_FOLDER = config["data_root"]["DATASET_FOLDER"]

    BN_INIT_DECAY = config["TRAIN"]["BN_INIT_DECAY"]
    BN_DECAY_DECAY_RATE = config["TRAIN"]["BN_DECAY_DECAY_RATE"]
    BN_DECAY_DECAY_STEP = config["TRAIN"]["BN_DECAY_DECAY_STEP"]
    BN_DECAY_CLIP = config["TRAIN"]["BN_DECAY_CLIP"]
    RESUME = config["TRAIN"]["RESUME"]
    HARD_NEGATIVES = {}
    TRAINING_LATENT_VECTORS = []
    TOTAL_ITERATIONS = 0
    #字典不能切片，train和test后期再处理
    TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
    # TEST_QUERIES = get_queries_dict(TEST_FILE)

    train()