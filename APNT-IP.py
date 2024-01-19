import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import sys
from torchvision import transforms
from vit import Transformer



parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 200) # PaviaU=103；salinas=204
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=4)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')
parser.add_argument("-a","--trans_layer", type = int, default = 2)

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(0)

def _init_():
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./classificationMap'):
        os.makedirs('./classificationMap')
_init_()

# load source domain data set
with open(os.path.join('../datasets',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data'] # (77592, 9, 9, 128)
labels_train = source_imdb['Labels'] # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape) # (77592, 9, 9, 128)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) #(9, 9, 128, 77592)
print(source_imdb['data'].shape) # (9, 9, 128, 77592)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set
test_data = '../datasets/IP/indian_pines_corrected.mat'
test_label = '../datasets/IP/indian_pines_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)

    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.Crop_and_resize(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(
            np.int64)


    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=100, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain


# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Mapping_Network(nn.Module):
    def __init__(self):
        super(Mapping_Network, self).__init__()
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, domain='source'):
        if domain == 'target':
            x = self.target_mapping(x)
        elif domain == 'source':
            x = self.source_mapping(x)

        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.final_feat_dim = FEATURE_DIM  # 64+32
        self.cls_token = nn.Parameter(torch.randn(1, 1, N_DIMENSION))
        self.drop_x = nn.Dropout(0.5)
        self.transformer = Transformer(N_DIMENSION, args.trans_layer, 8, 64, 1024, 0.1)


    def forward(self, x):  # x
        B, C, H, W =  x.shape
        x = x.view(B,C,H*W).transpose(1,2)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, cls_tokens), dim=1)
        x, att = self.transformer(x)
        x = x[:, -1, :]
        return x, att


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def cutmix(data, target, querys_attention, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    querys_attention = querys_attention[:, -1, 0:81]
    querys_attention = querys_attention.squeeze(dim=1).view(-1, 9, 9)

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    atten_part = querys_attention[:,bbx1:bbx2, bby1:bby2]
    atten = torch.sum(querys_attention, axis=1)
    atten = torch.sum(atten, axis=1)
    atten_part = torch.sum(atten_part, axis=1)
    atten_part = torch.sum(atten_part, axis=1)
    lam = 1 - atten_part/atten


    shuffled_attention = querys_attention[indices]
    atten_part_b = shuffled_attention[:, bbx1:bbx2, bby1:bby2]
    atten_b = torch.sum(shuffled_attention, axis=1)
    atten_b = torch.sum(atten_b, axis=1)
    atten_part_b = torch.sum(atten_part_b, axis=1)
    atten_part_b = torch.sum(atten_part_b, axis=1)
    lam_b =  atten_part_b / atten_b

    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]

    # Compute output
    target_a = target
    target_b = shuffled_target
    return data, target_a, target_b, lam.cuda(), lam_b.cuda()

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# run 10 times
nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
proto_loss = torch.zeros(1)
seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    seed_torch(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = Network()
    mapping_network = Mapping_Network()
    feature_encoder.apply(weights_init)
    mapping_network.apply(weights_init)
    feature_encoder.cuda()
    mapping_network.cuda()
    feature_encoder.train()
    mapping_network.train()
    feature_encoder_optim = torch.optim.Adam([{'params': feature_encoder.parameters()}, {'params': mapping_network.parameters()}], lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_tar, total_num = 0.0, 0.0, 0.0
    test_acc_list = []
    train_start = time.time()
    for episode in range(3001):  # EPISODE = 90000

        # source domain few-shot + domain adaptation
        if episode < 1000:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=False)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            supports = mapping_network(supports.cuda(), domain='source')
            querys = mapping_network(querys.cuda(), domain='source')

            # calculate features
            support_features, supp_att = feature_encoder(supports.cuda())  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_att = feature_encoder(querys.cuda())  # torch.Size([409, 32, 7, 3, 3])

            mix_querys, query_label_a, query_label_b, lam, lam_b = cutmix(querys, query_labels, query_att, alpha=1.0)
            mix_query_features, mix_query_outputs = feature_encoder(mix_querys.cuda())  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(mix_query_features, support_proto)
            f_loss = lam * F.cross_entropy(logits, query_label_a.long().cuda(), reduction="none") + lam_b * F.cross_entropy(logits, query_label_b.long().cuda(), reduction="none")
            f_loss = torch.mean(f_loss, axis=0)

            loss = f_loss

            # Update parameters
            feature_encoder.zero_grad()
            mapping_network.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_hit += torch.sum(torch.argmax(logits.cuda(), dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=False)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().next()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().next()  # (75,100,9,9)

            supports = mapping_network(supports.cuda(), domain='target')
            querys = mapping_network(querys.cuda(), domain='target')

            # calculate features
            support_features, supp_att = feature_encoder(supports.cuda())  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_att = feature_encoder(querys.cuda())  # torch.Size([409, 32, 7, 3, 3])

            mix_querys, query_label_a, query_label_b, lam, lam_b = cutmix(querys, query_labels, query_att, alpha=1.0)
            mix_query_features, mix_query_outputs = feature_encoder(mix_querys.cuda())  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(mix_query_features, support_proto)
            f_loss = lam * F.cross_entropy(logits, query_label_a.long().cuda(), reduction="none") + lam_b* F.cross_entropy(logits, query_label_b.long().cuda(), reduction="none")
            f_loss = torch.mean(f_loss, axis=0)

            loss = f_loss

            # Update parameters
            mapping_network.zero_grad()
            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()

            total_tar += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0 and episode < 1000:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}:  acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, 2*total_hit / total_num,  loss.item()))
        if (episode + 1) % 100 == 1 and episode >= 1000:  # display
            train_loss.append(loss.item())
            print('episode {:>3d}: acc {:6.4f}, loss: {:6.4f}'.format(episode + 1,  2*total_tar / total_num, loss.item()))

        if (episode + 1) % 200 == 1 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            mapping_network.eval()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().next()
            train_datas = mapping_network(Variable(train_datas).cuda(), domain='target')
            train_features, _ = feature_encoder(train_datas)  # (45, 160)

            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1, weights='distance')
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_datas = mapping_network(Variable(test_datas).cuda(), domain='target')
                test_features, _ = feature_encoder(test_datas)  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            mapping_network.train()
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(mapping_network.state_dict(),
                           str("./DFSL_mapping_network_" + "IP_" + str(iDataSet) + "iter_" + str(
                               TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                torch.save(feature_encoder.state_dict(),
                           str("./DFSL_feature_encoder_" + "IP_" + str(iDataSet) + "iter_" + str(
                               TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                print("save networks for episode:", episode + 1)
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = \
    best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = [0.5, 0.75, 1]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24, "./IP_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))