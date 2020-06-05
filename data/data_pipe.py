from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm


train_transform = trans.Compose([
    trans.Resize(100),
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_train_dataset(imgs_folder, train_transform=train_transform):
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_loader(datafolder, batch_size):
    ds, class_num = get_train_dataset(datafolder + '/imgs')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    return loader, class_num 
    
def load_bin(path, rootdir, transform=train_transform , image_size=[100, 100]):
    if not os.path.isdir(rootdir):
        os.makedirs(rootdir)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = '{}/{}'.format(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'imgs')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = os.path.join(save_path, str(label))
        if not os.path.isdir(label_path):
            os.makedirs(label_path)
        img.save(os.path.join(label_path, '{}.jpg'.format(idx)), quality=95)
