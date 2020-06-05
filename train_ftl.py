import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms as trans

import os
import cv2
import time
import math
import bcolz
import random
import logging
import numpy as np
from PIL import Image

from FTL_model import Encoder, Decoder, Distillation_R, FC_softmax
from data.data_pipe import get_train_loader, get_val_data
from verifacation import evaluate

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename="log_alter.txt")


data_folder = "data/faces_emore/imgs"
data_folder_stage2 = "data/faces_emore"
val_folder = "data/faces_emore"
save_path = "ckpt"
pretrained_model_path = 'ckpt'
assigned_epoch, assigned_step = 0, 39000

epoch_num = 100
batch_size = 128
device = torch.device("cuda:1")
#device = torch.device("cpu")
board_loss_every = 100
evaluate_every = 2000
save_every = 10000
transform = trans.Compose([ trans.Resize(100),
                            trans.ToTensor(),
                            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                          ])


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    output = torch.pow(output, 2)
    sum_output = torch.sum(output)
    return torch.div(sum_output, batch_size)

def verify(model, carray, issame, nrof_folds = 5):
    idx = 0
    embeddings = np.zeros([len(carray), 320])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size])
            g = model[0](batch.to(device))         
            embs = model[2](g)
            embeddings[idx:idx + batch_size] = embs.cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            g = model[0](batch.to(device))         
            embs = model[2](g)
            embeddings[idx:] = embs.cpu()
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    return accuracy.mean(), best_thresholds.mean()

def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    H = H.to(device)
    X_center =  torch.mm(H.float(), X.float())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    variance = torch.mm(components, torch.transpose(components, 0, 1))
    return variance

def UpdateStats(enc, data_folder, all_class, regular_class):
    center = dict()
    Q = torch.zeros((320, 320)).to(device)
    h = []
    ur_class_file = []
    with torch.no_grad():
        for idx, cand in enumerate(all_class):
            #if idx > 10:
            #    break
            cand_folder_path = os.path.join(data_folder, cand)
            c = torch.zeros((1, 320)).to(device)
            g_list = dict()
            for ind, img_name in enumerate(os.listdir(cand_folder_path)):
                img_path = os.path.join(cand_folder_path, img_name)
                img = cv2.imread(img_path)
                img_flip = cv2.flip(img, 1)    
                img = np.array(transform(Image.fromarray(img.astype(np.uint8))))
                img_flip = np.array(transform(Image.fromarray(img_flip.astype(np.uint8))))
                img_batch = torch.tensor(np.array([img, img_flip]))
                g_out = enc(img_batch.to(device))
                g_sum = torch.sum(g_out, 0)
                g_list[img_name] = g_out[0]
                c = c + g_sum
            if idx % 5000 == 0:
                logging.debug("UpdateStats Processing...{}".format(idx))
            center[cand] = torch.div(c, 2*(ind+1))
            if cand in regular_class:
                c = center[cand]
                dik_list, files, di_list = list(), list(), list()
                for name, g in g_list.items():
                    diff = g-c
                    files.append(name)
                    dik_list.append(diff)
                    di_list.append(torch.norm(diff))
                di_mean = torch.mean(torch.stack(di_list))
                for ind, (dik, di, name) in enumerate(zip(dik_list, di_list, files)):
                    Q = Q + torch.mm(torch.transpose(dik, 0, 1), dik)
                    if di > di_mean:
                        h.append([cand, g_list[name]])
            else:
                for name, g in g_list.items(): 
                    ur_class_file.append([cand, g])
    return center, PCA_svd(Q, 150), h, ur_class_file

def train(epochs):
    logging.debug("Prepare Data")
    
    all_class = [i for i in os.listdir(data_folder)]
    class_num = len(all_class)
    regular_class = []
    for idx, i in enumerate(all_class):
        if len(os.listdir(os.path.join(data_folder, i))) > 20:
            regular_class.append(i)
        if idx%20000 == 0:
            print("Processing...", idx)
    loader, class_num = get_train_loader(data_folder_stage2, batch_size)
    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_val_data(val_folder)

    ## Load Model
    enc = Encoder().to(device)
    dec = Decoder().to(device)
    r = Distillation_R().to(device)
    head = FC_softmax(320, class_num).to(device)

    enc.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'enc_{}_{}.pth'.format(assigned_epoch, assigned_step ))))
    dec.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'dec_{}_{}.pth'.format(assigned_epoch, assigned_step ))))
    r.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'r_{}_{}.pth'.format(assigned_epoch, assigned_step ))))
    head.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'head_{}_{}.pth'.format(assigned_epoch, assigned_step ))))
    enc.eval()
    dec.eval()
    r.eval()
    head.eval()
    model = [enc, dec, r]
   
    ## Set Training Criterion
    ce_loss = nn.CrossEntropyLoss()
    l2_loss = nn.MSELoss()
    optimizer = optim.Adam(model[0].parameters(), lr = 0.00001)
    optimizer.add_param_group({'params':model[1].parameters()})
    optimizer.add_param_group({'params':model[2].parameters()})
    optimizer.add_param_group({'params':head.parameters()})


    logging.debug("Start Training")
    ## Initial Training
    running_loss = 0
    step = 0
    acc_max = 0
 
    for e in range(epochs):
       for stage in [1, 2]:
            ## Initial Train For Stage 1 and 2            
            if stage == 1:
                center, Q, h, ur_class_file = UpdateStats(model[0], data_folder, all_class, regular_class)
                logging.debug("Center_num:{}, h_num:{}, ur_class_num:{}".format(len(center), len(h), len(ur_class_file)))
            else:
                iterr = iter(loader)
            ## Start Training            
            for step_stage in range(5000):
                if stage == 1:
                    regular_batch = random.sample(range(len(h)), batch_size)
                    ur_batch = random.sample(range(len(ur_class_file)), batch_size)
                    g_r_list, g_u_list, g_t_list, label_r_list, label_u_list, label_t_list = list(), list(), list(), list(), list(), list()
                    for idx_r, idx_u in zip(regular_batch, ur_batch):
                        ##Prepare data for First Batch ()
                        label_r, g_r = h[idx_r]
                        c_r = center[label_r]
                        g_r_list.append(g_r)
                        label_r_list.append(torch.tensor(int(label_r)))
                        ##Prepare data for Second Batch
                        label_u, g_u = ur_class_file[idx_u]
                        c_u = center[label_u]
                        g_u_list.append(g_u)
                        label_u_list.append(torch.tensor(int(label_u)))
                        ##Prepare data for Third Batch
                        g_t = c_u + torch.mm(Q, (g_r-c_r).t()).t()
                        g_t_list.append(g_t.view(-1))
                        label_t_list.append(torch.tensor(int(label_u)))
                    
                    g_r_list = torch.stack(g_r_list).to(device)
                    label_r_list = torch.stack(label_r_list).to(device)
                    g_u_list = torch.stack(g_u_list).to(device)
                    label_u_list = torch.stack(label_u_list).to(device)
                    g_t_list = torch.stack(g_t_list).to(device)
                    label_t_list = torch.stack(label_t_list).to(device)

                    for g, labels in zip([g_r_list, g_u_list, g_t_list], [label_r_list, label_u_list, label_t_list]):
                        optimizer.zero_grad()
                        embs = model[2](g)
                        fc_out = head(embs)
                        loss_ce = ce_loss(fc_out, labels)
                        loss_reg = l2_norm(fc_out)
                        loss = loss_ce + loss_reg * 0.25
                        loss.backward()
                        running_loss += loss.item() 
                        optimizer.step()
                    running_loss /= 3
                    
                elif stage == 2:
                    imgs, labels = next(iterr)
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    g = model[0](imgs)
                    img_decs = model[1](g)
                    with torch.no_grad():
                        embs = model[2](g)
                        fc_out = head(embs)
                    loss_ce = ce_loss(fc_out, labels)
                    loss_mse = l2_loss(imgs, img_decs)
                    loss_reg = l2_norm(fc_out)
                    loss = loss_ce + loss_mse + loss_reg * 0.25
                    loss.backward()
                    running_loss += loss.item()
                    optimizer.step()

                if step % board_loss_every == 0 and step != 0:
                    loss_board = running_loss / board_loss_every
                    printout = "stage:{}, step:{}, epoch:{}, train_loss:{}".format(stage, step, e, loss_board)
                    logging.debug(printout)
                    running_loss = 0

                if step % evaluate_every == 0 and step != 0:
                    accuracy, best_threshold = verify(model, agedb_30, agedb_30_issame)
                    printout = "dataset:age30db, acc:{}, best_threshold:{}".format(accuracy, best_threshold)
                    logging.debug(printout)
                    accuracy, best_threshold = verify(model, lfw, lfw_issame)
                    printout = "dataset:lfw, acc:{}, best_threshold:{}".format(accuracy, best_threshold)
                    logging.debug(printout)
                    accuracy, best_threshold = verify(model, cfp_fp, cfp_fp_issame)
                    printout = "dataset:cfp_fp, acc:{}, best_threshold:{}".format(accuracy, best_threshold)
                    logging.debug(printout)
                    if accuracy > acc_max:
                        torch.save(model[0].state_dict(), '{}/enc_alter_{}_{}.pth'.format(save_path, e, step))
                        torch.save(model[1].state_dict(), '{}/dec_alter_{}_{}.pth'.format(save_path, e, step))  
                        torch.save(model[2].state_dict(), '{}/r_alter_{}_{}.pth'.format(save_path, e, step))
                        torch.save(head.state_dict(), '{}/head_alter_{}_{}.pth'.format(save_path, e, step))
                        acc_max = accuracy
                    logging.debug("Save ckpt at epoch:{} step:{}".format(e, step))

                step += 1

                
if __name__ == '__main__':
    train(epoch_num)
