import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms as trans

import os
import math
import bcolz
import logging
import numpy as np

from tqdm import tqdm
from PIL import Image
from FTL_model import Encoder, Decoder, Distillation_R, FC_softmax
from data.data_pipe import get_train_loader, get_val_data
from verifacation import evaluate


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename="log_pre.txt")


data_folder = "data/faces_emore"
val_folder = "data/faces_emore"
pretrained = False
pretrained_model_path = 'ckpt'
assigned_epoch, assigned_step = 1, 71000 


batch_size = 128
device = torch.device("cuda:0")
#device = torch.device("cpu")
board_loss_every = 100
evaluate_every = 1000
save_every = 10000

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    output = torch.pow(output, 2)
    sum_output = torch.sum(output)
    return torch.div(sum_output, batch_size)


def verify(model, carray, issame, nrof_folds = 5, tta = False):
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

def train(epochs):
    logging.debug("Prepare Data")

    loader, class_num = get_train_loader(data_folder, batch_size)
    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_val_data(val_folder)

    ## Load Model
    enc = Encoder().to(device)
    dec = Decoder().to(device)
    r = Distillation_R().to(device)
    model = [enc, dec, r]
    head = FC_softmax(320, class_num).to(device)
    if pretrained:
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
    optimizer = optim.Adam(model[0].parameters(), lr = 0.0002)
    optimizer.add_param_group({'params':model[1].parameters()})
    optimizer.add_param_group({'params':model[2].parameters()})
    optimizer.add_param_group({'params':head.parameters()})

    ## Initial Training
    running_loss = 0
    step = 0
    acc_max = 0

    logging.debug("Start Training")
    for e in range(epochs):                             
        for imgs, labels in iter(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            g = model[0](imgs)
            img_decs = model[1](g)
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
                printout = "step:{}, epoch:{}, train_loss:{}".format(step, e, loss_board)
                logging.debug(printout)
                running_loss = 0.
            
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
                    torch.save(model[0].state_dict(), 'ckpt/enc_{}_{}.pth'.format(e, step))
                    torch.save(model[1].state_dict(), 'ckpt/dec_{}_{}.pth'.format(e, step))  
                    torch.save(model[2].state_dict(), 'ckpt/r_{}_{}.pth'.format(e, step))
                    torch.save(head.state_dict(), 'ckpt/head_{}_{}.pth'.format(e, step))
                    acc_max = accuracy
                    logging.debug("Save ckpt at epoch:{} step:{}".format(e, step))
            step += 1
                
if __name__ == '__main__':
    train(999999)
