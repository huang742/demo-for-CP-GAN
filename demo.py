# coding=utf-8
# from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from miscc.config import cfg, cfg_from_file



from nltk.tokenize import RegexpTokenizer
from PIL import Image
import codecs
import json
import imp
import sys
import math
import time
import pickle
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from miscc.utils import weights_init, load_params, copy_G_params

import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import opts

from model import RNN_ENCODER, CNN_ENCODER
#from DIY_resmodel import G_DCGAN, G_NET
from model import G_DCGAN, G_NET
from miscc.utils import mkdir_p
import torch
from copy import deepcopy
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.nn.utils.rnn import pack_padded_sequence
from miscc.config import cfg, cfg_from_file
import pprint

cuda = True

def load_models(nwords):
    cfg.TRAIN.NET_E = "./text_encoder/text_encoder.pth"

    text_encoder = RNN_ENCODER(nwords, nhidden=cfg.TEXT.EMBEDDING_DIM)
    # state_dict = \
    #     torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
    state_dict = \
        torch.load(cfg.TRAIN.NET_E)
    text_encoder.load_state_dict(state_dict)

    if cuda:
        text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()

    return text_encoder


def load_G():
    print(cfg.TRAIN.NET_G)
    cfg.TRAIN.NET_G = './test_models/test_netG.pth'
    if cfg.TRAIN.NET_G == '':
        print('Error: the path for morels is not found!')
    else:
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM == 1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
        else:
            netG = G_NET()
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)

        model_dir = cfg.TRAIN.NET_G
        state_dict = \
            torch.load(model_dir, map_location=lambda storage, loc: storage)
        # state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        if cuda:
            netG.cuda()
        netG.eval()
        print('Load G from: ', model_dir)
        # netG.eval()
        return netG





if __name__ == "__main__":
    opt = opts.parse_opt()
    captions = []
    cfg.DATA_DIR = '/home/user04/hzy/demo'
    save_dir = 'test_img'

    with codecs.open("./word_to_ix.json","r","utf-8") as f:
                word_to_ix = json.load(f)
    word_to_ix = dict(word_to_ix)

    text_encoder = load_models(len(word_to_ix) + 1)

    netG = load_G()
    batch_size = opt.batch_size
    criterion = nn.CrossEntropyLoss()

    nz = cfg.GAN.Z_DIM



    file = open("text.txt")
    for caption in file:
        caption = caption.lower()
        caption = caption.replace(',', '')
        # caption = caption.replace('.', '')
        caption = caption.replace('there', '')
        caption = caption.replace('are', '')
        caption = caption.replace('is', '')
        caption = caption.replace('that', '')
        caption = caption.replace('the', '')


        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(caption.lower())

        count = 1
        caption = [0 for i in range(50)]

        for n in range(len(tokens)):
            word = word_to_ix[tokens[n]]
            if word != None:
                caption[count] = int(word)
                count = count + 1
        captions.append(caption)

    noise = Variable(torch.FloatTensor(len(captions), nz))
    noise = noise.cuda()

    captions = torch.tensor(captions)

    cap_lens = []
    new_captions = torch.zeros(captions.size(0), captions.size(1))
    for i in range(captions.size(0)):
        for j in range(1, captions.size(1)):
            new_captions[i][j - 1] = captions[i][j]
    if cuda:
        new_captions = new_captions.long().cuda()
    else:
        new_captions = new_captions.long()

    for i in range(new_captions.size(0)):
        c_len = 0
        for j in range(new_captions.size(1)):
            if new_captions[i][j] != 0:
                c_len = c_len + 1
        cap_lens.append(c_len)
    cap_lens = torch.tensor(cap_lens)
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(cap_lens, 0, True)
    new_captions = new_captions[sorted_cap_indices]


    hidden = text_encoder.init_hidden(captions.size(0))
    real_words_embs, real_sent_emb = text_encoder(new_captions, sorted_cap_lens, hidden)
    real_words_embs, real_sent_emb = real_words_embs.detach(), real_sent_emb.detach()

    mask = (new_captions == 0)
    num_words = real_words_embs.size(2)
    if mask.size(1) > num_words:
        mask = mask[:, :num_words]

    noise.data.normal_(0, 1)
    fake_imgs, _, mu, logvar = netG(noise, real_sent_emb, real_words_embs, mask)



    for j in range(fake_imgs[-1].size(0)):
        img_name = "example"+str(j)+".png"
        save_path = '%s/%s'%(save_dir,img_name)
        # print(save_path)
        im = fake_imgs[-1][j].data.cpu().numpy()
        # print(im.shape)
        im = (im + 1.0) * 127.

        # print(im)
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        # print(im.shape)
        # print(im.shape)
        # print(im)
        im = Image.fromarray(im)
        # print(im)
        # fullpath = '%s.png' % (save_path)
        im.save(save_path)






