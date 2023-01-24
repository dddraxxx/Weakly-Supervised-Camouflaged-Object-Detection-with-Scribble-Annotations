#!/usr/bin/python3
#coding=utf-8

from functools import reduce
import os
import sys

#sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from skimage import img_as_ubyte, img_as_float
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from data import dataset
import time
import logging as logger
import json
import subprocess
GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

TAG = "test"
SAVE_PATH = TAG
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="test_%s.log"%('tmp'), filemode="w")

root = '/mnt/sdh/hrz/CodDataset'
DATASETS = [f'{root}/test/CAMO',f'{root}/test/CHAMELEON',f'{root}/test/COD10K',]

class DC(nn.Module):
    def __init__(self, h):
        super().__init__()
        w = h.weight
        b = h.bias
        self.w = w
        self.c = w.shape[1]
        self.b = b
    def forward(self, x):
        c1 = F.conv2d(x, self.w.transpose(0,1), padding=(1,1), groups=self.c)
        return c1.sum(1, keepdims=True) + self.b

class Test(object):
    def __init__(self, Dataset, datapath, Network):
        ## dataset
        self.datapath = datapath.split("/")[-1]
        print("Testing on %s"%self.datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network
        self.net.train(False)
        self.net.cuda()
        self.net.eval()

    def accuracy(self, map_path=None):
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            for image, mask, shape, maskpath in self.loader:
                start_time = time.time()
                if map_path is None:
                    image, mask            = image.cuda().float(), mask.cuda().float()
                    out2, _ = self.net(image, shape)
                    pred                   = torch.sigmoid(out2)
                    torch.cuda.synchronize()
                else:
                    file_name = maskpath[0].split('/')[-1].split('.')[0]
                    dataset_name = self.datapath
                    pred_name = os.path.join(map_path, dataset_name, file_name+'.png')
                    # read img as float
                    pred = cv2.imread(pred_name, cv2.IMREAD_GRAYSCALE)
                    pred = img_as_float(pred)
                    pred = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
                end_time = time.time()
                cost_time += end_time - start_time

                ## MAE
                cnt += 1
                mae += (pred-mask).abs().mean()
                ## F-Score
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred >= threshod[i]).float()
                    precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)
                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
                if cnt % 20 == 0:
                    fps = image.shape[0] / (end_time - start_time)
                    # print('MAE=%.6f, F-score=%.6f, fps=%.4f'%(mae/cnt, fscore.max()/cnt, fps))
            fps = len(self.loader.dataset) / cost_time
            msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f'%(self.datapath, mae/cnt, fscore.max()/cnt, len(self.loader.dataset), fps)
            print(msg)
            logger.info(msg)

    def save(self):
        with torch.no_grad():
            cost_time = 0
            cnt = 0
            mae = 0
            print('will save to ./map/{}'.format(EXP_NAME))
            # mkdir
            head = './map/{}/'.format(EXP_NAME) + self.cfg.datapath.split('/')[-1]
            if not os.path.exists(head):
                os.makedirs(head)
            for image, mask, (H, W), name in self.loader:
                start_time = time.perf_counter()
                out2, out_dst = self.net(image.cuda().float(), (H, W))
                torch.cuda.synchronize()
                cost_time += time.perf_counter() - start_time

                # out2 = F.interpolate(out2, size=(H, W), mode='bilinear', align_corners=False)
                pred = (torch.sigmoid(out2[0, 0])).cpu()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

                mae += (pred-mask).abs().mean()
                cnt += len(image)

                pred = pred.numpy()

                head     = './map/{}/'.format(EXP_NAME) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], img_as_ubyte(pred))

            fps = len(self.loader.dataset) / cost_time
            msg = '%s len(imgs)=%s, fps=%.4f'%(self.datapath, len(self.loader.dataset), fps)
            print('mae {}'.format(mae/cnt))
            print(msg)
            logger.info(msg)
        
    def change_json(self, json_path=None):
        js = json.load(open(json_path))
        k1 = next(iter(js.values()))
        k1[self.datapath]['path'] = os.path.abspath('./map/{}/'.format(EXP_NAME) + self.datapath)
        json.dump(js, open(json_path, 'w'), indent=4)

def cal_cod_metrics(js_m, js_d):
    js_m = os.path.abspath(js_m)
    js_d = os.path.abspath(js_d)
    os.chdir('./PySODEvalToolkit')
    os.system('python ./eval.py --method {} --dataset {} --record-txt ./results.txt'.format(js_m, js_d))
    os.chdir('../')



import os
from net import Net
from pathlib import Path as pa
EXP_NAME='trained'
JSON_METHOD = './PySODEvalToolkit/cod_method.json'
JSON_DATA = './PySODEvalToolkit/cod_dataset.json'

if __name__=='__main__':
    # set torch cuda device
    cfg = dataset.Config(datapath='000', mode='test')
    net = Net(cfg)
    path = 'model-best.pth'
    state_dict = torch.load(path)
    net.load_state_dict(state_dict, strict=True)
    print('complete loading: {}'.format(path))
    print('-----------------')
    print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))
    for e in DATASETS[:]:
        t =Test(dataset, e, net)
        # t.accuracy()
        t.save()
        t.change_json(JSON_METHOD)
    
    cal_cod_metrics(JSON_METHOD, JSON_DATA)
    print(EXP_NAME)