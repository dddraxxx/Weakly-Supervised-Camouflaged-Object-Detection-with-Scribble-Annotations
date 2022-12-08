#!/usr/bin/python3
#coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset, DataLoader
from lib.data_prefetcher import DataPrefetcher

class Config(object):
    def __init__(self, **kwargs):
        if kwargs.get('label_dir') is None:
            kwargs['label_dir'] = 'Scribble'
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean      = np.array([[[124.55, 118.90, 102.94]]])
            self.std       = np.array([[[ 56.77,  55.97,  57.50]]])
        elif 'DUT-OMRON' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.61, 121.86, 114.92]]])
            self.std       = np.array([[[ 58.10,  57.16,  61.09]]])
        elif 'MSRA-10K' in self.kwargs['datapath']:
            self.mean      = np.array([[[115.57, 110.48, 100.00]]])
            self.std       = np.array([[[ 57.55,  54.89,  55.30]]])
        elif 'MSRA-B' in self.kwargs['datapath']:
            self.mean      = np.array([[[114.87, 110.47,  95.76]]])
            self.std       = np.array([[[ 58.12,  55.30,  55.82]]])
        elif 'SED2' in self.kwargs['datapath']:
            self.mean      = np.array([[[126.34, 133.87, 133.72]]])
            self.std       = np.array([[[ 45.88,  45.59,  48.13]]])
        elif 'PASCAL-S' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.02, 112.75, 102.48]]])
            self.std       = np.array([[[ 59.81,  58.96,  60.44]]])
        elif 'HKU-IS' in self.kwargs['datapath']:
            self.mean      = np.array([[[123.58, 121.69, 104.22]]])
            self.std       = np.array([[[ 55.40,  53.55,  55.19]]])
        elif 'SOD' in self.kwargs['datapath']:
            self.mean      = np.array([[[109.91, 112.13,  93.90]]])
            self.std       = np.array([[[ 53.29,  50.45,  48.06]]])
        elif 'THUR15K' in self.kwargs['datapath']:
            self.mean      = np.array([[[122.60, 120.28, 104.46]]])
            self.std       = np.array([[[ 55.99,  55.39,  56.97]]])
        elif 'SOC' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.48, 111.78, 101.27]]])
            self.std       = np.array([[[ 58.51,  56.73,  56.38]]])
        else:
            #raise ValueError
            self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
            self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])
            # self.std, self.mean = np.array([0.1861761914527739, 0.19748777412623036, 0.2032849354904543])[None,None]*255, np.array([0.3320486163733052, 0.432231354815684, 0.449829585669272])[None,None]*255

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_name = cfg.datapath.split('/')[-1]
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            if cfg.mode == 'train':
              for line in lines:
                  imagepath = cfg.datapath +'/'+cfg.mode+ '/Image/' + line.strip() + '.jpg'
                  maskpath  = cfg.datapath + '/'+cfg.mode+f'/{cfg.label_dir}/'  + line.strip() + '.png'
                  self.samples.append([imagepath, maskpath])
            else:
              for line in lines:
                  imagepath = cfg.datapath + '/Image/' + line.strip() + '.jpg'
                  maskpath  = cfg.datapath +'/GT/'  + line.strip() + '.png'
                  self.samples.append([imagepath, maskpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.RandomHorizontalFlip(),
                                                    transform.RandomCrop(320, 320),
                                                    transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose(*([transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.ToTensor()
                                                ]+ [transform.RandomNoise(1e-1)]))
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath = self.samples[idx]
        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        H, W, C             = mask.shape
        if self.cfg.mode == 'train':
            image, mask         = self.transform(image, mask)
            mask[mask == 0.] = 255.
            mask[mask == 2.] = 0.
        else:
            image, _         = self.transform(image, mask)
            mask = torch.from_numpy(mask.copy()).permute(2,0,1)
            mask = mask.mean(dim=0, keepdim=True)
            mask /= 255
        # print(image.max(), image.min())
        return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='test', datapath='/dataC/qhd/cod/CodDataset')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)
    batch_idx = -1
    image, mask = prefetcher.next()
    image = image[0].permute(1,2,0).cpu().numpy()*cfg.std + cfg.mean
    mask  = mask[0].cpu().numpy()
    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.subplot(122)
    plt.imshow(mask)
    input()

