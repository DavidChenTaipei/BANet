#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys
sys.path.insert(0, '.')
import os
from tqdm import tqdm
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import save_checkpoint
from configs import cfg
from lib.cityscapes_cv2 import get_data_loader
from tools.evaluate import eval_model
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.models.banet import model
from torch.optim.lr_scheduler import ExponentialLR,ReduceLROnPlateau
writer = SummaryWriter('./runs/')

# apex
has_apex = False ##True

try:
    from apex import amp, parallel
except ImportError:
    has_apex = False

## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True




def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument('--name',dest = 'store_name',type=str)
    parse.add_argument('--epoch-to-train', dest = 'epoch_to_train',type=int)
    return parse.parse_args()

args = parse_args()



def set_model():
    net = model(channel=1024)
    if not args.finetune_from is None:
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    net.cuda()
    net.train()
    criteria = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    return net, criteria

def set_optimizer(model): 
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_meters(epoch):
    time_meter = TimeMeter(epoch)
    loss_meter = AvgMeter('loss')
    return time_meter, loss_meter

class ReduceLROnPlateauPatch(ReduceLROnPlateau):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]

def train(epoch,optim,net,criteria,lr_schdr):
    ## dataset
    dl = get_data_loader(
            cfg.im_root, cfg.train_im_anns,
            cfg.ims_per_gpu, cfg.scales, cfg.cropsize,mode='train')

    ## meters
    time_meter, loss_meter= set_meters(epoch)
    ## train loop
    for it, (im, lb) in enumerate(tqdm(dl)):
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        aux_loss, main_loss = net(im)
        aux_criteria = criteria(aux_loss, lb)
        main_criteria = criteria(main_loss, lb)
        loss = main_criteria+0.1*aux_criteria
        '''if has_apex:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:'''
        loss = loss.mean()
        loss.backward()
        optim.step()
        
        time_meter.update()
        loss_meter.update(loss.item())
        lr_schdr.step()
        return lr_schdr,time_meter,loss_meter
with open('lr_record.txt','r+') as m:
    lr  = m.read()
    lr = lr.replace('\n',' ')
    x = lr.split(' ')
    while ('' in x):
        x.remove('')
    lr_start = eval(x[-1])

def main():
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-train'.format('banet'), cfg.respth)
    
    best_prec1=(-1)
    logger = logging.getLogger()
    

    ## model
    net, criteria= set_model()
    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)
    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.epoch*371, warmup_iter=cfg.warmup_iters*371,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    for epoch in range(cfg.start_epoch,args.epoch_to_train):
        lr_schdr,time_meter,loss_meter = train(epoch,optim,net,criteria,lr_schdr)
        if True:
        #if ((epoch+1)!=cfg.epoch):
            lr = lr_schdr.get_lr()
            print(lr)
            lr = sum(lr) / len(lr)
            loss_avg = print_log_msg(
                epoch, cfg.epoch, lr, time_meter, loss_meter)
            writer.add_scalar('loss',loss_avg,epoch + 1)
            
        if ((epoch+1)==cfg.epoch) or ((epoch+1)==args.epoch_to_train):
        #if ((epoch+1)%1==0) and ((epoch+1)>cfg.warmup_iters):    
            torch.cuda.empty_cache()
            heads, mious,miou = eval_model(net,ims_per_gpu=2,im_root=cfg.im_root,im_anns=cfg.val_im_anns,it=epoch)       
            filename = osp.join(cfg.respth, args.store_name)
            state = net.state_dict()
            save_checkpoint(state,False,filename=filename)
            #writer.add_scalar('mIOU',miou,epoch+1)
            with open('lr_record.txt','w') as m:
                print('lr to store',lr)
                m.seek(0)
                m.write((str(epoch+1)+'   '))
                m.write(str(lr))
                m.truncate()
                m.close()
            with open('best_miou.txt', 'r+') as f:
                best_miou = f.read()
                #print(best_miou)
                best_miou = best_miou.replace('\n',' ')
                x = best_miou.split(' ')
                while ('' in x):
                    x.remove('')
                best_miou = eval(x[-1])
                is_best = miou> best_miou
                if is_best:
                    best_miou = miou
                    print('Is best? : ',is_best)
                    f.seek(0)
                    f.write((str(epoch+1)+'   '))
                    f.write(str(best_miou))
                    f.truncate()
                    f.close()
                    save_checkpoint(state,is_best,filename) 
            print('Have Stored Checkpoint')
        #if((epoch+1)==cfg.epoch) or ((epoch+1)==args.epoch_to_train):
            state = net.state_dict() 
            torch.cuda.empty_cache()
            #heads, mious = eval_model(net, 2, cfg.im_root, cfg.val_im_anns,it=epoch)
            logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
            save_checkpoint(state,False,filename)
            print('Have Saved Final Model')
            break

if __name__ == "__main__":
    main()

