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
# import torch.distributed as dist
from torch.utils.data import DataLoader
from utils import save_checkpoint
from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_cv2 import get_data_loader
from tools.evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from lib.models.MPM import StripPooling
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
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')




def parse_args():
    parse = argparse.ArgumentParser()
#    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
#    parse.add_argument('--port', dest='port', type=int, default=44554,)
    #parse.add_argument('--epoch-to-train', dest = 'epoch_to_train',type=int)
    #parse.add_argument('--epoch-start-at', dest= 'epoch_start_at',type=int)
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    parse.add_argument('--name',dest = 'store_name',type=str)
    parse.add_argument('--epoch-to-train', dest = 'epoch_to_train',type=int)
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.model]



def set_model():
    net = model_factory[cfg.model_type](19)
    if not args.finetune_from is None:
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    #if cfg.use_sync_bn: net = set_syncbn(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_sssyncbn(net):
    if has_apex:
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model): #.cpu().data.numpy()):
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
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters

class ReduceLROnPlateauPatch(ReduceLROnPlateau): #, _LRScheduler):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]

def train(epoch,optim,net,criteria_pre, criteria_aux,lr_schdr):
    ## dataset
    dl = get_data_loader(
            cfg.im_root, cfg.train_im_anns,
            cfg.ims_per_gpu, cfg.scales, cfg.cropsize,mode='train') #, distributed=is_dist) <--has been removed

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters(epoch)
    ## train loop
    for it, (im, lb) in enumerate(tqdm(dl)):
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        logits, *logits_aux = net(im)
        loss_pre = criteria_pre(logits, lb)
        loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        if has_apex:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optim.step()
#         torch.cuda.synchronize()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
        lr_schdr.step()
        #lr_schdr.step(loss.item())
        #lr = lr_schdr.get_lr()  #last_lr()
        #print('lr',lr)
    ## dump the final model and evaluate the result
    
    #filename = osp.join(cfg.respth, 'MPM_at_the_end_checkpoint')
    #logger.info('\nsave models to {}'.format(filename))
    #state = net.module.state_dict()
    return lr_schdr,time_meter,loss_meter,loss_pre_meter,loss_aux_meters

with open('lr_record.txt','r+') as m:
    lr  = m.read()
    lr = lr.replace('\n',' ')
    x = lr.split(' ')
    #print('eeeee',lr)
    while ('' in x):
        x.remove('')
    #print('x',x)
    lr_start = eval(x[-1])
def main():
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-train'.format(cfg.model_type), cfg.respth)
    
    best_prec1=(-1)
    logger = logging.getLogger()
    

    ## model
    net, criteria_pre, criteria_aux = set_model()
    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)

    ## meters
#    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.epoch*371, warmup_iter=cfg.warmup_iters*371,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)
    #lr_schdr = ExponentialLR(optim,gamma=0.9999)
    #lr_schdr = ReduceLROnPlateauPatch(optim,mode='min', factor=0.9999, patience=2)
##########    
    for epoch in range(cfg.start_epoch,args.epoch_to_train): #(cfg.start_epoch, cfg.epoch):
        lr_schdr,time_meter,loss_meter,loss_pre_meter,loss_aux_meters = train(epoch,optim,net,criteria_pre, criteria_aux,lr_schdr)
        if True:
        #if ((epoch+1)!=cfg.epoch):
            print('1')
            lr = lr_schdr.get_lr()
            print(lr)
            lr = sum(lr) / len(lr)
            loss_avg = print_log_msg(
                epoch, cfg.epoch, lr, time_meter, loss_meter,
                loss_pre_meter, loss_aux_meters)
            writer.add_scalar('loss',loss_avg,epoch + 1)
        if ((epoch+1)==cfg.epoch) or ((epoch+1)==args.epoch_to_train):
        #if ((epoch+1)%1==0) and ((epoch+1)>cfg.warmup_iters):    
            torch.cuda.empty_cache()
            heads, mious,miou = eval_model(net,ims_per_gpu=2,im_root=cfg.im_root,im_anns=cfg.val_im_anns,it=epoch)       
            #miou = 0.9
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
                #print('eeeee',x)
                while ('' in x):
                    x.remove('')
                #print('x',x)
                best_miou = eval(x[-1]) #(best_miou)
                #print('best miou from file',best_miou)
                is_best = miou> best_miou
                if is_best:
                    best_miou = miou
                    print('Is best? : ',is_best)
                    f.seek(0)
                    f.write((str(epoch+1)+'   '))
                    f.write(str(best_miou))
                    f.truncate()
                    f.close()
            #writer.add_scalar('mIOU',miou,epoch+1)
            filename = osp.join(cfg.respth, args.store_name)
            state = net.state_dict()
            save_checkpoint(state,False,filename=filename)
            filename = filename+'_'+str(epoch+1)
            logger.info('\nsave models to {}'.format(filename))
            save_checkpoint(state,is_best,filename) 
            print('Have Stored Checkpoint')
        if((epoch+1)==cfg.epoch) or ((epoch+1)==args.epoch_to_train):
            print('3')
            logger.info('\nevaluating the final model')
            filename = osp.join(cfg.respth,'_final_model')
            state = net.state_dict()
           
            torch.cuda.empty_cache()
            #heads, mious = eval_model(net, 1, cfg.im_root, cfg.val_im_anns,it)
            #heads, mious = eval_model(net, 2, cfg.im_root, cfg.val_im_anns,it=epoch)
            logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
            save_checkpoint(state,False,filename)
            print('Have Saved Final Model')
            break

if __name__ == "__main__":
    main()
