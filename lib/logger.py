#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import logging

def setup_logger(name, logpth):
    logfile = '{}-{}.log'.format(name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def print_log_msg(it, max_iter, lr, time_meter, loss_meter):
    t_intv, eta = time_meter.get()
    loss_avg, _ = loss_meter.get()
    msg = ', '.join([
        'iter: {it}/{max_it}',
        'lr: {lr:4f}',
        'eta: {eta}',
        'time: {time:.2f}',
        'loss: {loss:.4f}',
    ]).format(
        it=it+1,
        max_it=max_iter,
        lr=lr,
        time=t_intv,
        eta=eta,
        loss=loss_avg)
    logger = logging.getLogger()
    logger.info(msg)
    return loss_avg
