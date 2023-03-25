#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import argparse
import os
from datetime import datetime
import logging
import warnings

from utils.logger import setlogger
from utils.train import trainer


print(torch.__version__)
warnings.filterwarnings('ignore')

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model parameters
    parser.add_argument('--model_name', type=str, default='NJTN', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='Electric_locomotive', choices=['Electric_locomotive', 'QPZZ-II'])
    parser.add_argument('--signal_size', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=400)
    parser.add_argument('--normalizetype', type=str, default='mean-std', choices=['-1-1', 'mean-std'], help='normalization type')
    parser.add_argument('--base_model', type=str, default='cnn_1d', choices=['cnn_1d'])

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--last_batch', type=bool, default=False, help='whether using the last batch')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')

    # distance loss
    parser.add_argument('--trade_off_distance', type=str, default='Step', help='regularization coefficient')
    parser.add_argument('--lam_distance', type=float, default=1, help='this is used for Cons')

    # adversarial loss
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')
    parser.add_argument('--trade_off_adversarial', type=str, default='Step', help='regularization coefficient')
    parser.add_argument('--lam_adversarial', type=float, default=1, help='this is used for Cons')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='40, 80', help='the learning rate decay for step and stepLR')

    #
    parser.add_argument('--epoch', type=int, default=120)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = trainer(args, save_dir)
    trainer.setup()
    trainer.train()



