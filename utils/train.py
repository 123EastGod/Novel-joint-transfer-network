#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
import os
import time
import warnings
import numpy as np
import torch
import logging
from torch import nn, optim
from datasets import Simulation_data, Electric_locomotive_data
from models.cnn_1d import *
from models.AdversarialNet import *
from loss.JAN import JAN

class trainer(object):
    def __init__(self, args, save_dir):
        self.save_dir = save_dir
        self.args = args

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # load source_dataset
        self.source_dataset = {}
        self.source_dataset['train'], self.source_dataset['val'], self.source_dataset['num_classes'] = Simulation_data(
            normalizetype=args.normalizetype, signal_size=args.signal_size, num_samples=args.num_samples).data_split()

        # load target_dataset
        self.target_dataset = {}
        if args.data_name == 'Electric_locomotive':
            self.target_dataset['train'], self.target_dataset['val'] = Electric_locomotive_data(
                normalizetype=args.normalizetype, signal_size=args.signal_size, num_samples=args.num_samples).data_split()
        else:
            raise Exception("data not implement")

        self.datasets = {'source_train': self.source_dataset['train'], 'source_val': self.source_dataset['val'],
                         'target_train': self.target_dataset['train'], 'target_val': self.target_dataset['val'],
                         'num_classes': self.source_dataset['num_classes']}

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[
                                                               1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

        # define model
        if args.base_model == 'cnn_1d':
            self.model = cnn_features()
        else:
            raise Exception("model not implement")

        self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.outputdim(), args.bottleneck_num),
                                              nn.ReLU(inplace=True), nn.Dropout())
        self.classifier_layer = nn.Linear(args.bottleneck_num, self.datasets['num_classes'])

        self.softmax_layer = nn.Softmax(dim=1)

        self.max_iter = len(self.dataloaders['source_train']) * args.epoch

        self.AdversarialNet = AdversarialNet(in_feature=args.bottleneck_num,
                                                                        hidden_size=args.hidden_size, max_iter=self.max_iter,
                                                                        trade_off_adversarial=args.trade_off_adversarial,
                                                                        lam_adversarial=args.lam_adversarial
                                                                        )

        # Define the learning parameters
        parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                          {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                          {"params": self.classifier_layer.parameters(), "lr": args.lr},
                          {"params": self.AdversarialNet.parameters(), "lr": args.lr}]

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Invert the model and define the loss
        self.model.to(self.device)
        self.bottleneck_layer.to(self.device)
        self.classifier_layer.to(self.device)
        self.AdversarialNet.to(self.device)
        self.softmax_layer = self.softmax_layer.to(self.device)

        self.distance_loss = JAN
        self.criterion_source_weight = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        """
        Training process
        :return:
        """

        args = self.args

        step = 0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0

        for epoch in range(args.epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])
            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # Set model to train mode or test mode
                if phase == 'source_train':
                    self.model.train()
                    self.bottleneck_layer.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    self.bottleneck_layer.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train':
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, _ = iter_target.next()
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    if (step + 1) % len_target_loader == 0:
                        iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        features = self.model(inputs)
                        features = self.bottleneck_layer(features)
                        outputs = self.classifier_layer(features)
                        if phase != 'source_train':
                            logits = outputs
                            loss = self.criterion(logits, labels)
                        else:
                            logits = outputs.narrow(0, 0, labels.size(0))
                            classifier_loss = self.criterion(logits, labels)

                            # Calculate the distance metric
                            softmax_out = self.softmax_layer(outputs)
                            distance_loss = self.distance_loss([features.narrow(0, 0, labels.size(0)),
                                                                softmax_out.narrow(0, 0, labels.size(0))],
                                                               [features.narrow(0, labels.size(0),
                                                                                inputs.size(0) - labels.size(0)),
                                                                softmax_out.narrow(0, labels.size(0),
                                                                                   inputs.size(0) - labels.size(0))],
                                                               )

                            # Calculate the domain adversarial
                            # weight assignment
                            weight_dd = torch.zeros(inputs.size(0)).to(self.device)
                            source_adversarial_out = self.AdversarialNet(features).narrow(0, 0, labels.size(0))
                            source_domain_label = torch.ones(1).float().to(self.device)
                            j = 0
                            for i in source_adversarial_out:
                                source_inconsistent_loss = self.criterion_source_weight(i.view(-1), source_domain_label)
                                weight_dd[j] = source_inconsistent_loss
                                j += 1
                            weight_dd = (weight_dd - (torch.min(weight_dd))) / (torch.max(weight_dd) - torch.min(weight_dd))
                            weight_dd = weight_dd.cuda().view(-1)
                            # weights for target domain are equal to 1
                            for j in range(labels.size(0), inputs.size(0)):
                                weight_dd[j] = 1
                            weight_dd = weight_dd.detach()

                            domain_label_source = torch.ones(labels.size(0)).float()
                            domain_label_target = torch.zeros(inputs.size(0) - labels.size(0)).float()
                            adversarial_label = torch.cat((domain_label_source, domain_label_target), dim=0).to(
                                self.device)
                            adversarial_out = self.AdversarialNet(features)
                            adversarial_loss = nn.BCELoss(weight=weight_dd.view(-1).to(self.device))\
                                (adversarial_out.view(-1), adversarial_label.view(-1))

                            # Calculate the trade off parameter lam
                            if args.trade_off_distance == 'Cons':
                                lam_distance = args.lam_distance
                            elif args.trade_off_distance == 'Step':
                                lam_distance = 2 / (1 + math.exp(-10 * (epoch/args.epoch))) - 1
                            else:
                                raise Exception("trade_off_distance not implement")

                            # loss
                            loss = classifier_loss + lam_distance * distance_loss + adversarial_loss

                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()










