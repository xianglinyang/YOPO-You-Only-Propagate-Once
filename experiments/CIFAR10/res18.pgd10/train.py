from config import config, args
from dataset import create_train_dataset, create_test_dataset
from network import create_network

from lib.utils.misc import save_args, save_checkpoint, load_checkpoint
from lib.training.train import train_one_epoch, eval_one_epoch

import torch
import json
import time
import numpy as np
from tensorboardX import SummaryWriter
import argparse

import os
from collections import OrderedDict


if __name__ == "__main__":

    DEVICE = torch.device('cuda:{}'.format(args.d))
    torch.backends.cudnn.benchmark = True

    net = create_network()
    state_dict = torch.load("E:\\DVI_exp_data\\resnet18_cifar10\\Model\\Epoch_100\\subject_model.pth")
    net.load_state_dict(state_dict)
    net.to(DEVICE)

    criterion = config.create_loss_function().to(DEVICE)

    optimizer = config.create_optimizer(net.parameters())
    lr_scheduler = config.create_lr_scheduler(optimizer)

    ds_train = create_train_dataset(args.batch_size)
    ds_val = create_test_dataset(args.batch_size)

    TrainAttack = config.create_attack_method(DEVICE)
    EvalAttack = config.create_evaluation_attack_method(DEVICE)

    # save dir
    save_dir = os.path.join(".", "Model")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # eval before training
    eval_one_epoch(net, ds_val, DEVICE, EvalAttack)
    torch.save(net.state_dict(), os.path.join(save_dir, "{:d}.pth".format(0)))

    for n_epoch in range(1, 51, 1):

        descrip_str = 'Training epoch:{}/{} -- lr:{}'.format(n_epoch, config.num_epochs, lr_scheduler.get_last_lr())
                                                                           # lr_scheduler.get_lr()[0])
        train_one_epoch(net, ds_train, optimizer, criterion, DEVICE,
                        descrip_str, TrainAttack, adv_coef = args.adv_coef)
        # if config.val_interval > 0 and n_epoch % config.val_interval == 0:
        #     eval_one_epoch(net, ds_val, DEVICE, EvalAttack)
        eval_one_epoch(net, ds_val, DEVICE, EvalAttack)

        lr_scheduler.step()

        if n_epoch % 2 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, "{:d}.pth".format(n_epoch)))
