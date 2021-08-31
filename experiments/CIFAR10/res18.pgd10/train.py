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
    # for reproduce purpose
    torch.manual_seed(1331)
    DEVICE = torch.device('cuda:{}'.format(args.d))
    torch.backends.cudnn.benchmark = True

    net = create_network()
    # 190-th epoch of cifar10 resnet18 training
    state_dict = torch.load("/home/xianglin/DVI_exp_data/case_studies/adv_training_batch/subject_model.pth")
    net.load_state_dict(state_dict)
    net.to(DEVICE)

    criterion = config.create_loss_function().to(DEVICE)

    optimizer = config.create_optimizer(net.parameters())
    lr_scheduler = config.create_lr_scheduler(optimizer)

    ds_train = create_train_dataset(args.batch_size)
    ds_val = create_test_dataset(args.batch_size)
    # save dataset
    # train_data = torch.Tensor([]).to(DEVICE)
    # train_label = torch.Tensor([]).to(DEVICE)
    # test_data = torch.Tensor([]).to(DEVICE)
    # test_label = torch.Tensor([]).to(DEVICE)
    # for i, (data, labels) in enumerate(ds_train, 0):
    #     train_data = torch.cat((train_data, data.to(DEVICE).detach()), 0)
    #     train_label = torch.cat((train_label, labels.to(DEVICE).detach()), 0)
    # for i, (data, labels) in enumerate(ds_val, 0):
    #     test_data = torch.cat((test_data, data.to(DEVICE).detach()), 0)
    #     test_label = torch.cat((test_label, labels.to(DEVICE).detach()), 0)
    # torch.save(train_data, os.path.join("." , "training_dataset_data.pth"))
    # torch.save(test_data, os.path.join(".", "testing_dataset_data.pth"))
    # torch.save(train_label, os.path.join(".", "training_dataset_label.pth"))
    # torch.save(test_label, os.path.join(".", "testing_dataset_label.pth"))

    TrainAttack = config.create_attack_method(DEVICE)
    EvalAttack = config.create_evaluation_attack_method(DEVICE)

    # save dir
    save_dir = os.path.join(".", "Model")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # eval before training
    eval_one_epoch(net, ds_val, DEVICE, EvalAttack)
    epoch_dir = os.path.join(save_dir, "Epoch_{}".format(0))
    if not os.path.exists(epoch_dir):
        os.mkdir(epoch_dir)
    # torch.save(net.state_dict(), os.path.join(epoch_dir, "subject_model.pth"))

    for n_epoch in range(1, 100, 1):

        descrip_str = 'Training epoch:{}/{} -- lr:{}'.format(n_epoch, config.num_epochs,lr_scheduler.get_last_lr())
                                                                           # lr_scheduler.get_lr()[0])
        adv_dataset, adv_labels = train_one_epoch(net, ds_train, optimizer, criterion, DEVICE, descrip_str, TrainAttack,
                                                  adv_coef = args.adv_coef, save=False)
        eval_one_epoch(net, ds_val, DEVICE, EvalAttack)

        lr_scheduler.step()
        epoch_dir = os.path.join(save_dir, "Epoch_{}".format(n_epoch))
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)
        torch.save(net.state_dict(), os.path.join(epoch_dir, "subject_model.pth"))
        torch.save(adv_dataset, os.path.join(epoch_dir, "adv_dataset.pth"))
        torch.save(adv_labels, os.path.join(epoch_dir, "adv_labels.pth"))