import os
import sys
father_dir = os.path.join('/',  *os.path.realpath(__file__).split(os.path.sep)[:-2])
#print(father_dir)
if not father_dir in sys.path:
    sys.path.append(father_dir)
from lib.utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm

def train_one_epoch(net, batch_generator, optimizer,
                    criterion, DEVICE=torch.device('cuda:0'),
                    descrip_str='Training', AttackMethod = None, adv_coef = 1.0, save=False):
    '''

    :param attack_freq:  Frequencies of training with adversarial examples. -1 indicates natural training
    :param AttackMethod: the attack method, None represents natural training
    :return:  None    #(clean_acc, adv_acc)
    '''
    net.train()
    pbar = tqdm(batch_generator)
    advacc = -1
    advloss = -1
    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descrip_str)
    adv_dataset = torch.Tensor([]).to(DEVICE)
    adv_labels = torch.Tensor([]).to(DEVICE)
    for i, (data, label) in enumerate(pbar):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        pbar_dic = OrderedDict()
        TotalLoss = 0

        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, data, label)
            # save adv results for DVI
            adv_dataset = torch.cat((adv_dataset, adv_inp.detach()),0)
            adv_labels = torch.cat((adv_labels.detach(), label), 0)

            optimizer.zero_grad()
            net.train()
            pred = net(adv_inp)
            loss = criterion(pred, label)

            acc = torch_accuracy(pred, label, (1,))
            advacc = acc[0].item()
            advloss = loss.item()
            #TotalLoss = TotalLoss + loss * adv_coef
            (loss * adv_coef).backward()


        pred = net(data)

        loss = criterion(pred, label)
        #TotalLoss = TotalLoss + loss
        loss.backward()
        #TotalLoss.backward()
        #param = next(net.parameters())
        #grad_mean = torch.mean(param.grad)

        optimizer.step()
        acc = torch_accuracy(pred, label, (1,))
        cleanacc = acc[0].item()
        cleanloss = loss.item()
        if save:
            batch_dir = os.path.join(".", "Model", "Epoch_{}".format(i))
            if not os.path.exists(batch_dir):
                os.mkdir(batch_dir)
            net.eval()
            torch.save(net.state_dict(), os.path.join(batch_dir, "subject_model.pth"))
            torch.save(adv_inp.detach(), os.path.join(batch_dir, "adv_dataset.pth"))
            torch.save(label.detach(), os.path.join(batch_dir, "adv_labels.pth"))
            net.train()
        #pbar_dic['grad'] = '{}'.format(grad_mean)
        pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['loss'] = '{:.2f}'.format(cleanloss)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(advacc)
        pbar_dic['Advloss'] = '{:.2f}'.format(advloss)
        pbar.set_postfix(pbar_dic)
    return adv_dataset, adv_labels


def eval_one_epoch(net, batch_generator,  DEVICE=torch.device('cuda:0'), AttackMethod = None):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    for (data, label) in pbar:
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        with torch.no_grad():
            pred = net(data)
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item())

        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, data, label)

            with torch.no_grad():
                pred = net(adv_inp)
                acc = torch_accuracy(pred, label, (1,))
                adv_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(adv_accuracy.mean)

        pbar.set_postfix(pbar_dic)

        adv_acc = adv_accuracy.mean if AttackMethod is not None else 0
    return clean_accuracy.mean, adv_acc
