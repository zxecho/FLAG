import os
import copy
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from param_options import args_parser
from Fed_FamData_load import get_dataset
from Fed_test import evaluate
from networks import VGG
from function_utils import clip_gradient, set_lr, save_model, plot_confusion_matrix, loss_plot, mkdir
from function_utils import save2json, save2pkl


def IdptTraining(args, IdptModel, name, dataset_loader, save_path):
    local_loss_list = []
    local_acc_list = []
    eval_acc_list = []
    pre_acc = 0
    best_acc = 0

    Epochs = args.c_training_epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(IdptModel.parameters(), lr=args.fed_c_lr, momentum=0.9,
                                weight_decay=args.L2_weight_decay)
    # optimizer = torch.optim.Adam(IdptModel.parameters(), lr=args.fed_c_lr, weight_decay=args.L2_weight_decay)
    with tqdm(range(Epochs)) as tq:
        tq.set_description('Local {} Training Epoch: '.format(name))
        for epoch in tq:
            # IdptModel.cuda()
            IdptModel.train()
            train_loss = 0
            correct = 0
            total = 0

            if epoch > args.fed_c_decay_start >= 0:
                frac = (epoch - args.fed_c_decay_start) // args.fed_c_decay_steps
                decay_factor = args.fed_c_lr_decay ** frac
                current_lr = args.fed_c_lr * decay_factor
                set_lr(optimizer, current_lr)  # set the decayed rate
            else:
                current_lr = args.fed_c_lr

            # print('Idpt learning_rate: %s' % str(current_lr))

            tq.set_postfix(Lr=current_lr)

            for batch_idx, (inputs, targets) in enumerate(dataset_loader):
                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                # inputs, targets = Variable(inputs), Variable(targets)
                outputs = IdptModel(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if args.IdptC_grad_clip:
                    clip_gradient(optimizer, 0.1)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            pre_acc = torch.true_divide(correct, total)

            local_loss_list.append(train_loss / (batch_idx + 1))
            local_acc_list.append(pre_acc)

            # 保存模型的文件夹名称
            # file_name = save_pth_pre.split('/')[2]
            file_name = save_path.split('/')
            file_name = file_name[2] + '/' + file_name[3]

            if pre_acc > best_acc:
                best_acc = pre_acc
                save_model(IdptModel, file_name, 'VGG11_{}'.format(name))

            # 模型评估
            idpt_test_acc = evaluate(args, FedCNet, testset_loader, save_path_pre, 'All_IID',
                                     save_file_name='VGG11_Classifier_eval', epoch='one-time')

            eval_acc_list.append(idpt_test_acc)

        # tqdm打印信息
        tq.set_postfix(Acc=best_acc)

        fig, axs = plt.subplots()
        axs.plot(range(len(local_loss_list)), local_loss_list, label='Training loss')
        axs.set_xlabel('Solo classifier training epochs')
        axs.plot(range(len(local_acc_list)), local_acc_list, label='Test accuracy')
        plt.savefig(save_path + 'Idpt{}_classifier_training_loss&acc.png'.format(name))
        plt.cla()
        plt.close()

    return eval_acc_list


args = args_parser()

args.training_dataset_name = 'FER48_mixed_IID_training'          # 'FashionMNIST_IID_training'

# 做实验
exp_total_time = 1
cross_validation_sets = 1

results_saved_file = 'test_results'
results_plot_file = 'plot_results'
model_saved_file = 'saved_model'

args.exp_name = 'Idpt_classifier_{}_All_IID_size32_channel1'.format(args.dataset)

# 存储主文件路径
result_save_file = './{}/'.format(results_saved_file) + args.exp_name + '/'
mkdir(result_save_file)

save_path_pre = result_save_file
mkdir(save_path_pre)

device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
args.device = device

# 载入数据并进行构造
# 载入所有选定站点的数据集
trainset_loader = get_dataset(args,
                              data_file=args.dataset,
                              data_name='{}.h5'.format(
                                  args.training_dataset_name),
                              split='C_training')

testset_loader = get_dataset(args,
                             data_file=args.dataset,
                             data_name='{}.h5'.format(
                                 args.training_dataset_name),
                             split='C_testing')

# 建立fed G model
n_classes = len(args.dataset_label[args.dataset])

# FedCNet = DNetClassifier(n_classes, args.num_channels, args.init_imgsize)
FedCNet = VGG('VGG11', args.num_channels, n_classes)

if args.use_cuda:
    FedCNet.cuda()

file_name = save_path_pre.split('/')
save_model_file = file_name[2] + '/' + file_name[3]

print('Start training!\n')

print('===========Training process===========')
best_acc_list = IdptTraining(args, FedCNet, name='All_IID', dataset_loader=trainset_loader, save_path=save_path_pre)

# 保存联邦模型在各个本地训练集训练训练过程
idpt_fig, axs = plt.subplots()
axs.plot(range(len(best_acc_list)), best_acc_list)
plt.savefig(save_path_pre + 'all_fams_fed_eval_acc.png')
plt.cla()
# save2json(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
save2pkl(save_path_pre, best_acc_list, 'FedNet_eval_acc')
