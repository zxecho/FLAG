import os
import copy
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import torch

from param_options import args_parser
from Fed_FamData_load import get_fed_fam_dataset
from Fed_Local_train import Local_CGAN_Train, IdptTraining, Local_C_with_DA_Train
from Fed_test import evaluate, FedModelEvaluate, FedG_eval
from function_utils import norm, save_model, loss_plot, mkdir, weights_init_v2
from function_utils import save2json, save2pkl, loadjson, count_vars, get_labels_stat, get_global_lb_weights
from FedAvg import FedWeightedAvg, FedAvg

from ACGAN_test import ACG_eval_computation, G_eval

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def fed_main(args, save_path=''):
    if 'mnist' in args.dataset:
        from network32 import Generator, Discriminator, DNetClassifier
    elif args.dataset in ['cifar20']:
        from networks import Generator, Discriminator, DNetClassifier

    # 初始化必要对象
    n_classes = len(args.dataset_label[args.dataset])
    fixed_noise = None
    # 保存路径
    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    # 载入数据并进行构造
    # 载入所有选定站点的数据集
    c_trainset_loader_list, c_training_fm_n, c_f_n_list = get_fed_fam_dataset(args,
                                                                              data_name='{}.h5'.format(
                                                                                  args.training_dataset_name),
                                                                              split='C_training',
                                                                              batch_size=args.mix_bs)

    testset_loader_list, testing_fm_n, f_n_list = get_fed_fam_dataset(args,
                                                                      data_name='{}.h5'.format(
                                                                          args.training_dataset_name),
                                                                      split='C_testing')
    # 用于GAN训练
    gan_trainset_loader_list, gan_training_fm_n, gan_f_n_list = get_fed_fam_dataset(args,
                                                                                    data_name='{}.h5'.format(
                                                                                        args.training_dataset_name),
                                                                                    split='GAN_training')
    # 对每个家庭的标签进行处理,提取家庭拥有的标签
    each_family_label_stat, lb_counter_list = get_labels_stat(args.dataset, n_classes, args.training_dataset_lb_stat)
    global_lb_weights = deepcopy(lb_counter_list)

    # 建立fed G model
    FedGNet = Generator(n_classes, args.latent_dim, args.num_channels)
    FedDNet = Discriminator(n_classes, args.num_channels, args.filter_num)
    FedCNet = DNetClassifier(n_classes, args.num_channels, args.filter_num)

    print('Number of each model parameters: G: {} D: {} C: {}'.format(count_vars(FedGNet),
                                                                      count_vars(FedDNet),
                                                                      count_vars(FedCNet)))

    if args.resume:
        pretrained_model_path = './saved_model/{}/FedGNet.pkl'.format(save_model_file)
        if os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path)
            FedGNet.load_state_dict(checkpoint)
        else:
            print('[Local] Can not Load classifier weights to discriminator.')
    else:
        # Init model weights
        FedGNet.apply(weights_init_v2)
        try:
            FedDNet.apply(weights_init_v2)
        except:
            FedDNet.weight_init(0.0, 0.02)

    if args.use_cuda:
        FedGNet.to(args.device)
        FedDNet.to(args.device)
        FedCNet.to(args.device)

    # 为每个客户建立的独立个体拷贝
    family_GNets_list = []
    family_CNets_list = []
    family_DNets_list = []

    for _ in range(c_training_fm_n):
        # 为每个站点新建一个独立的个体网络
        if args.idpt_usrs_training or args.if_Local_train_C_with_DA:
            f_C = deepcopy(FedCNet)
            family_CNets_list.append(f_C)
        f_D = deepcopy(FedDNet)
        family_DNets_list.append(f_D)

    print('Network :\n', FedGNet)

    FedGNet.train()

    #  ########## training #############

    # 所有参与方进行独立的本地训练
    if args.idpt_usrs_training:
        idpt_eval_acc_list = []
        print('Start local family model training!\n')
        for idp_N, train_dataset, test_dataset, f_name in zip(family_CNets_list, c_trainset_loader_list,
                                                              testset_loader_list, f_n_list):
            local_name = '{}'.format(f_name)
            print('[Idpt]==========={} Training process==========='.format(local_name))
            IdptTraining(args, idp_N, name=local_name, dataset_loader=train_dataset, save_path=save_path)
            idpt_test_acc = evaluate(args, idp_N, test_dataset, save_path, local_name,
                                     save_file_name='idpt_eval_info', epoch='one-time')
            idpt_eval_acc_list.append(idpt_test_acc)
        # 保存每个家庭单独在本地训练集训练训练过程
        idpt_fig, axs = plt.subplots()
        axs.plot(range(len(idpt_eval_acc_list)), idpt_eval_acc_list, 'o')
        plt.savefig(save_path + 'all_fams_idpt_C_eval_acc.png')
        plt.cla()
        # save2json(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
        save2pkl(save_path, idpt_eval_acc_list, 'idpt_classifier_eval_acc')

    # 写入训练过程数据
    fw_name = save_path + 'Fed_main_training_' + 'log.txt'
    fw_fed_main = open(fw_name, 'w+')
    fw_fed_main.write('iter\t loss\t  Eval acc\t class_eval_acc\t total_emd\t \n')

    # 联邦学习主循环
    # 用于记录联邦训练过程数据
    labels_history_counter = dict()
    g_loss_train = []
    d_loss_train = []
    best_c_acc = 0
    best_totoal_emd = 0
    c_acc_list = []
    total_emd_list = []
    best_epoch = 0
    # 联邦学习主循环
    # 将每个家庭的分类器模型的部分权重数据迁移到判别器模型
    if args.init_FedD_with_C:
        idpt_pretrained_model = ''
        if not args.idpt_usrs_training:
            if 'mnist' in args.dataset:
                idpt_pretrained_model = args.idpt_pre_trained_c_model
            elif args.dataset in ['cifar20']:
                idpt_pretrained_model = ''
        for fam_DNet, fam_name in zip(family_DNets_list, f_n_list):
            pretrained_model_path = './saved_model/{}/0/idpt_{}.pkl'.format(idpt_pretrained_model, fam_name)
            if os.path.exists(pretrained_model_path):
                checkpoint = torch.load(pretrained_model_path)
                fam_DNet.load_state_dict(checkpoint, strict=False)
            else:
                print('[Local] Can not Load classifier weights to discriminator.')

    for fed_iter in range(args.epochs):  # 暂时取消self.args.local_ep *

        if fed_iter > args.G_decay_start >= 0:
            # G的lr衰减
            g_frac = (fed_iter - args.G_decay_start) // args.G_decay_every
            g_decay_factor = args.fed_g_lr_decay ** g_frac
            current_g_lr = args.fed_g_lr * g_decay_factor
        if fed_iter > args.D_decay_start >= 0:
            # D的lr衰减
            d_frac = (fed_iter - args.D_decay_start) // args.D_decay_every
            d_decay_factor = args.fed_g_lr_decay ** d_frac
            current_d_lr = args.fed_d_lr * d_decay_factor
        else:
            current_g_lr = args.fed_g_lr
            current_d_lr = args.fed_d_lr

        g_w_locals, d_w_locals = [], []
        local_d_losses_list, local_g_losses_list = [], []
        current_clients_weights_dict = dict()
        # 用于随机抽取指定数量的参与者加入联邦学习
        m = min(args.num_users, gan_training_fm_n)  # 对随机选择的数量进行判断限制
        m = max(m, 1)
        idxs_users = np.random.choice(gan_training_fm_n, m, replace=False)
        for idx in idxs_users:
            client_name = f_n_list[idx]
            g_w_l, d_w_l, local_d_loss, local_g_loss, \
            train_no = Local_CGAN_Train(args,
                                        globalGModel=deepcopy(FedGNet),
                                        globalDModel=FedDNet if args.if_Fed_train_D else family_DNets_list[idx],
                                        dataset_loader=gan_trainset_loader_list[idx],
                                        dataset_lb_stat=each_family_label_stat[client_name],
                                        local_g_lr=current_g_lr,
                                        local_d_lr=current_d_lr,
                                        save_path=save_path,
                                        name=client_name)
            # 记录并更新用于聚合权重列表，同时取出当前采样到的参与方的权重
            global_lb_weights, current_clients_weights_dict = get_global_lb_weights(client_name, each_family_label_stat,
                                                                                    global_lb_weights,
                                                                                    current_clients_weights_dict)
            # 记录weights
            g_w_locals.append(copy.deepcopy(g_w_l))
            d_w_locals.append(copy.deepcopy(d_w_l))
            # 记录loss
            local_g_losses_list.append(local_g_loss)
            local_d_losses_list.append(local_d_loss)

        # 使用联邦学习算法更新全局权重
        if args.weights_avg:
            g_w_glob = FedWeightedAvg(g_w_locals, current_clients_weights_dict, use_soft=False)
            if args.if_Fed_train_D:
                d_w_glob = FedWeightedAvg(d_w_locals, current_clients_weights_dict, use_soft=False)
        else:
            g_w_glob = FedAvg(g_w_locals)
            if args.if_Fed_train_D:
                d_w_glob = FedAvg(d_w_locals)

        # 全局模型载入联邦平均化之后的模型参数
        FedGNet.load_state_dict(g_w_glob)
        if args.if_Fed_train_D:
            FedDNet.load_state_dict(d_w_glob)

        # 打印训练过程的loss
        g_loss_avg = sum(local_g_losses_list) / len(local_g_losses_list)
        d_loss_avg = sum(local_d_losses_list) / len(local_d_losses_list)

        # 在tqdm上打印过程信息
        print('[Main federated learning] Avg_G_loss={}, Avg_D_loss={} \n'.format(g_loss_avg.item(), d_loss_avg.item()))

        g_loss_train.append(g_loss_avg)
        d_loss_train.append(d_loss_avg)

        # 进行federated G model 评估 计算classification score 和 EMD distance
        # FedG_eval(FedGNet, FedDNet, save_path, args.latent_dim, n_row=10, main_epoch=fed_iter)
        class_eval_acc, eval_emd, total_emd = ACG_eval_computation(args, FedGNet,
                                                                   save_path=save_path + '/FedGNet_eval/',
                                                                   name='FedGNet',
                                                                   dataset_loader_list=testset_loader_list,
                                                                   n_classes=n_classes,
                                                                   main_epoch=fed_iter)

        # 保存模型评估情况
        c_acc_list.append(class_eval_acc)
        total_emd_list.append(total_emd)

        # 保存联邦模型 FedGNet & FedDNet
        save_model(FedGNet, save_model_file, 'FedGNet')

        if best_c_acc < class_eval_acc:
            best_c_acc = class_eval_acc
            best_epoch = fed_iter
            # 进行最好训练模型的生成器采样
            G_eval(args, FedGNet,
                   save_path=save_path + '/Saved_best_FedGNet_eval/',
                   n_classes=n_classes, main_epoch=fed_iter)

            # 保存联邦模型在各个本地训练集训练训练过程
            idpt_fig, axs = plt.subplots()
            axs.plot(range(len(eval_emd)), eval_emd)
            axs.set_ylabel('EMD')
            axs.set_xlabel('Families')
            plt.savefig(save_path + 'fedG_eval_each_family_emd.png')
            plt.cla()
            plt.close()
            # save2json(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
        fw_fed_main.write('{}\t {:.5f}\t {:.5f}\t {:.5f}\t {:.5f}\t \n'.format(fed_iter, g_loss_avg, d_loss_avg,
                                                                               class_eval_acc, total_emd))
    # 保存联邦学习的评估过程统计
    idpt_fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].plot(range(len(c_acc_list)), c_acc_list)
    axs[0].set_ylabel('Classification Scores')
    axs[1].plot(range(len(total_emd_list)), total_emd_list)
    axs[1].set_ylabel('Total EMD')
    axs[1].set_xlabel('Communications')
    plt.savefig(save_path + 'fedG_eval_acc_score&emd.png')
    plt.cla()
    plt.close()
    save2pkl(save_path, c_acc_list, 'FedGNet_eval_acc_scores')
    save2pkl(save_path, total_emd_list, 'FedGNet_eval_emd')

    # print('Best test acc: ', best_acc, 'Best acc eval epoch: ', best_epoch)
    # save2json(save_path, {'best_acc': best_acc, 'best acc epoch': best_epoch}, 'best_result&epoch')

    # 绘制曲线
    fig, axs = plt.subplots()
    axs.plot(range(len(g_loss_train)), g_loss_train, label='Federated G training loss')
    axs.plot(range(len(g_loss_train)), d_loss_train, label='Federated D training loss')
    plt.legend()
    plt.savefig(save_path + 'FedGNet_training_loss.png')
    plt.cla()

    # 关闭写入
    fw_fed_main.close()
    plt.cla()  # 清除之前绘图
    plt.close()

    # ----------
    # Training local classifier with DA or Not
    # ----------
    if args.if_Local_train_C_with_DA:
        print('[Main] Local classifier trianing with DA...')
        local_DAC_eval_acc_list = []
        for fam_c, train_dataset, test_dataset, local_name in zip(family_CNets_list, c_trainset_loader_list,
                                                                  testset_loader_list, f_n_list):

            # 载入之前本地单独训练的模型
            if args.resume_idpt_model:
                pretrained_model_path = './saved_model/{}/Idpt_{}.pkl'.format(save_model_file, local_name)
                if os.path.exists(pretrained_model_path):
                    checkpoint = torch.load(pretrained_model_path)
                    fam_c.load_state_dict(checkpoint, strict=False)
                else:
                    print('[Local] Can not Load pre-trained classifier weights to current empty classifier.')

            Local_C_with_DA_Train(args, globalGModel=FedGNet,
                                  globalCModel=fam_c,
                                  dataset_loader=train_dataset,
                                  dataset_lb_stat=each_family_label_stat[local_name],
                                  save_path=save_path,
                                  name=local_name)
            idpt_test_acc = evaluate(args, fam_c, test_dataset, save_path, local_name,
                                     save_file_name='Local_C_with_DA_eval_info', epoch='one-time')
            local_DAC_eval_acc_list.append(idpt_test_acc)

        # 保存每个家庭单独在本地使用增强数据训练集训练的过程
        idpt_fig, axs = plt.subplots()
        axs.plot(f_n_list, local_DAC_eval_acc_list, 'o')
        plt.xticks(rotation=45)
        plt.savefig(save_path + 'all_fams_local_DAC_eval_acc.png')
        plt.cla()
        plt.close()
        # save2json(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
        save2pkl(save_path, local_DAC_eval_acc_list, 'local_DAclassifier_eval_acc')
    else:
        local_DAC_eval_acc_list = 0

    # 清空GPU缓存
    torch.cuda.empty_cache()

    return local_DAC_eval_acc_list


def Train_Local_C_with_DA(args, save_path):
    if args.dataset in ['mnist']:
        from network32 import Generator, Discriminator, DNetClassifier
    elif args.dataset in ['cifar20']:
        from networks import Generator, Discriminator, DNetClassifier

    # 初始化必要对象
    n_classes = len(args.dataset_label[args.dataset])
    # 对每个家庭的标签进行处理,提取家庭拥有的标签
    each_family_label_stat = get_labels_stat(args.dataset, args.training_dataset_lb_stat)

    # 保存路径
    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    # 载入所有选定站点的数据集
    c_trainset_loader_list, c_training_fm_n, c_f_n_list = get_fed_fam_dataset(args,
                                                                              data_name='{}.h5'.format(
                                                                                  args.training_dataset_name),
                                                                              split='C_training',
                                                                              batch_size=args.mix_bs)

    testset_loader_list, testing_fm_n, f_n_list = get_fed_fam_dataset(args,
                                                                      data_name='{}.h5'.format(
                                                                          args.training_dataset_name),
                                                                      split='C_testing')
    print('[Main] Local classifier trianing with DA...')
    FedGNet = Generator(n_classes, args.latent_dim, args.num_channels)
    FedCNet = DNetClassifier(args.n_classes, args.num_channels, args.filter_num)

    if args.use_cuda:
        FedGNet.to(args.device)
        FedCNet.to(args.device)

    # 为每个客户建立的独立个体拷贝
    family_CNets_list = []

    for _ in range(c_training_fm_n):
        # 为每个站点新建一个独立的个体网络
        if args.idpt_usrs_training or args.if_Local_train_C_with_DA:
            f_C = deepcopy(FedCNet)
            family_CNets_list.append(f_C)

    # load Federated G net
    pretrained_model_path = './saved_model/{}/FedGNet.pkl'.format(save_model_file)
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        FedGNet.load_state_dict(checkpoint, strict=False)
    else:
        print('[Local] Can not Load pre-trained generator weights.')

    local_DAC_eval_acc_list = []
    for fam_c, train_dataset, test_dataset, local_name in zip(family_CNets_list, c_trainset_loader_list,
                                                              testset_loader_list, f_n_list):

        # 载入之前本地单独训练的模型
        if args.resume_idpt_model:
            pretrained_model_path = './saved_model/{}/Idpt_{}.pkl'.format(save_model_file, local_name)
            if os.path.exists(pretrained_model_path):
                checkpoint = torch.load(pretrained_model_path)
                fam_c.load_state_dict(checkpoint, strict=False)
            else:
                print('[Local] Can not Load pre-trained classifier weights to current empty classifier.')

        Local_C_with_DA_Train(args, globalGModel=FedGNet,
                              globalCModel=fam_c,
                              dataset_loader=train_dataset,
                              dataset_lb_stat=each_family_label_stat[local_name],
                              save_path=save_path,
                              name=local_name)
        idpt_test_acc = evaluate(args, fam_c, test_dataset, save_path, local_name,
                                 save_file_name='Local_C_with_DA({})_eval_info'.format(args.mix_ritio),
                                 epoch='one-time')
        local_DAC_eval_acc_list.append(idpt_test_acc)

    # 保存每个家庭单独在本地使用增强数据训练集训练的过程
    idpt_fig, axs = plt.subplots()
    axs.plot(f_n_list, local_DAC_eval_acc_list, 'o')
    plt.xticks(rotation=45)
    plt.savefig(save_path + 'all_fams_local_DAC({})_eval_acc.png'.format(args.mix_ritio))
    plt.cla()
    plt.close()
    print('All participants average: ', np.mean(local_DAC_eval_acc_list))
    save2pkl(save_path, local_DAC_eval_acc_list, 'local_DAclassifier({})_eval_acc'.format(args.mix_ritio))

    return local_DAC_eval_acc_list


if __name__ == "__main__":
    args = args_parser()

    # 做实验
    exp_total_time = 3
    cross_validation_sets = 1

    results_saved_file = 'results'
    results_plot_file = 'plot_results'
    model_saved_file = 'saved_model'
    # 'oulu_Fed_NonIID_L4_training', 'BU3DFE_Fed_NonIID_L4_training', 'jaffe_Fed_NonIID_L4_training'
    params_test_list = [False]  # 用于设置不同参数
    test_param_name = 'init_FedD_with_C'
    args.if_Fed_train_D = True
    args.weights_avg = False
    args.init_FedD_with_C = False

    for param in params_test_list:
        # print('**  {} params test: {}  **'.format(test_param_name, param))
        # args.local_ep = param
        # args.training_dataset_name = param
        # args.testing_dataset_name = param
        # 用于设置不同的参数量
        # args.fed_d_lr_decay = param   # decay rate
        # args.fed_g_lr_decay = param  # decay rate
        # args.fed_d_lr = param
        # args.fed_d_lr_decay = param[1]
        # args.mix_ritio = param
        # args.init_FedD_with_C = param
        # 用于多参数实验
        # args.exp_name = 'FedDA_CGAN_{}_FedG_MixAllLalels_baseline_T1({}{})'.format(args.training_dataset_name,
        #                                                                    test_param_name, param)
        args.exp_name = 'FedDA_vanilla_CGAN_{}_MixAllLalels_baseline_T2'.format(args.training_dataset_name)
        ex_params_settings = {
            'algo_name': 'FedDA',
            # GAN算法相关参数
            'latent_dim': args.latent_dim,
            'init_input_size': args.init_imgsize,
            'filter_number': args.filter_num,
            # 实验相关参数
            'if_WeightAvg': args.weights_avg,
            'if_DAC_grad_clip': args.DAC_grad_clip,
            'if_Fed_train_D': args.if_Fed_train_D,
            'if_Fed_train_C': args.if_Fed_train_C,
            'init_FedD_with_C': args.init_FedD_with_C,
            'dataset': args.dataset,
            'dataset_label': args.dataset_label[args.dataset],
            'mix ratio': args.mix_ritio,
            # 训练相关参数
            'exp_total_time': exp_total_time,
            'epochs': args.epochs,
            'local_ep': args.local_ep,
            'local_bacthsize': args.local_bs,
            'activate_function': 'LeakReLU',
            'optimizer': args.optimizer_type,
            'participant num': args.num_users,
            'fed_g_lr': args.fed_g_lr,
            'fed_g_lr_decay_fractor': args.fed_g_lr_decay,
            'fed_d_lr': args.fed_d_lr,
            'fed_d_lr_decay_fractor': args.fed_d_lr_decay,
            'L2_weight_decay': args.L2_weight_decay,
            'G_decay_every_step': args.G_decay_start,
            'G_decay_start': args.G_decay_every,
            'D_decay_every_step': args.D_decay_start,
            'D_decay_start': args.D_decay_every,
            'idpt_lr': args.local_lr,
            # DA TRAINING
            # 'DA_c_decay_steps': args.fed_c_decay_steps,
            # 'fed_c_lr_decay': args.fed_c_lr_decay,
            'DA_c_lr': args.local_lr,
        }

        # 存储主文件路径
        result_save_file = './{}/'.format(results_saved_file) + args.exp_name + '/'
        mkdir(result_save_file)

        # 保存参数配置
        params_save_name = result_save_file + 'params_settings.json'
        with open(params_save_name, 'w+') as jsonf:
            json.dump(ex_params_settings, jsonf)

        exp_eval_r_list = []

        for exp_t in range(exp_total_time):
            print('******* Training time {} *******'.format(exp_t))
            save_path_pre = result_save_file + str(exp_t) + '/'
            mkdir(save_path_pre)
            if args.if_Fed_train_GAN:
                eval_result = fed_main(args, save_path_pre)
                exp_eval_r_list.append(np.mean(eval_result).item())
            else:
                eval_result = Train_Local_C_with_DA(args, save_path_pre)
                exp_eval_r_list.append(np.mean(eval_result).item())

        if args.if_Fed_train_GAN:
            print('Exps classifier results:  ', exp_eval_r_list, 'Avg best acc: ', np.mean(exp_eval_r_list).item())
            save2json(result_save_file,
                      {'Exps_results': exp_eval_r_list, 'avg_best_acc': np.mean(exp_eval_r_list).item()},
                      args.dataset + '_all_avg_acc_with_DA({})'.format(args.mix_ritio))
