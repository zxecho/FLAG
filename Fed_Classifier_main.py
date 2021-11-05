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
from Fed_Local_train import IdptTraining, Local_C_Train
from Fed_test import evaluate, FedModelEvaluate
from function_utils import norm, save_model, loss_plot, mkdir, weights_init
from function_utils import save2json, save2pkl, get_labels_stat
from FedAvg import FedWeightedAvg, FedAvg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def fed_main(args, save_path=''):

    if 'mnist' in args.dataset:
        from network32 import Generator, Discriminator, DNetClassifier
    elif args.dataset in ['cifar20']:
        from networks import Generator, Discriminator, DNetClassifier

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    args.device = device

    # 载入数据并进行构造
    # 载入所有选定站点的数据集
    trainset_loader_list, training_fm_n, f_n_list = get_fed_fam_dataset(args,
                                                                        data_name='{}.h5'.format(
                                                                            args.training_dataset_name),
                                                                        split='C_training')

    testset_loader_list, testing_fm_n, f_n_list = get_fed_fam_dataset(args,
                                                                      data_name='{}.h5'.format(
                                                                          args.training_dataset_name),
                                                                      split='C_testing')

    # 建立fed G model
    n_classes = len(args.dataset_label[args.dataset])
    # 对每个家庭的标签进行处理,提取家庭拥有的标签
    each_family_label_stat = get_labels_stat(dataset_name=args.dataset,
                                             training_dataset_name=args.training_dataset_lb_stat)

    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    # 使用数据扩充 data augment
    # load Federated G net
    GNet_file_path = 'Fed_vs_Idpt_mnist_fed_fn20_lb1_NonIID_training_T1'
    pretrained_model_path = './saved_model/{}/FedGNet.pkl'.format(GNet_file_path)
    FedGNet = Generator(n_classes, args.latent_dim, args.num_channels)
    FedCNet = DNetClassifier(n_classes, args.num_channels, args.filter_num)

    if args.use_cuda:
        FedCNet.to(args.device)
        FedGNet.to(args.device)
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        FedGNet.load_state_dict(checkpoint, strict=False)
        print('[Load G model] Load generator network model.')
    else:
        print('[Load G model] Can not Load classifier weights to discriminator.')

    # 为每个客户建立的独立个体拷贝
    family_CNets_list = []

    for _ in range(training_fm_n):
        # 为每个站点新建一个独立的个体网络
        f_C = deepcopy(FedCNet)
        family_CNets_list.append(f_C)

    print('Federated Classifier network :\n', FedCNet)

    FedCNet.train()

    #  ########## training #############

    # 所有参与方进行独立的本地训练
    if args.idpt_usrs_training:
        idpt_eval_acc_list = []
        print('Start local family model training!\n')
        for idp_N, train_dataset, test_dataset, f_name in zip(family_CNets_list, trainset_loader_list,
                                                              testset_loader_list, f_n_list):
            local_name = '{}'.format(f_name)
            print('[Idpt]==========={} Training process==========='.format(local_name))
            IdptTraining(args, idp_N, name=local_name, dataset_loader=train_dataset, save_path=save_path)
            idpt_test_acc = evaluate(args, idp_N, test_dataset, save_path, local_name,
                                     save_file_name='Family_Idpt_eval', epoch='one-time')
            idpt_eval_acc_list.append(idpt_test_acc)
        # 保存每个家庭单独在本地训练集训练训练过程
        idpt_fig, axs = plt.subplots()
        axs.plot(range(len(idpt_eval_acc_list)), idpt_eval_acc_list, 'o')
        plt.savefig(save_path + 'all_fams_idpt_C_eval_acc.png')
        plt.cla()
        # save2json(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
        save2pkl(save_path, idpt_eval_acc_list, 'idpt_classifier_eval_acc')
        # 清除之前独立学习的主循环所占用的显存空间
        torch.cuda.empty_cache()

    # 写入训练过程数据
    fw_name = save_path + 'Fed_main_training_' + 'log.txt'
    fw_fed_main = open(fw_name, 'w+')
    fw_fed_main.write('iter\t loss\t  Eval acc\t \n')

    # 联邦学习主循环
    # 用于记录联邦训练过程数据
    loss_train = []
    best_acc = 0
    best_epoch = 0
    fed_eval_acc_list = []
    # 联邦学习主循环
    with tqdm(range(args.c_training_epochs)) as tq:
        for fed_iter in tq:  # 暂时取消self.args.local_ep *
            tq.set_description('Federated Updating')
            if fed_iter > args.decay_start >= 0:
                frac = (fed_iter - args.decay_start) // args.decay_every
                decay_factor = args.fed_c_lr_decay ** frac
                current_lr = args.fed_c_lr * decay_factor
            else:
                current_lr = args.fed_g_lr

            w_locals = []
            local_losses_list = []
            usrs_weights = []
            # 用于随机抽取指定数量的参与者加入联邦学习
            m = min(args.num_users, training_fm_n)  # 对随机选择的数量进行判断限制
            m = max(m, 1)
            idxs_users = np.random.choice(training_fm_n, m, replace=False)
            for idx in idxs_users:
                local_name = f_n_list[idx]
                w_l, local_loss, train_no = Local_C_Train(args,
                                                          globalCModel=FedCNet,
                                                          globalGModel=FedGNet,
                                                          dataset_loader=trainset_loader_list[idx],
                                                          dataset_lb_stat=each_family_label_stat[local_name],
                                                          client_name=local_name,
                                                          lr=current_lr,
                                                          save_path=save_path)

                # 记录weights
                w_locals.append(copy.deepcopy(w_l))
                # 记录参与方的模型参数权重
                usrs_weights.append(train_no)
                # 记录loss
                local_losses_list.append(local_loss)
            # 使用联邦学习算法更新全局权重
            if args.weights_avg:
                w = norm(usrs_weights)
                w_glob = FedWeightedAvg(w_locals, w, use_soft=False)
            else:
                w_glob = FedAvg(w_locals)

            # 全局模型载入联邦平均化之后的模型参数
            FedCNet.load_state_dict(w_glob)

            # 打印训练过程的loss
            loss_avg = sum(local_losses_list) / len(local_losses_list)

            loss_train.append(loss_avg)
            # 进行federated model 评估

            eval_acc, family_Nets_list = FedModelEvaluate(args, FedCNet, testset_loader_list, save_path, 'FedCNet',
                                                          epoch=fed_iter)
            eval_acc = eval_acc.numpy().item()
            fed_eval_acc_list.append(eval_acc)

            # 在tqdm上打印过程信息
            tq.set_postfix(Avg_loss=loss_avg, Avg_acc=eval_acc)

            # 保存模型
            if best_acc < eval_acc:
                best_acc = eval_acc
                best_epoch = fed_iter
                # save_model_file = save_path.split('/')[2]
                save_model(FedCNet, save_model_file, 'FedCNet')

                # 保存联邦模型在各个本地训练集训练训练过程
                idpt_fig, axs = plt.subplots()
                axs.plot(range(len(family_Nets_list)), family_Nets_list, 'x')
                plt.savefig(save_path + 'all_fams_fed_eval_acc.png')
                plt.cla()
                plt.close()
                # save2json(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
                save2pkl(save_path, family_Nets_list, 'FedNet_eval_acc')
            fw_fed_main.write('{}\t {:.5f}\t {}\t \n'.format(fed_iter, loss_avg, eval_acc))

    # print('Best test acc: ', best_acc, 'Best acc eval epoch: ', best_epoch)
    # save2json(save_path, {'best_acc': best_acc, 'best acc epoch': best_epoch}, 'best_result&epoch')

    # 绘制曲线
    fig, axs = plt.subplots()
    axs.plot(range(len(loss_train)), loss_train, label='Federated Classifier training loss')
    axs.set_xlabel('Communications')
    axs.set_ylabel('Federated training loss')
    plt.legend()
    plt.savefig(save_path + 'FedCNet_training_loss.png')
    # 防止图像绘制太多，关闭不必要的plots
    plt.cla()
    plt.close()

    fig, axs = plt.subplots()
    loss_plot(axs, fed_eval_acc_list, 'FedCNet evaluationg accuracy')
    # plt.savefig(save_path + 'fed_{}.eps'.format(args.epochs))
    plt.savefig(save_path + 'FedCNet_evaluationg_accuracy.png')
    # 保存数据到本地
    save2json(save_path, {'fed_eval_acc': fed_eval_acc_list}, '{}_FedCNet_eval_acc'.format(args.exp_name))

    # 关闭写入
    fw_fed_main.close()
    plt.cla()  # 清除之前绘图
    plt.close()
    # 清空GPU缓存
    torch.cuda.empty_cache()

    return best_acc


if __name__ == "__main__":
    args = args_parser()

    # 做实验
    exp_total_time = 1
    cross_validation_sets = 1

    results_saved_file = 'results'
    results_plot_file = 'plot_results'
    model_saved_file = 'saved_model'
    # 'oulu_Fed_NonIID_L4_training', 'BU3DFE_Fed_NonIID_L4_training', 'jaffe_Fed_NonIID_L4_training'
    params_test_list = [0]  # 用于设置不同参数
    test_param_name = 'local_ep'

    args.c_training_epochs = 200
    args.if_Local_train_C_with_DA = False

    for param in params_test_list:
        # print('**  {} params test: {}  **'.format(test_param_name, param))
        # args.local_ep = param
        # args.training_dataset_name = param
        # args.testing_dataset_name = param
        args.exp_name = 'Fed_vs_Idpt_{}_T1'.format(args.training_dataset_name)
        ex_params_settings = {
            'algo_name': 'FedClassifier',
            # 实验相关参数
            'if_WeightAvg': args.weights_avg,
            'if_grad_clip': args.IdptC_grad_clip,
            'dataset_name': args.training_dataset_name,
            'dataset_label': args.dataset_label[args.dataset],
            # 训练相关参数
            'exp_total_time': exp_total_time,
            'epochs': args.epochs,
            'local_ep': args.local_ep,
            'local_bacthsize': args.local_bs,
            'optimizer': 'SGD',
            'participant num': args.num_users,
            'fed_c_lr': args.fed_c_lr,
            'fed_c_lr_decay': args.fed_c_lr_decay,
            'L2_weight_decay': args.L2_weight_decay,
            'decay_every_step': args.decay_every,
            'idpt_lr': args.local_lr,
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
            print('******* Training epoch {} *******'.format(exp_t))
            save_path_pre = result_save_file + str(exp_t) + '/'
            mkdir(save_path_pre)
            eval_result = fed_main(args, save_path_pre)
            exp_eval_r_list.append(eval_result)

        print('Exps results:  ', exp_eval_r_list, 'Avg best acc: ', np.mean(exp_eval_r_list))
        save2json(result_save_file, {'Exps_results': exp_eval_r_list, 'avg_best_acc': np.mean(exp_eval_r_list)},
                  args.dataset+'_all_avg_acc')