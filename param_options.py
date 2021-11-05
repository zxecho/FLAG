import argparse
from typing import Dict

datasets_label = {'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                  'fashion_mnist_fed': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                  'cifar20': ['bottles', 'bowls', 'cans', 'cups', 'plates',
                              'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                              'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                              'bed', 'chair', 'couch', 'table', 'wardrobe', 'fake'],
                  'FER': ['Anger', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']}

dataset_mean_std = {'oulu_Fed_NonIID_L4_training': {'mean': [0.36418, 0.36418, 0.36418],
                                                    'std': [0.20384, 0.20384, 0.20384]},
                    'BU3DFE_Fed_NonIID_L4_training': {'mean': [0.27676, 0.27676, 0.27676],
                                                      'std': [0.26701, 0.26701, 0.26701]},
                    'CK_Fed_NonIID_L4_training': {'mean': [0.51194, 0.51194, 0.51194],
                                                  'std': [0.28913, 0.28913, 0.28913]},
                    'jaffe_Fed_NonIID_L4_training': {'mean': [0.43192, 0.43192, 0.43192],
                                                     'std': [0.27979, 0.27979, 0.27979]},
                    'robot_FE_family_dataset': {'mean': [0.35028, 0.35028, 0.35028],
                                                'std': [0.18529, 0.18529, 0.18529]},
                    'cifar20': {'mean': [0.50708, 0.48655, 0.44092],
                                'std': [0.26733, 0.25644, 0.27615]},
                    }


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Federated Training')

    # general params
    parser.add_argument('--model', type=str, default='', help='CNN architecture')
    parser.add_argument('--n_classes', type=int, default=6, help="num of classes")
    parser.add_argument('--dataset', type=str, default='FER', help='mnist, cifar20, fashion_mnist, FER')
    parser.add_argument('--training_dataset_name', type=str, default='FER48_fed_NonIID_training',
                        help='training dataset name ('
                             'fashion_mnist_fed_fn20_lb1_NonIID_training'   # fashion-mnist
                             'FER48_fed_NonIID_training'    # FER
                             'mnist_fed_fn20_lb1_NonIID_training, '
                             'mnist_fed2_fn20_lb1_NonIID_training,'     # mnist 用
                             'mnist_fed2_fn10_lb1_NonIID_training)')
    parser.add_argument('--training_dataset_lb_stat', type=str, default='FER48_fed_NonIID_training_stat',
                        help='(FER48_fed_NonIID_training_stat'          # FER
                             'mnist_fed_fn20_lb1_NonIID_stat,'
                             'mnist_fed2_fn20_lb1_NonIID_stat,'         # mnist
                             'mnist_fed2_fn10_lb1_NonIID_stat,'
                             'fashion_mnist_fed_fn20_lb1_NonIID_stat)')     # fashion-mnist
    parser.add_argument('--idpt_pre_trained_c_model', type=str,
                        default='FedDA_ACGAN_FER48_fed_NonIID_training_FedG_MixAllLalels_Test',
                        help='(FedDA_ACGAN_mnist_fed_fn20_lb1_NonIID_training_FedGD_T1,'       # mnist
                             'Fed_vs_Idpt_fashion_mnist_fed_fn20_lb1_NonIID_training_T1)')  # fashion_mnist
    parser.add_argument('--testing_dataset_name', type=str, default='fed_test_MNIST_1', help='test dataset name')
    parser.add_argument('--dataset_label', type=Dict, default=datasets_label, help='datasets label')
    parser.add_argument('--dataset_mean_std', default=dataset_mean_std, type=Dict,
                        help='mean and std of data set for normalization')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--test_bs', default=64, type=int, help='test batch_size')
    parser.add_argument('--cut_size', default=44, type=int, help='cut size')
    # other arguments
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name')
    parser.add_argument('--saved_path', type=str, default='./saved_model/', help='Experiment name')
    parser.add_argument('--use_amp', action='store_true', help='whether AUTOMATIC MIXED PRECISION(混合精度) or not')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether using cuda')
    parser.add_argument('--device', type=str, default='cuda:0', help='which device to use')
    parser.add_argument('--save_frequency', type=int, default=10, help='save model frequency')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    # 实验相关
    # ===========  常用实验调节门阀  =====================
    parser.add_argument('--idpt_usrs_training', type=bool, default=False, help='If train family ')
    parser.add_argument('--if_Fed_train_GAN', type=bool, default=True, help='If train FED GAN net')
    parser.add_argument('--if_Local_train_C_with_DA', type=bool, default=True, help='If train family classifier net locally with DA data')
    parser.add_argument('--if_Fed_train_D', type=bool, default=False, help='If train family discriminator net federally')
    parser.add_argument('--init_FedD_with_C', type=bool, default=True, help='whether using Classifier model to init D weights')
    parser.add_argument('--mix_ritio', type=int, default=1, help='the ratio of real samples and fake samples')
    parser.add_argument('--mix_bs', default=64, type=int, help='test batch_size')

    # =============== 其他备用门阀 =================
    parser.add_argument('--resume', '-r', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--IdptC_grad_clip', type=bool, default=False, help='If clip gradients')
    parser.add_argument('--DAC_grad_clip', type=bool, default=False, help='If clip gradients when training C with DA')
    parser.add_argument('--if_Fed_train_C', type=bool, default=True, help='If train family classifier net federally')
    parser.add_argument('--resume_idpt_model', type=bool, default=False, help='If load idpt classifier model')

    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--latent_dim', type=int, default=100, help="dim of latent")
    parser.add_argument('--filter_num', type=int, default=32, help="initial the number of filters")
    parser.add_argument('--init_imgsize', type=int, default=32, help="convTran init size")
    parser.add_argument('--init_size', type=int, default=32, help="convTran init size")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--weights_avg', type=bool, default=True, help="if use weights avg")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    # Clasifier training params
    parser.add_argument('--c_training_epochs', type=int, default=50, help="local classifier training epochs")

    #  ========= optimizer 相关 ==========
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='The type of optimizer')
    parser.add_argument('--L2_weight_decay', type=float, default=1e-5, help="L2 weight decay")
    parser.add_argument('--G_decay_start', type=int, default=10, help="learning rate decay start")
    parser.add_argument('--G_decay_every', type=int, default=5, help="learning rate decay ")
    parser.add_argument('--D_decay_start', type=int, default=10, help="learning rate decay start")
    parser.add_argument('--D_decay_every', type=int, default=5, help="learning rate decay ")
    # local model parms
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--local_lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--local_lr_decay', type=float, default=0.8, help="D learning rate decay rate")

    # 专门用于联邦的生成网络
    parser.add_argument('--fed_g_lr', type=float, default=1e-3, help="Fed G learning rate")
    parser.add_argument('--fed_g_lr_decay', type=float, default=0.9, help="Fed G learning rate decay rate")
    parser.add_argument('--fed_d_lr', type=float, default=1e-3, help="Fed D learning rate")
    parser.add_argument('--fed_d_lr_decay', type=float, default=0.9, help="Fed D learning rate decay rate")
    parser.add_argument('--fed_c_lr', type=float, default=2e-3, help="Fed C learning rate")
    parser.add_argument('--fed_c_decay_start', type=int, default=5, help="Fed C learning rate decay start")
    parser.add_argument('--fed_c_lr_decay', type=float, default=0.95, help="Fed C learning rate decay rate")
    parser.add_argument('--fed_c_decay_steps', type=int, default=5, help="Fed C learning rate decay every N steps")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--n_critic', type=int, default=5, help="WGAN n critic")
    parser.add_argument('--clip_value', type=int, default=0.01, help="WGAN discriminator clip value")

    args = parser.parse_args()
    return args
