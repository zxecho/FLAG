import os
import copy
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from param_options import args_parser
from Fed_FamData_load import get_dataset
from Fed_test import evaluate, fixed_sampled_z
from networks import DNetClassifier, VGG_Models, ConvTranGenerator, ConvTranDiscriminator
# from FedDA.networks import Generator, Discriminator   # for image size 64
from network32 import Generator, Discriminator
from function_utils import save2json, save2pkl, loadjson
from function_utils import clip_gradient, set_lr, save_model, plot_confusion_matrix, save2json, mkdir
from function_utils import generate_augment_samples, weights_init_v2
from Fed_test import show_G_eval

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def WGPGAN_Train(args, save_path):
    """
    联邦下的本地模型训练函数, Wasserstain GP GAN
    :param args: 参数配置
    :param save_path: 保存路径
    :return:
    """

    # 初始化必要对象
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.use_cuda else "cpu")
    fixed_z = fixed_sampled_z(args.n_classes, args.latent_dim)  # 采样的固定随机向量
    # 保存路径
    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    # 载入数据并进行构造
    # 用于GAN训练
    trainset_loader = get_dataset(args,
                                  data_name='{}.h5'.format(
                                      args.training_dataset_name),
                                  split='GAN_training')

    # 有多少类数据
    n_classes = len(args.dataset_label[args.dataset]) - 1
    Epochs = 500

    # 用于保存模型
    file_name = save_path.split('/')
    file_name = file_name[2] + '/' + file_name[3]

    globalGModel = Generator(n_classes, args.latent_dim, args.num_channels)
    globalDModel = Discriminator(args.n_classes, args.num_channels, args.filter_num)
    # globalDModel = MyIncepResDiscriminator(args.n_classes, args.num_channels, args.filter_num)
    # Init model weights
    globalGModel.apply(weights_init_v2)
    try:
        globalDModel.apply(weights_init_v2)
    except:
        globalDModel.weight_init(0.0, 0.02)

    if args.use_cuda:
        globalGModel.to(device)
        globalDModel.to(device)

    # 写入训练过程数据
    fw_name = save_path + 'Fed_main_training_' + 'log.txt'
    fw_fed_main = open(fw_name, 'w+')
    fw_fed_main.write('iter\t loss\t  Eval acc\t class_eval_acc\t total_emd\t \n')

    # --------------- 先训练conditional GAN网络 ---------------
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # adversarial_loss = torch.nn.MSELoss()

    # labels_processing
    real_labels = 0.7 + 0.5 * torch.rand(n_classes, device=device)
    fake_labels = 0.3 * torch.rand(n_classes, device=device)

    # Optimizers
    optimizer_G = torch.optim.Adam(globalGModel.parameters(), lr=args.fed_g_lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(globalDModel.parameters(), lr=args.fed_d_lr, betas=(args.b1, args.b2))

    # ----------
    #  Training
    # ----------

    # 用于记录训练过程
    local_g_training_losses_list = []
    local_d_training_losses_list = []

    with tqdm(range(Epochs)) as tq:

        # 记录评估
        emd_list = []
        c_scores_list = []
        best_c_acc = 0

        globalGModel.train()
        globalDModel.train()

        for epoch in tq:

            # 用于记录训练过程中的loss变化
            local_g_training_loss = 0
            local_d_training_loss = 0

            for i, (inputs, targets) in enumerate(trainset_loader):
                # Adversarial ground truths
                # valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                # fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                batch_size = inputs.shape[0]

                # Configure input
                # real_imgs = Variable(imgs.type(Tensor))
                real_imgs = inputs.to(device)
                targets = targets.to(device)

                # Sample noise as generator input
                # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # 采用一维向量转换为图像
                '''
                z = torch.tensor(np.random.normal(0, 1, (inputs.shape[0], args.latent_dim)),
                                 dtype=torch.float32, device='cuda')
                gen_labels = torch.LongTensor(np.random.randint(0, n_classes, inputs.shape[0])).cuda()
                '''
                # 直接采用图像进行反卷积
                # z = torch.randn((batch_size, args.latent_dim), device=device).view(-1, args.latent_dim, 1, 1)

                # gen_rand = (torch.rand(batch_size, 1) * args.n_classes).type(torch.LongTensor).squeeze()
                # gen_labels = onehot[gen_rand]
                # real_labels_fill = fill[targets]
                # gen_labels_fill = fill[gen_rand]

                # 对标签进行顺滑处理
                real_label = real_labels[i % n_classes]
                fake_label = fake_labels[i % n_classes]

                if i % 25 == 0:
                    real_label, fake_label = fake_label, real_label

                fake_class_labels = n_classes * torch.ones((batch_size,), dtype=torch.long, device=device)

                # if args.use_cuda:
                #     gen_labels = gen_labels.cuda()
                #     real_labels_fill = real_labels_fill.cuda()
                #     gen_labels_fill = gen_labels_fill.cuda()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                # For real data
                validity_label = torch.full((batch_size,), real_label, device=device)

                real_pvalidity, real_plabels = globalDModel(real_imgs)

                errD_real_val = adversarial_loss(real_pvalidity, validity_label)
                errD_real_label = F.nll_loss(real_plabels, targets)
                errD_real = errD_real_val + errD_real_label
                errD_real.backward()
                # D_x = real_pvalidity.mean().item()

                # For fake data
                noise = torch.randn(batch_size, args.latent_dim, device=device)
                sample_labels = torch.randint(0, n_classes, (batch_size,), device=device, dtype=torch.long)

                # Generate a batch of images
                gen_imgs = globalGModel(noise, sample_labels)

                validity_label.fill_(fake_label)

                fake_pvalidity, fake_plabels = globalDModel(gen_imgs.detach())

                errD_fake_val = adversarial_loss(fake_pvalidity, validity_label)
                errD_fake_label = F.nll_loss(fake_plabels, fake_class_labels)
                errD_fake = errD_fake_val + errD_fake_label
                errD_fake.backward()
                # D_G_z1 = fake_pvalidity.mean().item()

                d_loss = errD_real + errD_fake

                # WGAN
                # d_loss = - D_x + D_G_z1 + errD_fake_label + errD_real_label
                # Source WGAN
                # d_loss = -torch.mean(globalDModel(real_imgs, real_labels_fill)) \
                #          + torch.mean(globalDModel(gen_imgs.detach(), gen_labels_fill))

                # d_loss.backward()
                optimizer_D.step()

                # Clip weights of discriminator WGAN
                # for p in globalDModel.parameters():
                #     p.data.clamp_(-args.clip_value, args.clip_value)

                # -----------------
                #  Train Generator
                # -----------------
                # if i % args.n_critic == 0:
                optimizer_G.zero_grad()

                noise = torch.randn(batch_size, args.latent_dim, device=device)
                sample_labels = torch.randint(0, n_classes, (batch_size,), device=device, dtype=torch.long)

                validity_label.fill_(1)

                # Generate a batch of images
                gen_imgs = globalGModel(noise, sample_labels)

                fake_pvalidity, fake_plabels = globalDModel(gen_imgs)

                errG_val = adversarial_loss(fake_pvalidity, validity_label)
                errG_label = F.nll_loss(fake_plabels, sample_labels)

                # D_G_z2 = pvalidity.mean().item()

                g_loss = errG_val + errG_label

                # Loss measures generator's ability to fool the discriminator
                # validity = globalDModel(gen_imgs, gen_labels)
                # g_loss = - torch.mean(globalDModel(gen_imgs, gen_labels_fill))
                g_loss.backward()
                optimizer_G.step()

                local_d_training_loss += d_loss
                local_g_training_loss += g_loss

            local_g_training_losses_list.append(local_g_training_loss.cpu().detach().numpy() / (i + 1))
            local_d_training_losses_list.append(local_d_training_loss.cpu().detach().numpy() / (i + 1))

            # 本地模型评估
            # LocalG_eval(globalGModel, save_path, name, args.latent_dim, n_row=10, main_epoch=main_epoch)
            G_eval(args, globalGModel, save_path + '/GNet_eval/', 'GModel',
                                  trainset_loader, n_classes=n_classes, main_epoch=epoch)

            # c_scores_list.append(c_score)
            # emd_list.append(emd)
            #
            # # 保存模型
            # if best_c_acc < c_score:
            #     best_c_acc = c_score
            #     best_epoch = epoch
            if epoch % args.save_frequency == 0:
                # 保存联邦模型 FedGNet & FedDNet
                save_model(globalGModel, save_model_file, 'GNet')
                save_model(globalDModel, file_name, 'DNet')

            fw_fed_main.write('{}\t {:.5f}\t {:.5f}\t {:.5f}\t {:.5f}\t \n'.format(epoch, g_loss, d_loss, 0, 0))

        fig, axs = plt.subplots()
        axs.plot(range(len(local_g_training_losses_list)), local_g_training_losses_list,
                 label='G training loss')
        axs.plot(range(len(local_d_training_losses_list)), local_d_training_losses_list,
                 label='D training loss')
        plt.legend()
        plt.savefig(save_path + 'GNet_training_loss.png')

        # 关闭写入
        fw_fed_main.close()
        plt.cla()  # 清除之前绘图
        plt.close()


def LocalG_eval(args, FedGNet, save_path, name, dataset_loader, main_epoch, z=None):
    n_row = args.n_classes
    if z is None:
        # sampling random vector z if z is None
        temp_z_ = torch.randn(n_row, args.latent_dim)
        fixed_z_ = temp_z_
        fixed_y_ = torch.zeros(n_row, 1)
        for i in range(n_row - 1):
            fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
            temp = torch.ones(n_row, 1) + i
            fixed_y_ = torch.cat([fixed_y_, temp], 0)

        fixed_z_ = fixed_z_.view(-1, args.latent_dim, 1, 1)
        fixed_y_label_ = torch.zeros(n_row ** 2, n_row)
        fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
        fixed_y_label_ = fixed_y_label_.view(-1, n_row, 1, 1)
        fixed_z_ = fixed_z_.cuda()
        fixed_z_.requires_grad = False
        fixed_y_label_ = fixed_y_label_.cuda()
        fixed_y_label_.requires_grad = False
    else:
        # Use fixed sampled random vector z
        fixed_z_ = z.fixed_z_
        fixed_y_label_ = z.fixed_y_label_

    mkdir(save_path)

    show_G_eval(FedGNet, fixed_z_, fixed_y_label_, main_epoch, n_classes=args.n_classes, n_channels=args.num_channels,
                show=False, save_path=save_path, save_name=name)

    # ==============================
    # Computer classification score
    # ==============================
    # Load pretrained classifier model VGG11
    from networks import VGG
    VGG_classifier = VGG('VGG11', args.num_channels, n_row)

    pretrained_model_path = 'saved_model/Idpt_classifier_cifar20_All_IID_test/VGG11_All_IID.pkl'
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        VGG_classifier.load_state_dict(checkpoint, strict=True)
        VGG_classifier.cuda()
    else:
        print('[Local] Can not Load VGG pre-trained classifier weights to classifier model.')

    # 生成样本
    synthetic_data, synthetic_lb = generate_augment_samples(FedGNet, args.test_bs,
                                                            args.init_imgsize, args.latent_dim, args.n_classes)

    # 进行分类
    total = 0
    correct = 0
    FedGNet.eval()
    with torch.no_grad():
        outputs = VGG_classifier(synthetic_data)
        _, predicted = torch.max(outputs.data, 1)
        total += synthetic_data.size(0)
        correct += predicted.eq(synthetic_lb.data).cpu().sum()

    acc = np.true_divide(correct, total)

    # ====================
    # Compute EMD
    # ====================
    t_total = 0
    t_values = 0
    f_total = 0
    f_values = 0
    for inputs, targets in dataset_loader:
        batch_size = inputs.shape[0]
        # 生成样本
        synthetic_data, synthetic_lb = generate_augment_samples(FedGNet, batch_size, args.init_imgsize,
                                                                args.latent_dim, args.n_classes)

        if args.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            t_outputs = VGG_classifier(inputs)
            t_value, t_predicted = torch.max(t_outputs.data, 1)
            t_total += targets.size(0)
            t_values += t_value.cpu().numpy().sum().item()

            f_outputs = VGG_classifier(synthetic_data)
            f_value, f_predicted = torch.max(f_outputs.data, 1)
            f_total += synthetic_lb.size(0)
            f_values += f_value.cpu().numpy().sum().item()

    EMD = t_values / t_total - f_values / f_total

    return acc, EMD


def G_eval(args, G, save_path, n_classes, main_epoch):
    # 初始化必要对象
    device = args.device
    noise = torch.randn(n_classes, args.latent_dim, device=device)
    labels = torch.arange(0, n_classes, dtype=torch.long, device=device)

    mkdir(save_path)

    with torch.no_grad():
        gen_images = G(noise, labels).detach()
    # show images
    images = make_grid(gen_images)
    images = images.cpu().numpy()
    images = images / 2 + 0.5
    plt.imshow(np.transpose(images, axes=(1, 2, 0)))
    plt.axis('off')
    plt.savefig(save_path + '/' + 'Epoch{}_eval'.format(str(main_epoch)) + ".png")


def G_eval_computation(args, G, save_path, name, dataset_loader, n_classes, main_epoch):
    # 初始化必要对象
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    noise = torch.randn(args.test_bs, 100, device=device)
    labels = torch.randint(0, n_classes, (args.test_bs,), device=args.device, dtype=torch.long)

    mkdir(save_path)

    with torch.no_grad():
        gen_images = G(noise, labels).detach()
    # show images
    show_imgs = gen_images[:24]
    images = make_grid(show_imgs)
    images = images.cpu().numpy()
    images = images / 2 + 0.5
    plt.imshow(np.transpose(images, axes=(1, 2, 0)))
    plt.axis('off')
    plt.savefig(save_path + '/' + 'Epoch{}_eval'.format(str(main_epoch)) + ".png")

    # ==============================
    # Computer classification score
    # ==============================
    acc = 0
    # Load pretrained classifier model VGG11
    from networks import VGG
    VGG_classifier = VGG('VGG11', args.num_channels, n_classes)

    pretrained_model_path = 'saved_model/Idpt_classifier_cifar20_All_IID_test/VGG11_All_IID.pkl'
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        VGG_classifier.load_state_dict(checkpoint, strict=True)
        VGG_classifier.cuda()
    else:
        print('[Local] Can not Load VGG pre-trained classifier weights to classifier model.')

    # 生成样本
    synthetic_data, synthetic_lb = get_synthetic_samples(G, args.test_bs, args.latent_dim, n_classes)

    synthetic_data = synthetic_data.cuda()
    synthetic_lb = synthetic_lb.cuda()

    # 进行分类
    total = 0
    correct = 0
    with torch.no_grad():
        outputs = VGG_classifier(synthetic_data)
        _, predicted = torch.max(outputs.data, 1)
        total += synthetic_data.size(0)
        correct += predicted.eq(synthetic_lb.data).cpu().sum()

    acc = np.true_divide(correct, total)

    # ====================
    # Compute EMD
    # ====================
    t_total = 0
    t_values = 0
    f_total = 0
    f_values = 0
    for inputs, targets in dataset_loader:
        batch_size = inputs.shape[0]
        # 生成样本
        synthetic_data, synthetic_lb = get_synthetic_samples(G, batch_size, args.latent_dim, n_classes)

        if args.use_cuda:
            synthetic_data = synthetic_data.cuda()
            synthetic_lb = synthetic_lb.cuda()
            inputs = inputs.cuda()
            targets = targets.cuda()
        with torch.no_grad():
            t_outputs = VGG_classifier(inputs)
            t_value, t_predicted = torch.max(t_outputs.data, 1)
            t_total += targets.size(0)
            t_values += t_value.cpu().numpy().sum().item()

            f_outputs = VGG_classifier(synthetic_data)
            f_value, f_predicted = torch.max(f_outputs.data, 1)
            f_total += synthetic_lb.size(0)
            f_values += f_value.cpu().numpy().sum().item()

    EMD = t_values / t_total - f_values / f_total

    return acc, EMD


def ACG_eval_computation(args, FedGNet, save_path, name, dataset_loader_list, n_classes, main_epoch):
    # 初始化必要对象
    device = args.device
    noise = torch.randn(args.test_bs, args.latent_dim, device=device)
    labels = torch.randint(0, n_classes, (args.test_bs,), device=device, dtype=torch.long)

    mkdir(save_path)

    with torch.no_grad():
        gen_images = FedGNet(noise, labels).detach()
    # show images
    show_imgs = gen_images[:32]
    images = make_grid(show_imgs)
    images = images.cpu().numpy()
    images = images / 2 + 0.5
    plt.imshow(np.transpose(images, axes=(1, 2, 0)))
    plt.axis('off')
    plt.savefig(save_path + '/' + 'Epoch{}_eval'.format(str(main_epoch)) + ".png")
    plt.cla()
    plt.close()

    # ==============================
    # Computer classification score
    # ==============================
    # Load pretrained classifier model VGG11
    from networks import VGG
    VGG_classifier = VGG('VGG11', args.num_channels, n_classes)

    if 'mnist' in args.dataset or 'FER' in args.dataset:
        pretrained_model_path = 'saved_model/Idpt_classifier_{}_All_IID_size32_channel1/VGG11_All_IID.pkl'.format(args.dataset)
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path)
        VGG_classifier.load_state_dict(checkpoint, strict=True)
        VGG_classifier.to(device)
    else:
        print('[Local] Can not Load VGG pre-trained classifier weights to classifier model.')

    # 生成样本
    synthetic_data, synthetic_lb = get_synthetic_samples(FedGNet, args.test_bs, args.latent_dim, args.n_classes, device)

    # 进行分类
    total = 0
    correct = 0
    FedGNet.eval()
    with torch.no_grad():
        outputs = VGG_classifier(synthetic_data)
        _, predicted = torch.max(outputs.data, 1)
        total += synthetic_data.size(0)
        correct += predicted.eq(synthetic_lb.data).cpu().numpy().sum()

    acc = torch.true_divide(correct, total).cpu().item()

    # ====================
    # Compute EMD
    # ====================
    EMD_list = []
    gt_total = 0
    gt_values = 0
    gf_total = 0
    gf_values = 0
    global_EMD = None
    for dataset_loader in dataset_loader_list:
        t_total = 0
        t_values = 0
        f_total = 0
        f_values = 0
        for inputs, targets in dataset_loader:
            batch_size = inputs.shape[0]
            # 生成样本
            synthetic_data, synthetic_lb = get_synthetic_samples(FedGNet, batch_size, args.latent_dim,
                                                                 n_classes, device)

            if args.use_cuda:
                inputs = inputs.to(device)
                targets = targets.to(device)

            with torch.no_grad():
                t_outputs = VGG_classifier(inputs)
                t_value, t_predicted = torch.max(t_outputs.data, 1)
                t_total += targets.size(0)
                t_values += t_value.cpu().numpy().sum().item()

                f_outputs = VGG_classifier(synthetic_data)
                f_value, f_predicted = torch.max(f_outputs.data, 1)
                f_total += synthetic_lb.size(0)
                f_values += f_value.cpu().numpy().sum().item()

                gt_total += t_total
                gt_values += t_values

                gf_total += f_total
                gf_values += f_values

        EMD = t_values / t_total - f_values / f_total
        EMD_list.append(EMD)

        global_EMD = gt_values / gt_total - gf_values / gf_total

    return acc, EMD_list, global_EMD


def get_synthetic_samples(G, size, latent_dim, n_classes, device):
    noise = torch.randn(size, latent_dim)
    labels = torch.randint(0, n_classes, (size,), dtype=torch.long)

    noise = noise.to(device)
    labels = labels.to(device)

    G.eval()
    with torch.no_grad():
        gen_images = G(noise, labels).detach()

    return gen_images, labels


if __name__ == "__main__":
    args = args_parser()

    # 做实验
    exp_total_time = 1
    cross_validation_sets = 1

    results_saved_file = 'test_results'
    results_plot_file = 'plot_results'
    model_saved_file = 'saved_model'

    args.training_dataset_name = 'CIFAR20_IID_training'

    args.exp_name = 'AC_GAN_{}_IID_BN_size32_catLable&Z'.format(args.dataset)

    # 存储主文件路径
    result_save_file = './{}/'.format(results_saved_file) + args.exp_name + '/'
    mkdir(result_save_file)

    WGPGAN_Train(args, result_save_file)
