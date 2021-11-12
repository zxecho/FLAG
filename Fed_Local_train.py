import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

from torchvision import transforms
from function_utils import clip_gradient, set_lr, save_model, plot_confusion_matrix, save2json, mkdir
from function_utils import generate_augment_samples, get_ACGAN_synthetic_samples
from Fed_test import show_G_eval

# from FedDA.data_processing import MnistDataset, get_each_participant_dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

label_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def Local_GAN_Train(args, globalGModel, globalDModel, dataset_loader, save_path, name, main_epoch=0):
    """
    联邦下的本地模型训练函数
    :param main_epoch: 主训练循环的轮次
    :param globalDModel:
    :param name:
    :param globalGModel:
    :param args: 参数配置
    :param dataset_loader: 当前参与者拥有的训练数据
    :param save_path: 保存路径
    :return:
    """
    # 有多少类数据
    n_classes = len(args.dataset_label[args.dataset])
    # label preprocess
    onehot = torch.zeros(args.n_classes, args.n_classes)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(args.n_classes, 1), 1).view(
        args.n_classes, args.n_classes, 1, 1)
    fill = torch.zeros([args.n_classes, args.n_classes, args.init_imgsize, args.init_imgsize])
    for i in range(args.n_classes):
        fill[i, i, :, :] = 1

    # 用于保存模型
    file_name = save_path.split('/')
    file_name = file_name[2] + '/' + file_name[3]

    # --------------- 先训练conditional GAN网络 ---------------
    # Loss function
    # adversarial_loss = torch.nn.BCELoss()
    adversarial_loss = torch.nn.MSELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(globalGModel.parameters(), lr=args.fed_g_lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(globalDModel.parameters(), lr=args.fed_d_lr, betas=(args.b1, args.b2))

    # ----------
    #  Training
    # ----------

    local_g_training_losses_list = []
    local_d_training_losses_list = []

    for epoch in range(args.local_ep):
        globalGModel.train()
        globalDModel.train()
        # 用于记录训练过程中的loss变化
        local_g_training_loss = 0
        local_d_training_loss = 0
        for i, (inputs, targets) in enumerate(dataset_loader):
            # Adversarial ground truths
            # valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            # fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            batch_size = inputs.shape[0]

            valid = torch.ones([inputs.shape[0], 1], requires_grad=False, dtype=torch.float32, device='cuda')
            fake = torch.zeros([inputs.shape[0], 1], requires_grad=False, dtype=torch.float32, device='cuda')

            # Configure input
            # real_imgs = Variable(imgs.type(Tensor))
            real_imgs = inputs.cuda()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # 采用一维向量转换为图像
            '''
            z = torch.tensor(np.random.normal(0, 1, (inputs.shape[0], args.latent_dim)),
                             dtype=torch.float32, device='cuda')
            gen_labels = torch.LongTensor(np.random.randint(0, n_classes, inputs.shape[0])).cuda()
            '''
            # 直接采用图像进行反卷积
            z = torch.randn((batch_size, args.latent_dim), device='cuda' if args.use_cuda else 'cpu').view(-1,
                                                                                                           args.latent_dim,
                                                                                                           1, 1)
            gen_rand = (torch.rand(batch_size, 1) * args.n_classes).type(torch.LongTensor).squeeze()
            gen_labels = onehot[gen_rand]
            real_labels_fill = fill[targets]
            gen_labels_fill = fill[gen_rand]

            if args.use_cuda:
                gen_labels = gen_labels.cuda()
                real_labels_fill = real_labels_fill.cuda()
                gen_labels_fill = gen_labels_fill.cuda()

            # Generate a batch of images
            gen_imgs = globalGModel(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            # validity = globalDModel(gen_imgs, gen_labels)
            g_loss = adversarial_loss(globalDModel(gen_imgs, gen_labels_fill), valid)
            local_g_training_loss += g_loss
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(globalDModel(real_imgs, real_labels_fill), valid)
            fake_loss = adversarial_loss(globalDModel(gen_imgs.detach(), gen_labels_fill), fake)
            d_loss = (real_loss + fake_loss) / 2
            local_d_training_loss += d_loss
            d_loss.backward()
            optimizer_D.step()

            local_g_training_losses_list.append(local_g_training_loss.cpu().item() / (i + 1))
            local_d_training_losses_list.append(local_d_training_loss.cpu().item() / (i + 1))

    # 本地模型评估
    # LocalG_eval(globalGModel, save_path, name, args.latent_dim, n_row=10, main_epoch=main_epoch)
    show_LocalG_eval(globalGModel, save_path, name, args.latent_dim, main_epoch=epoch, n_row=10)

    return globalGModel.state_dict(), globalDModel.state_dict(), \
           np.mean(local_g_training_losses_list), \
           np.mean(local_d_training_losses_list), \
           dataset_loader.dataset.number


def Local_CGAN_Train(args, globalGModel, globalDModel, dataset_loader, dataset_lb_stat,
                     local_g_lr=1e-4, local_d_lr=1e-4,
                     save_path='', name='', main_epoch=0):
    """
    Use vanillia conditional GAN
    :param local_d_lr:
    :param local_g_lr:
    :param args:
    :param globalGModel:
    :param globalDModel:
    :param dataset_loader:
    :param dataset_lb_stat:
    :param save_path:
    :param name:
    :param fixed_z:
    :param main_epoch:
    :return:
    """

    # 保存路径
    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    # 有多少类数据
    n_classes = len(args.dataset_label[args.dataset])

    # 用于保存模型
    file_name = save_path.split('/')
    file_name = file_name[2] + '/' + file_name[3]

    # --------------- 先训练conditional GAN网络 ---------------
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # adversarial_loss = torch.nn.MSELoss()
    # auxiliary_loss = torch.nn.CrossEntropyLoss()
    # F.nll_loss

    # labels_processing
    # real_labels = 0.7 + 0.5 * torch.rand(n_classes, device=args.device)
    # fake_labels = 0.3 * torch.rand(n_classes, device=args.device)

    # Optimizers
    optimizer_G = torch.optim.Adam(globalGModel.parameters(), lr=local_g_lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(globalDModel.parameters(), lr=local_d_lr, betas=(args.b1, args.b2))

    # optimizer_G = torch.optim.SGD(globalGModel.parameters(), lr=local_g_lr, momentum=0.95)
    # optimizer_D = torch.optim.SGD(globalDModel.parameters(), lr=local_d_lr)

    # ----------
    #  Training
    # ----------

    # 用于记录训练过程
    local_g_training_losses_list = []
    local_d_training_losses_list = []

    with tqdm(range(args.local_ep)) as tq:

        # 记录评估
        emd_list = []
        c_scores_list = []
        best_c_acc = 0
        tq.set_description('{} training'.format(name))
        globalGModel.train()
        globalDModel.train()

        for epoch in tq:

            # 用于记录训练过程中的loss变化
            local_g_training_loss = 0
            local_d_training_loss = 0

            for i, (inputs, targets) in enumerate(dataset_loader):
                batch_size = inputs.shape[0]

                # Configure input
                # real_imgs = Variable(imgs.type(Tensor))
                real_imgs = inputs.to(args.device)
                targets = targets.to(args.device)

                # 直接采用图像进行反卷积
                # z = torch.randn((batch_size, args.latent_dim), device=device).view(-1, args.latent_dim, 1, 1)

                # gen_rand = (torch.rand(batch_size, 1) * args.n_classes).type(torch.LongTensor).squeeze()
                # gen_labels = onehot[gen_rand]
                # real_labels_fill = fill[targets]
                # gen_labels_fill = fill[gen_rand]

                # 对标签进行顺滑处理
                # real_label = real_labels[i % n_classes]
                # fake_label = fake_labels[i % n_classes]
                #
                # if i % 25 == 0:
                #     real_label, fake_label = fake_label, real_label

                # Not smooth processing
                real_label = 1.0
                fake_label = 0.0

                validity_label = torch.full((batch_size,), real_label, device=args.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                # For real data

                real_pvalidity = globalDModel(real_imgs, targets)

                errD_real_val = adversarial_loss(real_pvalidity, validity_label)
                errD_real = errD_real_val * 0.5
                errD_real.backward()
                # D_x = real_pvalidity.mean().item()

                # For fake data
                random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
                sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)
                noise = torch.randn(batch_size, args.latent_dim, device=args.device)

                # Generate a batch of images
                gen_imgs = globalGModel(noise, sample_labels)

                validity_label.fill_(fake_label)

                fake_pvalidity = globalDModel(gen_imgs.detach(), sample_labels)

                errD_fake_val = adversarial_loss(fake_pvalidity, validity_label)
                errD_fake = errD_fake_val * 0.5
                errD_fake.backward()
                # D_G_z1 = fake_pvalidity.mean().item()

                d_loss = errD_real + errD_fake

                # d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                noise = torch.randn(batch_size, args.latent_dim, device=args.device)
                # sample_labels = torch.randint(0, n_classes, (batch_size,), device=args.device, dtype=torch.long)
                random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
                sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)

                validity_label.fill_(real_label)

                # Generate a batch of images
                gen_imgs = globalGModel(noise, sample_labels)

                fake_pvalidity = globalDModel(gen_imgs, sample_labels)

                errG_val = adversarial_loss(fake_pvalidity, validity_label)

                g_loss = 0.5 * errG_val

                g_loss.backward()
                optimizer_G.step()

                local_d_training_loss += d_loss
                local_g_training_loss += g_loss

            local_g_training_losses = local_g_training_loss.cpu().detach().numpy() / (i + 1)
            local_d_training_losses = local_d_training_loss.cpu().detach().numpy() / (i + 1)

            # 本地模型评估
            # LocalG_eval(globalGModel, save_path, name, args.latent_dim, n_row=0, main_epoch=main_epoch)
            AC_GAN_eval(args, globalGModel, save_path + '/GNet_eval/', name, n_classes=n_classes)

    # 保存联邦模型 LocalGNet & LocalDNet
    save_model(globalGModel, file_name, 'Local_{}_G'.format(name))
    save_model(globalDModel, file_name, 'Local_{}_D'.format(name))

    return globalGModel.state_dict(), globalDModel.state_dict(), \
           local_d_training_losses, local_g_training_losses, \
           dataset_loader.dataset.number


def Local_vanilla_ACGAN_Train(args, globalGModel, globalDModel, dataset_loader, dataset_lb_stat,
                              local_g_lr=1e-4, local_d_lr=1e-4,
                              save_path='', name='', main_epoch=0):
    """
    Use Aux Classifier GAN
    :param local_d_lr:
    :param local_g_lr:
    :param args:
    :param globalGModel:
    :param globalDModel:
    :param dataset_loader:
    :param dataset_lb_stat:
    :param save_path:
    :param name:
    :param fixed_z:
    :param main_epoch:
    :return:
    """

    # 保存路径
    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    # 有多少类数据
    n_classes = len(args.dataset_label[args.dataset])

    # 用于保存模型
    file_name = save_path.split('/')
    file_name = file_name[2] + '/' + file_name[3]

    # --------------- 先训练conditional GAN网络 ---------------
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # adversarial_loss = torch.nn.MSELoss()
    # auxiliary_loss = torch.nn.CrossEntropyLoss()
    # F.nll_loss

    # labels_processing
    # real_labels = 0.7 + 0.5 * torch.rand(n_classes, device=args.device)
    # fake_labels = 0.3 * torch.rand(n_classes, device=args.device)

    # Optimizers
    optimizer_G = torch.optim.Adam(globalGModel.parameters(), lr=local_g_lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(globalDModel.parameters(), lr=local_d_lr, betas=(args.b1, args.b2))

    # optimizer_G = torch.optim.SGD(globalGModel.parameters(), lr=local_g_lr, momentum=0.95)
    # optimizer_D = torch.optim.SGD(globalDModel.parameters(), lr=local_d_lr)

    # ----------
    #  Training
    # ----------

    # 用于记录训练过程
    local_g_training_losses_list = []
    local_d_training_losses_list = []

    with tqdm(range(args.local_ep)) as tq:

        # 记录评估
        emd_list = []
        c_scores_list = []
        best_c_acc = 0
        tq.set_description('{} training'.format(name))
        globalGModel.train()
        globalDModel.train()

        for epoch in tq:

            # 用于记录训练过程中的loss变化
            local_g_training_loss = 0
            local_d_training_loss = 0

            for i, (inputs, targets) in enumerate(dataset_loader):
                batch_size = inputs.shape[0]

                # Configure input
                # real_imgs = Variable(imgs.type(Tensor))
                real_imgs = inputs.to(args.device)
                targets = targets.to(args.device)

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
                # real_label = real_labels[i % n_classes]
                # fake_label = fake_labels[i % n_classes]
                #
                # if i % 25 == 0:
                #     real_label, fake_label = fake_label, real_label

                # Not smooth processing
                real_label = 1.0
                fake_label = 0.0

                fake_class_labels = (n_classes - 1) * torch.ones((batch_size,), dtype=torch.long, device=args.device)
                validity_label = torch.full((batch_size,), real_label, device=args.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                # For real data

                real_pvalidity, real_plabels = globalDModel(real_imgs)

                errD_real_val = adversarial_loss(real_pvalidity, validity_label)
                errD_real_label = F.nll_loss(real_plabels, targets)
                errD_real = (errD_real_val + errD_real_label) * 0.5
                errD_real.backward()
                # D_x = real_pvalidity.mean().item()

                # For fake data
                random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
                sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)
                noise = torch.randn(batch_size, args.latent_dim, device=args.device)

                # Generate a batch of images
                gen_imgs = globalGModel(noise, sample_labels)

                validity_label.fill_(fake_label)

                fake_pvalidity, fake_plabels = globalDModel(gen_imgs.detach())

                errD_fake_val = adversarial_loss(fake_pvalidity, validity_label)
                errD_fake_label = F.nll_loss(fake_plabels, sample_labels)
                errD_fake = (errD_fake_val + errD_fake_label) * 0.5
                errD_fake.backward()
                # D_G_z1 = fake_pvalidity.mean().item()

                d_loss = errD_real + errD_fake

                # d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                noise = torch.randn(batch_size, args.latent_dim, device=args.device)
                # sample_labels = torch.randint(0, n_classes, (batch_size,), device=args.device, dtype=torch.long)
                random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
                sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)

                validity_label.fill_(real_label)

                # Generate a batch of images
                gen_imgs = globalGModel(noise, sample_labels)

                fake_pvalidity, fake_plabels = globalDModel(gen_imgs)

                errG_val = adversarial_loss(fake_pvalidity, validity_label)
                errG_label = F.nll_loss(fake_plabels, sample_labels)  # 错误 fake_class_labels

                # D_G_z2 = pvalidity.mean().item()

                g_loss = 0.5 * (errG_val + errG_label)

                g_loss.backward()
                optimizer_G.step()

                local_d_training_loss += d_loss
                local_g_training_loss += g_loss

            local_g_training_losses = local_g_training_loss.cpu().detach().numpy() / (i + 1)
            local_d_training_losses = local_d_training_loss.cpu().detach().numpy() / (i + 1)

            # 本地模型评估
            # LocalG_eval(globalGModel, save_path, name, args.latent_dim, n_row=0, main_epoch=main_epoch)
            AC_GAN_eval(args, globalGModel, save_path + '/GNet_eval/', name, n_classes=n_classes)

    # 保存联邦模型 LocalGNet & LocalDNet
    save_model(globalGModel, file_name, 'Local_{}_G'.format(name))
    save_model(globalDModel, file_name, 'Local_{}_D'.format(name))

    return globalGModel.state_dict(), globalDModel.state_dict(), \
           local_d_training_losses, local_g_training_losses, \
           dataset_loader.dataset.number


def Local_ACGAN_Train(args, globalGModel, globalDModel, dataset_loader, dataset_lb_stat,
                      local_g_lr=1e-4, local_d_lr=1e-4,
                      save_path='', name='', main_epoch=0):
    """
    Use Aux Classifier GAN
    :param local_d_lr:
    :param local_g_lr:
    :param args:
    :param globalGModel:
    :param globalDModel:
    :param dataset_loader:
    :param dataset_lb_stat:
    :param save_path:
    :param name:
    :param fixed_z:
    :param main_epoch:
    :return:
    """

    # 保存路径
    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    # 有多少类数据
    n_classes = len(args.dataset_label[args.dataset])

    # 用于保存模型
    file_name = save_path.split('/')
    file_name = file_name[2] + '/' + file_name[3]

    # --------------- 先训练conditional GAN网络 ---------------
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # adversarial_loss = torch.nn.MSELoss()
    # auxiliary_loss = torch.nn.CrossEntropyLoss()
    # F.nll_loss

    # labels_processing
    # real_labels = 0.7 + 0.5 * torch.rand(n_classes, device=args.device)
    # fake_labels = 0.3 * torch.rand(n_classes, device=args.device)

    # Optimizers
    optimizer_G = torch.optim.Adam(globalGModel.parameters(), lr=local_g_lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(globalDModel.parameters(), lr=local_d_lr, betas=(args.b1, args.b2))

    # optimizer_G = torch.optim.SGD(globalGModel.parameters(), lr=local_g_lr, momentum=0.95)
    # optimizer_D = torch.optim.SGD(globalDModel.parameters(), lr=local_d_lr)

    # ----------
    #  Training
    # ----------

    # 用于记录训练过程
    local_g_training_losses_list = []
    local_d_training_losses_list = []

    with tqdm(range(args.local_ep)) as tq:

        # 记录评估
        emd_list = []
        c_scores_list = []
        best_c_acc = 0
        tq.set_description('{} training'.format(name))
        globalGModel.train()
        globalDModel.train()

        for epoch in tq:

            # 用于记录训练过程中的loss变化
            local_g_training_loss = 0
            local_d_training_loss = 0

            for i, (inputs, targets) in enumerate(dataset_loader):
                batch_size = inputs.shape[0]

                # Configure input
                # real_imgs = Variable(imgs.type(Tensor))
                real_imgs = inputs.to(args.device)
                targets = targets.to(args.device)

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
                # real_label = real_labels[i % n_classes]
                # fake_label = fake_labels[i % n_classes]
                #
                # if i % 25 == 0:
                #     real_label, fake_label = fake_label, real_label

                # Not smooth processing
                real_label = 1.0
                fake_label = 0.0

                fake_class_labels = (n_classes - 1) * torch.ones((batch_size,), dtype=torch.long, device=args.device)
                validity_label = torch.full((batch_size,), real_label, device=args.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                # For real data

                real_pvalidity, real_plabels = globalDModel(real_imgs)

                errD_real_val = adversarial_loss(real_pvalidity, validity_label)
                errD_real_label = F.nll_loss(real_plabels, targets)
                errD_real = (errD_real_val + errD_real_label) * 0.5
                errD_real.backward()
                # D_x = real_pvalidity.mean().item()

                # For fake data
                random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
                sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)
                noise = torch.randn(batch_size, args.latent_dim, device=args.device)

                # Generate a batch of images
                gen_imgs = globalGModel(noise, sample_labels)

                validity_label.fill_(fake_label)

                fake_pvalidity, fake_plabels = globalDModel(gen_imgs.detach())

                errD_fake_val = adversarial_loss(fake_pvalidity, validity_label)
                errD_fake_label = F.nll_loss(fake_plabels, sample_labels)
                errD_fake = errD_fake_val * 0.5  # + errD_fake_label
                errD_fake.backward()
                # D_G_z1 = fake_pvalidity.mean().item()

                d_loss = errD_real + errD_fake

                # d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                noise = torch.randn(batch_size, args.latent_dim, device=args.device)
                # sample_labels = torch.randint(0, n_classes, (batch_size,), device=args.device, dtype=torch.long)
                random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
                sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)

                validity_label.fill_(real_label)

                # Generate a batch of images
                gen_imgs = globalGModel(noise, sample_labels)

                fake_pvalidity, fake_plabels = globalDModel(gen_imgs)

                errG_val = adversarial_loss(fake_pvalidity, validity_label)
                errG_label = F.nll_loss(fake_plabels, sample_labels)  # 错误 fake_class_labels

                # D_G_z2 = pvalidity.mean().item()

                g_loss = 0.5 * (errG_val + errG_label)

                g_loss.backward()
                optimizer_G.step()

                local_d_training_loss += d_loss
                local_g_training_loss += g_loss

            local_g_training_losses = local_g_training_loss.cpu().detach().numpy() / (i + 1)
            local_d_training_losses = local_d_training_loss.cpu().detach().numpy() / (i + 1)

            # 本地模型评估
            # LocalG_eval(globalGModel, save_path, name, args.latent_dim, n_row=0, main_epoch=main_epoch)
            AC_GAN_eval(args, globalGModel, save_path + '/GNet_eval/', name, n_classes=n_classes)

    # 保存联邦模型 LocalGNet & LocalDNet
    save_model(globalGModel, file_name, 'Local_{}_G'.format(name))
    save_model(globalDModel, file_name, 'Local_{}_D'.format(name))

    return globalGModel.state_dict(), globalDModel.state_dict(), \
           local_d_training_losses, local_g_training_losses, \
           dataset_loader.dataset.number


# 使用GAN增强的分类器模型训练
def Local_C_with_DA_Train(args, globalGModel, globalCModel, dataset_loader, dataset_lb_stat, save_path, name):
    """
    使用GAN生成的增强数据进行分类器的训练
    :param args:
    :param globalGModel:
    :param globalCModel:
    :param dataset_loader:
    :param save_path:
    :param name:
    :return:
    """
    pre_acc = 0
    best_acc = 0

    # 保存路径
    file_name = save_path.split('/')
    file_name = file_name[2] + '/' + file_name[3]

    # -----------训练本地的分类器-------------
    Epochs = args.c_training_epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(globalCModel.parameters(), lr=args.local_lr,
                                 weight_decay=args.L2_weight_decay)

    local_loss_list = []
    local_acc_list = []
    with tqdm(range(Epochs)) as tq:
        tq.set_description('{} Training Epoch: '.format(name))
        for epoch in tq:
            # print('\n Current Epoch: %d' % epoch)
            globalCModel.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(dataset_loader):

                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)

                # 采样增强数据集
                # aug_dataset, aug_labels = generate_augment_samples(globalGModel, int(args.batch_size / args.mix_ritio),
                #                                                    args.init_imgsize, args.latent_dim, args.n_classes)

                aug_dataset, aug_labels = get_ACGAN_synthetic_samples(globalGModel,
                                                                      int(args.mix_bs * args.mix_ritio),
                                                                      args.latent_dim,
                                                                      args.n_classes,
                                                                      dataset_lb_stat,
                                                                      args.device)

                # 进行数据集拼接
                inputs = torch.cat([inputs, aug_dataset], dim=0)
                targets = torch.cat([targets, aug_labels], dim=0)

                outputs = globalCModel(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                if args.DAC_grad_clip:
                    clip_gradient(optimizer, 0.1)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            pre_acc = torch.true_divide(correct, total)

            local_loss_list.append(train_loss / (batch_idx + 1))
            local_acc_list.append(pre_acc.item())

            # tqdm打印信息
            tq.set_postfix(Acc=pre_acc)

            # 保存模型
            if pre_acc > best_acc:
                best_acc = pre_acc
                save_model(globalCModel, file_name, 'DA({})_Classifier_{}'.format(args.mix_ritio, name))

    fig, axs = plt.subplots()
    axs.plot(range(len(local_loss_list)), local_loss_list, label='Training loss')
    axs.set_xlabel('Training epochs')
    axs.plot(range(len(local_acc_list)), local_acc_list, label='Test accuracy')
    img_save_dir = save_path + 'DA_Classifier_training_info/'
    mkdir(img_save_dir)
    plt.legend()
    plt.savefig(img_save_dir + 'DA_{}_classifier_training_loss.png'.format(name, args.mix_ritio))
    plt.cla()
    plt.close()

    return best_acc


# 联邦学习中本地分类器模型训练
def Local_C_Train(args, globalCModel, globalGModel, dataset_loader, dataset_lb_stat, client_name, lr, save_path):
    Epochs = args.local_ep
    optimizer = torch.optim.SGD(globalCModel.parameters(), lr=lr, momentum=0.9, weight_decay=args.L2_weight_decay)
    criterion = nn.CrossEntropyLoss()
    local_loss_list = []
    local_acc_list = []

    for epoch in range(Epochs):
        # print('\n Current Epoch: %d' % epoch)
        globalCModel.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataset_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            batch_size = inputs.shape[0]

            optimizer.zero_grad()
            if args.if_Local_train_C_with_DA:
                aug_dataset, aug_labels = get_ACGAN_synthetic_samples(globalGModel,
                                                                      int(batch_size * args.mix_ritio),
                                                                      args.latent_dim,
                                                                      args.n_classes,
                                                                      dataset_lb_stat,
                                                                      args.device)
                # 进行数据集拼接
                inputs = torch.cat([inputs, aug_dataset], dim=0)
                targets = torch.cat([targets, aug_labels], dim=0)

            outputs = globalCModel(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if args.DAC_grad_clip:
                clip_gradient(optimizer, 0.1)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        local_loss_list.append(train_loss / (batch_idx + 1))

        epoch_acc = 100. * correct / total
        local_acc_list.append(torch.true_divide(correct, total).item())
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(range(len(local_loss_list)), local_loss_list)
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Local training epochs')
    axs[1].plot(range(len(local_acc_list)), local_acc_list)
    axs[1].set_ylabel('Acc')
    axs[1].set_xlabel('Local training epochs')
    save_path = save_path + '/Family_CNet_eval/'
    mkdir(save_path)
    plt.savefig('./{}/{}_training_loss&acc.png'.format(save_path, client_name))
    plt.cla()
    plt.close()

    return globalCModel.state_dict(), train_loss / (batch_idx + 1), dataset_loader.dataset.number


# 本地分类器单独训练
def IdptTraining(args, IdptModel, name, dataset_loader, save_path):
    local_loss_list = []
    local_acc_list = []
    pre_acc = 0
    best_acc = 0

    Epochs = args.c_training_epochs
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(IdptModel.parameters(), lr=args.fed_c_lr, momentum=0.9,
    #                             weight_decay=args.L2_weight_decay)
    optimizer = torch.optim.Adam(IdptModel.parameters(), lr=args.fed_c_lr)
    with tqdm(range(Epochs)) as tq:
        tq.set_description('Local {} Training Epoch: '.format(name))
        for epoch in tq:
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
                    inputs, targets = inputs.to(args.device), targets.to(args.device)

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
                pre_acc = best_acc
                save_model(IdptModel, file_name, 'Idpt_{}'.format(name))

        # tqdm打印信息
        tq.set_postfix(Acc=best_acc)

        fig, axs = plt.subplots()
        axs.plot(range(len(local_loss_list)), local_loss_list, label='Training loss')
        axs.set_xlabel('Solo classifier training epochs')
        axs.plot(range(len(local_acc_list)), local_acc_list, label='Training accuracy')
        img_save_dir = save_path + 'idpt_training_info/'
        mkdir(img_save_dir)
        plt.legend()
        plt.savefig(img_save_dir + 'Idpt{}_classifier_training_loss&acc.png'.format(name))
        plt.cla()
        plt.close()

    return best_acc


def LocalG_eval_old(FedGNet, save_path, name, latent_dim, n_row, main_epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, latent_dim)), dtype=torch.float32).cuda()
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = torch.tensor(labels, dtype=torch.long).cuda()

    gen_imgs = FedGNet(z, labels)
    save_path = save_path + '/Family_GNet_eval/'
    mkdir(save_path)
    save_image(gen_imgs.data, save_path + "/{}_eval_result_%d.png".format(name, main_epoch),
               nrow=n_row, normalize=True)


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


def AC_GAN_eval(args, G, save_path, name, n_classes):
    # 初始化必要对象
    noise = torch.randn(n_classes, args.latent_dim, device=args.device)
    labels = torch.arange(0, n_classes, dtype=torch.long, device=args.device)

    mkdir(save_path)

    with torch.no_grad():
        gen_images = G(noise, labels).detach()
    # show images
    images = make_grid(gen_images)
    images = images.cpu().numpy()
    images = images / 2 + 0.5
    plt.imshow(np.transpose(images, axes=(1, 2, 0)))
    plt.axis('off')
    plt.savefig(save_path + '/' + '{}_local_eval'.format(name) + ".png")
    plt.cla()
    plt.close()


if __name__ == '__main__':
    from param_options import args_parser

    arg = args_parser()
    # IdptTraining()

    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    dataloaders = get_each_participant_dataset('fed_training_MNIST_1.h5', 64, transformer)

    LocalTrain(arg, None, dataloaders[0])
