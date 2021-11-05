"""
用于载入联邦问题家庭数据

其将原数据集按照被试划分，之后划分家庭

training set 是每个家庭数据的80%

testing set 每个家庭数据集20%
"""
from PIL import Image
import numpy as np
import h5py
import torch
# import transforms as transforms
import torchvision.transforms as transforms
import torch.utils.data as data
from param_options import args_parser


class VarTransformers:

    def __init__(self, args, split='training'):
        if split == 'C_training':
            if 'mnist' in args.dataset:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.init_imgsize),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            elif 'FER' in args.dataset:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.init_imgsize),
                    # transforms.Grayscale(3),
                    transforms.ToTensor(),
                    # BU3DFE mean=[0.27676, 0.27676, 0.27676], std=[0.26701, 0.26701, 0.26701]
                    # jaffe mean=[0.43192, 0.43192, 0.43192], std=[0.27979, 0.27979, 0.27979]
                    # oulu mean=[0.36418, 0.36418, 0.36418], std=[0.20384, 0.20384, 0.20384]
                    # ck-48  mean=[0.51194, 0.51194, 0.51194], std=[0.28913, 0.28913, 0.28913]
                    # transforms.Normalize(mean=args.dataset_mean_std[args.dataset]['mean'],
                    #                      std=args.dataset_mean_std[args.dataset]['std']),
                    transforms.Normalize(0.5, 0.5),
                ])
            elif 'cifar' in args.dataset:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.init_imgsize),
                    transforms.RandomCrop(args.init_imgsize, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=args.dataset_mean_std[args.dataset]['mean'],
                                         std=args.dataset_mean_std[args.dataset]['std']),
                ])
        elif split == 'C_testing':
            if 'mnist' in args.dataset:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.init_imgsize),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            elif 'FER' in args.dataset:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.init_imgsize),
                    transforms.ToTensor(),
                    # BU3DFE mean=[0.27676, 0.27676, 0.27676], std=[0.26701, 0.26701, 0.26701]
                    # jaffe mean=[0.43192, 0.43192, 0.43192], std=[0.27979, 0.27979, 0.27979]
                    # oulu mean=[0.36418, 0.36418, 0.36418], std=[0.20384, 0.20384, 0.20384]
                    # ck-48  mean=[0.51194, 0.51194, 0.51194], std=[0.28913, 0.28913, 0.28913]
                    # transforms.Normalize(mean=args.dataset_mean_std[args.dataset]['mean'],
                    #                      std=args.dataset_mean_std[args.dataset]['std']),
                    transforms.Normalize(0.5, 0.5),
                ])
            elif 'cifar' in args.dataset:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(args.init_imgsize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=args.dataset_mean_std[args.dataset]['mean'],
                                         std=args.dataset_mean_std[args.dataset]['std']),
                ])

        if split == 'GAN_training':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.init_imgsize),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif split == 'GAN_testing':
            self.transform = transforms.Compose([
                transforms.Resize(args.init_imgsize),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])


class CK(data.Dataset):
    """
    CK_48 famliy split Dataset.

    Set4_training 的数据结构
    group: Family0 ---> dataset: {"FEdata_pixel", "FEdata_label"}
    """

    def __init__(self, h5file, channels, split='training', transform=None):
        self.transform = transform
        self.h5data = h5file[split]
        self.channels = channels
        # self.data_list = []
        # self.labels_list = []
        # for ind in range(self.number):
        #     self.data_list.append(self.data['FEdata_pixel'][ind])
        #     self.labels_list.append(self.data['FEdata_label'][ind])
        #
        self.data_array = self.h5data['FEdata_pixel']
        self.label_array = self.h5data['FEdata_label']

        if 'Family' in split:
            self.number = self.data_array.shape[0]
        else:
            self.number = len(h5file[split])  # 该家庭有多少数据量

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data_array[index], self.label_array[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.channels == 3:
            if len(img.shape) < 3:
                img = img[:, :, np.newaxis]
                img = np.concatenate((img, img, img), axis=2)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.data_array.shape[0]


def get_fed_fam_dataset(args, data_name, split='training', batch_size=None):
    transform = VarTransformers(args, split).transform

    h5file = h5py.File('./dataset/{}/{}'.format(args.dataset, data_name), 'r', driver='core')
    data_split = 'training'

    if 'training' in split:
        data_split = 'training'
    elif 'testing' in split:
        data_split = 'testing'

    if batch_size is None:
        batch_size = args.batch_size

    h5fdata = h5file[data_split]
    fm_keys = h5fdata.keys()
    fm_n = len(fm_keys)

    loader_list = []
    f_n_list = []
    loader_dict = {}
    for fm_name in fm_keys:
        dataset = CK(h5fdata, channels=args.num_channels, split=fm_name, transform=transform)
        if 'training' == split:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_bs, shuffle=True)
        loader_list.append(dataloader)
        loader_dict[fm_name] = dataloader
        f_n_list.append(fm_name)

    return loader_list, fm_n, f_n_list


def get_dataset(args, data_file, data_name, split='training'):
    transform = VarTransformers(args, split).transform

    h5file = h5py.File('./dataset/{}/{}'.format(data_file, data_name), 'r', driver='core')

    if 'training' in split:
        data_split = 'training'
    elif 'testing' in split:
        data_split = 'testing'

    dataset = CK(h5file, channels=args.num_channels, split=data_split, transform=transform)
    if 'training' in split:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    elif 'testing' in split:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.test_bs, shuffle=True)

    return dataloader


if __name__ == '__main__':
    args = args_parser()
    trainset_loader_list, training_fm_n, f_n_list = get_fed_fam_dataset(args,
                                                                        data_name='{}.h5'.format(
                                                                            args.training_dataset_name),
                                                                        split='training')

    for dataset_loader in trainset_loader_list:
        for batch_idx, (inputs, targets) in enumerate(dataset_loader):
            print(inputs, targets)
