import os
import random
import numpy as np
import torch
import pickle
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
# import transforms as transforms
from torchvision import transforms
from function_utils import save2json, save2pkl, load_pkl_file, construct_queue, get_n_items_from_queue
from function_utils import plot_fam_stat

label_map = {9: 'bottles', 10: 'bowls', 16: 'cans', 28: 'cups', 61: 'plates',
             0: 'apples', 51: 'mushrooms', 53: 'oranges', 57: 'pears', 83: 'sweet peppers',
             22: 'clock', 39: 'computer keyboard', 40: 'lamp', 86: 'telephone', 87: 'television',
             5: 'bed', 20: 'chair', 25: 'couch', 84: 'table', 94: 'wardrobe'}

label_map2new = {9: 0, 10: 1, 16: 2, 28: 3, 61: 4,
                 0: 5, 51: 6, 53: 7, 57: 8, 83: 9,
                 22: 10, 39: 11, 40: 12, 86: 13, 87: 14,
                 5: 15, 20: 16, 25: 17, 84: 18, 94: 19}


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def sort_CIFAR100_dataset_and_select_part_data():
    """
    将原始的数据按照label进行划分
    :return:
    """
    training_dataset_path = 'E:/zx/FedLearning/FedDA/datasets/cifar-100-python/train'
    dataset_dict = unpickle(training_dataset_path)

    print(dataset_dict.keys())
    print('Finish loading!')

    # 将原本的表情数据集按照被试划分
    data_dict = dict()
    for data_name, fine_lb, coarse_lb, data in zip(dataset_dict[b'filenames'], dataset_dict[b'fine_labels'],
                                                   dataset_dict[b'coarse_labels'], dataset_dict[b'data']):

        if coarse_lb in [3, 4, 5, 6]:
            if fine_lb not in data_dict.keys():
                temp_dict = dict()
                temp_dict['data_name'] = []
                temp_dict['data'] = []
                temp_dict['coarse_label'] = []
                temp_dict['fine_label'] = []
                data_dict[fine_lb] = temp_dict

            # 选取室内场景物品部分类别数据（food contains, fruit and vegetables,
            # household electrical devices, household furniture）
            r = data[:1024].reshape(32, 32)
            g = data[1024:2048].reshape(32, 32)
            b = data[2048:].reshape(32, 32)
            image = np.dstack((r, g, b))

            data_dict[fine_lb]['data_name'].append(data_name)
            data_dict[fine_lb]['data'].append(image)
            data_dict[fine_lb]['fine_label'].append(label_map2new[fine_lb])
            data_dict[fine_lb]['coarse_label'].append(coarse_lb)

    save2pkl('./dataset/', data_dict, 'CIFAR20_indoor_sort_train')

    return data_dict


# =======================IID 数据===========================
def construct_federated_IID_datasets(dataset_name_postfix='', family_number=50, fam_max_size=200):
    # 载入原始数据
    training_dataset_path = './dataset/CIFAR20_indoor_sort_train.pkl'
    training_dataset = load_pkl_file(training_dataset_path)

    test_dataset_path = './dataset/CIFAR20_indoor_sort_test.pkl'
    test_dataset = load_pkl_file(test_dataset_path)

    print('Finish loading!')

    # 数据整合
    training_data_list = []
    training_label_list = []
    for k, v in zip(training_dataset.keys(), training_dataset.values()):
        training_data_list.extend(v['data'])
        training_label_list.extend(v['fine_label'])

    test_data_list = []
    test_label_list = []
    for k, v in zip(test_dataset.keys(), test_dataset.values()):
        test_data_list.extend(v['data'])
        test_label_list.extend(v['fine_label'])

    # 获取数据集大小
    training_num = len(training_label_list)
    test_num = len(test_label_list)
    # 将数据集转成数组结构
    training_label = np.array(training_label_list, dtype=np.int64)
    training_img = np.array(training_data_list)

    test_img = np.array(test_data_list)
    test_label = np.array(test_label_list, dtype=np.int64)
    # 构建随机索引
    random_training_idxes = [i for i in range(training_num)]
    random_test_idxes = [i for i in range(test_num)]
    # 随机化序列
    random.shuffle(random_training_idxes)
    random.shuffle(random_test_idxes)

    # 生成训练集家庭
    training_set_imglist = []
    training_set_lblist = []
    p = 0
    while p < training_num:
        m = min(training_num // family_number, fam_max_size)

        if p + m < training_num:
            idx = random_training_idxes[p:p + m]
            training_set_imglist.append(training_img[idx])
            training_set_lblist.append(training_label[idx])
            p = p + m
        else:
            idx = random_training_idxes[p:]
            training_set_imglist.append(training_img[idx])
            training_set_lblist.append(training_label[idx])
            break

    # 生成家庭测试集
    test_set_imglist = []
    test_set_lblist = []
    p = 0
    while p < test_num:
        m = test_num // family_number
        if p + m < training_num:
            idx = random_test_idxes[p:p + m]
            test_set_imglist.append(test_img[idx])
            test_set_lblist.append(test_label[idx])
            p = p + m
        else:
            idx = random_test_idxes[p:]
            # img_t.append(img[p:])
            # lb_t.append(label[p:])
            test_set_imglist.append(test_img[idx])
            test_set_lblist.append(test_label[idx])
            break

    # 统计每类数量
    families_stat_dict = dict()
    training_lb_stat_list = []
    for k, dataset in enumerate(training_set_lblist):
        temp = []
        for i in range(20):
            temp.append(dataset.tolist().count(i))
        training_lb_stat_list.append(temp)
        families_stat_dict['Family{}'.format(k)] = temp

    # 绘制数据集的统计图
    save2json('./dataset/', families_stat_dict, 'cifar20_Fed_IID_stat')  # 存储统计信息

    test_lb_stat_list = []
    for dataset in test_set_lblist:
        temp = []
        for i in range(20):
            temp.append(dataset.tolist().count(i))
        test_lb_stat_list.append(temp)

    # save all training data set to .h5
    with h5py.File('./dataset/CIFAR20_fed_IID_training.h5', 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")
        for i, f_train_data, f_train_lb, f_test_data, f_test_lb in zip(range(family_number),
                                                                       training_set_imglist, training_set_lblist,
                                                                       test_set_imglist, test_set_lblist):
            # create h5f group
            train_f_group = group_train.create_group('Family{}'.format(i))
            test_f_group = group_test.create_group('Family{}'.format(i))

            # training_label = np.array(f_train_lb, dtype=np.int64)
            # testing_label = np.array(f_test_lb, dtype=np.int64)

            train_f_group.create_dataset('FEdata_pixel', data=f_train_data)
            train_f_group.create_dataset('FEdata_label', data=f_train_lb)
            test_f_group.create_dataset('FEdata_pixel', data=f_test_data)
            test_f_group.create_dataset('FEdata_label', data=f_test_lb)

    h5f.close()
    check_h5('./dataset/CIFAR20_fed_IID_training.h5')


def construct_IID_datasets():
    # 载入原始数据
    training_dataset_path = './dataset/CIFAR20_indoor_sort_train.pkl'
    training_dataset = load_pkl_file(training_dataset_path)

    test_dataset_path = './dataset/CIFAR20_indoor_sort_test.pkl'
    test_dataset = load_pkl_file(test_dataset_path)

    print('Finish loading!')

    # 数据整合
    training_data_list = []
    training_label_list = []
    for k, v in zip(training_dataset.keys(), training_dataset.values()):
        training_data_list.extend(v['data'])
        training_label_list.extend(v['fine_label'])

    test_data_list = []
    test_label_list = []
    for k, v in zip(test_dataset.keys(), test_dataset.values()):
        test_data_list.extend(v['data'])
        test_label_list.extend(v['fine_label'])

    # 将数据集转成数组结构
    training_label = np.array(training_label_list, dtype=np.int64)
    training_img = np.array(training_data_list)

    test_img = np.array(test_data_list)
    test_label = np.array(test_label_list, dtype=np.int64)

    # save all training data set to .h5
    with h5py.File('./dataset/CIFAR20_IID_training.h5', 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")

        # training_label = np.array(f_train_lb, dtype=np.int64)
        # testing_label = np.array(f_test_lb, dtype=np.int64)

        group_train.create_dataset('FEdata_pixel', data=training_img)
        group_train.create_dataset('FEdata_label', data=training_label)
        group_test.create_dataset('FEdata_pixel', data=test_img)
        group_test.create_dataset('FEdata_label', data=test_label)

    h5f.close()
    check_h5('./dataset/CIFAR20_IID_training.h5')


def check_h5(fname):
    with h5py.File(fname, 'r') as f:
        for fkey in f.keys():
            print(f[fkey], fkey)

        print("======= 优雅的分割线 =========")

        for fm in f.keys():
            fm_group = f[fm]
            print('>>> Group: ', fm)
            for fm_p in fm_group.keys():
                print(fm_p, fm_group[fm_p])

    f.close()


if __name__ == '__main__':
    # Construct fed training data set
    # construct_fed_training_MNISTdata()
    # imgs, labels = load_fed_training_MNISTdata()
    # save_participant_data_as_h5('fed_training_MNIST_1', imgs, labels)

    # Construct training data set
    # construct_training_MNISTdata()
    # sort_CIFAR100_dataset_and_select_part_data()
    # construct_federated_NonIID_datasets()
    construct_federated_IID_datasets()
    # construct_IID_datasets()

    # construct testing data
