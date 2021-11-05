import os
import random
import numpy as np
import torch
import pickle
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import torch.utils.data as data
import transforms as transforms
from FedDA.function_utils import save2json, save2pkl, load_pkl_file

labels10 = {'bed': 0, 'cabinet': 1, 'chair': 2, 'door': 3, 'refrigerator': 4, 'sofa': 5, 'table': 6,
            'television': 7, 'water_dispencer': 8, 'wheelchair': 9}

label5 = {'bed': 0, 'chair': 1, 'door': 2, 'sofa': 3, 'table': 4}


def construct_pkl_datasets():
    raw_datasets_path = 'G:/ML_datasets/Indoor datasets/MYNursingHome/selected_nursinghome5/'
    labels = os.listdir(raw_datasets_path)
    dataset_dict = dict()
    labels_dict = dict()
    for idx, label in enumerate(labels):
        labels_dict[label] = idx
        dataset_dict[label] = dict()
        dataset_dict[label]['data'] = []
        dataset_dict[label]['labels'] = []
        data_dir = raw_datasets_path + '{}/'.format(label)
        data_list = os.listdir(data_dir)
        for img_name in data_list:
            img_path = data_dir + '/' + img_name
            img = Image.open(img_path)
            img = img.resize((256, 256))
            # img.show()
            img = np.array(img)
            dataset_dict[label]['data'].append(img)
            dataset_dict[label]['labels'].append(idx)

        print('Finished {} !'.format(label))

    print(labels_dict)

    save2pkl('./data/', dataset_dict, 'IndoorNursing_objects_n5_r256')

    return


def construct_IID_datasets():
    data_path = './dataset/IndoorNursing_objects_c5n200_r256.pkl'
    data = load_pkl_file(data_path)

    training_images = []
    training_labels = []

    test_images = []
    test_labels = []

    training_ratio = 0.8

    for k, v in zip(data.keys(), data.values()):
        data_len = len(v['data'])
        split_number = int(data_len*training_ratio)
        training_images.extend(v['data'][:split_number])
        training_labels.extend(v['labels'][:split_number])

        test_images.extend(v['data'][split_number:])
        test_labels.extend(v['labels'][split_number:])

        # 将数据集转成数组结构
    training_images = np.array(training_images)
    training_labels = np.array(training_labels, dtype=np.int64)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels, dtype=np.int64)

    # save all training data set to .h5
    with h5py.File('./dataset/IndoorNursing5_IID_training_c5n200.h5', 'w') as h5f:
        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")

        # training_label = np.array(f_train_lb, dtype=np.int64)
        # testing_label = np.array(f_test_lb, dtype=np.int64)

        group_train.create_dataset('FEdata_pixel', data=training_images)
        group_train.create_dataset('FEdata_label', data=training_labels)
        group_test.create_dataset('FEdata_pixel', data=test_images)
        group_test.create_dataset('FEdata_label', data=test_labels)

    h5f.close()
    check_h5('./dataset/IndoorNursing5_IID_training_c5n200.h5')


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
    # construct_pkl_datasets()
    construct_IID_datasets()