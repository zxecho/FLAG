import os
import random
import numpy as np
import torch
import pickle
import h5py
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from function_utils import save2json, save2pkl, load_pkl_file, construct_queue, get_n_items_from_queue
from function_utils import plot_fam_stat, mkdir

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

labels = [i for i in range(10)]


def sort_Fashion_MNIST_data():
    """
    将原始的数据按照label进行划分
    :return:
    """
    training_label_path = './raw_datasets/fashion_mnist/train-labels-idx1-ubyte'
    with open(training_label_path, 'rb') as lbf:
        training_labels = np.frombuffer(lbf.read(), dtype=np.uint8, offset=8)

    labels_n = len(set(training_labels))

    training_img_path = './raw_datasets/fashion_mnist/train-images-idx3-ubyte'
    with open(training_img_path, 'rb') as imgf:
        training_images = np.frombuffer(imgf.read(), dtype=np.uint8, offset=16).reshape(len(training_labels), 784)

    # test dataset
    test_label_path = './raw_datasets/fashion_mnist/t10k-labels-idx1-ubyte'
    with open(test_label_path, 'rb') as lbf:
        test_labels = np.frombuffer(lbf.read(), dtype=np.uint8, offset=8)

    test_img_path = './raw_datasets/fashion_mnist/t10k-images-idx3-ubyte'
    with open(test_img_path, 'rb') as imgf:
        test_images = np.frombuffer(imgf.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 784)

    print('Finish loading!')

    # show images
    # img = training_images[0].reshape(28, 28)
    # plt.imshow(img)
    # plt.show()

    # 将原本的表情数据集按照被试划分
    data_dict = dict()
    # Init data_dict for saving the dataset divided by label
    for i in range(labels_n):
        data_dict[i] = []

    for lb, data in zip(training_labels, training_images):
        data = np.array(data).reshape(28, 28)
        data_dict[lb].append(data)

    print('Label numbers: ', len(data_dict.keys()))

    save2pkl('./dataset/', data_dict, 'FashionMNIST_training_div_by_label')

    data_dict = dict()
    for i in range(labels_n):
        data_dict[i] = []
    for lb, data in zip(test_labels, test_images):
        data = np.array(data).reshape(28, 28)
        data_dict[lb].append(data)

    print('Label numbers: ', len(data_dict.keys()))

    save2pkl('./dataset/', data_dict, 'FashionMNIST_test_div_by_label')


# ======================= Non-IID 数据构造 ====================
def construct_federated_NonIID_datasets(dataset_name='', family_number=20, fam_max_size=1000):
    mkdir('./dataset/{}'.format(dataset_name))
    lb_num = len(labels)
    lb_select_max_n = 1
    training_data = load_pkl_file('./dataset/FashionMNIST_training_div_by_label.pkl')
    print('Finish loading fashion-mnist training data!')

    # ----------------- 构造训练集 -----------------
    # 用于统计每个家庭数据的数据分布情况
    families_stat_dict = dict()
    families_data_dict = dict()

    for lb in labels:
        # 将每个标签数据转化为队列
        training_data[lb] = construct_queue(training_data[lb])
    lb_select_set1 = [i % 10 for i in range(lb_num)]  # 用于强制每个用户使用一个标签的数据，Non-IID
    lb_select_set2 = np.random.randint(0, lb_num, family_number - lb_num)
    lb_select = lb_select_set1 + lb_select_set2.tolist()
    random.shuffle(lb_select)
    for i in range(family_number):
        lb_s = lb_select[i]
        tem_family_dict = {'data': dict(), 'stat': dict()}
        for j in range(lb_num):
            tem_family_dict['stat'][j] = 0
            tem_family_dict['data'][j] = []
        lb_n = np.random.randint(1, lb_select_max_n + 1)  # 当前家庭所具有的标签种类数
        # lb_select = random.sample(range(lb_num), lb_n)  # 随机选择出lb_n个标签
        m = 50000 // family_number  # 随机出当前数据集的个数
        m = min(m, fam_max_size)
        lb_data_n = m // lb_n
        # 将每个随机到的标签，从对应的数据集中取出m个
        # for lb_s in lb_s:
        tem_family_dict['data'][lb_s], fam_data_len = get_n_items_from_queue(training_data[lb_s], lb_data_n)
        if fam_data_len != lb_data_n:
            tem_family_dict['stat'][lb_s] = fam_data_len
        else:
            tem_family_dict['stat'][lb_s] = lb_data_n

        families_stat_dict['Client{}'.format(i + 1)] = [tem_family_dict['stat'][l] for l in range(lb_num)]
        families_data_dict['Client{}'.format(i + 1)] = tem_family_dict['data']

    # 绘制数据集的统计图
    save2json('./dataset/{}'.format(dataset_name), families_stat_dict, '{}_fn{}_lb{}_NonIID_stat'.format(dataset_name,
                                                                                                         family_number,
                                                                                                         lb_select_max_n))  # 存储统计信息
    plot_fam_stat(families_stat_dict, labels, '{}_fn{}_lb{}_NonIID'.format(dataset_name, family_number,
                                                                           lb_select_max_n))

    # ---------------- 构造测试集 --------------------
    test_data = load_pkl_file('./dataset/FashionMNIST_test_div_by_label.pkl')
    print('Finish loading fashion-mnist test data!')

    # ----------------- 构造测试集 -----------------
    # 用于统计每个家庭数据的数据分布情况
    families_test_stat_dict = dict()
    families_test_data_dict = dict()

    for lb in labels:
        # 将每个标签数据转化为队列
        test_data[lb] = construct_queue(test_data[lb])
    lb_select_list = labels  # 使用所有标签作为测试集
    m = fam_max_size * 0.25  # 每个参与方测试机的样本总数
    lb_data_n = int(m // len(lb_select_list))
    for i in range(family_number):
        tem_family_dict = {'data': dict(), 'stat': dict()}
        for j in range(lb_num):
            tem_family_dict['stat'][j] = 0
            tem_family_dict['data'][j] = []
        lb_n = np.random.randint(1, lb_select_max_n + 1)  # 当前家庭所具有的标签种类数
        # lb_select = random.sample(range(lb_num), lb_n)  # 随机选择出lb_n个标签
        # 将每个随机到的标签，从对应的数据集中取出m个
        for lb_s in lb_select_list:
            tem_family_dict['data'][lb_s], fam_data_len = get_n_items_from_queue(test_data[lb_s], lb_data_n)
            if fam_data_len != lb_data_n:
                tem_family_dict['stat'][lb_s] = fam_data_len
            else:
                tem_family_dict['stat'][lb_s] = lb_data_n

        families_test_stat_dict['Client{}'.format(i + 1)] = [tem_family_dict['stat'][l] for l in range(lb_num)]
        families_test_data_dict['Client{}'.format(i + 1)] = tem_family_dict['data']

    # 绘制数据集的统计图
    save2json('./dataset/{}'.format(dataset_name), families_stat_dict,
              '{}_fn{}_lb{}_NonIID_stat_test_data'.format(dataset_name,
                                                          family_number,
                                                          lb_select_max_n))  # 存储统计信息
    plot_fam_stat(families_test_stat_dict, labels, '{}_fn{}_lb{}_NonIID_test_data'.format(dataset_name, family_number,
                                                                                     lb_select_max_n))

    with h5py.File('./dataset/{}/{}_fn{}_lb{}_NonIID_training.h5'.format(dataset_name, dataset_name, family_number,
                                                                         lb_select_max_n), 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")
        for i, f_name, f_data, f_test_data in zip(range(family_number),
                                                             # training data
                                                             families_data_dict.keys(),
                                                             families_data_dict.values(),
                                                             # test data
                                                             families_test_data_dict.values()):
            # create h5f group
            train_f_group = group_train.create_group(f_name)
            test_f_group = group_test.create_group(f_name)
            # 用于记录当前家庭的训练集和测试集数据
            training_data_list = []
            training_label_list = []

            for label, data in zip(f_data.keys(), f_data.values()):
                training_label_list.extend([label] * len(data))
                training_data_list.extend(data)

            testing_data_list = []
            testing_label_list = []
            for label, data in zip(f_test_data.keys(), f_test_data.values()):
                testing_label_list.extend([label] * len(data))
                testing_data_list.extend(data)

            training_data = np.array(training_data_list)
            training_label = np.array(training_label_list, dtype=np.int64)

            testing_data = np.array(testing_data_list)
            testing_label = np.array(testing_label_list, dtype=np.int64)

            train_f_group.create_dataset('FEdata_pixel', data=training_data)
            train_f_group.create_dataset('FEdata_label', data=training_label)
            test_f_group.create_dataset('FEdata_pixel', data=testing_data)
            test_f_group.create_dataset('FEdata_label', data=testing_label)

    h5f.close()
    check_h5('./dataset/{}/{}_fn{}_lb{}_NonIID_training.h5'.format(dataset_name, dataset_name, family_number,
                                                                   lb_select_max_n))


def construct_IID_datasets(dataset_name=''):
    training_label_path = './raw_datasets/fashion_mnist/train-labels-idx1-ubyte'
    with open(training_label_path, 'rb') as lbf:
        training_labels = np.frombuffer(lbf.read(), dtype=np.uint8, offset=8)

    labels_n = len(set(training_labels))

    training_img_path = './raw_datasets/fashion_mnist/train-images-idx3-ubyte'
    with open(training_img_path, 'rb') as imgf:
        training_images = np.frombuffer(imgf.read(), dtype=np.uint8, offset=16).reshape(len(training_labels), 784)

    # test dataset
    test_label_path = './raw_datasets/fashion_mnist/t10k-labels-idx1-ubyte'
    with open(test_label_path, 'rb') as lbf:
        test_labels = np.frombuffer(lbf.read(), dtype=np.uint8, offset=8)

    test_img_path = './raw_datasets/fashion_mnist/t10k-images-idx3-ubyte'
    with open(test_img_path, 'rb') as imgf:
        test_images = np.frombuffer(imgf.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 784)

    print('Finish loading raw data!')

    # 制作训练集和测试集
    training_set_imglist, training_set_lblist = [], []
    test_set_imglist, test_set_lblist = [], []
    # training dataset
    for lb, data in zip(training_labels, training_images):
        data = np.array(data).reshape(28, 28)
        training_set_imglist.append(data)
        training_set_lblist.append(lb)

    # training dataset
    for lb, data in zip(test_labels, test_images):
        data = np.array(data).reshape(28, 28)
        test_set_imglist.append(data)
        test_set_lblist.append(lb)

    # save all training data set to .h5
    with h5py.File('./dataset/{}/FashionMNIST_IID_training.h5'.format(dataset_name), 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")

        training_data = np.array(training_set_imglist)
        training_label = np.array(training_set_lblist, dtype=np.int64)

        testing_data = np.array(test_set_imglist)
        testing_label = np.array(test_set_lblist, dtype=np.int64)

        group_train.create_dataset('FEdata_pixel', data=training_data)
        group_train.create_dataset('FEdata_label', data=training_label)
        group_test.create_dataset('FEdata_pixel', data=testing_data)
        group_test.create_dataset('FEdata_label', data=testing_label)

    h5f.close()
    check_h5('./dataset/{}/FashionMNIST_IID_training.h5'.format(dataset_name))

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
    set_name = 'fashion_mnist_fed'

    # sort_Fashion_MNIST_data()
    # construct_federated_NonIID_datasets(dataset_name=set_name)
    construct_IID_datasets(dataset_name=set_name)
