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
from function_utils import plot_fam_stat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

labels = [i for i in range(10)]


def sort_training_MNIST_data():
    """
    将原始的数据按照label进行划分
    :return:
    """
    training_img_path = './data/mnist/MNIST/processed/training.pt'
    img, label = torch.load(training_img_path)
    img = img.detach().numpy()
    label = label.detach().numpy()
    print(label)
    print('Finish loading!')

    # 将原本的表情数据集按照被试划分
    data_dict = dict()
    # Init data_dict for saving the dataset divided by label
    for i in range(len(labels)):
        data_dict[i] = []

    for lb, data in zip(label, img):
        data_dict[lb].append(data)

    print('Person numbers: ', len(data_dict.keys()))
    # print(fs_dict)
    # for k in fs_dict.keys():
    #     print('{}: {}'.format(k, len(fs_dict[k].keys())))

    save2pkl('./dataset/', data_dict, 'MNIST_training_div_label')

    return data_dict


# ======================= Non-IID 数据构造 ====================
def construct_federated_NonIID_datasets(dataset_name='', family_number=10, fam_max_size=1000):
    lb_num = len(labels)
    lb_select_max_n = 1
    training_data = load_pkl_file('./dataset/MNIST_training_div_label.pkl')
    print('Finish loading!')

    # ----------------- 构造训练集 -----------------
    # 用于统计每个家庭数据的数据分布情况
    families_stat_dict = dict()
    families_data_dict = dict()

    for lb in labels:
        # 将每个标签数据转化为队列
        training_data[lb] = construct_queue(training_data[lb])
    lb_select_set1 = [i % 10 for i in range(lb_num)]    # 用于强制每个用户使用一个标签的数据，Non-IID
    lb_select_set2 = np.random.randint(0, lb_num, family_number-lb_num)
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

        families_stat_dict['Client{}'.format(i+1)] = [tem_family_dict['stat'][l] for l in range(lb_num)]
        families_data_dict['Client{}'.format(i + 1)] = tem_family_dict['data']

    # 绘制数据集的统计图
    save2json('./dataset/', families_stat_dict, '{}_fn{}_lb{}_NonIID_stat'.format(dataset_name,
                                                                                  family_number, lb_select_max_n))  # 存储统计信息
    plot_fam_stat(families_stat_dict, labels, '{}_fn{}_lb{}_NonIID'.format(dataset_name, family_number,
                                                                           lb_select_max_n))

    # ---------------- 构造测试集 --------------------
    test_img_path = './raw_datasets/mnist/test.pt'
    test_img, test_label = torch.load(test_img_path)
    test_img = test_img.detach().numpy()
    test_label = test_label.detach().numpy()
    test_num = len(test_label)
    test_m = test_num // family_number
    # 生成每个家庭的测试集
    test_set_imglist = []
    test_set_lblist = []
    # p = 0
    # while p < test_num:
    #     # img_t = []
    #     # lb_t = []
    #     if p + test_m < test_num:
    #         test_set_imglist.append(test_img[p:p + test_m])
    #         test_set_lblist.append(test_label[p:p + test_m])
    #         p = p + test_m
    #     else:
    #         # img_t.append(img[p:])
    #         # lb_t.append(label[p:])
    #         test_set_imglist.append(test_img[p:])
    #         test_set_lblist.append(test_label[p:])
    #         break

    # save all training data set to .h5
    for i in range(family_number):
        sampled_index = random.sample(range(test_num), int(fam_max_size * 0.25))
        test_set_imglist.append(test_img[sampled_index])
        test_set_lblist.append(test_label[sampled_index])

    with h5py.File('./dataset/{}_fn{}_lb{}_NonIID_training.h5'.format(dataset_name, family_number,
                                                                           lb_select_max_n), 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")
        for i, f_name, f_data, f_test_data, f_test_lb in zip(range(family_number),
                                                             families_data_dict.keys(), families_data_dict.values(),
                                                             test_set_imglist, test_set_lblist):
            # create h5f group
            train_f_group = group_train.create_group(f_name)
            test_f_group = group_test.create_group(f_name)
            # 用于记录当前家庭的训练集和测试集数据
            training_data_list = []
            training_label_list = []

            for label, data in zip(f_data.keys(), f_data.values()):
                training_label_list.extend([label] * len(data))
                training_data_list.extend(data)

            training_data = np.array(training_data_list)
            training_label = np.array(training_label_list, dtype=np.int64)

            testing_label = np.array(f_test_lb, dtype=np.int64)

            train_f_group.create_dataset('FEdata_pixel', data=training_data)
            train_f_group.create_dataset('FEdata_label', data=training_label)
            test_f_group.create_dataset('FEdata_pixel', data=f_test_data)
            test_f_group.create_dataset('FEdata_label', data=testing_label)

    h5f.close()
    check_h5('./dataset/{}_fn{}_lb{}_NonIID_training.h5'.format(dataset_name, family_number,
                                                                           lb_select_max_n))


# =======================IID 数据===========================
def construct_federated_IID_datasets(dataset_name_postfix='', family_number=50, fam_max_size=1000):
    # 载入原始数据
    training_img_path = './data/mnist/MNIST/processed/training.pt'
    training_img, training_label = torch.load(training_img_path)
    training_img = training_img.detach().numpy()
    training_label = training_label.detach().numpy()

    test_img_path = './data/mnist/MNIST/processed/test.pt'
    test_img, test_label = torch.load(test_img_path)
    test_img = test_img.detach().numpy()
    test_label = test_label.detach().numpy()

    print('Finish loading!')

    training_num = training_label.shape[0]
    test_num = test_label.shape[0]

    # 生成训练集家庭
    training_set_imglist = []
    training_set_lblist = []
    p = 0
    while p < training_num:
        m = min(training_num // family_number, fam_max_size)

        # img_t = []
        # lb_t = []
        if p + m < training_num:
            training_set_imglist.append(training_img[p:p + m])
            training_set_lblist.append(training_label[p:p + m])
            p = p + m
        else:
            # img_t.append(img[p:])
            # lb_t.append(label[p:])
            training_set_imglist.append(training_img[p:])
            training_set_lblist.append(training_label[p:])
            break

    # 生成家庭测试集
    test_set_imglist = []
    test_set_lblist = []
    p = 0
    while p < test_num:
        m = test_num // family_number
        if p + m < training_num:
            test_set_imglist.append(test_img[p:p + m])
            test_set_lblist.append(test_label[p:p + m])
            p = p + m
        else:
            # img_t.append(img[p:])
            # lb_t.append(label[p:])
            test_set_imglist.append(test_img[p:])
            test_set_lblist.append(test_label[p:])
            break

    # save all training data set to .h5
    with h5py.File('./dataset/MNIST_fed_IID_training.h5', 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")
        for i, f_train_data, f_train_lb, f_test_data, f_test_lb in zip(range(family_number),
                                                                       training_set_imglist, training_set_lblist,
                                                                       test_set_imglist, test_set_lblist):
            # create h5f group
            train_f_group = group_train.create_group('Family{}'.format(i))
            test_f_group = group_test.create_group('Family{}'.format(i))

            training_label = np.array(f_train_lb, dtype=np.int64)
            testing_label = np.array(f_test_lb, dtype=np.int64)

            train_f_group.create_dataset('FEdata_pixel', data=f_train_data)
            train_f_group.create_dataset('FEdata_label', data=training_label)
            test_f_group.create_dataset('FEdata_pixel', data=f_test_data)
            test_f_group.create_dataset('FEdata_label', data=testing_label)

    h5f.close()
    check_h5('./dataset/MNIST_fed_IID_training.h5')


def construct_IID_datasets():
    # 载入原始数据
    training_img_path = './raw_datasets/training.pt'
    training_img, training_label = torch.load(training_img_path)
    training_img = training_img.detach().numpy()
    training_label = training_label.detach().numpy()

    test_img_path = './raw_datasets/test.pt'
    test_img, test_label = torch.load(test_img_path)
    test_img = test_img.detach().numpy()
    test_label = test_label.detach().numpy()

    print('Finish loading!')

    # save all training data set to .h5
    with h5py.File('./dataset/MNIST_IID_training.h5', 'w') as h5f:

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
    check_h5('./dataset/MNIST_IID_training.h5')


def construct_testing_MNISTdata():
    training_img_path = './data/mnist/MNIST/processed/test.pt'
    img, label = torch.load(training_img_path)
    img = img.detach().numpy()
    label = label.detach().numpy()
    print(label)
    print('Finished!')

    testing_num = label.shape[0]
    print('testing_num', testing_num)

    # 数据保存到本地，方便调用
    fed_training_dataset = {'img_data': img, 'label': label}
    with open('./data/fed_test_MNIST_1.pkl', 'wb') as pkf:
        pickle.dump(fed_training_dataset, pkf)
    pkf.close()


def load_pkl_MNISTdata(fname):
    with open('./data/{}'.format(fname), 'rb') as pkf:
        data_dict = pickle.load(pkf)
    pkf.close()
    return data_dict['img_data'], data_dict['label']


def load_fed_training_MNISTdata():
    with open('./data/fed_training_MNIST_1.pkl', 'rb') as pkf:
        data_dict = pickle.load(pkf)
    pkf.close()
    return data_dict['img_data'], data_dict['label']


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
    set_name = 'mnist_fed2'
    # construct_fed_training_MNISTdata()
    # imgs, labels = load_fed_training_MNISTdata()
    # save_participant_data_as_h5('fed_training_MNIST_1', imgs, labels)

    # Construct training data set
    # construct_training_MNISTdata()
    # sort_training_MNIST_data()
    # construct_IID_datasets()
    construct_federated_NonIID_datasets(dataset_name=set_name)
    # construct_federated_IID_datasets()
    # imgs, labels = load_pkl_MNISTdata('training_MNIST.pkl')
    # save_all_data_as_h5('training_MNIST_1', 'training', imgs, labels)

    # Construct test data set
    # construct_testing_MNISTdata()
