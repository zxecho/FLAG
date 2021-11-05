import os
import random
import json
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

labels = [i for i in range(6)]


# 对原始数据进行读取，和重新排列
def sort_FER_data(dataset_name):
    # 将原本的表情数据集按照被试划分，所有数据{‘People_ID’: [{'data_name':label}, {}, ..., {}]}, 并构建联邦数据集

    fs_dict = dict()

    lbs_prodict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5}

    ck_data_path = 'raw_data/CK/'
    bu3dfe_data_path = 'raw_data/BU3DFE/'

    ck_mark_len = 4
    bu3dfe_mark_len = 5

    lbs = os.listdir(ck_data_path)
    print(lbs)

    # CK 数据集
    for k in range(len(lbs)):
        lb_name = lbs[k]
        lb_dir = ck_data_path + lb_name  # label
        fs = os.listdir(lb_dir)  # data file name
        print('Current processing label dir: {} {}'.format(lb_dir, len(fs)))

        lb = lbs_prodict[lb_name]

        p_mark = fs[0][:ck_mark_len]
        if p_mark not in fs_dict.keys():
            fs_dict[p_mark] = dict()  # 用于存储不同人的表情字典数据集

        for i in range(len(fs)):
            n_mark = fs[i][:ck_mark_len]
            img_data = get_img(fs[i], lb_dir)
            if p_mark == n_mark:
                if lb not in fs_dict[p_mark].keys():
                    fs_dict[p_mark][lb] = []
                fs_dict[p_mark][lb].append(img_data)
            else:
                p_mark = n_mark
                if p_mark not in fs_dict.keys():
                    fs_dict[p_mark] = dict()
                if lb not in fs_dict[p_mark].keys():
                    fs_dict[p_mark][lb] = []
                fs_dict[p_mark][lb].append(img_data)

    # BU3DFE数据集
    for k in range(len(lbs)):
        lb_name = lbs[k]
        lb_dir = bu3dfe_data_path + lb_name  # label
        fs = os.listdir(lb_dir)  # data file name
        print('Current processing label dir: {} {}'.format(lb_dir, len(fs)))

        lb = lbs_prodict[lb_name]

        p_mark = fs[0][:bu3dfe_mark_len]
        if p_mark not in fs_dict.keys():
            fs_dict[p_mark] = dict()  # 用于存储不同人的表情字典数据集

        for i in range(len(fs)):
            n_mark = fs[i][:bu3dfe_mark_len]
            img_data = get_img(fs[i], lb_dir)
            if p_mark == n_mark:
                if lb not in fs_dict[p_mark].keys():
                    fs_dict[p_mark][lb] = []
                fs_dict[p_mark][lb].append(img_data)
            else:
                p_mark = n_mark
                if p_mark not in fs_dict.keys():
                    fs_dict[p_mark] = dict()
                if lb not in fs_dict[p_mark].keys():
                    fs_dict[p_mark][lb] = []
                fs_dict[p_mark][lb].append(img_data)

    print('Person numbers: ', len(fs_dict.keys()))

    save2pkl('./dataset/{}/'.format(dataset_name), fs_dict, '{}_div_by_people'.format(dataset_name))


# 对原始数据进行读取，混合
def mix_FER_data(dataset_name):
    # 将原本的表情数据集按照被试划分，所有数据{‘People_ID’: [{'data_name':label}, {}, ..., {}]}, 并构建联邦数据集

    fs_dict = dict()

    lbs_prodict = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sadness': 4, 'surprise': 5}

    ck_data_path = 'raw_data/CK/'
    bu3dfe_data_path = 'raw_data/BU3DFE/'

    lbs = os.listdir(ck_data_path)
    print(lbs)

    # CK 数据集
    for k in range(len(lbs)):
        lb_name = lbs[k]
        lb_dir = ck_data_path + lb_name  # label
        fs = os.listdir(lb_dir)  # data file name
        print('Current processing label dir: {} {}'.format(lb_dir, len(fs)))

        lb = lbs_prodict[lb_name]

        if lb not in fs_dict.keys():
            fs_dict[lb] = []  # 用于存储不同人的表情字典数据集

        for i in range(len(fs)):
            img_data = get_img(fs[i], lb_dir)
            fs_dict[lb].append(img_data)

    # BU3DFE数据集
    for k in range(len(lbs)):
        lb_name = lbs[k]
        lb_dir = bu3dfe_data_path + lb_name  # label
        fs = os.listdir(lb_dir)  # data file name
        print('Current processing label dir: {} {}'.format(lb_dir, len(fs)))

        lb = lbs_prodict[lb_name]

        if lb not in fs_dict.keys():
            fs_dict[lb] = []  # 用于存储不同人的表情字典数据集

        for i in range(len(fs)):
            img_data = get_img(fs[i], lb_dir)
            fs_dict[lb].append(img_data)

    print('Person numbers: ', len(fs_dict.keys()))

    save2pkl('./dataset/{}/'.format(dataset_name), fs_dict, '{}_mixed(48x48)'.format(dataset_name))


# ======================= Non-IID 数据构造 ====================
def construct_federated_NonIID_datasets_v1(dataset_name='', family_number=20, fam_max_size=1000):
    fpt = 'dataset/FER/FER_div_by_people.pkl'
    source_data = load_pkl_file(fpt)

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

    # ----------------- 构造训练集 -----------------
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


def construct_federated_NonIID_datasets_v2(dataset_name):
    """"
    （ID，label）均随机抽取
    按照被试划分，构建联邦问题数据集，以家庭为单位，每个家庭所有成员样本混合，每个家庭的0.8为训练集，0.2为测试集
    :parameter
        source_data_sat: 原始表情数据的按被试划分的统计 dictionary
    """
    fpt = 'dataset/FER/FER_div_by_people.pkl'
    source_data = load_pkl_file(fpt)
    labels = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    label_map = [i for i in range(len(labels))]

    keys_array = np.array([k for k in source_data.keys()])
    # 打乱排序
    keys_array = np.random.permutation(keys_array)
    n = len(source_data.keys())
    # 生成训练集家庭
    training_set_list = []
    p = 0
    while p <= n:
        m = np.random.randint(2, 5)  # 2-4个人组成一个家庭
        t = []
        if p + m < n:
            for i in range(m):
                t.append(keys_array[p + i])
        else:
            for i in range(n - p):
                t.append(keys_array[p + i])
            training_set_list.append(t)
            break
        p = p + m
        training_set_list.append(t)
    print('Training families: ', training_set_list)

    training_rate = 0.8
    # 用于统计数据集（ID, label）概况
    train_ID_dict = dict()
    train_label_dict = {k: 0 for k in label_map}
    test_ID_dict = dict()
    test_label_dict = {k: 0 for k in label_map}

    # 用于总体统计当前数据集每个家庭的训练集与测试集的状况
    fam_train_stat_dict = dict()
    fam_test_stat_dict = dict()

    # save all training data set to .h5
    with h5py.File('./dataset/{}/FER48_fed_NonIID_training.h5'.format(dataset_name), 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")
        for i, fam in enumerate(training_set_list):
            train_f_group = group_train.create_group("Client{}".format(i))
            test_f_group = group_test.create_group("Client{}".format(i))

            # 用于记录家庭所有成员数据
            fam_data_list = []
            fam_len = 0

            # 将家庭成员数据合并
            for p in fam:
                fam_data_list.append(source_data[p])  # 获取当前背视的数据

            # 构造local data set
            ''' 1. 所有家庭数据随机打乱'''
            # fam_pds = np.random.permutation(fam_pds)
            ''' 2. 随机选择每个家庭的label'''
            # select local data set by labels for constructing Non-IID data set
            # select_ls = np.random.permutation(label_map)
            # train_labels = select_ls[:4]
            # test_labels = select_ls[4:]
            # cur_f_training_set = get_selected_lbs_data(fam_pds, train_labels)
            # cur_f_test_set = get_selected_lbs_data(fam_pds, test_labels)
            ''' 3. 每个家庭数据进行截断'''
            # training_num = int(fam_l * training_rate)
            # cur_p_training_set = fam_pds[:training_num]
            # cur_p_test_set = fam_pds[training_num:]

            ''' 4. Dirichlet 采样 '''
            cur_f_training_data_set, cur_f_training_lbs_set, cur_f_test_data_set, cur_f_test_lbs_set, \
            f_train_stat, f_test_stat = get_data_by_dirichlet_dist(
                fam_data_list,
                label_map,
                training_rate,
                i,
                dataset_name,
                10.0)

            fam_train_stat_dict['Client{}'.format(i)] = [f_train_stat[k] for k in label_map]
            fam_test_stat_dict['Client{}'.format(i)] = [f_test_stat[k] for k in label_map]

            training_data = np.array(cur_f_training_data_set)
            training_label = np.array(cur_f_training_lbs_set, dtype=np.int64)

            test_data = np.array(cur_f_test_data_set)
            test_label = np.array(cur_f_test_lbs_set, dtype=np.int64)

            train_f_group.create_dataset('FEdata_pixel', data=training_data)
            train_f_group.create_dataset('FEdata_label', data=training_label)
            test_f_group.create_dataset('FEdata_pixel', data=test_data)
            test_f_group.create_dataset('FEdata_label', data=test_label)

    h5f.close()

    # 存储统计数据
    save2json('./dataset/{}'.format(dataset_name), fam_train_stat_dict, 'FER48_fed_NonIID_training_stat')  # 存储统计信息
    save2json('./dataset/{}'.format(dataset_name), fam_test_stat_dict, 'FER48_fed_NonIID_test_stat')  # 存储统计信息

    plot_fam_stat(fam_train_stat_dict, label_map, dataset_name + '_train')
    plot_fam_stat(fam_test_stat_dict, label_map, dataset_name + '_test')

    check_h5('./dataset/{}/FER48_fed_NonIID_training.h5'.format(dataset_name))


def construct_IID_datasets(dataset_name=''):
    fpt = 'dataset/FER/FER_mixed(48x48).pkl'
    source_data = load_pkl_file(fpt)
    print('Finish loading raw data!')

    training_rate = 0.8

    # 制作训练集和测试集
    training_set_imglist, training_set_lblist = [], []
    test_set_imglist, test_set_lblist = [], []

    for k, v in zip(source_data.keys(), source_data.values()):
        data_len = len(v)
        training_n = int(data_len * training_rate)
        random.shuffle(v)
        training_set_imglist.extend(v[:training_n])
        training_set_lblist.extend([k] * training_n)

        test_set_imglist.extend(v[training_n:])
        test_set_lblist.extend([k] * (data_len - training_n))

    # save all training data set to .h5
    with h5py.File('./dataset/{}/FER48_mixed_IID_training.h5'.format(dataset_name), 'w') as h5f:
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
    check_h5('./dataset/{}/FER48_mixed_IID_training.h5'.format(dataset_name))


'''
Utils Function =========================================================================
'''


def get_data_by_dirichlet_dist(fam_data_list, label_list, train_rate, fam_No, save_name, Concentration=10.):
    # 用于当前家庭数据的统计, 并画图
    cur_fam_train_dict = dict()
    cur_fam_test_dict = dict()

    ln = len(label_list)  # 标签长度
    training_data_list = []
    training_lbs_list = []
    test_data_list = []
    test_lbs_list = []
    fam_data_dict_set_by_lb = get_label_dict_set(fam_data_list, label_list)
    p = torch.tensor([1 / ln for _ in range(ln)])  # 初始化Dirichlet采样概率
    # 初始化Dirichlet 采样函数
    dist = torch.distributions.dirichlet.Dirichlet(Concentration * p)
    p = dist.sample()  # 采样概率
    fam_lp = np.array([round(p.item(), 1) for p in p])  # 根据采样得到的概率
    for lb, lp in zip(label_list, fam_lp):
        cur_lb_data = fam_data_dict_set_by_lb[lb]  # 当前标签的家庭成员所有数据
        lsn = 0
        if lp > 0:
            train_rate = max((1 - lp), train_rate)
            lsn = int(len(cur_lb_data) * train_rate)  # 当前样本需要采样个数

        cur_fam_train_dict[lb] = lsn
        cur_fam_test_dict[lb] = len(cur_lb_data) - lsn

        # 将原本标签对应的所有数据随机化
        ramdom_lb_fam_data = np.random.permutation(cur_lb_data)
        # 将随机选择的训练集和测试集数据和标签都加入列表，便于生成数据集
        training_data_list.extend(ramdom_lb_fam_data[:lsn])
        training_lbs_list.extend([lb] * cur_fam_train_dict[lb])
        test_data_list.extend(ramdom_lb_fam_data[lsn:])
        test_lbs_list.extend([lb] * cur_fam_test_dict[lb])

    save2json('./dataset/{}/client_stat/train/'.format(save_name), cur_fam_train_dict,
              'client{}_train_stat_info'.format(fam_No))
    save2json('./dataset/{}/client_stat/test/'.format(save_name), cur_fam_test_dict,
              'client{}_test_stat_info'.format(fam_No))

    return training_data_list, training_lbs_list, test_data_list, test_lbs_list, cur_fam_train_dict, cur_fam_test_dict


def get_label_dict_set(data_list, label_list):
    # 按照标签分类，将一个家庭的所有数据保存到一个字典中 {‘label1’: [{}, {}...{}], 'label2':[{}, {}...{}1]}
    t = {lb: [] for lb in label_list}
    for data in data_list:
        for k, v in zip(data.keys(), data.values()):
            t[k].extend(v)
    return t


def get_img(img_info, src_dataset):
    img_path = './{}/{}'.format(src_dataset, img_info)
    img = Image.open(img_path).convert('L')
    img = img.resize((48, 48))
    array_img = np.array(img)
    if len(array_img.shape) < 3:
        array_img = np.expand_dims(array_img, axis=2)
    return array_img


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


def save_to_json(fname, data):
    with open(fname, 'w') as file_obj:
        json.dump(data, file_obj)


if __name__ == '__main__':
    set_name = 'FER'
    sort_FER_data(set_name)
    # mix_FER_data(set_name)
    construct_federated_NonIID_datasets_v2(dataset_name=set_name)
    # construct_IID_datasets(dataset_name=set_name)
