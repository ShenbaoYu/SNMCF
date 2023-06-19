# -*- coding: utf-8 -*-
"""
功能：对数据集做基本数据处理和统计
"""


import random
import numpy as np
from numpy.core.fromnumeric import shape


def missing_stu_exe(stu_exe, miss_rate):
    """
    功能: 将(存在NaN的数据)学生作答矩阵做缺失值处理, 缺失值用NaN填充
    方法: 1. 先找出所有有作答记录的位置; 2. 再随机从这些位置中按照缺失比例做随机缺失，记录下缺失的位置作为测试集合
    
    Inputs:
    -------
    :param stu_exe --> numpy.ndarray
        学生-习题原始作答矩阵
        row: 习题
        col: 学生
    :param miss_rate --> float
        缺失比例

    Outputs:
    -------
    :return train_data --> numpy.ndarray
        学生-习题作答矩阵
        row: 习题
        col: 学生
    :return test_loc --> list([exe_id, stu_id],...)
        缺失的位置集合
    """

    stu_num = stu_exe.shape[1]  # 获得学生作答矩阵的行和列
    train_data = stu_exe.copy()

    test_loc = list()
    for stu in range(stu_num):
        nan_index = list(np.where(np.isnan(train_data[:,stu]))[0])
        del_num = int((len(train_data[:,stu]) - len(nan_index)) * miss_rate)
        _ = [i for i, j in enumerate(train_data[:,stu]) if i not in nan_index]  # the list that consists of index can be deleted
        random.shuffle(_)
        while del_num > 0 and len(_) > 0:
            del_index = _.pop()
            res_log = train_data[del_index,:].copy()  # copy the exercise response log
            res_log[stu] = np.NaN  # attemp to delete this record
            if np.isnan(res_log).all():
                # if the exercise (ID = del_index) has no anwser records
                # after delete this response log, ignore it.
                continue
            else:
                train_data[del_index][stu] = np.NaN
                test_loc.append([del_index, stu])
                del_num -= 1
    
    return train_data, test_loc


def matrix_miss_fill(stu_exe_miss):
    """
    功能: 对缺失的学生作答数据做填充处理
    方法1: 0填充
    方法2: 0/1随机填充

    Inputs:
    -------
    :param stu_exe_miss --> numpy.ndarray
        带有缺失数据的学生-习题作答矩阵
        row: 习题
        col: 学生

    Outputs:
    -------
    :return stu_exe_fill --> numpy.ndarray
        填充后的学生-习题作答矩阵
        row: 习题
        col: 学生
    """

    stu_exe_fill = stu_exe_miss.copy()
    # stu_exe_fill[np.isnan(stu_exe_fill)] = np.random.randint(0,2)  # 0/1随机填充
    stu_exe_fill[np.isnan(stu_exe_fill)] = 0  # 用0填充缺失值
    
    return stu_exe_fill


def matrix_miss_fill_GBE(stu_exe_miss):
    """
    功能: 对缺失的学生作答数据做填充处理
    方法: Global Baseline Estimation
        1. 获取整个数据集的平均得分 r(avg)
        2. 获取习题Exn的平均得分 Exn(avg)
        3. 获取学生Sm的平均得分 Sm(avg)
        4. 填充缺失值 r(nm) =  Exn(avg) + Sm(avg) - r(avg)
    
    Inputs:
    -------
    :param stu_exe_miss --> numpy.ndarray
        带有缺失数据的学生-习题作答矩阵(np.NaN)
        row: 习题
        col: 学生

    Outputs:
    -------
    :return stu_exe_fill --> numpy.ndarray
        填充后的学生-习题作答矩阵
        row: 习题
        col: 学生
    """

    shape = stu_exe_miss.shape
    stu_exe_fill = stu_exe_miss.copy()  

    exe_avg = dict()  # 每一道习题的平均得分
    stu_avg = dict()  # 每一个学生的平均得分
    ans_avg = 0  # 整个数据集的平均得分

    # 记录缺失位置信息
    miss_coo = np.argwhere(np.isnan(stu_exe_fill)).tolist()

    # 计算整个数据集的平均得分
    ans_avg = stu_exe_fill[stu_exe_fill>=0].sum() / (shape[0] * shape[1] - len(miss_coo))

    # 记录每一道习题的平均得分
    for _ in range(shape[0]):
        exe_list = stu_exe_fill[_]  # 提取习题_的所有作答情况
        ans_total = exe_list[exe_list>=0].sum()  # 记录该习题的得分和
        count = len(exe_list) - len(np.argwhere(np.isnan(exe_list)))  # 记录非空元素个数
        exe_avg[_] = ans_total / count

    # 记录每一个学生的平均得分
    for _ in range(shape[1]):
        stu_list = stu_exe_fill[:,_]  # 提取学生_的所有的作答情况
        ans_total = stu_list[stu_list>=0].sum()  # 记录该学生的得分和
        count = len(stu_list)- len(np.argwhere(np.isnan(stu_list)))  # 记录非空元素个数
        stu_avg[_] = ans_total / count

    # 填充缺失数据
    for coo in miss_coo:
        stu_exe_fill[coo[0]][coo[1]] = exe_avg[coo[0]] + stu_avg[coo[1]] - ans_avg
    
    stu_exe_fill[stu_exe_fill < 0] = 0
    stu_exe_fill[np.isnan(stu_exe_fill)] = 0   # 如果数据太稀疏，仍然有出现NaN的情况，则用0填充
    
    return stu_exe_fill


def stu_exe_repe_col(stu_exe, repe_rate):
    """
    功能: 修改学生作答矩阵
    方法: 随机复制某些列到其它列, 使得作答矩阵看起来更"相似"

    Inputs:
    -------
    :param stu_exe --> numpy.ndarray
        学生作答矩阵
        row: 习题
        col: 学生

    :param repe_rate --> float
        复制占比

    Outputs:
    -------
    :return stu_exe --> numpy.ndarray
        处理后的学生作答矩阵
        row: 习题
        col: 学生
    """

    col = stu_exe.shape[1]

    repe_num = col * repe_rate

    count = 0
    rand_col_1 = np.random.randint(0, col)
    while count <= repe_num:
        rand_col_2 = np.random.randint(0, col)
        if rand_col_1 != rand_col_2:
            stu_exe[:, rand_col_2] = stu_exe[:, rand_col_1]

        count += 1

    return stu_exe


def stu_exe_repe_row(stu_exe, repe_rate):
    """
    功能: 修改学生作答矩阵
    方法: 随机复制某些行到其它行, 使得作答矩阵看起来更"相似"

    Inputs:
    -------
    :param stu_exe --> numpy.ndarray
        学生作答矩阵
        row: 习题
        col: 学生

    :param repe_rate --> float
        复制占比

    Outputs:
    -------
    :return stu_exe --> numpy.ndarray
        处理后的学生作答矩阵
        row: 习题
        col: 学生
    """

    row = stu_exe.shape[0]

    repe_num = row * repe_rate

    count = 0
    rand_row_1 = np.random.randint(0, row)
    while count <= repe_num:
        rand_row_2 = np.random.randint(0, row)
        if rand_row_1 != rand_row_2:
            stu_exe[rand_row_2] = stu_exe[rand_row_1]

        count += 1

    return stu_exe