# -*- coding: utf-8 -*-
"""
功能：定义并且计算评价指标
"""


import math
import numpy as np
import pandas as pd
from scipy import stats

from pandas.core.frame import DataFrame


def cal_accuracy_q(matrix_ori, matrix_pre):
    """
    功能: 计算预测的Q矩阵的准确率
    计算方法: 判断有多少个元素相等

    Inputs:
    -------
    :param matrix_ori --> numpy.ndarray
        原始矩阵

    :param matrix_pre --> numpy.ndarray
        预测矩阵

    Outputs:
    -------
    :return cor_rate --> float
        准确率
    """

    shape = matrix_ori.shape

    cor_rate = 0  # 初始化准确率

    # 对拟合的数据四舍五入取整
    matrix_pre_round = matrix_pre.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix_pre_round[i][j] = round(matrix_pre_round[i][j])  # 四舍五入取整

    # 计算取整后的预测矩阵的准确率
    for i in range(shape[0]):
        for j in range(shape[1]):
            if matrix_pre_round[i][j] == matrix_ori[i][j]:
                cor_rate += 1
    cor_rate = cor_rate / (shape[0] * shape[1])

    return cor_rate


def cal_accuracy_obj(matrix_ori, matrix_pre, miss_coo, exe_desc):
    """
    功能:
        计算预测矩阵matrix_pre的客观题正确率
    计算方法:
        客观题判断有多少的元素相等

    Inputs:
    -------
    :param matrix_ori --> numpy.ndarray
        原始矩阵

    :param matrix_pre --> numpy.ndarray
        预测矩阵

    :param miss_coo --> list()
        缺失数据的位置

    :param exe_desc --> dict()
        习题信息的描述

    Outputs:
    -------
    :return cor_rate_obj --> float
        客观题准确率
    """
    shape = matrix_ori.shape

    # 对拟合的客观题数据做取整处理
    matrix_pre_round = matrix_pre.copy()
    # 注: matrix的row表示习题
    for i in range(shape[0]):  # 遍历习题
        if exe_desc[i] == 'Sub':
            continue  # 如果是主观题, 不做处理
        for j in range(shape[1]):
            matrix_pre_round[i][j] = round(matrix_pre_round[i][j])  # 四舍五入取整

    cor_rate_obj = 0  # 初始化客观题准确率
    miss_count_obj = 0  # 初始化客观题预测数量

    for _ in miss_coo:

        if exe_desc[_[0]] == 'Obj':
            # 计算客观题的准确率
            miss_count_obj += 1
            if matrix_pre_round[_[0]][_[1]] == matrix_ori[_[0]][_[1]]:
                cor_rate_obj += 1

    try:
        cor_rate_obj = cor_rate_obj / miss_count_obj
    except ZeroDivisionError:
        # 如果没有客观题被预测
        cor_rate_obj = None

    return cor_rate_obj


def cal_mae(matrix_ori, matrix_pre, miss_coo):
    """
    功能: 计算预测矩阵的MAE

    Inputs:
    -------
    :param matrix_ori --> numpy.ndarray
        原始矩阵

    :param matrix_pre --> numpy.ndarray
        预测矩阵

    :param miss_coo --> list()
        缺失数据的位置

    Outputs:
    -------
    :return mae --> float
    """
    mae = 0

    for _ in miss_coo:
        mae += abs(matrix_ori[_[0]][_[1]] - matrix_pre[_[0]][_[1]])
    mae = mae / len(miss_coo)

    return mae


def cal_rmse(matrix_ori, matrix_pre, miss_coo):
    """
    功能: 计算预测矩阵的RMSE

    Inputs:
    -------
    :param matrix_ori --> numpy.ndarray
        原始矩阵

    :param matrix_pre --> numpy.ndarray
        预测矩阵

    :param miss_coo --> list()
        缺失数据的位置

    Outputs:
    -------
    :return rmse --> float
    """
    rmse = 0

    for _ in miss_coo:
        rmse += math.pow(abs(matrix_ori[_[0]][_[1]] - matrix_pre[_[0]][_[1]]), 2)

    rmse = math.sqrt(rmse / len(miss_coo))

    return rmse


def cal_doa(stu_kn_pro, matrix_ori, miss_coo, q_matrix):
    """
    功能: 评价学生对知识点的诊断程度
    评价方法: 一致性程度(Degree of Agreement, DOA)

    Inputs:
    -------
    :param stu_kn_pro --> numpy.ndarray
        学生对知识点的诊断矩阵

    :param matrix_ori --> numpy.ndarray
        学生作答原始矩阵

    :param miss_coo --> list()
        缺失数据的位置

    :param q_matrix --> numpy.ndarray
        Q矩阵

    Outputs:
    -------
    :return kn_doa_list --> dict(kn:doa_value,...)
        每个知识点诊断结果的DOA
    """

    kn_doa_list = dict()  # 初始化doa

    # 初始化每个知识点下每个学生的ID以及相应的知识点诊断情况和真实答题情况
    kn_stu_pro_ans = dict()  # {kn: {exe:[学生ID, 知识点诊断结果, 答题结果]}}

    [exe_num, kn_num] = q_matrix.shape
    # 遍历每一个知识点k
    for kn in range(kn_num):
        kn_stu_pro_ans[kn] = dict()
        # 遍历Q矩阵中涉及了该知识点的习题
        for exe in range(exe_num):
            if q_matrix[exe][kn] == 1:
                kn_stu_pro_ans[kn][exe] = list()
                # 遍历待预测位置是否出现了该习题
                for miss_p in miss_coo:
                    if miss_p[0] == exe:
                        # 如果有则提取出相应的学生,
                        # 并提取该学生的知识点诊断程度和该生对习题的作答情况
                        stu = miss_p[1]  # 学生
                        kn_pro = stu_kn_pro[stu][kn]  # 知识点诊断结果
                        ans = matrix_ori[exe][stu]  # 习题作答结果
                        kn_stu_pro_ans[kn][exe].append([stu, kn_pro, ans])

    # 计算每个知识点的DOA
    # 遍历每个知识点
    for kn, value in kn_stu_pro_ans.items():
        kn_doa_list[kn] = 0
        delta_kn = 0
        delta_exe = 0

        # 遍历知识点涉及的每道习题
        for spa_list in value.values():
            # 依次对比该习题下所有学生的知识点诊断结果和习题答题结果
            for i in range(len(spa_list)):
                stu_a = spa_list[i][0]  # 学生A
                for j in range(len(spa_list)):

                    stu_b = spa_list[j][0]  # 学生B

                    # 不同的两个学生
                    if stu_a != stu_b:

                        a_pro = round(spa_list[i][1], 5)  # 学生A的知识点诊断结果, 取小数点后5位
                        b_pro = round(spa_list[j][1], 5)  # 学生B的知识点诊断结果, 取小数点后5位
                        a_ans = spa_list[i][2]  # 学生A的习题真实解答结果
                        b_ans = spa_list[j][2]  # 学生B的习题真实解答结果

                        # 如果学生A的知识点诊断结果>学生B
                        if a_pro > b_pro:
                            delta_kn += 1
                            # 且 学生A的作答结果>学生B
                            if a_ans > b_ans:
                                delta_exe += 1

        try:
            doa = delta_exe / delta_kn
            kn_doa_list[kn] = doa
        except ZeroDivisionError:
            kn_doa_list[kn] = np.NaN

    return kn_doa_list


def cal_diag_krc(exe_desc, matrix_ori, miss_coo, stu_kn_pro, q_matrix):
    """
    功能: 评价学生对知识点的诊断程度
    评价方法: 基于AUC的知识点-响应一致性系数 (Knowledge-Response Consistency Coefficient)

    Inputs:
    -------
    :param exe_desc --> dict()
        习题的信息描述(主观题/客观题)

    :param matrix_ori --> numpy.ndarray
        学生作答原始矩阵
        row: 习题
        col: 学生

    :param miss_coo --> list()
        缺失数据的位置
    
    :param stu_kn_pro --> numpy.ndarray
        学生对知识点的诊断矩阵
        row: 学生
        col: 知识点

    :param q_matrix --> numpy.ndarray
        Q矩阵

    Outputs:
    -------
    :return kn_krc_list --> dict(kn:doa_value,...)
        每个知识点诊断结果的KRC
    """

    # 初始化每个知识点的krc
    kn_krc_list_obj = dict()  # 客观题
    kn_krc_list_sub = dict()  # 主观题
    kn_krc_list = dict()  # 客观题+主观题

    exe_num, kn_num = q_matrix.shape

    """ --- 创建 kn_exe_stu --- """
    kn_exe_stu_pair = dict()  # 初始化每个知识点下在测试集中涉及的所有"习题-学生对"
    for kn in range(kn_num):
        kn_exe_stu_pair[kn] = list()
    for _ in miss_coo:
        exe = _[0]
        for kn in range(kn_num):
            if q_matrix[exe][kn] == 1:
                kn_exe_stu_pair[kn].append(_)

    """ --- 创建 kn_stu_pro_ans --- """
    # 初始化每个知识点下的每个学生的知识点掌握情况和习题答题情况
    kn_stu_pro_ans_obj = dict()  # 客观题
    kn_stu_pro_ans_sub = dict()  # 主观题
    for kn in range(kn_num):
        kn_stu_pro_ans_obj[kn] = list()
        kn_stu_pro_ans_sub[kn] = list()

    # 遍历测试数据中某个知识点下所有的"习题-学生对", 存储知识点掌握情况和答题情况
    for kn, pairs in kn_exe_stu_pair.items():
        for _ in pairs:
            exe = _[0]
            stu = _[1]
            pro = round(stu_kn_pro[stu][kn], 5)  # 获得学生的知识点诊断结果
            ans = matrix_ori[exe][stu]  # 获取学生的真实作答结果

            # 如果习题是客观题
            if exe_desc[exe] == 'Obj':
                kn_stu_pro_ans_obj[kn].append((pro, ans))
            # 如果是主观题
            elif exe_desc[exe] == 'Sub':
                kn_stu_pro_ans_sub[kn].append((pro, ans))

    """ --- 计算KRC --- """
    # 计算每个知识点的KRC数值(客观题)
    for kn in range(kn_num):
        if not len(kn_stu_pro_ans_obj[kn]):
            kn_krc_list_obj[kn] = np.NaN
            continue
        kn_krc_list_obj[kn] = __cal_binary_krc(kn_stu_pro_ans_obj[kn], label=[0.0, 1.0])

    # 计算每个知识点的KRC数值(主观题)
    for kn in range(kn_num):
        # 遍历该知识点的所有(知识点掌握程度, 主观题答题结果)
        pairs = kn_stu_pro_ans_sub[kn]

        if not len(pairs):
            kn_krc_list_sub[kn] = np.NaN
            continue

        stu_pro_ans = dict()  # 根据不同的主观题答题结果存储(知识点掌握程度, 主观题答题结果)
        for _ in pairs:
            try:
                stu_pro_ans[_[1]].append(_)
            except:
                stu_pro_ans[_[1]] = list()
                stu_pro_ans[_[1]].append(_)

        # 每两种答题结果计算一次Binary KRC数值
        krc = 0
        count = 0
        labels = list(stu_pro_ans.keys())
        k = len(labels)
        for i in range(0, k):
            label_1 = labels[i]
            for j in range(i+1, k):
                label_2 = labels[j]
                _pro_ans = stu_pro_ans[label_1] + stu_pro_ans[label_2]
                krc += __cal_binary_krc(_pro_ans, [label_1, label_2])
                count += 1

        try:
            kn_krc_list_sub[kn] = krc / count
        except ZeroDivisionError:
            kn_krc_list_sub[kn] = np.NaN

    # 将知识点的客观题KRC和主观题KRC合并
    for kn in range(kn_num):
        kn_krc_obj = kn_krc_list_obj[kn]  # 知识点kn的客观题KRC
        kn_krc_sub = kn_krc_list_sub[kn]  # 知识点kn的主观题KRC

        if kn_krc_obj is not np.NaN and kn_krc_sub is not np.NaN:
            kn_krc_list[kn] = (kn_krc_obj + kn_krc_sub) / 2
        elif kn_krc_obj is np.NaN and kn_krc_sub is not np.NaN:
            kn_krc_list[kn] = kn_krc_sub
        elif kn_krc_sub is np.NaN and kn_krc_obj is not np.NaN:
            kn_krc_list[kn] = kn_krc_obj

    return kn_krc_list


def __cal_binary_krc(stu_pro_ans, label):
    """
    功能: 基于二分类AUC原理计算每个知识点诊断结果的Binary KRC值

    Inputs:
    -------
    :param stu_pro_ans --> list[(pro1, ans1),(pro2, ans2),...]
        某个知识点下的所有学生的知识点掌握程度以及相应的答题情况

    :param label --> list[类标签1, 类标签2]

    Outputs:
    -------
    :return krc value --> float
    """

    rank = 0  # 初始化排序值
    num_pos = 0  # 初始化回答正确的个数
    num_neg = 0  # 初始化回答错误的个数

    # 升序排序
    stu_pro_ans.sort(key=lambda x: x[0])
    # stu_pro_ans.sort(key=lambda x: (x[0], x[1]))

    # 计算KRC
    for i in range(len(stu_pro_ans)):
        ans = stu_pro_ans[i][1]
        if ans == max(label):
            rank = rank + i + 1
            num_pos += 1
        elif ans == min(label):
            num_neg += 1
    if num_pos == 0:
        krc = 0
    elif num_neg == 0:
        krc = 1
    else:
        krc = (rank - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)

    return krc


def sort(arr):
    """
    功能: 对列表对象内的元素(元祖)进行排序
    方法: 冒泡排序

    Inputs:
    -------
    :param arr --> list[(x1,y1),(x2,y2),...,(xn,yn)]

    Outputs:
    -------
    :return arr --> list[(x1,y1),(x2,y2),...,(xn,yn)]
    """

    for i in range(0, len(arr)):
        for j in range(0, len(arr)-1):
            if arr[j][0] >= arr[j+1][0]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr


def cal_sim(matrix):
    """
    功能: 计算矩阵的行和列的相似度

    Inputs:
    -------
    :param matrix --> numpy.ndarray
        待计算的矩阵

    Outputs:
    -------
    :return sim_row --> numpy.ndarray
        matrix矩阵的行相似性矩阵

    :return sim_col --> numpy.ndarray
        matrix矩阵的列相似性矩阵
    """

    [row, col] = matrix.shape
    # 初始化相似度矩阵
    sim_row = np.zeros(shape=(row, row))
    sim_col = np.zeros(shape=(col, col))

    sim_row_ave = 0
    sim_col_ave = 0

    # 计算行与行之间的相似度
    count = 0
    for i in range(row):
        for j in range(i, row):
            sim_row[i, j] = __sim_cos(matrix[i], matrix[j])
            sim_row_ave += sim_row[i, j]
            count += 1

    print("SIM ROW AVE:", sim_row_ave/count)

    # 计算列与列之间的相似度
    count = 0
    for i in range(col):
        for j in range(i, col):
            sim_col[i, j] = __sim_cos(matrix[:, i], matrix[:, j])
            sim_col_ave += sim_col[i, j]
            count += 1
    print("SIM COL AVE:", sim_col_ave/count)

    return sim_row, sim_col


def __sim_cos(vec_1, vec_2):
    """
    功能: 计算两个向量之间的余弦相似度

    Inputs:
    -------
    :param vec_1 --> numpy.ndarray

    :param vec_2 --> numpy.ndarray

    Outputs:
    -------
    :return sim_cos --> float
    """

    assert len(vec_1) == len(vec_2), \
        "The lengths of two vectors are not equal"

    num = __bit_product_sum(vec_1, vec_2)
    den = np.sqrt(__bit_product_sum(vec_1, vec_1)) * np.sqrt(__bit_product_sum(vec_2, vec_2))
    sim_cos = 0

    if den == 0:
        np.seterr(invalid='ignore')
    else:
        sim_cos = num / den

    return sim_cos


def __bit_product_sum(x, y):
    """
    功能: 计算两个向量的内积

    Inputs:
    -------
    :param x,y
        待计算内积的两个向量

    Outputs:
    -------
    return 内积的计算结果
    """
    return sum([item[0] * item[1] for item in zip(x, y)])


def cal_diag_rank_correlation(matrix_ori, miss_coo, stu_kn_pro, q_matrix):
    """
    功能: 评价学生对知识点的诊断程度
    评价方法: 斯皮尔曼等级相关系数 (Spearman rank correlation)

    Inputs:
    -------
    :param matrix_ori --> numpy.ndarray
        学生作答原始矩阵
        row: 习题
        col: 学生

    :param miss_coo --> list()
        缺失数据的位置
    
    :param stu_kn_pro --> numpy.ndarray
        学生对知识点的诊断矩阵
        row: 学生
        col: 知识点

    :param q_matrix --> numpy.ndarray
        Q矩阵


    Outputs:
    -------
    :return rho
        相关系数
    """

    rho = 0  # 初始化斯皮尔曼等级相关系数

    kn_num = q_matrix.shape[1]

    kn_exe_stu = dict()

    for kn in range(kn_num):
        kn_exe_stu[kn] = dict()

        for _ in miss_coo:
            exe, stu = _[0], _[1]  # 获取习题ID和学生ID
            if q_matrix[exe][kn] == 1:
                try:
                    kn_exe_stu[kn][exe].append(stu)
                except:
                    kn_exe_stu[kn][exe] = list()
                    kn_exe_stu[kn][exe].append(stu)

    _count = 0
    for kn, exe_stus in kn_exe_stu.items():
        for exe, stus in exe_stus.items():
            kn_pro_list, score_list = [], []
            for stu in stus:
                kn_pro_list.append(stu_kn_pro[stu][kn])  # 传入该学生的知识点掌握程度
                score_list.append(matrix_ori[exe][stu])  # 传入该学生的真实作答结果
            _ = {
                'pro': kn_pro_list,
                'ans': score_list
            }
            data = DataFrame(_)  # 转换数据类型
            a = data['pro'].rank()  # 获取对知识点掌握情况的排名
            b = data['ans'].rank()  # 获取对答题结果的排名

            # 计算斯皮尔曼等级相关系数
            if len(a) <= 1 or len(b) <= 1:
                continue
            rho += 1 - ((sum((a-b)*(a-b))*6) / (len(a)*(len(a)*len(a)-1)))
            _count += 1

    return  rho / _count


def cal_diag_hca(stu_kn_pro, know_graph, conf_level):
    """
    功能: 量化学生对知识点的掌握情况和Hierarchical Cognitive Assumption (HCA) 的匹配程度
    方法: Wilcoxon-signed-rank-test Passing Ratio (PR)

    inputs:
    -------
    :param conf_level --> float
        the confidence level for P-value
    """

    stu_wil = {}  # the Wilcoxon-signed-rank-test result for each student

    for stu in range(stu_kn_pro.shape[0]):
        m_pa, m_ch = [], []  # the mastery degree of parent knowledge with the corresponding child knowledge
        for edge in know_graph.values.tolist():
            m_pa.append(stu_kn_pro[stu][edge[0]])  # the mastery degree of parent node
            m_ch.append(stu_kn_pro[stu][edge[1]])  # the mastery degree of child node
        if not sum(np.array(m_pa) - np.array(m_ch)) == 0.0:
            I = stats.wilcoxon(m_pa, m_ch, correction = True, alternative='greater')
            if I.pvalue <= conf_level:
                stu_wil[stu] = 1
            else:
                stu_wil[stu] = 0
    
    if len(stu_wil.values()) > 0:
        pr = sum(list(stu_wil.values())) /len(list(stu_wil.values()))
    else:
        pr = None
    return pr


def cal_simpson_div(stu_kn_pro):
    """
    FUNCTION: calculate the students' knowleddge concept richness
    based on Simpson's Diversity Index (sdi)
    """
    sdi = 0  # initialization

    stu_num = stu_kn_pro.shape[0]
    for stu in range(stu_num):
        # calculate the sdi for each student
        types, counts = np.unique(stu_kn_pro[stu],return_counts=True)
        type_prob = counts / sum(counts)
        sdi += sum(type_prob**2)

    return sdi / stu_num
