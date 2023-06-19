# -*- coding: utf-8 -*-
"""
其它辅助工具
"""

import numpy as np
from numpy import linalg as la

def read_problem_desc(filename):
    """
    功能: 读取习题描述: 1. 主观题; 2. 客观题; 3. 习题的满分值

    Inputs:
    -------
    :param filename --> .txt
        格式:
        No.	Type	Full Score
        1	Obj	    3
        2	Obj	    4
        3	Obj	    5
        ......
    
    Ouputs:
    -------
    :return exe_desc --> dict(习题ID:主观题/客观题,...)
    """

    exe_desc = dict()
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            desc = [i for i in lines.split()]
            if len(desc) > 3:
                continue
            exe_desc[int(desc[0])-1] = desc[1]

    return exe_desc


def build_problem_desc(stu_exe):
    """
    功能：从学生-习题作答矩阵中构建习题描述
    
    Inputs:
    -------
    :param stu_exe --> numpy.ndarray
        row: 习题
        col: 学生

    Output:
    :return exe_description --> dict(exe ID:Obj/Sub,...)
        Obj: 客观题
        Sub: 主观题
    """

    exe_num = stu_exe.shape[0]

    # 先初始化全部为客观题 --> Obj
    problem_description = dict()  # 初始化
    for exe_id in range(exe_num):
        problem_description[exe_id] = 'Obj'

    # 找出stu_exe的元素在(0,1)之间的位置，如果存在，视作主观题
    sub_coo = np.argwhere((stu_exe > 0) & (stu_exe < 1)).tolist()
    for coo in sub_coo:
        problem_description[coo[0]] = 'Sub'

    return problem_description


def read_test_loc(filename):
    """
    功能：读取测试集合的坐标

    Inputs:
    -------
    :param filename: --> .txt
        测试集合坐标数据文件

    Outputs:
    -------
    :return coo_test --> numpy.ndarray
        测试集合
    """

    _ = list()
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            _.append([int(i) for i in lines.split()])

    test_loc = list()
    for coo in _:
        test_loc.append(tuple(coo))

    return test_loc


def get_rank(x_matrix):
    """
    功能: 获得矩阵 x_matrix 的秩
    方法: SVD

    Inputs:
    -------
    :param x_matrix --> numpy.ndarray
        待求矩阵
    """

    left_u, sigma, right_vt = la.svd(x_matrix)
    sin_deg = 0
    print("\n学生作答矩阵SVD后的奇异值向量为:\n", sigma)
    for _ in sigma:
        sin_deg += _
    print("第一个奇异值占所有奇异值的比重:", sigma[0] / sin_deg)