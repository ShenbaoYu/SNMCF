# -*- coding: utf-8 -*-
"""
SNMCF model training and testing
"""

import os
import sys
import warnings
import numpy as np
import snmcf
import time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from SidePackage import evaluation as ev
from SidePackage import preprocessing as pre
from SidePackage import auxiliary as aux




def snmcf_train(train_data, train_fill, q_m, rank, alpha, lu, lv, le, gbe=False):
    """
    FUNCTION: model training

    Inputs:
    -------
    :param train_data --> numpy.ndarray
        the student-exercise data matrix which is used for training (randomly delete some entries).
        row: exercises
        col: students
    
    :param train_fill --> numpy.ndarray
        the student-exercise data matrix whose missing entries are filled.

    :param q_m --> numpy.ndarray
        the Q-matrix
        row: exercises
        col: knowledge concepts
    
    :param alpha --> float

    :param lu--> float

    :param lv --> float

    :param le --> float

    Outputs:
    -------
    :return stu_pro --> numpy.ndarray
        the predicted mastery for each knowledge concept for all students

    :return stu_exe_pre --> numpy.ndarray
        the predicted matrix of the student-exercise matrix

    :return q_pre --> numpy.ndarray
        the predicted matrix of the Q-matrix

    """

    print("\nSNMCF hyperparameters: \nrank: %d\nAlpha: %.2f\nLambda U: %.2f\nLambda V: %.2f\nLambda E: %.2f"
          % (rank, alpha, lu, lv, le))

    if gbe is True:
        w = snmcf.cal_weight_matrix(train_fill)
    else:
        w = snmcf.cal_weight_matrix(train_data)
    
    # --- model training ---
    start = time.time()
    # training based on the Multiplicative Update Rules
    u, v, e = snmcf.fit_data_mult(train_fill, q_m, w, rank, alpha, lu, lv, le)
    # training based on the Projected Gradient Method
    # u, v, e = snmcf.fit_data_prograd(train_fill, q_m, w, rank, alpha,lu, lv, le)

    # calculate the mastery of knowledge concepts for all students
    stu_pro = snmcf.stu_kn_diagnose_area(u, v)
    # the predicted student-exercise matrix
    stu_exe_pre = snmcf.cal_matrix_pre(left_latent_matrix=e, right_latent_matrix=u)
    # the predicted Q-matrix
    q_pre = snmcf.cal_matrix_pre(left_latent_matrix=e, right_latent_matrix=v)
    end = time.time()
    print('TIME:%.5f' %(end - start))
    
    return stu_pro, stu_exe_pre, q_pre


def snmcf_test(q_m, q_pre, stu_exe, stu_exe_pre, stu_pro, know_graph, test_loc, prob_desc, cl):
    """
    Function: model testing
    """
    
    # the accuracy of Q-matrix
    accuracy_q = ev.cal_accuracy_q(q_m, q_pre)
    # the accuracy of the predicted student performance on objective exercises
    accuracy_obj = ev.cal_accuracy_obj(stu_exe, stu_exe_pre, test_loc, prob_desc)
    # the mean absolute error
    mae = ev.cal_mae(stu_exe, stu_exe_pre, test_loc)
    # the root mean square error
    rmse = ev.cal_rmse(stu_exe, stu_exe_pre, test_loc)
    # the KRC for diagnosis results
    kn_krc_list = ev.cal_diag_krc(prob_desc, stu_exe, test_loc, stu_pro, q_m)
    krc = np.mean([x for x in kn_krc_list.values()])
    # Spearman rank correlation coefficient 
    rho = ev.cal_diag_rank_correlation(stu_exe, test_loc, stu_pro, q_m)
    # Simpson diversity index
    sdi = ev.cal_simpson_div(stu_pro)
    print("ACCURACY (Obj): %.5f, ACCURACY(Q): %.5f MAE: %.5f, RMSE: %.5f, KRC: %.5f, SpearmanRHO: %.5f, SimpsonIndex: %.5f" 
    % (accuracy_obj, accuracy_q, mae, rmse, krc, rho, sdi))
    
    # the Wilcoxon-signed-rank-test Passing Ratio (PR)
    pr = None
    if not know_graph is None: pr = ev.cal_diag_hca(stu_pro, know_graph, conf_level=cl)
    if not pr is None:
        print("The Passing Ratio (PR) is %.5f, confidience level is %.2f" % (pr, cl))
    else:
        print("TEST FAILED using Wilcoxon-signed-rank-test Passing Ratio")




if __name__ == '__main__':

    MISS_R = 0.2  # the testing ratio

    # --- HYPER PARAMETERS ---
    # for the SNMCF framework
    RANK = 1
    ALPHA = 5
    LU = 0
    LE = 0
    LV = 0

    is_GBE = False  # fill missing value using the Global Baseline Estimation method (default:False)
    CL = 0.05  # the confidience level

    DATASET = input("\nplease choose a dataset: [FrcSub, Math1, Math2, Quanlang, Quanlang-s, A0910, Junyi, Junyi-s]: ")
    print("dataset %s is choosed" % DATASET)
    if DATASET not in ['FrcSub', 'Math1', 'Math2', 'Quanlang', 'Quanlang-s', 'A0910', 'Junyi', 'Junyi-s']:
        warnings.warn("dataset does not exist.")
        exit()
    
    # 1. student-exercise matrix (row: exercises, col: students)
    stu_exe = ((np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/data.txt")).astype(float)).T
    # 2. Q-matrix
    q_m = np.loadtxt(BASE_DIR + "/Data/" + DATASET +"/q.txt", dtype=int)
    # 3. problem description
    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(stu_exe)   
    # 4. the knowledge dependency map
    know_graph = None
    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/hier.csv"):
        know_graph = pd.read_csv(BASE_DIR + "/Data/" + DATASET +'/hier.csv')  # get the prerequisite graph

    # divdide the student-exercise matrix into traning data and testing data
    is_divide = input("re-divide the dataset? (yes or no): ")
    if is_divide == "yes":
        # get the training set and testing index of student-exercise matrix
        train_data, test_loc = pre.missing_stu_exe(stu_exe, MISS_R)
        np.savetxt(BASE_DIR + "/SNMCF/data/train@" + DATASET + ".txt", train_data, fmt='%.4f')
        np.savetxt(BASE_DIR + "/SNMCF/data/test@" + DATASET + ".txt", np.array(test_loc), delimiter=' ', fmt='%s')
        print("the data division has been completed.")
    elif is_divide == "no":
        pass
    else:
        warnings.warn("illegal input!")
        exit()      
    # get the training set
    train_data = ((np.loadtxt(BASE_DIR + "/SNMCF/data/train@" + DATASET + ".txt")).astype(float))
    # get the testing index
    test_loc = aux.read_test_loc(BASE_DIR + "/SNMCF/data/test@" + DATASET + ".txt")
    
    # filling the missing value
    if DATASET in ['FrcSub', 'Math1', 'Math2']:
        train_fill = pre.matrix_miss_fill_GBE(train_data)  # using the Global Baseline Estimation (GBE)
        is_GBE = True
    elif DATASET in ['Quanlang', 'A0910', 'Quanlang-s', 'Junyi', 'Junyi-s']:
        train_fill = pre.matrix_miss_fill(train_data)
    # -- training and testing ---
    stu_pro, stu_exe_pre, q_pre = snmcf_train(train_data, train_fill, q_m, RANK, ALPHA, LU, LV, LE, gbe=is_GBE)
    snmcf_test(q_m, q_pre, stu_exe, stu_exe_pre, stu_pro, know_graph, test_loc, prob_desc, CL)