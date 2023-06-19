"""
Sensitivity Analysis of the Parameters
"""

# NOTE: *** Grid search for parameter sensitivities analysis ***

# --- Step.1 Search the best rank ---
     
# --- Step.2 Grid search for hyper-parameter alpha ---
# Set LU, LE, LV = 0, 0, 0, and then
# ALPHA_SET = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100] for grid search alpha

# --- Step.3 Grid search for lu,lv and le, based on the optimal alpha value ---
# Choosing LE_SET or LU_SET or LV_SET in [0.001, 0.01, 0.1, 1, 5, 10]  
# for example, if you want to deterimine the best value of LU,
# please first give the default value of LE and LV,
# as well as the best ALPHA mentioned in Step.1
# then, grid search U in [0.001, 0.01, 0.1, 1, 5, 10] to test the model performance


import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import warnings
import numpy as np
import pandas as pd
from SidePackage import evaluation as ev
from SidePackage import preprocessing as pre
from SidePackage import auxiliary as aux
from SNMCF import main



def out_to_file(path, model_name):

    class logger(object):
        
        def __init__(self, file_name, path):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, file_name), mode='a', encoding='utf8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            pass
    
    sys.stdout = logger(model_name + '.log', path=path)


DATASET = 'FrcSub'
print("dataset %s is choosed" % DATASET)
MISS_R = 0.2  # the testing ratio
is_GBE = False
CL = 0.05

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

out_to_file(BASE_DIR + "/SNMCF/log/", 'para_analysis@' + DATASET)




# --- Step 1: Search the best RANK ---

# ALPHA = 0.5
# LU, LV, LE = 0, 0, 0
# max_rank = min(min(stu_exe.shape), min(q_m.shape))
# for r in range(1, max_rank):
#     stu_pro, stu_exe_pre, q_pre = main.snmcf_train(train_data, train_fill, q_m, r, ALPHA, LU, LV, LE, gbe=is_GBE)
#     main.snmcf_test(q_m, q_pre, stu_exe, stu_exe_pre, stu_pro, know_graph, test_loc, prob_desc, cl=CL)

# --- Step.2 Grid search for hyper-parameter alpha ---
# Set LU, LE, LV = 0, 0, 0, and then
# ALPHA_SET = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100] for grid search alpha

# best_rank = 2
# alpha_set = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
# LU, LV, LE = 0, 0, 0
# for alpha in alpha_set:
#     stu_pro, stu_exe_pre, q_pre = main.snmcf_train(train_data, train_fill, q_m, best_rank, alpha, LU, LV, LE, gbe=is_GBE)
#     main.snmcf_test(q_m, q_pre, stu_exe, stu_exe_pre, stu_pro, know_graph, test_loc, prob_desc, cl=CL)

# --- Step.3 Grid search for lu,lv and le, based on the optimal alpha value ---
# best_rank = 2
# best_alpha = 0.5
# best_lu = 10
# best_lv = 1
# for le in [0.001, 0.01, 0.1, 1, 5, 10]:
#     stu_pro, stu_exe_pre, q_pre = main.snmcf_train(train_data, train_fill, q_m, best_rank, best_alpha, best_lu, best_lv, le, gbe=is_GBE)
#     main.snmcf_test(q_m, q_pre, stu_exe, stu_exe_pre, stu_pro, know_graph, test_loc, prob_desc, cl=CL)