# -*- coding: utf-8 -*-
"""
the main module of the SNMCF framework

Objective Function:
min ||W*(X - EU)|| + alpha||Q-EV|| + lambdaE||E|| + lambdaU||U|| + lambdaV||V||
s.t E > 0, U > 0, V > 0

X: The student scoring matrix;
    row: exercises, column: students
Q: The Q-matrix;
    row:exercises, column: knowledge concepts
E: The exercise characteristic matrix
U: The student proficiency matrix
V: The knowledge concept requirement matrix
*: Hadamard product

NOTE: we solve this objective function using two methods:
    (1) Updated Rules --> fit_data_mult()
    (2) Projected Gradient Methods --> fit_data_prograd()
"""


import numpy as np
import math
import time

MAX_ITER = 500  # The maximum number of iterations
CRI = 0.001  # or 0.001. The difference between the value of the objective function
THETA = 0.01  # theta value of the line search in the projected gradient-based method
BETA = 0.1  # the step-size of the line search in the projected gradient-based method


def cal_weight_matrix(student_exe):
    """
    FUNCTION: get the weight matrix of the student scoring matrix: W,
    where the element value equals 0 if the student has no response in the exercise, otherwise, 1.

    Inputs:
    -------
    :param student_exe --> numpy.ndarray
        the student scoring matrix

    Outputs:
    -------
    :return weight_matrix --> numpy.ndarray
        the weight matrix
    """

    weight_matrix = np.ones(shape=student_exe.shape)  # initialization
    weight_matrix[np.isnan(student_exe)] = 0

    return weight_matrix.astype(np.int)


def fit_data_mult(student_exe, q_matrix, weight_matrix, n_factors, alpha, lambda_u, lambda_v, lambda_e,
                  max_iter=MAX_ITER, criteria=CRI):
    """
    FUNCTION: update solutions based on "Multiplicative Update Rules"

    Inputs:
    -------
    :param student_exe --> numpy.ndarray
        the student scoring matrix

    :param q_matrix --> numpy.ndarray
        the Q matrix

    :param weight_matrix --> numpy.ndarray
        the weight matrix

    :param n_factors --> int
        the rank

    :param alpha --> float

    :param lambda_u --> float

    :param lambda_v --> float

    :param lambda_e --> float

    :param max_iter --> int

    :param criteria --> float

    Outputs:
    -------
    :return u --> numpy.ndarray
        the student proficiency matrix
    
    :return v --> numpy.ndarray
        the knowledge concept requirement matrix
    
    :return e --> numpy.ndarray
        the exercise characteristic matrix
    """

    print("Fitting data...")

    exe_num = len(student_exe)  # the number of exercises: N
    student_num = len(student_exe[0])  # the number of students: M
    knowledge_num = len(q_matrix[0])  # the number of knowledge concepts: K

    # initialize E, U, and V
    e = np.random.uniform(0, 1, size=(exe_num, n_factors))
    u = np.random.uniform(0, 1, size=(n_factors, student_num))
    v = np.random.uniform(0, 1, size=(n_factors, knowledge_num))

    # calculate the objective function value
    objective_value = math.pow(np.linalg.norm(weight_matrix * (student_exe-np.dot(e, u)), ord='fro'), 2) \
                      + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(e, v), ord='fro'), 2) \
                      + lambda_e * math.pow(np.linalg.norm(e, ord='fro'), 2) \
                      + lambda_u * math.pow(np.linalg.norm(u, ord='fro'), 2) \
                      + lambda_v * math.pow(np.linalg.norm(v, ord='fro'), 2)

    # ---- iterate over the Multiplicative Update Rule ----

    start_time = time.time()

    convergence = False
    i = 0
    # print("Iteration %d = %s" % (i, objective_value))

    while (not convergence) and (i < max_iter):

        """ calculate the next parameter values """
        _u = update_u(student_exe, weight_matrix, e, u, lambda_u)
        _v = update_v(q_matrix, e, v, alpha, lambda_v)
        _e = update_e(student_exe, q_matrix, weight_matrix, _u, _v, e, alpha, lambda_e)

        """ calculate the next objective function value """
        _objective_value = math.pow(np.linalg.norm(weight_matrix * (student_exe - np.dot(_e, _u)), ord='fro'), 2) \
                           + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(_e, _v), ord='fro'), 2) \
                           + lambda_e * math.pow(np.linalg.norm(_e, ord='fro'), 2) \
                           + lambda_u * math.pow(np.linalg.norm(_u, ord='fro'), 2) \
                           + lambda_v * math.pow(np.linalg.norm(_v, ord='fro'), 2)

        """ update U, E, and V """
        u = _u
        v = _v
        e = _e

        """ is convergent? """
        convergence = abs(objective_value - _objective_value) < criteria
        objective_value = _objective_value  # update the objective function value

        i += 1

        # print("Iteration %d = %s" % (i, _objective_value))
        
        if i == max_iter:
            print('Maximum iterations reached.')
    
    end_time = time.time()

    gap = end_time - start_time
    # print("The PARAMETERS HAVE BEEN UPDATED. TIME = ", gap)

    return u, v, e


def update_u(student_exe, weight_matrix, e, u, lambda_u):
    """
    FUNCTION: update the student proficiency matrix

    Inputs:
    -------
    :param student_exe --> numpy.ndarray
        the student scoring matrix

    :param weight_matrix --> numpy.ndarray
        the weight matrix

    :param e --> numpy.ndarray
        the exercise characteristic matrix of the previous round E

    :param u --> numpy.ndarray
        the student proficiency matrix of the previous round U

    :param lambda_u --> float

    Outputs:
    -------
    :return _u --> numpy.ndarray
        the updated student proficiency matrix
    """

    # NOTE: when the data is sparse, it is should be add the regularization term to avoid the NaN value.

    _u = u * (np.dot(e.T, weight_matrix*student_exe)
              / (np.dot(e.T, weight_matrix*np.dot(e, u)) + lambda_u*u))

    
    # When a student has no answer records on all exercises,
    # we give a small bias to avoid the NaN value.
    bias = 0.00001
    _u = _u + bias

    return _u


def update_v(q_matrix, e, v, alpha, lambda_v):
    """
    FUNCTION: update the knowledge concept requirement matrix

    Inputs:
    -------
    :param q_matrix --> numpy.ndarray
        the Q matrix

    :param e --> numpy.ndarray
        the exercise characteristic matrix of the previous round E

    :param v --> numpy.ndarray
        the knowledge concept requirement matrix of the previous round V 

    :param alpha --> float

    :param lambda_v --> float

    Outputs:
    -------
    :return _v --> numpy.ndarray
        the updated knowledge concept requirement matrix
    """
    _v = v * ((alpha*np.dot(e.T, q_matrix))
              / (alpha*np.dot(np.dot(e.T, e), v) + lambda_v*v))
    
    # When a student has no answer records on all exercises,
    # we give a small bias to avoid the NaN value.
    bias = 0.00001
    _v = _v + bias

    return _v


def update_e(student_exe, q_matrix, weight_matrix, u, v, e, alpha, lambda_e):
    """
    FUNCTION: update the exercise characteristic matrix E

    Inputs:
    -------
    :param student_exe --> numpy.ndarray
        the student scoring matrix

    :param q_matrix --> numpy.ndarray
        the Q matrix

    :param weight_matrix --> numpy.ndarray
        the weight matrix

    :param u --> numpy.ndarray
        the student proficiency matrix of the previous round U

    :param v --> numpy.ndarray
        the knowledge concept requirement matrix of the previous round V 

    :param e --> numpy.ndarray
        the exercise characteristic matrix of the previous round E

    :param alpha --> float

    :param lambda_e --> float

    Outputs:
    -------
    :return _e --> numpy.ndarray
        the updated exercise characteristic matrix
    """

    _e = e * ((np.dot(weight_matrix*student_exe, u.T) + alpha*np.dot(q_matrix, v.T))
              / (np.dot(weight_matrix*np.dot(e, u), u.T) + alpha*np.dot(np.dot(e, v), v.T) + lambda_e*e))

    # When a student has no answer records on all exercises,
    # we give a small bias to avoid the NaN value.
    bias = 0.00001
    _e = _e + bias

    return _e


def fit_data_prograd(student_exe, q_matrix, weight_matrix,
                     n_factors, alpha, lambda_u, lambda_v, lambda_e,
                     max_iter=MAX_ITER, criteria=CRI):
    """
    FUNCTION: update solutions based on "Projected Gradient Method"

    Inputs:
    -------
    :param student_exe --> numpy.ndarray
        the student scoring matrix

    :param q_matrix --> numpy.ndarray
        the Q matrix

    :param weight_matrix --> numpy.ndarray
        the weight matrix

    :param n_factors --> int
        the rank

    :param alpha --> float

    :param lambda_u --> float

    :param lambda_v --> float

    :param lambda_e --> float

    :param max_iter --> int

    :param criteria --> float

    Outputs:
    -------
    :return u --> numpy.ndarray
        the student proficiency matrix

    :return v --> numpy.ndarray
        the knowledge concept requirement matrix

    :return e --> numpy.ndarray
        the exercise characteristic matrix
    """

    print("Fitting data...")

    exe_num = len(student_exe)
    student_num = len(student_exe[0])
    knowledge_num = len(q_matrix[0])

    # initialize E, U, and V
    e = np.random.uniform(0, 1, size=(exe_num, n_factors))
    u = np.random.uniform(0, 1, size=(n_factors, student_num))
    v = np.random.uniform(0, 1, size=(n_factors, knowledge_num))

    # calculate the objective function value
    objective_value = math.pow(np.linalg.norm(weight_matrix * (student_exe-np.dot(e, u)), ord='fro'), 2) \
                      + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(e, v), ord='fro'), 2) \
                      + lambda_e * math.pow(np.linalg.norm(e, ord='fro'), 2) \
                      + lambda_u * math.pow(np.linalg.norm(u, ord='fro'), 2) \
                      + lambda_v * math.pow(np.linalg.norm(v, ord='fro'), 2)
    
    # ---- iterate over the Projected Gradient Method ----

    start_time = time.time()

    convergence = False
    i = 0
    # print("Iteration %d = %s" % (i, objective_value))

    while (not convergence) and (i < max_iter):

        # calculate the gradient of U
        grad_u = -2 * np.dot(e.T, weight_matrix*student_exe) \
                 +2 * np.dot(e.T, weight_matrix*np.dot(e, u)) \
                 +2 * lambda_u*u
        # line search the next value of U
        _u = line_search_u(weight_matrix, student_exe, q_matrix, 
                           alpha, lambda_u, lambda_v, lambda_e,
                           grad_u, u, v, e)
        # calculate the gradient of V
        grad_v = -2 * alpha * (np.dot(e.T, q_matrix) - np.dot(np.dot(e.T, e), v)) \
                 +2 * lambda_v*v
        # line search the next value of V
        _v = line_search_v(weight_matrix, student_exe, q_matrix, 
                           alpha, lambda_u, lambda_v, lambda_e,
                           grad_v, _u, v, e)
        # calculate the gradient of E
        grad_e = -2 * np.dot(weight_matrix*student_exe, _u.T) \
                 +2 * np.dot(weight_matrix*np.dot(e, _u), _u.T) \
                 +alpha * (-2*np.dot(q_matrix, _v.T) + 2*np.dot(np.dot(e, _v), _v.T)) \
                 +2 * lambda_e*e
        # line search the next value of E
        _e = line_search_e(weight_matrix, student_exe, q_matrix,
                           alpha, lambda_u, lambda_v, lambda_e, 
                           grad_e, _u, _v, e)

        """ calculate the objective function value """
        _objective_value = math.pow(np.linalg.norm(weight_matrix * (student_exe - np.dot(_e, _u)), ord='fro'), 2) \
                           + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(_e, _v), ord='fro'), 2) \
                           + lambda_e * math.pow(np.linalg.norm(_e, ord='fro'), 2) \
                           + lambda_u * math.pow(np.linalg.norm(_u, ord='fro'), 2) \
                           + lambda_v * math.pow(np.linalg.norm(_v, ord='fro'), 2)

        """ update the parameters """
        u = _u
        v = _v
        e = _e

        """ is convergent? """
        convergence = abs(objective_value - _objective_value) < criteria
        objective_value = _objective_value

        i += 1

        # print("Iteration %d = %s" % (i, _objective_value))
        
        if i == max_iter:
            print('Maximum iterations reached.')

    end_time = time.time()

    gap = end_time - start_time
    print("Time = ", gap)

    return u, v, e


def line_search_u(weight_matrix, student_exe, q_matrix, 
                  alpha, lambda_u, lambda_v, lambda_e,
                  grad_u, u_0, v_0, e_0, theta = THETA):
    """
    FUNCTION: calculate the U value in the nexit iteration
    RULE: Armijo rule along the projection arc
    """

    _step = 1.0
    # calculate the objecive function value based on u_0
    obj_u_0 = math.pow(np.linalg.norm(weight_matrix * (student_exe-np.dot(e_0, u_0)), ord='fro'), 2) \
              + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(e_0, v_0), ord='fro'), 2) \
              + lambda_e * math.pow(np.linalg.norm(e_0, ord='fro'), 2) \
              + lambda_u * math.pow(np.linalg.norm(u_0, ord='fro'), 2) \
              + lambda_v * math.pow(np.linalg.norm(v_0, ord='fro'), 2)

    is_continue = True
    u_1 = None
    _c = 1

    while is_continue:
        # calculate the new U value: u_1
        u_1 = np.maximum(u_0 - _step * grad_u, 0)
        
        # calculate the objecive function value based on u_1
        obj_u_1 = math.pow(np.linalg.norm(weight_matrix * (student_exe-np.dot(e_0, u_1)), ord='fro'), 2) \
                  + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(e_0, v_0), ord='fro'), 2) \
                  + lambda_e * math.pow(np.linalg.norm(e_0, ord='fro'), 2) \
                  + lambda_u * math.pow(np.linalg.norm(u_1, ord='fro'), 2) \
                  + lambda_v * math.pow(np.linalg.norm(v_0, ord='fro'), 2)
        
        # is satisfied Armijo rule?
        if obj_u_1 - obj_u_0 <= theta *  sum(sum(grad_u * (u_1-u_0))):
            is_continue = False
        else:
            _step = math.pow(BETA, _c)  # shrink the step size
            # _step = _step / 2
            _c += 1
    
    return u_1


def line_search_v(weight_matrix, student_exe, q_matrix, 
                  alpha, lambda_u, lambda_v, lambda_e,
                  grad_v, u_0, v_0, e_0, theta = THETA):

    """
    FUNCTION: calculate the V value in the nexit iteration
    RULE: Armijo rule along the projection arc
    """

    _step = 1.0
    # calculate the objecive function value based on v_0
    obj_v_0 = math.pow(np.linalg.norm(weight_matrix * (student_exe-np.dot(e_0, u_0)), ord='fro'), 2) \
              + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(e_0, v_0), ord='fro'), 2) \
              + lambda_e * math.pow(np.linalg.norm(e_0, ord='fro'), 2) \
              + lambda_u * math.pow(np.linalg.norm(u_0, ord='fro'), 2) \
              + lambda_v * math.pow(np.linalg.norm(v_0, ord='fro'), 2)
    
    is_continue = True
    v_1 = None
    _c = 1

    while is_continue:
         # calculate the new V value: v_1
        v_1 = np.maximum(v_0 - _step * grad_v, 0)
        
        # calculate the objecive function value based on v_1
        obj_v_1 = math.pow(np.linalg.norm(weight_matrix * (student_exe-np.dot(e_0, u_0)), ord='fro'), 2) \
                  + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(e_0, v_1), ord='fro'), 2) \
                  + lambda_e * math.pow(np.linalg.norm(e_0, ord='fro'), 2) \
                  + lambda_u * math.pow(np.linalg.norm(u_0, ord='fro'), 2) \
                  + lambda_v * math.pow(np.linalg.norm(v_1, ord='fro'), 2)
        
        # is satisfied Armijo rule?
        if obj_v_1 - obj_v_0 <= theta *  sum(sum(grad_v * (v_1-v_0))):
            is_continue = False
        else:
            _step = math.pow(BETA, _c)  # shrink the step size
            # _step = _step / 2
            _c += 1
    
    return v_1


def line_search_e(weight_matrix, student_exe, q_matrix, 
                  alpha, lambda_u, lambda_v, lambda_e,
                  grad_e, u_0, v_0, e_0, theta = THETA):
    
    """
    FUNCTION: calculate the E value in the nexit iteration
    RULE: Armijo rule along the projection arc
    """

    _step = 1.0
    # calculate the objecive function value based on e_0
    obj_e_0 = math.pow(np.linalg.norm(weight_matrix * (student_exe-np.dot(e_0, u_0)), ord='fro'), 2) \
              + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(e_0, v_0), ord='fro'), 2) \
              + lambda_e * math.pow(np.linalg.norm(e_0, ord='fro'), 2) \
              + lambda_u * math.pow(np.linalg.norm(u_0, ord='fro'), 2) \
              + lambda_v * math.pow(np.linalg.norm(v_0, ord='fro'), 2)
    
    is_continue = True
    e_1 = None
    _c = 1

    while is_continue:
        # calculate the new E value: e_1
        e_1 = np.maximum(e_0 - _step * grad_e, 0)
        
        # calculate the objecive function value based on e_1
        obj_e_1 = math.pow(np.linalg.norm(weight_matrix * (student_exe-np.dot(e_1, u_0)), ord='fro'), 2) \
                  + math.pow(alpha * np.linalg.norm(q_matrix - np.dot(e_1, v_0), ord='fro'), 2) \
                  + lambda_e * math.pow(np.linalg.norm(e_1, ord='fro'), 2) \
                  + lambda_u * math.pow(np.linalg.norm(u_0, ord='fro'), 2) \
                  + lambda_v * math.pow(np.linalg.norm(v_0, ord='fro'), 2)
        
        # is satisfied Armijo rule?
        if obj_e_1 - obj_e_0 <= theta *  sum(sum(grad_e * (e_1-e_0))):
            is_continue = False
        else:
            _step = math.pow(BETA, _c)  # shrink the step size
            # _step = _step / 2
            _c += 1
    
    return e_1


def cal_matrix_pre(left_latent_matrix, right_latent_matrix):
    """
    FUNCTION: calculate the predicted scoring matrix based on
    X[i][j] = U[i][:]*V[:][j]

    Inputs:
    -------
    :param left_latent_matrix --> numpy.ndarray

    :param right_latent_matrix --> numpy.ndarray

    Outputs:
    -------
    return matrix_pre --> numpy.ndarray
    """

    matrix_pre = np.dot(left_latent_matrix, right_latent_matrix)

    return matrix_pre


def cal_matrix_pre_bias(matrix_ori, miss_col, left_latent_matrix, right_latent_matrix):
    """
    FUNCTION: calculate the predicted scoring matrix based on
    X[i][j] = U[i][:]*V[:][j] + be[i] + bu[j] + mu

    Inputs:
    -------
    :param matrix_ori --> numpy.ndarray

    :param miss_col --> list
        the positions of all missing values

    :param left_latent_matrix --> numpy.ndarray

    :param right_latent_matrix --> numpy.ndarray

    Outputs:
    -------
    return matrix_pre --> numpy.ndarray
    """
    shape = matrix_ori.shape

    mu = 0  # the mean value of the student scoring matrix
    be = dict()  # the exercise bias
    bu = dict()  # the student bias

    for i in range(shape[0]):
        be[i] = 0
        count = 0
        for j in range(shape[1]):
            if (i, j) in miss_col:
                continue
            mu += matrix_ori[i][j]
            be[i] += matrix_ori[i][j]
            count += 1
        be[i] = be[i] / count

    for j in range(shape[1]):
        bu[j] = 0
        count = 0
        for i in range(shape[0]):
            if (i, j) in miss_col:
                continue
            bu[j] += matrix_ori[i][j]
            count += 1
        bu[j] = bu[j] / count

    mu = mu / (shape[0]*shape[1] - len(miss_col))

    # hyper-parameters
    w_mf = 0.5
    w_be = 0.2
    w_bu = 0.2
    w_mu = 0.1

    matrix_pre = np.zeros(shape=list(shape), dtype=float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix_pre[i][j] = w_mf * (np.dot(left_latent_matrix[i, :], right_latent_matrix[:, j])) \
                               + w_be * be[i] + w_bu * bu[j] + w_mu * mu

    return matrix_pre


def stu_kn_diagnose_area(u, v):
    """
    FUNCTION: the cognitivte diagnosis (measuring students' knowledge concept proficiency)
    
    METHOD:
    1. using the point coverage function: cov(stu) and cov(kn)
    2. calculate (cov(kn) ∩ cov(stu)) / cov(kn)

    Inputs:
    -------
    :param u --> numpy.ndarray
        the student proficiency matrix

    :param v --> numpy.ndarray
        the knowledge concept requirement matrix

    Outputs:
    -------
    :return stu_kn_pro --> numpy.ndarray
        the student-knowledge concept proficiency matricx
    """

    shape_u = u.shape
    shape_v = v.shape

    threshold = 1.0 * math.pow(10, -5)

    # initialize the student-knowledge concept proficiency matricx
    stu_kn_pro = np.zeros(shape=(shape_u[1], shape_v[1]), dtype=float)
    rank = shape_u[0]  # get the rank (the number of topic-skill)

    for k in range(shape_v[1]):  # walk through all knowledge concepts

        # calculate the cov value of knowledge concepts, excluding the dimension value being 0
        cov_kn = 1
        is_zero = list()  # the dimensions that have 0 values
        for _ in range(rank):  # walk through all ranks
            _e_kn = v[_, k]
            if _e_kn < threshold:  # if < threshold, give 0 value
                is_zero.append(_)  # record the dimension being 0 value
            else:
                cov_kn = cov_kn * _e_kn

        for s in range(shape_u[1]):  # walk through all students

            # calculate the cov value of the student
            cov_stu = 1
            for _ in range(rank):
                if _ in is_zero:
                    continue
                else:
                    _e_stu = u[_, s]
                    cov_stu = cov_stu * _e_stu

            # calculate the student's knowledge concept proficiency
            _pro = min(cov_stu, cov_kn) / cov_kn
            stu_kn_pro[s][k] = _pro

    return stu_kn_pro


def stu_kn_diagnose_sim(u, v):
    """
    FUNCTION: the cognitivte diagnosis (measuring students' knowledge concept proficiency)

    METHOD: calculte the similarity between student and knowledge concept based on,
    their European distance  new coordinate (E)

    Inputs:
    -------
    :param u --> numpy.ndarray

    :param v --> numpy.ndarray

    Outputs:
    -------
    :return stu_kn_pro --> numpy.ndarray
    """

    shape_u = u.shape
    shape_v = v.shape
    rank = shape_u[0]

    stu_kn_pro = np.zeros(shape=(shape_u[1], shape_v[1]), dtype=float)

    for s in range(shape_u[1]):
        for k in range(shape_v[1]):
            dis = 0
            for _ in range(rank):
                dis += math.pow(abs(u[_, s] - v[_, k]), 2)
            dis = math.sqrt(dis)

            stu_kn_pro[s][k] = 1/dis

    return stu_kn_pro