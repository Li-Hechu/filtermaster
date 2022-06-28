# -*- coding:utf8 -*-

import numpy as np
from copy import deepcopy
from filtermaster.statistic import covariance


def infokalman(Fn,Cn,Vn,Wn,X0:np.matrix,Yn):
    """
    This is information kalman filter.
    The main difference between this filter and normal kalman filter is 'Kn', the covariance
    matrix of state x(n). In information kalman filter, we define an information matrix
    'Jn', which is the inverse of Kn, to replace 'Kn' in the progress. However, we will
    inverse four matrixs and the formula is more complex, so the calculate capacity is more
    high. And I don;t recommend you to use it when the rank of matrix is big.

    Parameters
    ----------
    Fn : the state transition matrix, if Fn are changable with time, it must be as long 
            as Yn. If it is a matrix or a list including one matrix, it will be seen as 
            the LTI system. So do the parameters: Cn, Vn, Wn.
    Cn : the observing matrix.
    Vn : the self-correltion matrix of error matrix(v1(n)) in state euqation.
    Wn : the self-correlation matrix of error matrix(v2(n)) in observing matrix.
    X0 : the start condition for iteration.
    Yn : the observing value you've already got, must be np.ndarray of one demontion.
    """
    # define the constant value
    fn = []
    cn = []
    vn = []
    wn = []
    yn = [deepcopy(Yn[0])]
    gn = []
    N = len(Yn)
    # precess Fn
    if len(Fn) == 1:
        fn = deepcopy(Fn[0])
    elif len(Fn) == N:
        fn = deepcopy(Fn)
    elif isinstance(Fn,np.matrix):
        fn = deepcopy(Fn)
    else:
        raise ValueError("the length of 'Fn' dosen't match")
    # process Cn
    if len(Cn) == 1:
        cn = deepcopy(Cn[0])
    elif len(Cn) == N:
        cn = deepcopy(Cn)
    elif isinstance(Cn,np.matrix):
        cn = deepcopy(Cn)
    else:
         raise ValueError("the length of 'Cn' dosen't match")
    # precess Vn
    if len(Vn) == 1:
        vn = deepcopy(Vn[0]) + np.matrix(np.eye(Vn[0].shape[0]))*np.float64("1e-10") # avoid the singular-matrix problem
    elif len(Vn) == N:
        vn = deepcopy(Vn)
    elif isinstance(Vn,np.matrix):
        vn = deepcopy(Vn)
    else:
         raise ValueError("the length of 'Vn' dosen't match")
    # process Wn
    if len(Wn) == 1:
        wn = deepcopy(Wn[0]) + np.matrix(np.eye(Wn[0].shape[0]))*np.float64("1e-10") # avoid the singular-matrix problem
    elif len(Wn) == N:
        wn = deepcopy(Wn)
    elif isinstance(Wn,np.matrix):
        wn = deepcopy(Wn)
    else:
         raise ValueError("the length of 'Wn' dosen't match")
    # initial condition
    x0_list = [deepcopy(X0[i,0]) for i in range(X0.shape[0])]
    mean = np.array(x0_list).mean()
    xn = np.matrix([[mean] for i in range(X0.shape[0])])
    # add a small matrix to information matrix 'Jn'  to make it can be reversed
    Jn = (np.matrix(covariance([x0_list],[x0_list])) + np.matrix(np.eye(X0.shape[0]))*np.float64("1e-20")).I
    # start iteration
    if len(Fn) == N:
        for i in range(1,N,1):
            # predict 'xn'
            xn_1 = fn[i-1]*xn
            # calculate the error
            alpha = Yn[i] - cn[i]*xn_1
            # predict kn
            Jn_1 = vn[i].I - vn[i-1].I*fn[i-1]*(Jn+fn[i-1].T*vn[i-1].I*fn[i-1]).I*fn[i-1].T*vn[i-1].I
            # calculate 'rn'
            Jn = Jn_1 + cn[i].T*wn[i]*cn[i]
            # caculate kalman gain
            gf = Jn.I*cn[i].T*wn[i].I
            # correct the error and get the final results
            xn = xn_1 + gf*alpha
            # store the result
            yn.append(cn[i]*xn)
            gn.append(gf)
    elif len(Fn) == 1:
        for i in range(1,N,1):
             # predict 'xn'
            xn_1 = fn*xn
            # calculate the error
            alpha = Yn[i] - cn*xn_1
            # predict kn
            Jn_1 = vn.I - vn.I*fn*(Jn+fn.T*vn.I*fn).I*fn.T*vn.I
            # calculate 'rn'
            Jn = Jn_1 + cn.T*wn*cn
            # caculate kalman gain
            gf = Jn.I*cn.T*wn.I
            # correct the error and get the final results
            xn = xn_1 + gf*alpha
            # store the result
            yn.append((cn*xn)[0,0])
            gn.append(gf)
    else:
        raise ValueError("error")
    # return 
    if len(Fn) == N:
        return np.array(yn,dtype=np.matrix)
    else:
        return np.array(yn,dtype=np.float64)

