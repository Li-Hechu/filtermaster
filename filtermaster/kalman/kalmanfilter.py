# -*- coding:utf8 -*-

import numpy as np
from copy import deepcopy
from filtermaster.statistic import covariance

def kalman(Fn,Cn,Vn,Wn,X0:np.matrix,Yn:np.ndarray):
    """
    This is a normal kalman filter.
    Kalman filter is a linear optimal filter, like wiener filter.
    In many actual situations, the signal can be determined by a state equation,
    for instance, the action of an object or original signal is structive. Kalman
    filter also needs another observing equation to describe the value you've got.
    However, the Kalman filter is only adapted to linear systems, for some non-linear
    systems, this filter is no longer suitable.

    the state equation :        X(n+1) = F(n)X(n) + v1(n)
    the observing equation :    y(n) = C(n)X(n) + v2(n)

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
        vn = deepcopy(Vn[0])
    elif len(Vn) == N:
        vn = deepcopy(Vn)
    elif isinstance(Vn,np.matrix):
        vn = deepcopy(Vn)
    else:
         raise ValueError("the length of 'Vn' dosen't match")
    # process Wn
    if len(Wn) == 1:
        wn = deepcopy(Wn[0])
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
    kn = np.matrix(covariance([x0_list],[x0_list]))
    I = np.matrix(np.eye(X0.shape[0]))
    # start iteration
    if len(Fn) == N:
        for i in range(1,N,1):
            # predict 'xn'
            xn_1 = fn[i-1]*xn
            # calculate the error
            alpha = Yn[i] - cn[i]*xn_1
            # predict kn
            kn_1 = fn[i]*kn*fn[i].T + vn[i-1]
            # calculate 'rn'
            rn = cn[i]*kn_1*cn[i].T + wn[i-1]
            # caculate kalman gain
            gf = kn_1*cn[i].T*rn.I
            # correct the error and get the final results
            xn = xn_1 + gf*alpha
            kn = (I-gf*cn[i])*kn_1
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
            kn_1 = fn*kn*fn.T + vn
            # calculate 'rn'
            rn = cn*kn_1*cn.T + wn
            # caculate kalman gain
            gf = kn_1*cn.T*rn.I
            # correct the error and get the final results
            xn = xn_1 + gf*alpha
            kn = (I-gf*cn)*kn_1
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
