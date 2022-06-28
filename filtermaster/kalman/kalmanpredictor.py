# -*- coding:utf8 -*-

import numpy as np
from copy import deepcopy
from filtermaster.statistic import covariance

def prekalman(Fn,Cn,Vn,Wn,X1:np.matrix,Yn):
    """
    This is kalman predictor.
    The difference between kalman predictoe and normal kalman filter is the predictor
    uses x(n)(observing value) to estimate x(n+1) while normal kalman to estimate
    x(n)(estimteed value). And the core progress is the same as normal kalman filter.

    Parameters
    ----------
    Fn : the state transition matrix, if Fn are changable with time, it must be as long 
            as Yn. If it is a matrix or a list including one matrix, it will be seen as 
            the LTI system. So do the parameters: Cn, Vn, Wn.
    Cn : the observing matrix.
    Vn : the self-correltion matrix of error matrix(v1(n)) in state euqation.
    Wn : the self-correlation matrix of error matrix(v2(n)) in observing matrix.
    X1 : the start condition for iteration.
    Yn : the observing value you've already got, must be np.ndarray of one demontion.
    """
    fn = []
    cn = []
    vn = []
    wn = []
    yn = [deepcopy(Yn[0]),deepcopy(Yn[1])]
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
    x1_list = [deepcopy(X1[i,0]) for i in range(X1.shape[0])]
    mean = np.array(x1_list).mean()
    xn = np.matrix([[mean] for i in range(X1.shape[0])])
    kn = np.matrix(covariance([x1_list],[x1_list]))
    I = np.matrix(np.eye(X1.shape[0]))
    # start iteration
    if len(Fn) == N:
        for i in range(1,N-1,1):
            # get kalman gain 'gp'
            rn = cn[i]*kn*cn[i].T + wn[i]
            gp = fn[i]*kn*cn[i].T*rn.I
            # calculate the error
            alpha = Yn[i] - cn[i]*xn
            # get predicted coefficients
            xn = fn[i]*xn + gp*alpha
            # update 'kn' 
            k_n = kn - fn[i].I*gp*cn[i]*kn
            kn = fn[i]*k_n*fn[i].T + vn[i]
            # store values
            yn.append(cn[i]*xn)
            gn.append(gp)
    elif len(Fn) == 1:
        # get kalman gain 'gp'
        for i in range(1,N-1,1):
            rn = cn*kn*cn.T + wn
            gp = fn*kn*cn.T*rn.I
            # calculate the error
            alpha = Yn[i] - cn*xn
            # get predicted coefficients
            xn = fn*xn + gp*alpha
            # update 'kn' 
            k_n = kn - fn.I*gp*cn*kn
            kn = fn*k_n*fn.T + vn
            # store values
            yn.append((cn*xn)[0,0])
            gn.append(gp)
    else:
        raise ValueError("error")
    # return 
    if len(Fn) == N:
        return np.array(yn,dtype=np.matrix)
    else:
        return np.array(yn,dtype=np.float64)
