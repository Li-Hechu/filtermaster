# -*- coding:utf8 -*-

import numpy as np
from copy import deepcopy


def ls(xn:list|np.ndarray,dn:list|np.ndarray,r:int=3,dtype:str="pre"):
    """
    This is a least square filter.
    It receives input values and expected values, then calculate the 
    coefficients of the filter, making the sum of squares least.
    Usually, the value matrix build by 'xn' have many windowing types.
    I strongly recommend you to use the pre-windowing type 'pre' to build
    value matrix. The back-windowing type 'back' isn't as accurate as
    pre-windowing type, only when the rank 'r' is as big as possible,
    could this filter can reach the optimal estiamting values. The type 
    'cov', which means the covariance windowing, and 'corr', which means 
    self-correlation windowing cannot estimate the output values as accurate 
    as possible. Thus, you'd better be careful when you use them. Another 
    interesting fact is that, the output values will approach a higher 
    accuracy as the rank 'r' getting bigger. Due to the matrix will be inversed
    in the calculating process, I don't recommend you to use this filter when
    the length of 'xn' is large, or it'll take a long time to get the result.
    In this case, you can use recursive least square(RLS) filter instead.

    Parameters
    ----------
    xn : the input values
    dn : the expeted values
    r : the rank of this filter
    dtype : the windowing type for value matrix build by 'xn'. There are four
                types: 'cov' for covariance windowing, 'corr' for self-correlation
                windowing, 'pre' for pre-windowing  and 'back' for back-windowing.
                Their performance is mentioned above.

    Returning values
    ----------
    It will return the coefficients of the filter and estimated values.

    Demonstration
    ----------
    w, y = ls([1,2,3,4,5,6],[0,1,2,3,4,5],r=4,dtype="pre")
    print(y)
    >>> [0.0 1.0 2.0 3.0 4.0 5.0]
    print(w)
    >>> [[3.90164092e-15]
         [1.00000000e+00]
         [7.99360578e-15]
         [6.66133815e-15]]
    """
    # type judgement
    if not isinstance(xn,(list,np.ndarray)):
        raise ValueError("'xn' must be a list or ndarray")
    if not isinstance(dn,(list,np.ndarray)):
        raise ValueError("'xn' must be a list or ndarray")
    if not isinstance(r,int):
        raise TypeError("'r' must be an integer")
    if len(xn) != len(dn):
        raise ValueError("'xn' and 'dn' must have the same length")
    if r <= 0 or r > len(xn):
        raise ValueError("'r' should between 0 and the length of 'xn'")
    # process the value window
    N = len(xn)
    A = []
    # covariance windowing
    if dtype == "cov":
        for i in range(r-1,N,1):
            A.append([])
            for j in range(i,i-r,-1):
                A[i+1-r].append(deepcopy(xn[j]))
    # self-correlation windowing
    elif dtype == "corr":
        A = [[0 for j in range(r)] for i in range(N+r-1)]
        for j in range(r):
            for i in range(N+r-1):
                if i < j or i > N + j - 1:
                    A[i][j] = 0
                else:
                    A[i][j] = deepcopy(xn[i-j])
    # pre-windowing
    elif dtype == "pre":
        for i in range(N):
            A.append([])
            for j in range(r):
                if i < j:
                    A[i].append(0)
                else:
                    A[i].append(deepcopy(xn[i-j]))
    # back-windowing
    elif dtype == "back":
        A = [[0 for j in range(r)] for i in range(N)]
        for j in range(r):
            for i in range(N):
                if i + r - j - 1 >= N:
                    A[i][j] = 0
                else:
                    A[i][j] = deepcopy(xn[i+r-j-1])
    else:
        raise ValueError(f"unsurpported window type '{dtype}' for the value matrix")
    # initialize A matrix
    A = np.matrix(A)
    # initialize dn matrix
    d = []
    if dtype == "cov":
        d = np.matrix([deepcopy(dn[i]) for i in range(r-1,N)]).T
    elif dtype == "corr":
        tmp = np.concatenate((deepcopy(dn),[0 for i in range(r-1)]),axis=0)
        d = np.matrix(tmp).T
        del tmp
    else:
        d = np.matrix(deepcopy(dn)).T
    # calculate the 'w'
    w = (A.T*A).I*A.T*d
    # get the results
    re = A*w
    # process the results according to 'dtype'
    if dtype == "cov":
        tp = [dn[i] for i in range(r-1)]
        for i in range(r-1,N):
            tp.append(deepcopy(re[i-r+1,0]))
        re = np.matrix(tp).T
        del tp
    if dtype == "corr":
        re = deepcopy(re[0:N,0])
    # turn matrix into an array
    ans = []
    for i in range(N):
        ans.append(re[i,0])
    ans = np.array(ans)
    # returning values
    return w,ans
