# -*- coding:utf8 -*-

import numpy as np
from copy import deepcopy
from filtermaster.statistic import correlation

def graord(xn,dn,count:int=15):
    """
    This function is used to calculate the coefficients of the filter and 
    the output values of a single moment in grand descend algorithm.

    Parameters
    ----------
    xn : the delaying values, reversing time sequence
    dn : the expected values, and must be a number
    count : the iteration times

    Returning Values
    ----------
    It'll return the coefficients of a difference equation in this moment and the
    output value of the filter.

    Demonstration
    ----------
    a=[-0.12,0.23,-0.05]
    w,y = graord(a,0.56)
    print(w)
    >>> [[0.40515699]
         [2.62287265]
         [1.019557  ]]
    
    print(y)
    >>> 0.50366402
    """
    N = len(xn)
    d_n = [dn]
    w = np.matrix([[1.0] for i in range(N)])
    rxd = np.matrix(correlation(xn,d_n,k0=False,out="sequence",inv=False)).T
    rn = np.matrix(correlation(xn,xn,out="matrix",inv=False))
    val,arr = np.linalg.eig(rn)
    miu = 1/val.max()
    del arr
    for i in range(count):
        w = w + miu*(rxd-rn*w)
    y = 0
    for i in range(N):
        y += xn[i] * w[i,0]
    return w,y



def grades(xn:list|np.ndarray,dn:list|np.ndarray,r:int=3,count:int=15):
    """
    This is a grand descend filter.
    It is based on wiener filter but have different ways to get
    the coeficients. The key point in grand descend algotithm is
    iteration. Every moment, this filter needs an initial coefficients
    'w', and it will approach a stable value through iteration, which 
    is what we expect. It is recommended that the count be 15. In order 
    to ensure the coefficients is stable, the count should not be 
    under 10

    Parameters
    ----------
    xn : the input values
    dn : the expected values
    r : the rank of this filter.
    count : itering times.

    Returning values
    ----------
    It'll return two values. The first is the 'w' in every single moment, and
    another is output values of this filter.

    Demonstration (you can copy the codes below and run it)
    ----------
    from filtermaster.diffequation import difeq
    import matplotlib.pyplot as plt
    import numpy as np
    a = np.linspace(0,5,50)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = difeq(b,[0.23,-2.12,1.05])
    w,y = grades(b,c,3)
    print(w)
    plt.plot(a,c)
    plt.plot(a,y)
    plt.show()
    """
    if len(xn) != len(dn):
        raise ValueError("'xn' and 'dn' must have the same length")
    N = len(xn)
    yn = []
    wn = []
    for i in range(r-1):
        yn.append(dn[i])
    for i in range(r-1,N,1):
        x_n = [deepcopy(xn[i-j]) for j in range(r)]
        w, y = graord(x_n,deepcopy(dn[i]),count)
        wn.append(w)
        yn.append(y)
    return np.array(wn),np.array(yn)
