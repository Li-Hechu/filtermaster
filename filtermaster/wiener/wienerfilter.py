# -*- coding:utf8 -*-

from filtermaster.statistic import correlation
import numpy as np
from copy import deepcopy


def wieord(xn:list|np.ndarray,dn):
    """
    This function can estimate the output value of a single moment
    in wiener filter, just need the delaying values x(n) and the expected
    values of this moment.

    Parameters
    ----------
    xn : the delaying values.
    dn : the expected values, and must be a number.

    Returning values
    ----------
    It'll return the coefficients of a difference equation in this moment and the
    output value of the filter.

    Demonstration
    ----------
    a = [0.23,0.77,-0.31]
    w,y = wieord(a,1.1)
    print(w)
    >>> [[ 1.05644994]
        [ 0.74515366]
        [-0.29623083]]

    print(y)
    >>> 0.90858335
    """
    d_n = [dn]
    rxd = np.matrix(correlation(xn,d_n,k0=False,out="sequence",inv=False)).T
    rn = np.matrix(correlation(xn,xn,out="matrix",inv=False))
    wn = rn.I*rxd
    yn = (wn.T*np.matrix(xn).T)[0,0]
    return wn,yn


def wiener(xn,dn,r:int=3):
    """
    Wiener filter is an optimal linear filter in statistical meaning.
    This filter needs 'dn', which is expected values. And it also
    needs 'xn', the input values. The final purpose is to make the
    mean error between output values 'yn' of this filter and 'dn' least. 
    'yn' is determind by a difference equation, and working out its 
    coeffecients is the point.

    Parameters
    ----------
    xn : the input values
    dn : the expected values
    r : the rank of a difference equation

    Returning values
    ----------
    It'll return two values. The first is the coefficients
    of a the filter in every moment, and another is the output
    values of this filter.

    Demonstration (you can copy the codes below and run it)
    ----------
    from filtermaster.diffequation import difeq
    import matplotlib.pyplot as plt
    import numpy as np
    a = np.linspace(0,5,50)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = difeq(b,[0.23,-2.12,1.05])
    w,y = wiener(b,c,3)
    print(w)
    plt.plot(a,c)
    plt.plot(a,y)
    plt.show()
    """
    if len(xn) != len(dn):
        raise ValueError("xn and dn must have the same length")
    yn = []
    wn = []
    for i in range(r-1):
        yn.append(dn[i])
    for i in range(r-1,len(xn)):
        w,y = wieord(deepcopy([xn[i-j] for j in range(r)]),deepcopy(dn[i]))
        wn.append(w)
        yn.append(y)
    return np.array(wn),np.array(yn)

