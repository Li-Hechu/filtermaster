# -*- coding:utf8 -*-

import numpy as np
from copy import deepcopy

def rls(xn:list|np.ndarray,dn:list|np.ndarray,r:int=3):
    """
    This is a recursive least square(RLS) filter.
    The priciple of this filter is based on LS filter. However, due to
    the heavy calculation quantity, LS filter isn't suitable when the 
    length of input values is huge. We also want to design a filter which 
    can be described by a recurrence formula like the grand descend filter 
    and LMS filter. Luckily, the RLS filter can satisfy our requirements.

    Parameters
    ----------
    xn : the input values
    dn : the expected values
    r : the rank of this filter.

    Returning values
    ----------
    It'll return two values. The first is the 'w' in every single moment, and
    another is output values of this filter.

    Demonstration (you can copy the codes below and run it)
    ----------
    from filtermaster.diffequation import difeq
    import matplotlib.pyplot as plt
    a = np.linspace(0,5,50)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = difeq(b,[0.23,-2.12,1.05])
    d = c + np.random.randn(50)
    w,y = rls(b,d,4)
    plt.plot(a,c)
    plt.plot(a,y)
    plt.show()
    """
    # judge whether the parameters is legal
    if len(xn) != len(dn):
        raise ValueError("'xn' and 'dn' must have the same length")
    if r <= 0 or r > len(xn):
        raise ValueError("'r' must be between 0 and the length of 'xn'")
    if not isinstance(r,int):
        raise ValueError("'r' must be an integer")
    # define the constant value
    lam = 0.8
    delta = np.float64("10")
    yn = []
    wn = []
    # define initial value for iteration
    fi = np.matrix(np.eye(r))*delta
    pn = fi.I
    w = np.matrix([[0] for i in range(r)])
    for i in range(r-1):
        yn.append(deepcopy(dn[i]))
    # start iteration
    for i in range(r-1,len(xn),1):
        # calculate the vector of input values
        x = np.matrix([deepcopy(xn[i-j]) for j in range(r)]).T
        # calculate RLS gain vector
        kn = (pn*x)/(lam+(x.I*pn*x)[0,0])
        # error in progreess
        eps = dn[i] - (w.T*x)[0,0]
        # update the coefficients
        w = w + kn*eps
        # update the gain vector
        pn = lam**(-1)*pn - lam**(-1)*kn*x.T*pn
        # calculate output values of filter
        y = (w.T*x)[0,0]
        # store the output value and coefficients
        yn.append(y)
        wn.append(w)
    # return
    return np.array(wn), np.array(yn)
