# -*- coding:utf8 -*-

import numpy as np
from copy import deepcopy
from filtermaster.statistic import correlation


def forward(xn:list|np.ndarray,r:int=3,er=0):
    """
    This is a forward linear predictor.
    It uses values in quantity of 'r' before 'xn' to estimate the output value 'xn'.
    And this filter can make the mean error least. It conforms to the autoregressive(AR)
    progress.

    Parameters
    ----------
    xn : the observing values
    r : the rank of AR module
    er : the variane of observing noise.

    Returning values
    ----------
    It'll retrurn the coefficients of every moment and the output values.

    Demonstration (you can copy the codes below and run it)
    ----------
    import matplotlib.pyplot as plt
    a = np.linspace(0,5,100)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = b+np.random.randn(100)
    w,y = forward(c,3,1.20)
    plt.plot(a,c)
    plt.plot(a,b)
    plt.plot(a,y)
    plt.show()
    """
    if len(xn) < r:
        raise ValueError("the length of 'xn' must be larger than 'r'")
    if r <= 0:
        raise ValueError("rank must be larger than 0")
    yn = []
    wn = []
    for i in range(r):
        yn.append(deepcopy(xn[i]))
    for i in range(r,len(xn),1):
        x = [deepcopy(xn[i-j-1]) for j in range(r)]
        d = [deepcopy(xn[i])]
        rn = np.matrix(correlation(x,x,out="matrix",inv=False)) + np.matrix(er*np.eye(r))
        rxd = np.matrix(correlation(x,d,out="sequence",k0=False,inv=False)).T
        w = rn.I*rxd
        wn.append(w)
        y = (w.T*np.matrix(x).T)[0,0]
        yn.append(y)
    return np.array(wn),np.array(yn)


def backward(xn:list|np.ndarray,r:int=3,er=0):
    """
    This is a backward linear predictor.
    It uses the values in quantity of 'r' ahead 'xn' to estimate the value of 'xn'.
    This filter seems to be unreal, however, it can work with the forward linear
    predictor to design a complete system.

    Parameters
    ----------
    xn : the observing values
    r : rank of this filter
    er : the variane of observing noise.

    Returning values
    ----------
    It'll retrurn the coefficients of every moment and the output values.

    Demonstration (you can copy the codes below and run it)
    ----------
    import matplotlib.pyplot as plt
    a = np.linspace(0,5,100)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    w,y = backward(b,3,1.20)
    plt.plot(a,b)
    plt.plot(a,y)
    plt.show()
    """
    if len(xn) < r:
        raise ValueError("the length of 'xn' must be larger than 'r'")
    if r <= 0:
        raise ValueError("rank must be larger than 0")
    yn = []
    wn = []
    for i in range(len(xn)-r):
        x = [deepcopy(xn[i+r-j]) for j in range(r)]
        d = [deepcopy(xn[i])]
        rn = np.matrix(correlation(x,x,out="matrix",inv=False)) + np.matrix(er*np.eye(r))
        rxd = np.matrix(correlation(x,d,out="sequence",k0=False,inv=False)).T
        w = rn.I*rxd
        wn.append(w)
        y = (w.T*np.matrix(x).T)[0,0]
        yn.append(y)
    for i in range(len(xn)-r,len(xn)):
        yn.append(deepcopy(xn[i]))
    return np.array(wn),np.array(yn)
