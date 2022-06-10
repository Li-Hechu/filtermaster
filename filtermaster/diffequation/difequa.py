# -*- coding:utf8 -*-

import numpy as np
import matplotlib.pyplot as plt

def difeq(
    x:np.ndarray|list,
    co:list|np.ndarray,
    inv:bool=True
    ):
    """
    This function can generate the values of a difference equation
    based on the coefficients and input values that are conveied in.

    Parameters
    ----------
    x : input values
    co : coefficients of a difference equation.
    inv : true for matching every last 'x' value with the first
            coefficient and false for the opposite situation.

    Returning value
    ----------
    a np.ndarray will be returned.

    Demonstration
    ----------
    import numpy as np
    import matplotlib.pyplot as plt

    a = np.linspace(0.5,50)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = difeq(b,[1.2,-0.34,-0.58],inv=True)
    plt.plot(a,c)
    plt.plot(a,b)
    plt.show()
    """
    if not isinstance(x,(list,np.ndarray)):
        raise TypeError("only receives list or ndarray")
    if not isinstance(x,(list,np.ndarray)):
        raise TypeError(f"unsupported type {type(co)}")
    r = len(co)
    re = []
    for i in range(r-1):
        re.append(x[i])
    if inv:
        for i in range(r-1,len(x)):
            tmp = 0
            for j in range(i,i-r,-1):
                tmp += x[j] * co[i-j]
            re.append(tmp)
    else:
        for i in range(r-1,len(x)):
            tmp = 0
            for j in range(i-r+1,i+1,1):
                tmp += x[j] *co[i-j]
            re.append(tmp)
    return np.array(re)
