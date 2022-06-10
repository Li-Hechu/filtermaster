# -*- coding:utf8 -*- 

import numpy as np
from copy import deepcopy

def lms(xn:list|np.ndarray,dn:list|np.ndarray,r:int=3):
    """
    This is a leat mean square(LMS) filter.
    Its advantages is obvious: its algorithm is simple, calculation quantity
    is low, and the coefficients can reach convergence. However the speed of
    convergenc is low and there is extral error.

    Parameters
    ----------
    xn : the input values
    dn : the expected values
    r : the rank of this filter.

    Returning Values
    ----------
    It'll return the coefficients of a difference equation in this moment and the
    output value of the filter.

    Demonstration (you can copy the codes below and run it)
    ----------
    from filtermaster.diffequation import difeq
    import matplotlib.pyplot as plt
    import numpy as np
    a = np.linspace(0,5,50)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = difeq(b,[0.23,-2.12,1.05]) 
    d = c + np.random.randn(50)
    w,y = lms(b,d,3)
    plt.plot(a,c)
    plt.plot(a,y)
    plt.show()
    """
    if len(xn) != len(dn):
        raise ValueError("'xn' and 'dn' must have the same length")
    if r <= 0 or r > len(xn):
        raise ValueError("'r' must be between 0 and the length of 'xn'")
    if not isinstance(r,int):
        raise ValueError("'r' must be an integer")
    # initialize the coefficients
    w = np.matrix([[1.0] for i in range(r)])
    # define the constant value
    wn = []
    yn = []
    # initialize yn
    for i in range(r-1):
        yn.append(dn[i])
    # iteration process
    for i in range(r-1,len(xn),1):
        # gain the delaying value
        x_n = np.matrix([deepcopy(xn[i-j]) for j in range(r)]).T
        # esimate 'rn'
        rn = x_n*x_n.T
        # estimate the output value
        y = (w.T*x_n)[0,0]
        # the error between 'rn' and 'yn'
        e_n = dn[i] - y
        # calculate the convergence value 'miu'
        tr = 0
        for i in range(r):
            tr += rn[i,i]
        miu = 1/tr
        # store the w and y in a list
        wn.append(w)
        yn.append(y)
        # iteration formula
        w = w + miu*x_n*e_n
    # returning value
    return np.array(wn),np.array(yn)

def nlms(xn:list|np.ndarray,dn:list|np.ndarray,r:int=3):
    """
    This is a normalized LMS(NLMS) filter.
    Compared with the LMS filter, the only change in NLMS filter is the
    iteration step 'miu'. Please see the source code for more information.

    Parameters
    ----------
    xn : the input values
    dn : the expected values
    r : the rank of this filter.

    Returning Values
    ----------
    It'll return the coefficients of a difference equation in this moment and the
    output value of the filter.

    Demonstration (you can copy the codes below and run it)
    ----------
    from filtermaster.diffequation import difeq
    import matplotlib.pyplot as plt
    import numpy as np
    a = np.linspace(0,5,50)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = difeq(b,[0.23,-2.12,1.05]) 
    d = c + np.random.randn(50)
    w,y = nlms(b,d,3)
    plt.plot(a,c)
    plt.plot(a,y)
    plt.show()
    """
    if len(xn) != len(dn):
        raise ValueError("'xn' and 'dn' must have the same length")
    if r <= 0 or r > len(xn):
        raise ValueError("'r' must be between 0 and the length of 'xn'")
    if not isinstance(r,int):
        raise ValueError("'r' must be an integer")
    # initialize the coefficients
    w = np.matrix([[1.0] for i in range(r)])
    # define con tant values
    wn = []
    yn = []
    # initialize yn
    for i in range(r-1):
        yn.append(dn[i])
    # iteration process
    for i in range(r-1,len(xn),1):
        # gain the delaying value
        x_n = np.matrix([deepcopy(xn[i-j]) for j in range(r)]).T
        # estimate the output value
        y = (w.T*x_n)[0,0]
        # the error between 'rn' and 'yn'
        e_n = dn[i] - y
        # calculate the convergence value 'miu'
        miu0 = 1
        sqr_xn = 0
        for i in range(r):
            sqr_xn += x_n[i,0]**2
        miu = miu0/sqr_xn
        # store the w and y in a list
        wn.append(w)
        yn.append(y)
        # iteration formula
        w = w + miu*x_n*e_n
    # returning value
    return np.array(wn),np.array(yn)

def llms(xn:list|np.ndarray,dn:list|np.ndarray,r:int=3):
    """
    This is a leaky LMS(LLMS) filter.
    When the x(n)*x(n) matrix is singular, the coefficients might not be
    convergent. Thus, we need the leaky LMS filter to solve the problem.

    Parameters
    ----------
    xn : the input values
    dn : the expected values
    r : the rank of this filter.

    Returning Values
    ----------
    It'll return the coefficients of a difference equation in this moment and the
    output value of the filter.

    Demonstration (you can copy the codes below and run it)
    ----------
    from filtermaster.diffequation import difeq
    import matplotlib.pyplot as plt
    import numpy as np
    a = np.linspace(0,5,50)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = difeq(b,[0.23,-2.12,1.05]) 
    d = c + np.random.randn(50)
    w,y = llms(b,d,3)
    plt.plot(a,c)
    plt.plot(a,y)
    plt.show()
    """
    if len(xn) != len(dn):
        raise ValueError("'xn' and 'dn' must have the same length")
    if r <= 0 or r > len(xn):
        raise ValueError("'r' must be between 0 and the length of 'xn'")
    if not isinstance(r,int):
        raise ValueError("'r' must be an integer")
    # initialize the coefficients
    w = np.matrix([[1.0] for i in range(r)])
    # define con tant values
    wn = []
    yn = []
    ga = 0.01    # gamma value in leaky LMS
    # initialize yn
    for i in range(r-1):
        yn.append(dn[i])
    # iteration process
    for i in range(r-1,len(xn),1):
        # gain the delaying value
        x_n = np.matrix([deepcopy(xn[i-j]) for j in range(r)]).T
        # estimate 'rn'
        rn = x_n * x_n.T
        # estimate the output value
        y = (w.T*x_n)[0,0]
        # the error between 'rn' and 'yn'
        e_n = dn[i] - y
        # calculate the convergence value 'miu'
        val,arr = np.linalg.eig(rn)
        del arr
        miu = 1/(val.max().real+ga)    # avoiding the conjugate number caused by the computer itself
        # store the w and y in a list
        wn.append(w)
        yn.append(y)
        # iteration formula
        w = (1-ga*miu)*w + miu*x_n*e_n
    # returning value
    return np.array(wn),np.array(yn)
    
def slms(xn:list|np.ndarray,dn:list|np.ndarray,r:int=3):
    """
    This is a sign LMS(SLMS) filter.
    Although the calculation quantity of LMS, NLMS and LLMS can be low, 
    we still want another filter can be more effective and reduce the 
    calculation quantity further. So, the sign LMS filter is designed.
    We just need to replace the error 'en' to a sign funtion, whose value
    is determined by the sign of 'en'.

    Parameters
    ----------
    xn : the input values
    dn : the expected values
    r : the rank of this filter.

    Returning Values
    ----------
    It'll return the coefficients of a difference equation in this moment and the
    output value of the filter.

    Demonstration (you can copy the codes below and run it)
    ----------
    from filtermaster.diffequation import difeq
    import matplotlib.pyplot as plt
    import numpy as np
    a = np.linspace(0,5,50)
    b = np.sin(2*np.pi*a-np.pi/3) + 2* np.cos(3*np.pi*a+np.pi/12)
    c = difeq(b,[0.23,-2.12,1.05]) 
    d = c + np.random.randn(50)
    w,y = slms(b,d,3)
    plt.plot(a,c)
    plt.plot(a,y)
    plt.show()
    """
    if len(xn) != len(dn):
        raise ValueError("'xn' and 'dn' must have the same length")
    if r <= 0 or r > len(xn):
        raise ValueError("'r' must be between 0 and the length of 'xn'")
    if not isinstance(r,int):
        raise ValueError("'r' must be an integer")
    # initialize the coefficients
    w = np.matrix([[1.0] for i in range(r)])
    # define the constant value
    wn = []
    yn = []
    # initialize yn
    for i in range(r-1):
        yn.append(dn[i])
    # iteration process
    for i in range(r-1,len(xn),1):
        # gain the delaying value
        x_n = np.matrix([deepcopy(xn[i-j]) for j in range(r)]).T
        # esimate 'rn'
        rn = x_n*x_n.T
        # estimate the output value
        y = (w.T*x_n)[0,0]
        # the error between 'rn' and 'yn'
        e_n = dn[i] - y
        # calculate the convergence value 'miu'
        tr = 0
        for i in range(r):
            tr += rn[i,i]
        miu = 1/tr
        # store the w and y in a list
        wn.append(w)
        yn.append(y)
        # iteration formula
        w = w + miu*x_n*sgn(e_n)/2
    # returning value
    return np.array(wn),np.array(yn)

def sgn(x):
    """
    sign funtion
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
