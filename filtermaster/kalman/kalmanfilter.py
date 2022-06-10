# -*- coding:utf8 -*-

import numpy as np

def kalman(data:np.ndarray|list=None,
           btype:str="d",
           v1:float|int=None,
           v2:float|int|np.ndarray|np.matrix=None,
           *,co:list=None,
           t:float|int=None,
           re:str="y"
           ):
    """
    Kalman filter is a optimal linear filter.
    In many actual situations, the signal can be determined by a state equation,
    for instance, the action of an object or original signal is structive. Kalman
    filter also needs another observing equation to describe the value you've got.
    However, the Kalman filter is only adapted to linear systems, for some non-linear
    systems, this filter is no longer suitable.
    Now, there are two typial systems kalman filter can be apllied to. One is a system 
    determined by a difference equation(Auroregressive progress, AR), another is a moving 
    object abided by Newton's Second Law.
    So, this function is designed to adapt to the two major situations mentioned above,
    and you can use the parameter 'type' to decide the model.

    Parameters
    ----------
    data : observing value of a system
    btype : the model you use, 'd' for differene equation(acquiescent), 'xv' for observing 
            displacement and velocity, and 'xva' will take acceleration into account.
    v1 : variance of devitation of state equation.
    v2 : variance of devitation of observing equation.
    co : (optional) coefficients of difference equation, only 'd' is given, and the program
            wil automatically build the matrix.
    t : (optional) the sampling interval of the moving system, only 'xv' or 'xva' is given.
    re : (optional) the returning value, 'y' for forcast value(acquiescent), 'yg' for Kalman 
            gain additionally.

    Returns
    ----------
    It'll return the ndarray.

    Demonstration
    ----------
    a = np.linspace(0,8,150)
    b = 2*np.sin(3*np.pi*a+np.pi/3) + np.cos(2*np.pi*a-np.pi/3) - 1.53*np.sin(2*np.pi*a+np.pi/6) + np.random.randn(150)
    re = kalman(b,co=[0.23,-1.23,0.98],v1=9,v2=0.45)
    plt.plot(a,b)
    plt.plot(a,re)
    plt.show()
    """
    # process parameters
    if data is None:
        raise TypeError("data shouldn't be None type.")
    if btype == "d" or btype == "xv" or btype == "xva":
        pass
    else:
        raise TypeError("unsurpported type '%s'" % btype)
    if re == "y" or re == "yg":
        pass
    else:
        raise TypeError("unrecognised returning type '%s'" % re)
    if not isinstance(v1,(int,float)) or not isinstance(v2,(float,int)):
        raise TypeError(f"'v1' and 'v2' both must be int or float, not {type(v1),type(v2)}")
    # define the constant
    rank = 0
    x0 = []
    k0 =[]
    I = []
    f = []
    q1 = []
    q2 = []
    c = []
    g_f = []
    y = [data[0]]
    # build relative matrix
    if btype == "d":
        if co is None or isinstance(co,(int,float)):
            raise TypeError("'co' must be list or ndarrar")
        rank = len(co)
        if t is not None:
            print("SurplusWarning: 't' isn't used in 'd' model.")
        # initiate the first matrix
        tmp = []
        temp = []
        a = data[0]/rank
        for i in range(rank):
            tmp.append([])
            temp.append([a])
            for j in range(rank):
                tmp[i].append(a)
        k0 = np.matrix(tmp)
        x0 = np.matrix(temp)
        del tmp,temp
        # build identity matrix
        I = np.matrix(np.eye(rank))
        # build the state transition matrix
        for i in range(rank):
            f.append([])
            for j in range(rank):
                if i != rank-1:
                    if j != i + 1:
                        f[i].append(0)
                    else:
                        f[i].append(1)
                else:
                    f[i].append(co[j])
        f = np.matrix(f)
        # build the self-correlation matrix of devitation of state equation
        for i in range(rank):
            q1.append([])
            for j in range(rank):
                q1[i].append(0)
        q1[rank-1][rank-1] = v1
        q1 = np.matrix(q1)
        # build the variance matrix of observing equation
        q2 = np.matrix(v2)
        # build the observing coefficient matrix
        c = [0 for i in range(rank)]
        c[rank-1] = 1
        c = np.matrix(c)
    else:
        if t is None or isinstance(t,(list,np.ndarray)):
            raise TypeError(f"'t' should be int or float not {type(t)}")
        if co is not None:
            print("SurplusWarning: 'co' isn't used in 'xv' and 'xva' models.")
        # 'xv' model
        if btype=="xv":
            # initiate the first matrix
            x0 = np.matrix([[0],[0]])
            k0 = np.matrix([[0,0],[0,0]])
            # build identity matrix
            I = np.matrix([[1,0],[0,1]])
            # build the state transition matrix
            f = np.matrix([[1,t],[0,1]])
            # build the self-correlation matrix of devitation of state equation
            q1 = np.matrix([[t**2*v1,t*v1],[t*v1,v1]])
            # build the variance matrix of observing equation
            q2 = np.matrix(v2)
            # build the observing coefficient matrix
            c = np.matrix([1,0])
        # 'xva' model
        else:
            # initiate the first matrix
            x0 = np.matrix([[0],[0],[0]])
            k0 = np.matrix([[0,0,0],[0,0,0],[0,0,0]])
            # build identity matrix
            I = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
            # build the state transition matrix
            f = np.matrix([[1,t,t**2/2],[0,1,t],[0,0,1]])
            # build the self-correlation matrix of devitation of state equation
            q1 = np.matrix([[t**4/4*v1,t**3/2*v1,t**2/2*v1],[t**3/2*v1,t**2*v1,t*v1],[t**2/2*t,t*v1,v1]])
            # build the variance matrix of observing equation
            q2 = np.matrix(v2)
            # build the observing coefficient matrix
            c = np.matrix([1,0,0])
    # core of kalman filter
    for i in range(len(data)-1):
            x_n_n_1 = f*x0
            alpha = data[i] - c*x_n_n_1
            k_n_n_1 = f*k0*f.T + q1
            rn = c*k_n_n_1*c.T + q2
            gf = k_n_n_1*c.T*rn.I
            x_n = x_n_n_1 + gf*alpha
            k_n = (I-gf*c)*k_n_n_1
            x0 = x_n
            k0 = k_n
            y.append((c*x0)[0,0])
            g_f.append(gf)
    # returning value
    if re == "y":
        return np.array(y)
    else:
        return np.array(y),np.array(g_f)
