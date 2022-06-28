# -*- coding:utf8 -*-

import numpy as np
from filtermaster.kalman.kalmanfilter import kalman
from filtermaster.kalman.kalmanpredictor import prekalman
from filtermaster.kalman.informationkalman import infokalman
from filtermaster.kalman.stablekalman import stakalman

def ltikalman(data:np.ndarray|list=None,
              btype:str="d",
              v1:float|int=None,
              v2:float|int|np.ndarray|np.matrix=None,
              model:str="n",
              *,co:list=None,
              t:float|int=None
              ):
    """
    This function is designed to solve the typical models in kalman filter through setting a few 
    parameters and it'll build the relavant matrixs for you automatically. One is a system 
    determined by a difference equation(Auroregressive progress, AR), and another is a moving 
    object abided by Newton's Second Law.
    You can use the parameter 'btype' to decide the system type and 'model' to decide which
    kind of kalman filter you'd like to use. There are four models in total: normal kalman,
    kalman predictor, infomation kalman and stable kalman. What I need to shed light on is
    that the stable kalman filter dosen't involve alpha-beta filter or alpha-beta-gamma
    filter, just the universal type. If you want to use alpha-beta or alpha-beta-gamma filter,
    please import thses two functions through sentence 'from filtermaster.kalman import albeta'
    or 'from filtermaster.kalman import albegamma'.

    Parameters
    ----------
    data : observing value of a system
    btype : the model you use, 'd' for differene equation(acquiescent), 'xv' for observing 
            displacement and velocity, and 'xva' will take acceleration into account.
    v1 : variance of devitation of state equation.
    v2 : variance of devitation of observing equation.
    model : which type of kalman filter you'd like to use. 'n' for normal kalman, 'p' for
            kalman predictor, 'i' for information kalman and 's' for stable kalman.
    co : (optional) coefficients of difference equation, only 'd' is given, and the program
            wil automatically build the matrix.
    t : (optional) the sampling interval of the moving system, only 'xv' or 'xva' is given.

    Returns
    ----------
    It'll return the filted values or kalman gain additionally.

    Demonstration (you can copy the code below and run it)
    ----------
    import matplotlib.pyplot as plt
    a = np.linspace(0,8,150)
    b = 2*np.sin(3*np.pi*a+np.pi/3) + np.cos(2*np.pi*a-np.pi/3) - 1.53*np.sin(2*np.pi*a+np.pi/6)
    c = b +  np.random.randn(150)
    y = ltikalman(c,co=[0.23,-1.43,1.98],v1=0.03,v2=9)
    plt.plot(a,c)
    plt.plot(a,b)
    plt.plot(a,y)
    plt.show()
    """
    # process parameters
    if data is None:
        raise TypeError("data shouldn't be None type.")
    if btype == "d" or btype == "xv" or btype == "xva":
        pass
    else:
        raise TypeError("unsupported type '%s'" % btype)
    if model == "n":
        pass
    elif model == "p":
        pass
    elif model == "i":
        pass
    elif model == "s":
        pass
    else:
        raise TypeError(f"unsupported type {model} for 'model'")
    if not isinstance(v1,(int,float)) or not isinstance(v2,(float,int)):
        raise TypeError(f"'v1' and 'v2' both must be int or float, not {type(v1),type(v2)}")
    # define the constant values
    rank = 0
    x0 = []
    f = []
    q1 = []
    q2 = []
    c = []
    g = []
    y = []
    # build relative matrix
    if btype == "d":
        if co is None or isinstance(co,(int,float)):
            raise TypeError("'co' must be list or ndarray")
        rank = len(co)
        if t is not None:
            print("SurplusWarning: 't' isn't used in 'd' model.")
        # initiate the first matrix
        if model != "p":
            x0 = np.matrix([[0] for i in range(rank)])
            x0[rank-1,0] = data[0]
        else:
            x0 = np.matrix([[0] for i in range(rank)])
            x0[rank-1,0] = data[0]
            x0[rank-2,0] = data[1]
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
            # build the state transition matrix
            f = np.matrix([[1,t],[0,1]])
            # build the self-correlation matrix of devitation of state equation
            q1 = np.matrix([[t**4/4*v1,t**3/2*v1],[t**3/2*v1,t**2*v1]])
            # build the variance matrix of observing equation
            q2 = np.matrix(v2)
            # build the observing coefficient matrix
            c = np.matrix([1,0])
        # 'xva' model
        else:
            # initiate the first matrix
            x0 = np.matrix([[0],[0],[0]])
            # build the state transition matrix
            f = np.matrix([[1,t,t**2/2],[0,1,t],[0,0,1]])
            # build the self-correlation matrix of devitation of state equation
            q1 = np.matrix([[t**4/4*v1,t**3/2*v1,t**2/2*v1],[t**3/2*v1,t**2*v1,t*v1],[t**2/2*t,t*v1,v1]])
            # build the variance matrix of observing equation
            q2 = np.matrix(v2)
            # build the observing coefficient matrix
            c = np.matrix([1,0,0])
    # core of kalman filter
    if model == "n":
        y = kalman([f],[c],[q1],[q2],x0,data)
    elif model == "p":
        y = prekalman([f],[c],[q1],[q2],x0,data)
    elif model == "i":
        y = infokalman([f],[c],[q1],[q2],x0,data)
    elif model == "s":
        y = stakalman(f,c,q1,q2,x0,data)
    else:
        return False
    # returning value
    return y
