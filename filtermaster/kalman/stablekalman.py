# -*- coding:utf8 -*-

import numpy as np

def stakalman(Fn:np.matrix,Cn:np.matrix,Vn:np.matrix,Wn:np.matrix,X0:np.matrix,Yn):
    """
    This is stable kalman filter.
    This filter is fit for the LTI system, which means that the state transition matrix 'Fn',
    state error matrix 'v1',observing matrix 'Cn', and observing error matrix 'v2' are fixed.
    As we all know, in kalman filter, the gain "Gf' tends to be a stable value, so as the 
    self-correlation matrix 'Kn'. So we can calcute these two stable values ahead. Finally, 
    the calculating formula can get the estimated value for just a single step. 
    Above all, the caculation capacity is low, the only thing you will do is to calculate
    'Kn' by solving a Riccati equation and then caculate 'Gf'. It is fit for embedded system,
    which requaires high speed for getting results.

    Parameters
    ----------
     Fn : the state transition matrix.
    Cn : the observing matrix.
    Vn : the self-correltion matrix of error matrix(v1(n)) in state euqation.
    Wn : the self-correlation matrix of error matrix(v2(n)) in observing matrix.
    X0 : the start condition for iteration.
    Yn : the observing value you've already got, must be np.ndarray of one demontion.
    """
    # judge whether the parameters is type of np.matrix
    if not isinstance(Fn,np.matrix):
        raise TypeError(f"Fn must be matrix, not {type(Fn)}")
    if not isinstance(Cn,np.matrix):
        raise TypeError(f"Cn must be matrix, not {type(Cn)}")
    if not isinstance(X0,np.matrix):
        raise TypeError(f"X0 must be matrix, not {type(X0)}")
    if not isinstance(Vn,np.matrix):
        raise TypeError(f"Vn must be matrix, not {type(Vn)}")
    if not isinstance(Wn,np.matrix):
        raise TypeError(f"Wn must be matrix, not {type(Wn)}")
    # solve matrix K
    K = dare(Fn.T,Cn.T,Vn,Wn)
    # solve stable value R
    R = Cn*K*Cn.T + Wn
    # solve kalman gain
    Gf = K*Cn.T*R.I
    # define identity matrix
    I = np.matrix(np.eye(Fn.shape[0]))
    # define returning matrix
    yn = [Yn[0]]
    # stable kalman filter
    for i in range(1,len(Yn),1):
        X0 = (I - Gf*Cn)*Fn*X0 + Gf*Yn[i]
        yn.append((Cn*X0)[0,0])
    # return 
    return np.array(yn, dtype=np.float64)

def albeta(v1:float,v2:float,x0:np.matrix,T:float,Yn):
    """
    This is alpha-beta filter.

    Parameters
    ----------
    v1 : the varinace of error matrix in state equation
    v2 : the variance of error matrix in observing equation
    X0 : the start condition for iteration.
    T : the gap of the system.
    Yn : the observing value you've already got, must be np.ndarray of one demontion.
    """
    # define the constant value
    lam = v1*2*T**2/v2**2
    # define retuning value
    yn = [Yn[0]]
    # solve alpha and beta
    alpha = -1/8*(lam**2 + 8*lam -(lam+4)*(lam**2 + 8*lam)**0.5)
    beta = 1/4*(lam**2 + 4*lam - lam*(lam**2 + 8*lam)**0.5)
    # define coefficient matrixs of uploading formula
    fn = np.matrix([[1-alpha,(1-alpha)*T],[-beta/T,1-beta]])
    gn = np.matrix([[alpha],[beta/T]])
    # get the results
    for i in range(1,len(Yn),1):
        x0 = fn*x0 + gn*Yn[i]
        yn.append(x0[0,0])
    return np.array(yn,dtype=np.float64)

def albegamma(v1:float,v2:float,x0:np.matrix,T:float,Yn):
    """
    This is alpha-beta-gamma filter.

    Parameters
    ----------
    v1 : the varinace of error matrix in state equation
    v2 : the variance of error matrix in observing equation
    X0 : the start condition for iteration.
    T : the gap of the system.
    Yn : the observing value you've already got, must be np.ndarray of one demontion.
    """
    # define variable values
    lam = v1**2*T**2/v2**2
    b = lam/2 - 3
    c = lam/2 + 3
    p = c - b**2/3
    q = 2*b**3/27 - b*c/3 - 1
    z = ((-q + (q**2 + 4*p**3/27)**0.5)/2)**(1/3)
    s = z - p/(3*z) - b/3
    # solve alpha, beta and gamma
    alpha = 1 - s**2
    beta = 2*(1 - s)**2
    gamma = 2*lam*s
    # define retuning value
    yn = [Yn[0]]
    # define coefficient matrixs of uploading formula
    fn = np.matrix([[1-alpha,(1-alpha)*T,(1-alpha)*T**2/2],[-1*beta/T,1-beta,T*(1-beta/2)],[-1*gamma**2/(2*T**2),-1*gamma/(2*T),1-gamma/4]])
    gn = np.matrix([[alpha],[beta/T],[gamma/(2*T**2)]])
    # get the results
    for i in range(1,len(Yn),1):
        x0 = fn*x0 + gn*Yn[i]
        yn.append(x0[0,0])
    # return
    return np.array(yn,dtype=np.float64)


def dare(A:np.matrix,B:np.matrix,Q:np.matrix,R:np.matrix)->np.matrix:
    """
    This function can solve Riccati equation of discrete type.
    """
    # judge whether the parameters is type of np.matrix
    if not isinstance(A,np.matrix):
        raise TypeError(f"A must be matrix, not {type(A)}")
    if not isinstance(B,np.matrix):
        raise TypeError(f"B must be matrix, not {type(B)}")
    if not isinstance(Q,np.matrix):
        raise TypeError(f"Q must be matrix, not {type(Q)}")
    if not isinstance(R,np.matrix):
        raise TypeError(f"R must be matrix, not {type(R)}")
    # initialize
    P = Q
    # define constant value
    At = A.T
    Bt = B.T
    # start iteration
    while True:
        # calculate the new P matrix
        P_new = At*P*A - At*P*B*(R+Bt*P*B).I*Bt*P*A + Q
        # iteration break condition
        if abs((P_new - P).max()) < np.float64("1e-8"):
            P = P_new
            break
        else:
            P = P_new
            continue
    return P