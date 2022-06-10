# -*- coding:utf8 -*-

"""
This module is used to calculate the results in mathematical statistics.
And the most common calculation are the correlation and covariance. Thus,
this module provide two functions ,'correlation' and 'covariance' to finish
the work.
"""

from filtermaster.statistic.corre import *
from filtermaster.statistic.cov import *

def correlation(
    xn:Union[list,ndarray],
    yn:Union[list,ndarray],
    k0:bool = True,
    out:str ="sequence",
    inv:bool = False
    ):
    """
    This function can calculate the self-correlation or cross-correlation sequence
    as well as matrix.
    If 'xn' and 'yn' are the same sequences, it'll return the self-correlation value, or 
    the cross-correlation value.

    Parameters
    ----------
    xn : one time series of forward direction.
    yn : another time series of forward direction.
    k0 : the parameter for output control. It determines whether only returning the
           first half of the sequence, only output model is 'sequence'.
    out : the returning type of this function, 'sequence' for a series and 'matrix' for a
            matrix.
    inv : whether inverse the time series.

    Returning values
    ----------
    It'll return np.array(one demontion) or np.matrix(two demontion), determined by the 'out'
    parameter.

    Demonstration
    ----------
    a = [1,2,4,2,1]
    b = [1,2,1]
    print(correlation(a,b,out="matrix"))
    >>> [[9, 12, 9, 4, 1], 
        [4, 9, 12, 9, 4], 
        [1, 4, 9, 12, 9], 
        [0, 1, 4, 9, 12], 
        [0, 0, 1, 4, 9]]
    
    a = [1,2,4,2,1]
    b = [1,2,1]
    print(correlation(a,b,k0=True,out="sequence"))
    >>> [0,0,1,4,9,12,9,4,1]
    print(correlation(a,b,k0=False,out="sequence"))
    >>> [9,12,9,4,1]
    """
    return corre(xn,yn,k0,out,inv)

def covariance(
    xn:Union[list,ndarray],
    yn:Union[list,ndarray],
    ):
    """
    This function an calculate the covariance seuqnece or matrix, determined by
    the demontions of 'xn' and 'yn'. Once the two paameters are conveied into the
    function, it can automatically recognise the demontions.

    Parameters
    ----------
    xn : one time series of forward diretion.
    yn : another time series of forward direction.

    Returning values
    ----------
    If it is one demontion, a float will be returned, or return a matrix.

    Demonstration
    =========
    a = [[1,2,3,1],[-3,1,2,0],[0,0,1,-3],[0,3,7,1]]
    b = [[3,-1,1,2],[1,0,2,7],[-1,1,2,-3],[0,2,-3,9]]
    print(covariance(a,b))
    >>> [[0.375, 0.0, -1.0, -2.875],
        [0.625, 0.25, -2.0, 3.875],
        [0.0625, 1.375, -4.625, 7.8125],
        [1.6875, -0.375, -1.875, 5.9375]]

    a = [1,2,3,1]
    b = [3,-1,1,2]
    print(covariance(a,b))
    >>> -0.6875
    """
    return cov(xn,yn)