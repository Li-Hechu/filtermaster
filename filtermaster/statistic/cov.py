# -*- coding:utf8 -*-
from numpy import ndarray,matrix,array
from typing import Union
from copy import deepcopy
from math import fsum

class covError(Exception):
    def __init__(self,errorinfo):
        super().__init__(self)
        self.error = errorinfo

    def __str__(self):
        return self.error


def cov(
    xn:Union[list,ndarray],
    yn:Union[list,ndarray],
    ):
    try:
        return listcov(xn,yn)
    except TypeError:
        return matcov(xn,yn)


def matcov(
    xn:Union[list,ndarray],
    yn:Union[list,ndarray]
    ):
    if len(xn[0]) != len(yn[0]) or len(xn) != len(yn):
        raise covError("xn and yn must have the same dimension")
    row = len(xn)
    column = len(xn[0])
    result = [[0 for i in range(row)] for j in range(column)]
    xn = array(xn).T
    yn = array(yn).T
    for i in range(column):
        for j in range(column):
            result[i][j] = listcov(xn[i],yn[j])
    return matrix(deepcopy(result))


def listcov(
    xn:Union[list,ndarray],
    yn:Union[list,ndarray]
    ):
    if len(xn) != len(yn):
        raise covError("xn and yn must have the same length")
    N = len(xn)
    
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(N):
        sum1 += xn[i]
        sum2 += yn[i]
    xmean = float(sum1)/N
    ymean = float(sum2)/N
    for i in range(N):
        sum3 += fsum([(xn[i]-xmean)])*fsum([(yn[i]-ymean)])
    return sum3/len(xn)
