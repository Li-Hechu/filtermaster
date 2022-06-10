# -*- coding:utf8 -*-

from numpy import ndarray,array,matrix
from typing import Union
from copy import *

class correError(Exception):

    def __init__(self,errorinfo):
        super().__init__(self)
        self.error = errorinfo

    def __str__(self):
        return self.error

def corre(
    x:Union[list,ndarray],
    y:Union[list,ndarray],
    k0:bool = True,
    out:str ="sequence",
    inv:bool = False,
    ):
    if isinstance(x,Union[list,ndarray]) and isinstance(y,Union[list,ndarray]):
        pass
    else:
        raise correError("xn and yn must be list or ndarray")

    if out == "sequence" or out == "matrix":
        pass
    else:
        raise correError("the value of 'out' must be 'sequence' or 'matrix'")

    xn = []
    yn = []
    if inv:
        for i in range(len(x)-1,-1,-1):
            xn.append(deepcopy(x[i]))
        for i in range(len(y)-1,-1,-1):
            yn.append(deepcopy(y[i]))
    else:
        xn = deepcopy(x)
        yn = deepcopy(y)

    if len(xn) > len(yn):
        yn.extend([0 for i in range(len(xn)-len(yn))])
    elif len(xn) < len(yn):
        xn.extend([0 for i in range(len(yn)-len(xn))])
    else:
        pass
    temp = [0 for i in range(len(xn))]
    temp.extend(yn)
    yn = temp
    yn.extend([0 for i in range(len(xn))])
    
    N = len(xn)
    result= []
    for k in range(-N+1,N):
        num = 0
        for n in range(0,N):
            num += xn[n]*yn[N+n-k]
        result.append(num)

    if out == "sequence":
        if k0:
            return array(deepcopy(result))
        else:
            return array(deepcopy(result[N-1:]))
    elif out == "matrix":
        ans = [[0 for i in range(N)] for j in range(N)]
        for i in range(N-1,2*N-1,1):
            for j in range(i,i-N,-1):
                ans[i+1-N][i-j] = result[j]
        return matrix(deepcopy(ans))
 