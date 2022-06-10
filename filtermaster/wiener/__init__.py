# -*- coding:utf8 -*-

"""
This module is used to design the linear optimal filter.
It includes the wiener filter, linear prediction filter
and least square filter.
"""

from filtermaster.wiener.wienerfilter import wiener,wieord
from filtermaster.wiener.linpre import forward, backward
from filtermaster.wiener.leasqr import ls
