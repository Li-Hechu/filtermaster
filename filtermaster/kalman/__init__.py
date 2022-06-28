# -*- coding:utf8 -*-

"""
This module includes kalman filter and its extention.

In most cases, we can just use funtion 'ltikalman' to design the filter,
which provides many parameters to build one.

See each funtion in every python file.
"""

from filtermaster.kalman.kalmanfilter import kalman
from filtermaster.kalman.lti_kalman import ltikalman
from filtermaster.kalman.kalmanpredictor import prekalman
from filtermaster.kalman.informationkalman import infokalman
from filtermaster.kalman.stablekalman import stakalman, albeta, albegamma, dare

