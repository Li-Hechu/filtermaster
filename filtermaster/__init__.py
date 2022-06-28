# -*- coding:utf8 -*-

"""
The filtermaster module includes many types of filters and the file tree is below:

root filtermaster
     --adaptfilt
       --__init__.py
       --granddescend.py
       --leameansqr.py
       --releasqr.py
     --diffeuqtion
       --__init__.py
       --difequa.py
     --kalman
       --__init__.py
       --informationkalman.py
       --kalmnafilter.py
       --kalmanpredictor.py
       --lti_kalman.py
       --stablekalman.py
     --statistic
       --__init__.py
       --corre.py
       --cov.py
     --wiener
       --__init__.py
       --leasqr.py
       --linpre.py
       --wienerfilter.py

You can import the entire mudule through the sentence 'import filtermaster as fm',
and if you just want to use a specific module in filtermaster, you can use
'from filtermaster import XXX' to import it.

This module includes kalman filter, wiener filter and self-adaption filters
including grand descend filter, least mean square(LMS) filter 
and recursive least square(RLS) filter.

See each of them in every module.
"""

from filtermaster.kalman import *
from filtermaster.statistic import *
from filtermaster.wiener import *
from filtermaster.diffequation import *
from filtermaster.adaptfilt import *
