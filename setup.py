# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:11:48 2017

@author: Yusen Ye
"""

import os
import sys
import shutil
from subprocess import call
from warnings import warn
from setuptools import setup

setup(name='MSTD',
version='2.0',
package_dir={'': 'src'},
packages=['MSTDlib'],
package_data={
        # And include any *.msg files found in the 'hello' package, too:
        'MSTD': ['data/*','*.txt','data/Celltypes_blood_17_location/*'],
    },

include_package_data=True
#py_modules=['MSTDlib_test_v2'],

)

#get location of setup.py
# setup_dir = os.path.dirname(os.path.realpath(__file__))
#Copy test data
# data_dir = os.path.expanduser('~/.MSTD/data')
# if os.path.isdir(data_dir):
    # shutil.rmtree(data_dir)
# shutil.copytree(setup_dir + '/data/', data_dir)