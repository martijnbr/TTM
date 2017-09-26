# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:06:09 2017

@author: bruggink
"""

def test(v1, *args, **kwargs):
    print(v1)
    print(kwargs)
    
test(**{'v1':5, 'v2':2})