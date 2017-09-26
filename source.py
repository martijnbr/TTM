# -*- coding: utf-8 -*-

import numpy as np

class PulsedGausianLaser:
    def __init__(self, P, tau, r=1e-3):
        self.P = P / (np.pi * r**2)
        self.r = r      # spot radius
        self.tau = tau
    
    def Intensity(self, t):
        return self.P * np.e**(-2.77 * (t/self.tau)**2)
        #return np.full((len(zLocations)), 0)