# -*- coding: utf-8 -*-

import numpy as np

class TTM:
    """
    Class used for Two Temperature Model (TTM) simulations
    """

    def __init__(self, _sample, _laser):
        """
        Define material and starting temperatures
        
        Params:
            @Sample _sample      The Sample instance to use
            
        Returns
            Null        
        """
        
        self.sample = _sample
        self.laser = _laser
        
        
    def Evolve(self, t, dt):
        """
        Evolve TTM one time step. Time step size defined in dt
        
        Params:
            @float dt       Time step size
            
        Returns:
            Null
        """        

        # get material parameters
        Ce = self.sample.Ce()
        Cl = self.sample.Cl()
        Ke = self.sample.Ke()
        
        self.CheckStability(Ke, Ce, dt, self.sample.dz)
    
        # determine evolution matrices (electron difusion, el-ph coupling, source)
        el_difus = self.ElectronDifusion(Ke)        
        elph_coup = self.ElPhCoupling()
        source = self.Source(t) 
        
        # determine time evolution
        
        dt_e = np.divide(el_difus - elph_coup + source, Ce) * dt
        dt_l = np.divide(elph_coup, Cl) * dt        
        
        # evolve
        self.sample.Te += dt_e
        self.sample.Tl += dt_l
    
        
    def ElPhCoupling(self):
        g = self.sample.g
        return g * (self.sample.Te - self.sample.Tl)
        
    def ElectronDifusion(self, Ke):
        Te = self.sample.Te
        dif = np.zeros_like(Te)
        
        # Bulk
        for i in range(1, len(Te) -1):
            dif[i] =  Ke[i] * (Te[i+1] - 2*Te[i] + Te[i-1]) 
            dif[i] += (Ke[i+1] - Ke[i-1]) * (Te[i+1] - Te[i-1]) / 4.
            
        # Boundary
        dif[0] = Ke[0] * (Te[1] - Te[0]) + (Ke[1] - Ke[0]) * (Te[1] - Te[0])
        dif[-1] = Ke[-1] * (Te[-2] - Te[-1]) #+ (Ke[-1] - Ke[-2]) * (Te[-1] - Te[-2])
        
        return dif / self.sample.dz**2
                
    def Source(self, t):
        zLocs = self.sample.zLocations
        
        A = self.sample.A
        alp = self.sample.alpha
        I = self.laser.Intensity(t)
        P = [I * np.e**(-alp * zLocs[i]) for i in range(len(zLocs))]
        P = np.array(P)
        return P * A / (1 - np.e**(-alp * self.sample.d))

    def CheckStability(self, Ke, Ce, dt, dz):
        val = np.divide(Ke, Ce) * dt / dz**2
        if(np.max(val) > 0.5):
            raise Exception("Stability check failed, gamma: {0}".format(np.max(val)))
        