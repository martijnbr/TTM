# -*- coding: utf-8 -*-

"""
To Do:
    1 Material init nX, nY, nZ are not integers, gives trouble in matrix 
    creation. Casting to int destroys spatial spacing. Solution?: fit dr into 
    sample dimensions until dimensions exceeded

"""

import numpy
    
class Sample:
    
    def __init__(self, g, Ae, Al, Ke0, l, R, T, Te0=300., Tl0=300.):
        # Heat dissipation terms
        self.g = g            # el-ph coupling
        self.Ae = Ae           # Heat capacity constant of electrons. Ce = Ae * Te
        self.Al = Al           # Heat capacity constant of latice. Cl = Al * Tl
        self.Ke0 = Ke0         # Electron difusion constant
        
        # Heat absorbance terms
        self.lamda = l         # Optical penetration depth
        self.alpha = 1. / l      # Absorbed energy density
        self.R = R             # Reflectance
        self.T = T             # Transmittance
        self.A = 1-R-T         # Absorbance
        
        #Starting values
        self.Te0 = Te0
        self.Tl0 = Tl0
        
        # Dimension setting check variable
        self.dimensionsSet = False
                    
    def SetDimensions(self, d, nSteps):
        if self.dimensionsSet:
            raise Exception("Sample dimension can only be set once")
        self.dimensionsSet = True
        
        self.d = d
        self.nZSteps = int(nSteps)
        self.dz = d / nSteps       # d should always be a float
        
        self.Te = numpy.full((self.nZSteps), float(self.Te0))
        self.Tl = numpy.full((self.nZSteps), float(self.Tl0))
        
        self.zLocations = numpy.empty((self.nZSteps))
        for i in range(self.nZSteps):
            self.zLocations[i] = i * self.dz
    
    def Ce(self):
        self.AssertDimensionSet()
        return self.Ae * self.Te
    
    def Cl(self):
        self.AssertDimensionSet()
        #return 130*19.3e3
        return self.Al * self.Tl
    
    def Ke(self):
        self.AssertDimensionSet()
        return self.Ke0 * numpy.divide(self.Te, self.Tl)
    
    def AssertDimensionSet(self):
        if not self.dimensionsSet:
            raise Exception("Sample dimension not yet set")

# Au parameters from Hohlfeld et all 2000
Au = Sample(**{'g':2.0e16, 'Ae':67.96, 'Al':670., 'Ke0':318, 'l':15.3e-9, 
                 'R':0.408,'T':0})
