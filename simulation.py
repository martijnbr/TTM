# -*- coding: utf-8 -*-

import numpy as np

import samples
import ttm
import source


class Simulation:
    
    def __init__(self, sample, source):
        self.times = []
        self.data = []  
        self.source = source
        self.sample = sample
        self.model = ttm.TTM(sample, source)
        
    def Simulate(self, t0, dt, maxTime):
        t=t0
        self.t0 = t0
        self.maxTime = maxTime
        self.dt = dt
            
        while t < maxTime:
            self.times.append(t)
            self.data.append([self.sample.Te.copy(), self.sample.Tl.copy()])

            self.model.Evolve(t, dt)
            t += dt
            
            if(int(t/dt)%1e4 == 0):
                print("t: {0:.3}".format(t))
                
    def DisplayGraph(self, depth):
        import matplotlib.pyplot as plt
        
        Te = []
        Tl = []
        
        for i in range(len(self.data)):
            Te.append(self.data[i][0][depth])
            Tl.append(self.data[i][1][depth])
        x = np.arange(self.t0, self.maxTime, self.dt)
        
        plt.figure()
        plt.title("d= {0}m".format(self.sample.d))
        plt.plot(x, Te, label="Electron temperature")
        plt.plot(x, Tl, label="Latice temperature")
        plt.legend()
        plt.show()
        
    def DisplayEnergyEvolution(self):
        import matplotlib.pyplot as plt
        
        Ee = []
        El = []
        Et = []
        
        scale = 10**4
        
        for i in range(len(self.data) / scale):
            Ee.append(0)
            El.append(0)
            
            for j in range(len(self.data[i*scale][0])):
                Ee[i] += self.sample.Ae * self.data[i*scale][0][j]**2
                El[i] += self.sample.Al * self.data[i*scale][1][j]**2
            
            Et.append(Ee[i] + El[i])
        
        x = np.arange(self.t0, self.maxTime, self.dt * scale)
        
        plt.figure()
        plt.title("d= {0}m".format(self.sample.d))
        plt.plot(x, Ee, label="Electron energy")
        plt.plot(x, El, label="Latice energy")
        plt.plot(x, Et, label="Total energy")
        plt.show()
        
    def GetData(self):
        return self.data
        
    def SaveData(self, fileName):
        return self
    

maxTime = 5e-12
t0 = -100e-15
dt = 1e-17

sample = samples.Au
sample.SetDimensions(100e-9, 100)

source = source.PulsedGausianLaser(1e15, 30e-15, r=1e-3)

sim = Simulation(sample, source)
sim.Simulate(t0, dt, maxTime);


