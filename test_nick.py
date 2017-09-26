import numpy as np
import matplotlib.pyplot as plt



plt.close('all')  #purge all existing figure windows


#Plot parameters
plot_params = {'font.size': 50, 
           'legend.fontsize': 32, 
           'xtick.major.pad': 3.5, 
           'ytick.major.pad': 3.5, 
           'grid.linewidth': 2, 
           'grid.alpha': 0.4, 
           'xtick.major.size': 15.0, 
           'xtick.major.width': 2.5, 
           'ytick.major.width': 2.5, 
           'ytick.major.size': 15.0,
           'ytick.major.pad' : 15}
           
plt.rcParams.update(plot_params) 

#experiment parameters
F = 0.02*(93e-6)/(np.pi*(1.5e-4)**2)  #Jm^-2; laser fluence (E_p/A) multiplied by the fraction that is absorbed
tau = 30e-15  #s; pulse duration parameter
ds = [20e-9, 0]  #layer thicknesses in m
t0 = 100e-15  #pulse delay in s (timing of the peak)

#numerical parameters
t_end = 25e-12  #end time in s
Nt = int(3e5)  #number of time steps
Nzs = [9,0]  #number of space steps per layer
Nz = np.sum(Nzs)  #total number of space steps

#material properties
#Gold:
Ae1 = 71.  #Jm^-3K^-2; Ce = Ae*Te      the dot makes it a float; we don't want this to be integer
Ke01 = 318. #Wm^-1K^-1; Ke = Ke0*Te/Tl; thermal conductivity
g1 = 2.1e16 #Wm-3K^-1
alpha1 = (16.3e-9)**(-1)  #inverse optical penetration depth (1/alpha is 1/e-distance for the intensity)
Cl1 = 130*19.3e3  #J/(kgK) * kg/m^3 = J/(m^3K) (specific heat times density). This is the bulk heat capacity of gold and should be representative for the lattice. taken from engineeringtoolbox.com
#eps = -1.05+5.62j   #complex permittivity taken from refractiveindex.info for 400 nm wavelength. not used right now.
#Platinum
Ae2 = 740.
Ke02 = 73.
g2 = 110e16
alpha2 = (11.2e-9)**(-1)
Cl2 = 130*21.4e3
#Tin
#Ae3 = 84. 
#Ke03 = 67.
#g3 = 0
#alpha3 = (13.1e-9)**(-1)  #Calculated using alpha=4pi/lambda * Im(n(w)) where Im(n(w))=k(w) which is the absorption coefficient found at refractiveindex.info for 1300 nm
#Cl3 = 228*7.27e3

#preparation work
z_vec1 = np.linspace(0,ds[0],Nzs[0],endpoint=False)  #array containing all the z-positions in the first layer
z_vec2 = np.linspace(ds[0],ds[0]+ds[1],Nzs[1],endpoint=False)  #second layer
z_vec = np.concatenate((z_vec1,z_vec2))   #total position array
t_vec = np.linspace(0,t_end,Nt,endpoint=False)  #array with time values
dt = (t_vec[-1]-t_vec[0])/(Nt-1)   #time step
dz = (z_vec[-1]-z_vec[0])/(Nz-1)   #z-step
dep_energy_vec = np.zeros(Nt)    #this will track the cumulative deposited laser fluence as a function of time
electron_energy = np.zeros(Nt)   #for cumulative energy in the electron system
lattice_energy = np.zeros(Nt)   #cumulative lattice energy
system_energy = np.zeros(Nt)    #cumulative energy in electron system + lattice
Te_mat,Tl_mat = np.zeros((Nt,Nz)),np.zeros((Nt,Nz))              #pre-allocation for storing temperatures as functions of time and space
Cl = np.concatenate((Cl1*np.ones(Nzs[0]),Cl2*np.ones(Nzs[1])))     #array giving Cl as function of space (step function). below the sme for Ke0, Ae and g
Ke0 = np.concatenate((Ke01*np.ones(Nzs[0]),Ke02*np.ones(Nzs[1])))
Ae = np.concatenate((Ae1*np.ones(Nzs[0]),Ae2*np.ones(Nzs[1])))
g = np.concatenate((g1*np.ones(Nzs[0]),g2*np.ones(Nzs[1])))

def calc_source(t):   #given t, this function returns the power per unit volume (W/m^3) as a function of z, using a gaussian in time and a decaying exponential in z
    return alpha1*F/(np.sqrt(np.pi)*tau)*np.exp(-((t-t0)/tau)**2)*np.exp(-alpha1*z_vec)
    
def calc_T(Te,Tl,Ce,Cl,Ke,P):   #this function runs over z and calculates Te and Tl based on their values in the previous time step
    Te_out,Tl_out = np.zeros(Nz),np.zeros(Nz)   #pre-allocation
    for j in range(Nz):   #we loop over all z
        Tl_out[j] = Tl[j] + dt*g[j]/Cl[j]*(Te[j]-Tl[j])   #lattice temperature is not concerned with its neighbors because we don't consider diffusion here
        if (not j == 0) and (not j == Nz-1):   #we check if we are not dealing with an edge point...
            Te_out[j] = Te[j] + dt/(dz**2*Ce[j])*(Ke[j+1]-Ke[j])*(Te[j+1]-Te[j])+Ke[j]*dt/(dz**2*Ce[j])*(Te[j+1]-2*Te[j]+Te[j-1])-g[j]*dt/Ce[j]*(Te[j]-Tl[j])+dt*P[j]/Ce[j]
        else:  #... if we are dealing with an edge point then we need to get rid of some terms
            if j == 0:  #for the very first (z=0) point, we use Te[j-1] = Te[j], so Te[j+1]-2*Te[j]+Te[j-1] --> Te[j+1]-Te[j]. in this way we enforce the boundary condition that says dT/dz = 0
                Te_out[j] = Te[j] + dt/(dz**2*Ce[j])*(Ke[j+1]-Ke[j])*(Te[j+1]-Te[j])+Ke[j]*dt/(dz**2*Ce[j])*(Te[j+1]-Te[j])-g[j]*dt/Ce[j]*(Te[j]-Tl[j])+dt*P[j]/Ce[j]
            else: #for the very last point, we remove some terms to the same effect of enforcing the boundary condition on this side.
                Te_out[j] = Te[j] +Ke[j]*dt/(dz**2*Ce[j])*(-Te[j]+Te[j-1])-g[j]*dt/Ce[j]*(Te[j]-Tl[j])+dt*P[j]/Ce[j]
    return Te_out,Tl_out
    
def check_stability(Ce,Ke):  #the stability criterion relies on Ce and Ke. Since they are variable, this subroutine is used to check. it is called every time step...
    for j in range(Nz):     # ... and is taken from Burden & Faires, Numerical Analysis, 6th ed, paragraph 12.2.
        C = Ce[j]
        K = Ke[j]
        criterion = K/C*dt/dz**2
        if criterion > 0.5:
            print 'warning! '+str(criterion)+' >= 1/2'
            
def calc_electron_energy(Te_vec): #calculates the energy in the electron system at one moment in time
    energy = 0
    for j in range(Nz):
        energy += 0.5*Ae[j]*(Te_vec[j]**2-300**2)*dz
    return energy
    
def calc_lattice_energy(Tl_vec):  #the same for the lattice energy
    energy = 0
    for j in range(Nz):
        energy += Cl[j]*(Tl_vec[j]-300)*dz
    return energy


dep_energy = 0   #initialize the total deposited energy to 0
for i in range(Nt):  #we run over time
    if not i == 0:
        Te,Tl = Te_mat[i-1],Tl_mat[i-1]   #previous time point values
    else:  #for the very first iteration, we don't refer to the previous values (they don't exist), but rather use 300 K as an initial condition
        Te,Tl = 300*np.ones(Nz),300*np.ones(Nz)
    Ce = Ae*Te
    Ke = Ke0*Te/Tl
    P = calc_source(t_vec[i])   #we get the laser power per unit volume as a function of z
    dep_energy += np.sum(P)*dt*dz    #we track the total deposited fluence (J/m^2) ...
    dep_energy_vec[i] = dep_energy  #... and store it in an array
    check_stability(Ce,Ke)   #this either does nothing or it prints a warning to the console
    Te_mat[i],Tl_mat[i] = calc_T(Te,Tl,Ce,Cl,Ke,P)  #the heart of the calculation. includes enforcing boundary conditions.
    electron_energy[i] = calc_electron_energy(Te_mat[i])   #we store the energy in the electron system, lattice and the total in three separate arrays
    lattice_energy[i] = calc_lattice_energy(Tl_mat[i])
    system_energy[i] = electron_energy[i] + lattice_energy[i]
    
#the rest of the script only does plotting
Z,T = np.meshgrid(1e9*z_vec,1e12*t_vec)   
plt.figure(figsize=(12, 10))
plt.subplot(121)
plt.pcolor(Z,T,Te_mat)
plt.colorbar()
plt.xlabel('z (nm)')
plt.ylabel('t (ps)')
plt.title('Electron temperature (K)')

plt.subplot(122)
plt.pcolor(Z,T,Tl_mat)
plt.colorbar()
plt.xlabel('z (nm)')
plt.title('Lattice temperature (K)')

plt.figure()
plt.plot(1e12*t_vec,electron_energy,'r',label='electron energy')
plt.plot(1e12*t_vec,lattice_energy,'b',label='lattice energy')
plt.plot(1e12*t_vec,system_energy,color='m',label='electron+lattice')
plt.plot(1e12*t_vec,dep_energy_vec,color='k',label='cumulative laser energy')
plt.legend(loc=[0.47,0.4])
plt.xlabel('time (ps)')
plt.ylabel('energy density (J/m^2)')
  


plt.figure()
plt.plot(T[:,5],Te_mat[:,5],'r')
plt.xlabel('t (ps)')
plt.ylabel('Electron temperature (K)')
plt.title('Electron temperature at z=10 nm')
plt.grid()

plt.figure()
plt.plot(T[:,5],Tl_mat[:,5],'r')
plt.xlabel('t (ps)')
plt.ylabel('Lattice temperature (K)')
plt.title('Lattice temperature at z=10 nm')
plt.grid()

