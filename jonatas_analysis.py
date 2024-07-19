
"""
Created on Mon Feb 16 12:00:56 2015

@author: msacchi
"""

from numba import jit
import numpy as np
import matplotlib.pyplot as plt

import sys  
sys.path.append('./src') 


@jit(nopython=True)
def radon_adjoint(d,Nt,dt,Nh,h,Np,p,href):

# Adjoint Time-domain Parabolic Radon Operator

# d(nt,nh): data
# dt      : sampling interval
# h(Nh)   : offset
# p(Np)   : curvature of parabola
# href    : reference offset
# Returns m(nt,np) the Radon coefficients 

# M D Sacchi, 2015,  Email: msacchi@ualberta.ca

 
    m=np.zeros((Nt,Np))

    for itau in range(0,Nt):
        for ih in range(0,Nh):
            for ip in range(0,Np):
                t = (itau)*dt+p[ip]*(h[ih]/href)**2
                it = int(t/dt)
                if it<Nt:
                    if it>0:
                        m[itau,ip] +=  d[it,ih]
    
    return m
        

def radon_forward(m,Nt,dt,Nh,h,Np,p,href):

# Forward Time-domain Parabolic Radon Transform

# m(nt,nh): Radon coefficients 
# dt      : sampling interval
# h(Nh)   : offset
# p(Np)   : curvature of parabola
# href    : reference offset
# Returns d(nt,nh) the synthetized data from the Radon coefficients

# M D Sacchi, 2015,  Email: msacchi@ualberta.ca

    d=np.zeros((Nt,Nh))
      
    for itau in range(0,Nt):
        for ih in range(0,Nh):
            for ip in range(0,Np):
                t = (itau)*dt+p[ip]*(h[ih]/href)**2
                it=int(t/dt)
                if it<Nt:
                    if it>=0:
                        d[it,ih] +=  m[itau,ip]                   
    return d
    

def radon_cg(d,m0,Nt,dt,Nh,h,Np,p,href,Niter):
         
# LS Radon transform. Finds the Radon coefficients by minimizing
# ||L m - d||_2^2 where L is the forward Parabolic Radon Operator.
# The solution is found via CGLS with operators L and L^T applied on the
# flight

# M D Sacchi, 2015,  Email: msacchi@ualberta.ca

    m = m0  
    
    s = d-radon_forward(m,Nt,dt,Nh,h,Np,p,href) # d - Lm
    pp = radon_adjoint(s,Nt,dt,Nh,h,Np,p,href)  # pp = L's 
    r = pp
    q = radon_forward(pp,Nt,dt,Nh,h,Np,p,href)
    old = np.sum(np.sum(r*r))
    print("iter","  res")
    
    for k in range(0,Niter):
         alpha = np.sum(np.sum(r*r))/np.sum(np.sum(q*q))
         m +=  alpha*pp
         s -=  alpha*q
         r = radon_adjoint(s,Nt,dt,Nh,h,Np,p,href)  # r= L's
         new = np.sum(np.sum(r*r))
         print(k, new)
         beta = new/old
         old = new
         pp = r + beta*pp
         q = radon_forward(pp,Nt,dt,Nh,h,Np,p,href) # q=L pp
           
    return m 

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 20:31:49 2015

@author: msacchi
"""
import numpy as np
import matplotlib.pyplot as plt

def wigb(d,dt,h,xcur,color):

# Plot wiggle seismic plot (python version of Xin-gong Li faumous wigb.m)

    [nt,nx] = np.shape(d)
    dmax = np.max(d)
    d = d/dmax
    t = np.linspace(0,(nt-1)*dt,nt)
    tmax = np.amax(t)
    hmin = np.amin(h)
    hmax = np.amax(h)
   
    c = xcur*np.mean(np.diff(h))

    plt.axis([hmin-2*c, hmax+2*c, tmax, 0.])
    d[nt-1,:]=0
    d[0,:]=0
    for k in range(0,nx):
        s =d[:,k]*c
        plt.plot(s+h[k], t, color,linewidth=1)
        b = h[k]+s.clip(min=0) 
        plt.fill(b,t,color)

    return
    
def ricker(dt,f0):    
         
#Ricker wavelet of central frequency f0 sampled every dt seconds

# M D Sacchi, 2015,  Email: msacchi@ualberta.ca

    nw = 2.5/f0/dt
    nw = 2*int(nw/2)
    nc = int(nw/2)
    a = f0*dt*3.14159265359
    n = a*np.arange(-nc,nc)
    b = n**2
    return (1-2*b)*np.exp(-b)


Np = 55        # Curvatures
Nt = 250
Nh = 25
dt = 4./1000.
dh = 20. 

p = np.linspace(-0.1,.2,Np)
h = np.linspace(0,(Nh-1)*dh,Nh)

d = np.zeros((Nt,Nh))
m = np.zeros((Nt,Np))

m0 = np.zeros((Nt,Np))
f0 = 14

wavelet = ricker(dt,f0)

Nw = len(wavelet)
href = h[Nh-1]

m[40:40+Nw,20]=wavelet
m[90:90+Nw,24]=-wavelet
m[95:95+Nw,14]=-wavelet
m[15:15+Nw,4]=wavelet

m[75:75+Nw,12]=-wavelet

# Get the data using the forward operator

d = radon_forward(m,Nt,dt,Nh,h,Np,p,href)    # Make data by forward Radon modelling d = L m

# Invert the Radon coefficients using LS

m = radon_cg(d,m0,Nt,dt,Nh,h,Np,p,href,10)  # Compute m via inversion using Conjugate Gradients 

dp = radon_forward(m,Nt,dt,Nh,h,Np,p,href)  # Predict data from inverted m

# -----------------------------------
# Rest of the stuff is for plotting
# -----------------------------------

xcur = 1.2
plt.figure(figsize=(13, 6), dpi=80)
           
plt.subplot(1,3,1)
wigb(d,dt,h,xcur,'b')
plt.title('Data')
plt.xlabel('Offset [m]')
plt.ylabel('Time [s]')

plt.subplot(1,3,2)
wigb(m,dt,p,xcur,'b')
plt.title('Radon')
plt.xlabel('Curvature [s]')
plt.ylabel('Time [s]')

plt.subplot(1,3,3)
wigb(dp,dt,h,xcur,'b')
plt.title('Predicted data')
plt.xlabel('Offset [m]')
plt.ylabel('Time [s]')

plt.tight_layout()
plt.show()