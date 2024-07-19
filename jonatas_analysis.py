import numpy as np
import matplotlib.pyplot as plt
import toolbox.visualizing as view
from toolbox import managing as mng


input_file = "data/overthrust_synthetic_seismic_data.sgy" # Synthetic data

data = mng.import_sgy_file(input_file)
key = 'cmp'
index = 286
seismic, dt, nt = view.data(data, key, index)
print('seismic ',np.shape(seismic))


Np = 55        # Curvatures
# Nt = 250
Nt = nt

Nh = 48
# dt = 4./1000.
dh = 20. 

p = np.linspace(-0.1,.2,Np)
h = np.linspace(0,(Nh-1)*dh,Nh)

# d = np.zeros((Nt,Nh))
d = seismic
m = np.zeros((Nt,Np))

print('m ', np.shape(m))

m0 = np.zeros((Nt,Np))
f0 = 14

wavelet = view.ricker(dt,f0)

Nw = len(wavelet)
href = h[Nh-1]

m[40:40+Nw,20]=wavelet
m[90:90+Nw,24]=-wavelet
m[95:95+Nw,14]=-wavelet
m[15:15+Nw,4]=wavelet

m[75:75+Nw,12]=-wavelet


# Get the data using the forward operator

#d = view.radon_forward(m,Nt,dt,Nh,h,Np,p,href)    # Make data by forward Radon modelling d = L m
# d = seismic
print('d ',np.shape(d))

plt.figure()
plt.imshow(d, aspect='auto')
plt.show()

# exit()
# Invert the Radon coefficients using LS

m = view.radon_cg(d,m0,Nt,dt,Nh,h,Np,p,href,10)  # Compute m via inversion using Conjugate Gradients 

dp = view.radon_forward(m,Nt,dt,Nh,h,Np,p,href)  # Predict data from inverted m



# -----------------------------------
# Rest of the stuff is for plotting
# -----------------------------------

xcur = 1.2
plt.figure(figsize=(13, 6), dpi=80)
           
plt.subplot(1,3,1)
view.wigb(d,dt,h,xcur,'b')
plt.title('Data')
plt.xlabel('Offset [m]')
plt.ylabel('Time [s]')

plt.subplot(1,3,2)
view.wigb(m,dt,p,xcur,'b')
plt.title('Radon')
plt.xlabel('Curvature [s]')
plt.ylabel('Time [s]')

plt.subplot(1,3,3)
view.wigb(dp,dt,h,xcur,'b')
plt.title('Predicted data')
plt.xlabel('Offset [m]')
plt.ylabel('Time [s]')

plt.tight_layout()
plt.show()