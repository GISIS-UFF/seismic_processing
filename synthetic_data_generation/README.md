# Synthetic Data Generation

A simple seismic modeling code was implemented to provide a seismic data to test the visualizing, filtering, sorting and stacking algorithms. 

**Requirements**:

    $ pip3 install numpy numba segyio matplotlib     

**Usage**:

    $ NUMBA_THREADING_LAYER='omp' python3 seismic_modeling_2D.py parameters.txt

The entire modeling takes **3110.3 seconds** to be concluded using an AMD Ryzen 5 2500U processing unit. 

## Theory

Introduction 

Main equation

$$ \left(\dfrac{\partial^2 P}{\partial x} + \dfrac{\partial^2 P}{\partial z}\right) - \dfrac{1}{v_p^2}\dfrac{\partial^2 P}{\partial t^2} = f(t)$$

Finite difference method in terms of signal processing

Focusing on the second derivative

$$f''(x) \approx d_x^2(\delta[n+1] - 2\delta[n] + \delta[n-1]) * f[n]$$

Explain 8E2T FDM operators in comparison of other orders

Write discrete equation solution

Dispersion condition

$$f_{max} \le \dfrac{v_{min}}{\alpha \, d_h}$$

Stability condition

$$d_t \le \dfrac{d_h}{\beta \, v_{max}}$$

Source Term: Ricker wavelet

$$f_c = \dfrac{f_{max}}{3\sqrt{\pi}}$$
$$t_0 = \dfrac{2\pi}{f_{max}}$$
$$t_d = t - t_0$$

$$f(t) = (1 - 2\pi(\pi f t_d)^2)e^{-\pi(\pi f t_d)^2}$$ 

Half derivative to remove cilindrical coordinate effects from 2D cartesian modeling

$$\mathcal{F}\left[\dfrac{\partial^\alpha f(t)}{\partial x^\alpha} \right] = (iw)^\alpha F(w)$$

$$f_h(t) = \mathcal{F}^{-1}\left[ (iw)^{\frac{1}{2}} F(w)\right]$$

Cerjan boundary conditions

$$b[i] = e^{-(d(n_b - i))^2}$$

## Experiment

The SEG/EAGE Overthrust model has dimensions of (x,z) = (20, 4.5) km regularly spaced with 25 meters containing 181 samples in depth and 801 samples laterally. We use hundred points in boundary conditions mantaining the model top with no absorption to keep multiples in synthetic data. The entire modeling has 301 shots and 396 receivers with 96 active receivers per shot.

![model_geometry](https://github.com/GISIS-UFF/seismic_processing/assets/44127778/34ae4949-7771-434c-9ce5-8dbe351c4a71)


The modeling time is setted in 3 seconds with 1501 samples and 2 ms of time spacing. The source has the same properties of length as modeling time with 30 Hz of maximum frequency.   

![wavelet](https://github.com/GISIS-UFF/seismic_processing/assets/44127778/0678f46f-c524-49f5-88bf-df887727211d)
