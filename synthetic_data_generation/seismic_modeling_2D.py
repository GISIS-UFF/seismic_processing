import numpy as np
import segyio as sgy
import matplotlib.pyplot as plt

from os import system, path
from time import time
from sys import argv

from numba import njit, prange

class Seismic_Modeling:

    def __init__(self):

        if len(argv) < 2 or not path.isfile(argv[1]):
            print("Parameter file not found!")
            print("To run this code properlly:")
            print("\n$ NUMBA_THREADING_LAYER='omp' python3 seismic_modeling_2D.py parameters.txt\n")
            exit()    

        self.file = argv[1]

        self.set_parameters()
        
        self.set_model()
        self.set_geometry()
        self.set_boundary()
        self.set_wavelet()
        self.set_damper()

        self.plot_wavelet()
        self.plot_geometry()

        ti = time()
        self.fdm_propagation()
        tf = time()

        print(f"\nRuntime = {tf - ti:.3f} s")

        self.build_segy_file()

    def catch_parameter(self, target):
        
        file = open(self.file, "r")
        
        for line in file.readlines():
            if line[0] != "#":
                splitted = line.split()
                if len(splitted) != 0:
                    if splitted[0] == target: 
                        return splitted[2]         
        file.close()

    def set_parameters(self):

        self.fdm_stencil = 4    

        self.nx = int(self.catch_parameter("n_samples_x"))
        self.nz = int(self.catch_parameter("n_samples_z"))
        self.nt = int(self.catch_parameter("time_samples"))

        self.dh = float(self.catch_parameter("model_spacing"))
        self.dt = float(self.catch_parameter("time_spacing"))

        self.fmax = float(self.catch_parameter("max_frequency"))

        self.nb = int(self.catch_parameter("boundary_samples"))
        self.factor = float(self.catch_parameter("attenuation_factor")) 

        self.vp_file = self.catch_parameter("velocity_model_file")

        self.total_shots = int(self.catch_parameter("total_shots"))
        self.total_nodes = int(self.catch_parameter("total_nodes"))

        self.nxx = self.nx + 2*self.nb
        self.nzz = self.nz + self.nb+self.fdm_stencil

        self.wavelet = np.zeros(self.nt)
        
        self.vp = np.zeros((self.nz, self.nx))
        self.Vp = np.zeros((self.nzz, self.nxx))

        self.damp2D = np.ones_like(self.Vp)

    def set_geometry(self):

        ishot = float(self.catch_parameter("shot_beg"))
        fshot = float(self.catch_parameter("shot_end"))
        selev = float(self.catch_parameter("shot_elevation"))

        self.sx = np.linspace(ishot, fshot, self.total_shots)
        self.sz = np.ones(self.total_shots) * selev 

        inode = float(self.catch_parameter("node_beg"))
        fnode = float(self.catch_parameter("node_end"))
        gelev = float(self.catch_parameter("node_elevation"))

        self.rx = np.linspace(inode, fnode, self.total_nodes)
        self.rz = np.ones(self.total_nodes) * gelev 

    def set_model(self):
        
        data = np.fromfile(self.vp_file, dtype = np.float32, count = self.nx * self.nz)            
        self.vp = np.reshape(data, [self.nz, self.nx], order = "F")

    def set_wavelet(self):

        fc = self.fmax / (3.0 * np.sqrt(np.pi))    

        self.tlag = 2.0*np.pi/self.fmax

        for n in range(self.nt):
            aux = np.pi*((n*self.dt - self.tlag)*fc*np.pi) ** 2.0 
            self.wavelet[n] = (1.0 - 2.0*aux)*np.exp(-aux)

        w = 2.0*np.pi*np.fft.fftfreq(self.nt, self.dt)

        self.wavelet = np.real(np.fft.ifft(np.fft.fft(self.wavelet)*np.sqrt(complex(0,1)*w))) 

    def set_boundary(self):

        self.Vp[self.fdm_stencil:self.nzz-self.nb, self.nb:self.nxx-self.nb] = self.vp[:,:]

        self.Vp[:self.fdm_stencil, self.nb:self.nxx-self.nb] = self.vp[0,:]

        for i in range(self.nb):
            self.Vp[self.nzz-i-1, self.nb:self.nxx-self.nb] = self.vp[-1,:]

        for j in range(self.nb):
            self.Vp[:,j] = self.Vp[:,self.nb]
            self.Vp[:,self.nxx-j-1] = self.Vp[:,-(self.nb+1)]

        self.vmax = np.max(self.Vp)
        self.vmin = np.min(self.Vp)

    def set_damper(self):
        
        damp1D = np.zeros(self.nb)

        for i in range(self.nb):   
            damp1D[i] = np.exp(-(self.factor*(self.nb - i))**2.0)

        for i in range(0, self.nzz-self.nb):
            self.damp2D[i, :self.nb] = damp1D
            self.damp2D[i, self.nxx-self.nb:self.nxx] = damp1D[::-1]

        for j in range(self.nb, self.nxx-self.nb):
            self.damp2D[self.nzz-self.nb:self.nzz, j] = damp1D[::-1]    

        for i in range(self.nb):
            self.damp2D[self.nzz-self.nb-1:self.nzz-i-1, i] = damp1D[i]
            self.damp2D[self.nzz-i-1, i:self.nb] = damp1D[i]

            self.damp2D[self.nzz-self.nb-1:self.nzz-i, self.nxx-i-1] = damp1D[i]
            self.damp2D[self.nzz-i-1, self.nxx-self.nb-1:self.nxx-i] = damp1D[i]

    def fdm_propagation(self):

        for self.shot_index in range(len(self.sx)):

            sIdx = int(self.sx[self.shot_index] / self.dh) + self.nb
            sIdz = int(self.sz[self.shot_index] / self.dh) + self.fdm_stencil    

            self.seismogram = np.zeros((self.nt, self.total_nodes))

            self.Upas = np.zeros_like(self.Vp)
            self.Upre = np.zeros_like(self.Vp)
            self.Ufut = np.zeros_like(self.Vp)

            for self.time_index in range(self.nt):

                self.show_modeling_status()

                self.Upre[sIdz,sIdx] += self.wavelet[self.time_index] / (self.dh*self.dh)

                laplacian = fdm_8E2T_scalar2D(self.Upre, self.nxx, self.nzz, self.dh)
                
                self.Ufut = laplacian*(self.dt*self.dt*self.Vp*self.Vp) - self.Upas + 2.0*self.Upre

                self.Upas = self.Upre * self.damp2D     
                self.Upre = self.Ufut * self.damp2D

                self.get_seismogram()

            self.seismogram.flatten("F").astype(np.float32, order = "F").tofile(f"seismogram_{self.nt}x{self.total_nodes}_shot_{self.shot_index+1}.bin")

    def show_modeling_status(self):

        beta = 2
        alpha = 3

        if self.time_index % int(self.nt / 100) == 0:
            system("clear")
            print("Seismic modeling in constant density acoustic media\n")
            
            print(f"Model samples: (z, x) = ({self.nz}, {self.nx})")
            print(f"Model spacing = {self.dh} m\n")

            print(f"Max velocity = {self.vmax:.1f} m/s")
            print(f"Min velocity = {self.vmin:.1f} m/s\n")
            
            print(f"Time samples = {self.nt} s")
            print(f"Time spacing = {self.dt} s")
            print(f"Max frequency = {self.fmax} Hz")

            print(f"\nHighest frequency without dispersion:")
            print(f"Frequency <= {self.vmin / (alpha*self.dh):.1f} Hz")

            print(f"\nHighest time step without instability:")
            print(f"Time step <= {self.dh / (beta*self.vmax):.5f} s")

            print(f"\nShot {self.shot_index+1} of {self.total_shots}")        
            print(f"\nModeling progress = {100*(self.time_index+1)/self.nt:.0f} %")        

    def get_seismogram(self):
        
        for k in range(self.total_nodes):
            rIdx = int(self.rx[k] / self.dh) + self.nb
            rIdz = int(self.rz[k] / self.dh) + self.fdm_stencil
            self.seismogram[self.time_index, k] = self.Upre[rIdz, rIdx]
    
    def plot_wavelet(self):
        
        nw = int(4*self.tlag/self.dt)
        t = np.arange(nw) * self.dt
        f = np.fft.fftfreq(self.nt, self.dt)

        fwavelet = np.fft.fft(self.wavelet)

        fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (10,8))

        ax[0].plot(t, self.wavelet[:nw] * 1.0 / np.max(self.wavelet))
        ax[0].set_xlim([0, nw*self.dt])
        ax[0].set_title("Wavelet filtered by half derivative technique", fontsize = 18)    
        ax[0].set_xlabel("Time [s]", fontsize = 15)    
        ax[0].set_ylabel("Normalized amplitude", fontsize = 18)    

        ax[1].plot(f, np.abs(fwavelet) * 1.0 / np.max(np.abs(fwavelet)), "o")
        ax[1].set_xlim([0, self.fmax])
        ax[1].set_title("Wavelet in frequency domain", fontsize = 18)    
        ax[1].set_xlabel("Frequency [Hz]", fontsize = 15)    
        ax[1].set_ylabel("Normalized amplitude", fontsize = 18)    

        fig.tight_layout()
        plt.savefig("wavelet.png")

    def plot_geometry(self):

        xloc = np.linspace(self.nb, self.nx + self.nb - 1, 11, dtype = int) 
        xlab = np.array((xloc - self.nb)*self.dh, dtype = int) 

        zloc = np.linspace(0, self.nz - 1, 7, dtype = int) 
        zlab = np.array(zloc*self.dh, dtype = int) 

        fig, ax = plt.subplots(1,1, figsize = (20,4.5))

        img = ax.imshow(self.Vp, aspect = "auto", cmap = "jet")
        cbar = fig.colorbar(img, ax = ax)
        cbar.set_label("Velocity [m/s]", fontsize = 12)

        ax.plot(np.arange(self.nx)+self.nb, np.ones(self.nx)+self.fdm_stencil-1, "-k")
        ax.plot(np.arange(self.nx)+self.nb, np.ones(self.nx)+self.fdm_stencil+self.nz-1, "-k")
        ax.plot(np.ones(self.nz)+self.nb-1, np.arange(self.nz)+self.fdm_stencil, "-k")
        ax.plot(np.ones(self.nz)+self.nb+self.nx-1, np.arange(self.nz)+self.fdm_stencil, "-k")

        ax.scatter(self.rx / self.dh + self.nb, self.rz / self.dh + self.fdm_stencil, color = "black")
        ax.scatter(self.sx / self.dh + self.nb, self.sz / self.dh + self.fdm_stencil, color = "red")
        
        ax.set_xticks(xloc)
        ax.set_xticklabels(xlab)

        ax.set_yticks(zloc)
        ax.set_yticklabels(zlab)

        ax.set_title("Model delimitations and geometry", fontsize = 18)
        ax.set_xlabel("Distance [m]", fontsize = 15)
        ax.set_ylabel("Depth [m]", fontsize = 15)

        fig.tight_layout()
        plt.savefig("model_geometry.png")

    def build_segy_file(self):

        spread = 96     
        spacing = 50.0

        SRCi = np.zeros(spread*self.total_shots, dtype = int)
        RECi = np.zeros(spread*self.total_shots, dtype = int)

        CMPi = np.zeros(spread*self.total_shots, dtype = int)
        CMPx = np.zeros(spread*self.total_shots, dtype = float)
        CMPy = np.zeros(spread*self.total_shots, dtype = float)

        OFFt = np.zeros(spread*self.total_shots, dtype = float)

        sx = np.zeros(spread*self.total_shots, dtype = float)
        sy = np.zeros(spread*self.total_shots, dtype = float)
        sz = np.zeros(spread*self.total_shots, dtype = float)

        rx = np.zeros(spread*self.total_shots, dtype = float)
        ry = np.zeros(spread*self.total_shots, dtype = float)
        rz = np.zeros(spread*self.total_shots, dtype = float)

        tsl = np.arange(spread*self.total_shots, dtype = int) + 1

        seismic = np.zeros((self.nt, spread*self.total_shots), dtype = np.float32)

        for i in range(self.total_shots):

            actives = slice(i*spread, i*spread + spread)

            data = np.fromfile(f"seismogram_1501x397_shot_{i+1}.bin", dtype = np.float32, count = self.nt*self.total_nodes)
            data = np.reshape(data, [self.nt, self.total_nodes], "F")

            seismic[:,actives] = data[:,i:i+spread]

            sx[actives] = self.sx[i]
            sy[actives] = 0.0
            sz[actives] = self.sz[i]

            rx[actives] = self.rx[i:i+spread]
            ry[actives] = 0.0
            rz[actives] = self.rz[i:i+spread]

            SRCi[actives] = i+1
            RECi[actives] = np.arange(spread, dtype = int) + i + 1
            CMPi[actives] = np.arange(spread, dtype = int) + 2*i + 1

            CMPx[actives] = sx[actives] - 0.5*(sx[actives] - rx[actives])    
            CMPy[actives] = sy[actives] - 0.5*(sy[actives] - ry[actives])

            OFFt[actives] = np.arange(spread)*spacing - 0.5*(spread-1)*spacing   

        sgy.tools.from_array2D("overthrust_synthetic_seismic_data.sgy", seismic.T, dt = int(self.dt*1e6))
        
        data = sgy.open("overthrust_synthetic_seismic_data.sgy", "r+", ignore_geometry = True)

        data.bin[sgy.BinField.JobID]                 = 1
        data.bin[sgy.BinField.LineNumber]            = 1
        data.bin[sgy.BinField.ReelNumber]            = 1
        data.bin[sgy.BinField.Interval]              = int(self.dt*1e6)
        data.bin[sgy.BinField.IntervalOriginal]      = int(self.dt*1e6)
        data.bin[sgy.BinField.Samples]               = self.nt
        data.bin[sgy.BinField.SamplesOriginal]       = self.nt
        data.bin[sgy.BinField.Format]                = 1
        data.bin[sgy.BinField.SortingCode]           = 1
        data.bin[sgy.BinField.MeasurementSystem]     = 1
        data.bin[sgy.BinField.ImpulseSignalPolarity] = 1

        for idx, key in enumerate(data.header):

            print(f"Adjusting trace header: {idx+1} of {data.tracecount} traces concluded")

            key.update({sgy.TraceField.TRACE_SEQUENCE_LINE                    : int(tsl[idx])      })
            key.update({sgy.TraceField.TRACE_SEQUENCE_FILE                    : int(SRCi[idx])     })
            key.update({sgy.TraceField.FieldRecord                            : int(SRCi[idx])     })
            key.update({sgy.TraceField.TraceNumber                            : int(RECi[idx])     })
            key.update({sgy.TraceField.EnergySourcePoint                      : 0                  })
            key.update({sgy.TraceField.CDP                                    : int(CMPi[idx])     })
            key.update({sgy.TraceField.CDP_TRACE                              : int(CMPi[idx])     })
            key.update({sgy.TraceField.TraceIdentificationCode                : int(tsl[idx])      })
            key.update({sgy.TraceField.NSummedTraces                          : 0                  })
            key.update({sgy.TraceField.NStackedTraces                         : 0                  })
            key.update({sgy.TraceField.DataUse                                : 0                  })
            key.update({sgy.TraceField.offset                                 : int(OFFt[idx]*100) })
            key.update({sgy.TraceField.ReceiverGroupElevation                 : int(rz[idx]*100)   })
            key.update({sgy.TraceField.SourceSurfaceElevation                 : int(sz[idx]*100)   })
            key.update({sgy.TraceField.SourceDepth                            : 0                  })
            key.update({sgy.TraceField.ReceiverDatumElevation                 : 0                  })
            key.update({sgy.TraceField.SourceDatumElevation                   : 0                  })
            key.update({sgy.TraceField.SourceWaterDepth                       : 0                  })
            key.update({sgy.TraceField.ElevationScalar                        : 100                })
            key.update({sgy.TraceField.SourceGroupScalar                      : 100                })
            key.update({sgy.TraceField.SourceX                                : int(sx[idx]*100)   })
            key.update({sgy.TraceField.SourceY                                : int(sy[idx]*100)   })
            key.update({sgy.TraceField.GroupX                                 : int(rx[idx]*100)   })
            key.update({sgy.TraceField.GroupY                                 : int(ry[idx]*100)   })
            key.update({sgy.TraceField.CoordinateUnits                        : 1                  })
            key.update({sgy.TraceField.WeatheringVelocity                     : 0                  })
            key.update({sgy.TraceField.SubWeatheringVelocity                  : 0                  })
            key.update({sgy.TraceField.SourceUpholeTime                       : 0                  })
            key.update({sgy.TraceField.GroupUpholeTime                        : 0                  })
            key.update({sgy.TraceField.SourceStaticCorrection                 : 0                  })
            key.update({sgy.TraceField.GroupStaticCorrection                  : 0                  })
            key.update({sgy.TraceField.TotalStaticApplied                     : 0                  })
            key.update({sgy.TraceField.LagTimeA                               : 0                  })
            key.update({sgy.TraceField.LagTimeB                               : 0                  })
            key.update({sgy.TraceField.DelayRecordingTime                     : 0                  })
            key.update({sgy.TraceField.MuteTimeStart                          : 0                  })
            key.update({sgy.TraceField.MuteTimeEND                            : 0                  })
            key.update({sgy.TraceField.TRACE_SAMPLE_COUNT                     : self.nt            })
            key.update({sgy.TraceField.TRACE_SAMPLE_INTERVAL                  : int(self.dt*1e6)   })
            key.update({sgy.TraceField.GainType                               : 1                  })
            key.update({sgy.TraceField.InstrumentGainConstant                 : 0                  })
            key.update({sgy.TraceField.InstrumentInitialGain                  : 0                  })
            key.update({sgy.TraceField.Correlated                             : 0                  })
            key.update({sgy.TraceField.SweepFrequencyStart                    : 0                  })
            key.update({sgy.TraceField.SweepFrequencyEnd                      : 0                  })
            key.update({sgy.TraceField.SweepLength                            : 0                  })
            key.update({sgy.TraceField.SweepType                              : 0                  })
            key.update({sgy.TraceField.SweepTraceTaperLengthStart             : 0                  })
            key.update({sgy.TraceField.SweepTraceTaperLengthEnd               : 0                  })
            key.update({sgy.TraceField.TaperType                              : 0                  })
            key.update({sgy.TraceField.AliasFilterFrequency                   : 0                  })
            key.update({sgy.TraceField.AliasFilterSlope                       : 0                  })
            key.update({sgy.TraceField.NotchFilterFrequency                   : 0                  })
            key.update({sgy.TraceField.NotchFilterSlope                       : 0                  })
            key.update({sgy.TraceField.LowCutFrequency                        : 0                  })
            key.update({sgy.TraceField.HighCutFrequency                       : 0                  })
            key.update({sgy.TraceField.LowCutSlope                            : 0                  })
            key.update({sgy.TraceField.HighCutSlope                           : 0                  })
            key.update({sgy.TraceField.YearDataRecorded                       : 0                  })
            key.update({sgy.TraceField.DayOfYear                              : 0                  })
            key.update({sgy.TraceField.HourOfDay                              : 0                  })
            key.update({sgy.TraceField.MinuteOfHour                           : 0                  })
            key.update({sgy.TraceField.SecondOfMinute                         : 0                  })
            key.update({sgy.TraceField.TimeBaseCode                           : 1                  })
            key.update({sgy.TraceField.TraceWeightingFactor                   : 0                  })
            key.update({sgy.TraceField.GeophoneGroupNumberRoll1               : 0                  })
            key.update({sgy.TraceField.GeophoneGroupNumberFirstTraceOrigField : 0                  })
            key.update({sgy.TraceField.GeophoneGroupNumberLastTraceOrigField  : 0                  })
            key.update({sgy.TraceField.GapSize                                : 0                  })
            key.update({sgy.TraceField.OverTravel                             : 0                  })
            key.update({sgy.TraceField.CDP_X                                  : int(CMPx[idx]*100) })
            key.update({sgy.TraceField.CDP_Y                                  : int(CMPy[idx]*100) })
            key.update({sgy.TraceField.INLINE_3D                              : 0                  })
            key.update({sgy.TraceField.CROSSLINE_3D                           : 0                  })
            key.update({sgy.TraceField.ShotPoint                              : 0                  })
            key.update({sgy.TraceField.ShotPointScalar                        : 0                  })
            key.update({sgy.TraceField.TraceValueMeasurementUnit              : 0                  })
            key.update({sgy.TraceField.TransductionConstantMantissa           : 0                  })
            key.update({sgy.TraceField.TransductionConstantPower              : 0                  })
            key.update({sgy.TraceField.TraceIdentifier                        : 0                  })
            key.update({sgy.TraceField.ScalarTraceHeader                      : 0                  })
            key.update({sgy.TraceField.SourceType                             : 0                  })
            key.update({sgy.TraceField.SourceEnergyDirectionMantissa          : 0                  })
            key.update({sgy.TraceField.SourceEnergyDirectionExponent          : 0                  })
            key.update({sgy.TraceField.SourceMeasurementUnit                  : 0                  })
            key.update({sgy.TraceField.UnassignedInt1                         : 0                  })
            key.update({sgy.TraceField.UnassignedInt2                         : 0                  })
            
        data.close()
        print("\nFile \033[31moverthrust_synthetic_seismic_data.sgy\033[m is ready!\n")
        system("rm *.bin")

@njit(parallel = True)
def fdm_8E2T_scalar2D(Upre, nxx, nzz, dh):

    laplacian = np.zeros_like(Upre)

    for index in prange(nxx*nzz):
        
        i = int(index % nzz)
        j = int(index / nzz)

        if (3 < i < nzz - 4) and (3 < j < nxx - 4):
            
            d2U_dx2 = (- 9.0*(Upre[i, j - 4] + Upre[i, j + 4]) \
                   +   128.0*(Upre[i, j - 3] + Upre[i, j + 3]) \
                   -  1008.0*(Upre[i, j - 2] + Upre[i, j + 2]) \
                   +  8064.0*(Upre[i, j + 1] + Upre[i, j - 1]) \
                   - 14350.0*(Upre[i, j])) / (5040.0*dh*dh)

            d2U_dz2 = (- 9.0*(Upre[i - 4, j] + Upre[i + 4, j]) \
                   +   128.0*(Upre[i - 3, j] + Upre[i + 3, j]) \
                   -  1008.0*(Upre[i - 2, j] + Upre[i + 2, j]) \
                   +  8064.0*(Upre[i - 1, j] + Upre[i + 1, j]) \
                   - 14350.0*(Upre[i, j])) / (5040.0*dh*dh)

            laplacian[i,j] = d2U_dx2 + d2U_dz2

    return laplacian

if __name__ == "__main__":
    Seismic_Modeling()

