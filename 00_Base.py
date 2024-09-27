# %%
import numpy as np
import warnings

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler

import matplotlib.pyplot as plt
import pandas as pd

from tabulate import tabulate


# %%
class MyModel:
    def __init__(self, TMAX=250):
        self.TMAX = TMAX
        ###
        self.MH = np.full((TMAX,), np.nan)
        self.MF = np.full((TMAX,), np.nan)
        self.MG = np.full((TMAX,), np.nan)
        self.VH = np.full((TMAX,), np.nan)
        self.VF = np.full((TMAX,), np.nan)
        self.VG = np.full((TMAX,), np.nan)
        self.C = np.full((TMAX,), np.nan)
        self.G = np.full((TMAX,), np.nan)
        self.Y = np.full((TMAX,), np.nan)
        self.N = np.full((TMAX,), np.nan)
        self.W = np.full((TMAX,), np.nan)
        self.P = np.full((TMAX,), np.nan)
        self.T = np.full((TMAX,), np.nan)
        self.p = np.full((TMAX,), np.nan)
        ###
        self.ay = 0.6
        self.av = 0.2
        self.d = 0.03
        self.W0 = 1.0
        self.beta = 1.0
        self.mu = 0.2
        self.tW = 0.35
        self.tP = 0.2
        self.tC = 0.2
        ###
        self.MH[0] = 1.0
        self.MF[0] = 0.0
        self.MG[0] = 1.0
        self.VH[0] = 1.0
        self.VF[0] = 0.0
        self.VG[0] = -1.0
        self.C[0] = 0.0
        self.G[0] = 0.0
        self.Y[0] = 0.0
        self.N[0] = 0.0
        self.W[0] = 0.0
        self.P[0] = 0.0
        self.T[0] = 0.0
        self.p[0] = 0.0

    def set_ext(self, **kwargs):
        for var, value in kwargs.items():
            setattr(self, var, value)

    def set_init(self, **kwargs):
        for var, value in kwargs.items():
            getattr(self, var)[0] = value

    def step(self, t):
        for _ in range(5):
            self.MH[t] = self.MH[t - 1] + self.P[t] + self.W[t] - self.C[t] - self.T[t]
            self.MF[t] = self.MF[t - 1] + self.C[t] + self.G[t] - self.W[t] - self.P[t]
            self.MG[t] = self.MG[t - 1] - (self.T[t] - self.G[t])
            self.VH[t] = self.MH[t]
            self.VF[t] = self.MF[t]
            self.VG[t] = -self.MG[t]
            self.C[t] = (
                self.ay * (1 - self.tW) * self.W[t - 1] + self.av * self.MH[t - 1]
            ) / (1 + self.tC)
            self.G[t] = self.d * self.Y[t - 1] + self.T[t - 1]
            self.Y[t] = self.C[t] + self.G[t]
            self.W[t] = self.W0 * self.N[t]
            self.N[t] = self.Y[t] / (self.beta * self.p[t])
            self.P[t] = self.C[t] + self.G[t] - self.W[t]
            self.T[t] = self.tW * self.W[t] + self.tP * self.P[t] + self.tC * self.C[t]
            self.p[t] = (1 + self.mu) * self.W0 / self.beta

    def run(self):
        for t in range(1, self.TMAX):
            self.step(t)

    def _bs(self):
        bs = np.zeros((self.TMAX, 2, 4))
        bs[:, 0, 0] = self.MH
        bs[:, 0, 1] = self.MF
        bs[:, 0, 2] = -self.MG
        bs[:, 1, 0] = -self.VH
        bs[:, 1, 1] = -self.VF
        bs[:, 1, 2] = -self.VG
        return bs

    def _tfm(self):
        tfm = np.zeros((self.TMAX - 1, 6, 3))
        tfm[:, 0, 0] = -self.C[1 : self.TMAX]
        tfm[:, 0, 1] = self.C[1 : self.TMAX]
        tfm[:, 1, 1] = self.G[1 : self.TMAX]
        tfm[:, 1, 2] = -self.G[1 : self.TMAX]
        tfm[:, 2, 0] = self.W[1 : self.TMAX]
        tfm[:, 2, 1] = -self.W[1 : self.TMAX]
        tfm[:, 3, 0] = self.P[1 : self.TMAX]
        tfm[:, 3, 1] = -self.P[1 : self.TMAX]
        tfm[:, 4, 0] = -self.T[1 : self.TMAX]
        tfm[:, 4, 2] = self.T[1 : self.TMAX]
        tfm[:, 5, 0] = -(self.MH[1 : self.TMAX] - self.MH[0 : self.TMAX - 1])
        tfm[:, 5, 1] = -(self.MF[1 : self.TMAX] - self.MF[0 : self.TMAX - 1])
        tfm[:, 5, 2] = self.MG[1 : self.TMAX] - self.MG[0 : self.TMAX - 1]
        return tfm

    def check(self):
        # hidden eq
        hec = np.allclose(0.0, self.VH + self.VF + self.VG)
        print(f"Hidden equation check: {hec}")
        bs = self._bs()
        bsc = np.all(
            [
                np.allclose(bs[:, :, 0:-1].sum(axis=2), bs[:, :, -1]),
                np.allclose(bs.sum(axis=1), 0),
            ]
        )
        print(f"Balance sheet check: {bsc}")
        tfm = self._tfm()
        tfmc = np.all(
            [
                np.allclose(tfm.sum(axis=1), 0),
                np.allclose(tfm.sum(axis=2), 0),
            ]
        )
        print(f"Transaction-Flow matrix check: {tfmc}")


# %%
m = MyModel(500)
# m.set_ext(**ext)
# m.set_init(**init)
m.run()
m.check()

plt.plot(m.N)
plt.show()

plt.plot(m.MG / m.Y)
plt.show()

plt.plot(m.G / m.Y)
plt.show()

# %%
