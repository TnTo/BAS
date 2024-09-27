# %%
import numpy as np
from numpy import fmax, fmin
import warnings

from black_it.calibrator import Calibrator
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.loss_functions.minkowski import MinkowskiLoss
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
        self.MFC = np.full((TMAX,), np.nan)
        self.MFK = np.full((TMAX,), np.nan)
        self.MG = np.full((TMAX,), np.nan)
        self.KC = np.full((TMAX,), np.nan)
        self.KK = np.full((TMAX,), np.nan)
        self.K = np.full((TMAX,), np.nan)
        self.VH = np.full((TMAX,), np.nan)
        self.VFC = np.full((TMAX,), np.nan)
        self.VFK = np.full((TMAX,), np.nan)
        self.VG = np.full((TMAX,), np.nan)
        self.CT = np.full((TMAX,), np.nan)
        self.GT = np.full((TMAX,), np.nan)
        self.YCT = np.full((TMAX,), np.nan)
        self.ICT = np.full((TMAX,), np.nan)
        self.IKT = np.full((TMAX,), np.nan)
        self.YKT = np.full((TMAX,), np.nan)
        self.NCT = np.full((TMAX,), np.nan)
        self.NKT = np.full((TMAX,), np.nan)
        self.NT = np.full((TMAX,), np.nan)
        self.C = np.full((TMAX,), np.nan)
        self.G = np.full((TMAX,), np.nan)
        self.YC = np.full((TMAX,), np.nan)
        self.IC = np.full((TMAX,), np.nan)
        self.IK = np.full((TMAX,), np.nan)
        self.YK = np.full((TMAX,), np.nan)
        self.NC = np.full((TMAX,), np.nan)
        self.NK = np.full((TMAX,), np.nan)
        self.N = np.full((TMAX,), np.nan)
        self.WFC = np.full((TMAX,), np.nan)
        self.WFK = np.full((TMAX,), np.nan)
        self.W = np.full((TMAX,), np.nan)
        self.W0 = np.full((TMAX,), np.nan)
        self.PFC = np.full((TMAX,), np.nan)
        self.PFK = np.full((TMAX,), np.nan)
        self.P = np.full((TMAX,), np.nan)
        self.T = np.full((TMAX,), np.nan)
        self.Y = np.full((TMAX,), np.nan)
        self.pC = np.full((TMAX,), np.nan)
        self.pK = np.full((TMAX,), np.nan)
        self.betaC = np.full((TMAX,), np.nan)
        self.betaK = np.full((TMAX,), np.nan)
        self.KCu = np.full((TMAX,), np.nan)
        self.KKu = np.full((TMAX,), np.nan)
        self.Ku = np.full((TMAX,), np.nan)
        self.cuC = np.full((TMAX,), np.nan)
        self.cuK = np.full((TMAX,), np.nan)
        self.cu = np.full((TMAX,), np.nan)
        ###
        self.ay = 0.55
        self.av = 0.3
        self.d = 0.03
        self.aW = 0.05
        self.aBetaC = 1.0
        self.aBetaK = 0.0
        self.mu = 0.2
        self.tW = 0.35
        self.tP = 0.2
        self.tC = 0.2
        self.cuT = 0.8
        self.dK = 0.1
        ###
        self.MH[0] = 1.0
        self.MFC[0] = 0.0
        self.MFK[0] = 0.0
        self.MG[0] = 1.0
        self.KC[0] = 0.1
        self.KK[0] = 0.1
        self.K[0] = 0.2
        self.VH[0] = 1.0
        self.VFC[0] = 0.1
        self.VFK[0] = 0.1
        self.VG[0] = -1.0
        self.W[0] = 0.0
        self.W0[0] = 1.0
        self.Y[0] = 0.0
        self.T[0] = 0.0
        self.N[0] = 0.0
        self.pC[0] = 0.1
        self.pK[0] = 1.0
        self.betaC[0] = 1.1
        self.betaK[0] = 1.1
        self.KCu[0] = 0.0
        self.KKu[0] = 0.0
        self.cuC[0] = 0.0
        self.cuK[0] = 0.0

    def set_ext(self, **kwargs):
        for var, value in kwargs.items():
            setattr(self, var, value)

    def set_init(self, **kwargs):
        for var, value in kwargs.items():
            getattr(self, var)[0] = value

    def step(self, t):
        for _ in range(5):
            self.MH[t] = (
                self.MH[t - 1]
                + self.P[t]
                + self.W[t]
                - self.pC[t] * self.C[t]
                - self.T[t]
            )
            self.MFC[t] = (
                self.MFC[t - 1]
                + self.pC[t] * (self.C[t] + self.G[t])
                - self.pK[t] * self.IC[t]
                - self.WFC[t]
                - self.PFC[t]
            )
            self.MFK[t] = (
                self.MFK[t - 1] + self.pK[t] * self.IC[t] - self.WFK[t] - self.PFK[t]
            )
            self.MG[t] = self.MG[t - 1] - (self.T[t] - self.pC[t] * self.G[t])
            self.VH[t] = self.MH[t]
            self.KC[t] = (1 - self.dK) * self.KC[t - 1] + self.IC[t]
            self.KK[t] = (1 - self.dK) * self.KK[t - 1] + self.IK[t]
            self.K[t] = self.KC[t] + self.KK[t]
            self.VFC[t] = self.MFC[t] + self.pK[t] * self.KC[t]
            self.VFK[t] = self.MFK[t] + self.pK[t] * self.KK[t]
            self.VG[t] = -self.MG[t]
            self.CT[t] = (
                self.ay * (1 - self.tW) * self.W[t - 1] + self.av * self.MH[t - 1]
            ) / self.pC[t - 1]
            self.GT[t] = (self.d * self.Y[t - 1] + self.T[t - 1]) / self.pC[t - 1]
            self.YCT[t] = self.CT[t] + self.GT[t]
            self.ICT[t] = fmax(
                0,
                self.KCu[t - 1] * (1 / self.cuT - 1 / self.cuC[t - 1])
                + self.dK * self.KC[t - 1],
            )
            self.IKT[t] = fmax(
                0,
                self.KKu[t - 1] * (1 / self.cuT - 1 / self.cuK[t - 1])
                + self.dK * self.KK[t - 1],
            )
            self.YKT[t] = self.ICT[t] + self.IKT[t]
            self.NCT[t] = fmin(1, fmin(self.KC[t - 1], self.YCT[t] / self.betaC[t - 1]))
            self.NKT[t] = fmin(1, fmin(self.KK[t - 1], self.YKT[t] / self.betaK[t - 1]))
            self.NT[t] = self.NCT[t] + self.NKT[t]
            self.N[t] = fmin(1, self.NT[t])
            self.NC[t] = fmax(0, self.NCT[t] * self.N[t] / self.NT[t])
            self.NK[t] = fmax(0, self.NKT[t] * self.N[t] / self.NT[t])
            self.W0[t] = self.W0[t - 1] * (1 + self.aW * self.N[t - 1])
            self.WFC[t] = self.W0[t] * self.NC[t]
            self.WFK[t] = self.W0[t] * self.NK[t]
            self.W[t] = self.WFC[t] + self.WFK[t]
            self.YC[t] = self.NC[t] * self.betaC[t - 1]
            self.C[t] = self.CT[t] * self.YC[t] / self.YCT[t]  # fmin((W+MH[-1])/pC)
            self.G[t] = self.YC[t] - self.C[t]
            self.YK[t] = self.NK[t] * self.betaK[t - 1]
            self.IC[t] = fmax(0, self.ICT[t] * self.YK[t] / self.YKT[t])
            self.IK[t] = self.YK[t] - self.IC[t]
            self.PFC[t] = (
                self.pC[t] * (self.C[t] + self.G[t])
                - self.WFC[t]
                - self.pK[t] * self.IC[t]
            )
            self.PFK[t] = self.pK[t] * self.IC[t] - self.WFK[t]
            self.P[t] = self.PFC[t] + self.PFK[t]
            self.T[t] = self.tW * self.W[t] + self.tP * self.P[t] + self.tC * self.C[t]
            self.Y[t] = self.YC[t] + self.YK[t]
            self.pK[t] = (1 + self.mu) * (1 + self.dK) * self.W0[t] / self.betaK[t - 1]
            self.pC[t] = (
                (1 + self.mu) * (self.W0[t] + self.pK[t] * self.dK) / self.betaC[t - 1]
            )
            self.betaC[t] = self.betaC[t - 1] * (1 + self.aBetaC * self.NK[t])
            self.betaK[t] = self.betaK[t - 1] * (1 + self.aBetaK * self.NK[t])
            self.KCu[t] = fmin(self.NC[t], self.KC[t - 1])
            self.KKu[t] = fmin(self.NK[t], self.KK[t - 1])
            self.Ku[t] = fmin(self.N[t], self.K[t - 1])
            self.cuC[t] = self.KCu[t] / self.KC[t - 1]
            self.cuK[t] = self.KKu[t] / self.KK[t - 1]
            self.cu[t] = self.Ku[t] / self.K[t - 1]

    def run(self):
        for t in range(1, self.TMAX):
            self.step(t)

    def _bs(self):
        bs = np.zeros((self.TMAX, 3, 5))
        bs[:, 0, 0] = self.MH
        bs[:, 0, 1] = self.MFC
        bs[:, 0, 2] = self.MFK
        bs[:, 0, 3] = -self.MG
        bs[:, 1, 1] = self.pK * self.KC
        bs[:, 1, 2] = self.pK * self.KK
        bs[:, 1, 4] = self.pK * self.K
        bs[:, 2, 0] = -self.VH
        bs[:, 2, 1] = -self.VFC
        bs[:, 2, 2] = -self.VFK
        bs[:, 2, 3] = -self.VG
        bs[:, 2, 4] = -(self.VH + self.VFC + self.VFK + self.VG)
        return bs

    def _tfm(self):
        tfm = np.zeros((self.TMAX - 1, 7, 4))
        tfm[:, 0, 0] = -self.pC[1 : self.TMAX] * self.C[1 : self.TMAX]
        tfm[:, 0, 1] = self.pC[1 : self.TMAX] * self.C[1 : self.TMAX]
        tfm[:, 1, 1] = self.pC[1 : self.TMAX] * self.G[1 : self.TMAX]
        tfm[:, 1, 3] = -self.pC[1 : self.TMAX] * self.G[1 : self.TMAX]
        tfm[:, 2, 1] = -self.pK[1 : self.TMAX] * self.IC[1 : self.TMAX]
        tfm[:, 2, 2] = self.pK[1 : self.TMAX] * self.IC[1 : self.TMAX]
        tfm[:, 3, 0] = self.W[1 : self.TMAX]
        tfm[:, 3, 1] = -self.WFC[1 : self.TMAX]
        tfm[:, 3, 2] = -self.WFK[1 : self.TMAX]
        tfm[:, 4, 0] = self.P[1 : self.TMAX]
        tfm[:, 4, 1] = -self.PFC[1 : self.TMAX]
        tfm[:, 4, 2] = -self.PFK[1 : self.TMAX]
        tfm[:, 5, 0] = -self.T[1 : self.TMAX]
        tfm[:, 5, 3] = self.T[1 : self.TMAX]
        tfm[:, 6, 0] = -(self.MH[1 : self.TMAX] - self.MH[0 : self.TMAX - 1])
        tfm[:, 6, 1] = -(self.MFC[1 : self.TMAX] - self.MFC[0 : self.TMAX - 1])
        tfm[:, 6, 2] = -(self.MFK[1 : self.TMAX] - self.MFK[0 : self.TMAX - 1])
        tfm[:, 6, 3] = self.MG[1 : self.TMAX] - self.MG[0 : self.TMAX - 1]
        return tfm

    def check(self):
        # hidden eq
        hec = np.allclose(self.pK * self.K, self.VH + self.VFC + self.VFK + self.VG)
        print(f"Hidden equation check: {hec}")
        bs = self._bs()
        bsc = np.all(
            [
                np.allclose(bs[:, :, -1], bs[:, :, 0:-1].sum(axis=2)),
                np.allclose(0.0, bs.sum(axis=1)),
            ]
        )
        print(f"Balance sheet check: {bsc}")
        tfm = self._tfm()
        tfmc = np.all(
            [
                np.allclose(0.0, tfm.sum(axis=1)),
                np.allclose(0.0, tfm.sum(axis=2)),
            ]
        )
        print(f"Transaction-Flow matrix check: {tfmc}")


# %%
m = MyModel(200)
m.run()
m.check()

plt.plot(m.N)
plt.plot(m.NK)
plt.show()

plt.plot(m.Y)
plt.show()

plt.plot(m.MG / m.Y)
plt.show()


# %%
class MyCalibrator:
    def __init__(self, bounds, bounds_step, target, name, model):
        self.model = model
        self.name = name
        self.bounds = bounds
        self.bounds_step = bounds_step
        self.target = target

    def _load_or_init(self):
        try:
            self.calibrator = Calibrator.restore_from_checkpoint(
                self.name, model=self.model
            )
        except FileNotFoundError:
            self.calibrator = Calibrator(
                samplers=[
                    HaltonSampler(batch_size=8),
                    RandomForestSampler(batch_size=8),
                    BestBatchSampler(batch_size=8),
                ],
                real_data=self.target,
                model=self.model,
                parameters_bounds=self.bounds,
                parameters_precision=self.bounds_step,
                ensemble_size=1,
                loss_function=MinkowskiLoss(),
                random_state=8686,
                saving_folder=self.name,
            )

    def run(self, n=50):
        self._load_or_init()
        params, _ = self.calibrator.calibrate(n_batches=n)
        self.best = params[0]
        print(self.best)


# %%
T = 1000
bi = 200
tgts = 4
# ay av betaC betaK mu
bounds = [[0.5, 0.1, 0.0, 0.0, 0.0], [1.0, 0.5, 5.0, 5.0, 1.0]]
bounds_step = [0.0001] * 5
# target = np.atleast_2d(np.full((T - bi,), 0.85)).T
target = np.zeros((T - bi, tgts))
target[:, :] = np.array(
    [
        [
            # 0.35,  # profit share
            # 0.50,  # Gvt spending to GDP ratio
            # 0.80,  # APC out of income
            # 0.50,  # APC out of wealth
            0.85,  # N
            0.15,  # NK
            0.02,  # inflation rate
            0.03,  # growth rate
        ]
    ]
)


def model(p, n, seed):

    m = MyModel(T)
    m.set_ext(ay=p[0], av=p[1], aBetaC=p[2], aBetaK=p[3], mu=p[4])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        m.run()

    res = np.zeros((T - bi, tgts))
    # res[:, 0] = m.Y[bi:T] / m.Y[(bi - 1) : (T - 1)] - 1
    # res[:, 0] = m.C[bi:T] / m.W[bi:T]
    # res[:, 1] = m.C[bi:T] / m.MH[bi:T]
    res[:, 0] = m.N[bi:T]
    res[:, 1] = m.NK[bi:T]
    res[:, 2] = m.pC[bi:T] / m.pC[(bi - 1) : (T - 1)] - 1
    res[:, 3] = m.Y[bi:T] / m.Y[(bi - 1) : (T - 1)] - 1

    return np.nan_to_num(res, nan=-1e10)
    # return res


cal = MyCalibrator(bounds, bounds_step, target, "cals/04_5p_4t", model)

# %%
cal.run(200)

# %%
p = cal.best

print(f"BEST PARAMETERS: {p}")
# [0.9596 0.3698 0.4895 0.2904]

m = MyModel(T)
m.set_ext(ay=p[0], av=p[1], aBetaC=p[2], aBetaK=p[3], mu=p[4])
m.run()
m.check()

# %%

plt.plot((m.C / m.W)[10:-1])
plt.title("APC - Income")
plt.show()

plt.plot((m.C / m.MH)[10:-1])
plt.title("APC - Wealth")
plt.show()

plt.plot(m.N[10:-1])
plt.plot(m.NC[10:-1])
plt.plot(m.NK[10:-1])
plt.title("Employment share")
plt.show()

plt.plot(m.IC[10:-1])
plt.plot(m.IK[10:-1])
plt.plot(m.YK[10:-1])
plt.title("Investment")
plt.legend(["C", "K", "YK"])
plt.show()

plt.plot(m.YC[10:-1])
plt.plot(m.YCT[10:-1])
plt.title("C Production and Demand")
plt.show()

plt.plot((m.Y[1:-1] / m.Y[0:-2] - 1)[10:-1])
plt.title("Growth rate")
plt.show()

plt.plot((m.pC[1:-1] / m.pC[0:-2] - 1)[10:-1])
plt.title("Inflation rate")
plt.show()

plt.plot(m.pC[10:-1])
plt.plot(m.pK[10:-1])
plt.title("Prices")
plt.legend(["C", "K"])
plt.show()

plt.plot(m.betaC[10:-1])
plt.plot(m.betaK[10:-1])
plt.title("Productivity")
plt.legend(["C", "K"])
plt.show()

plt.plot(m.MH[10:-1])
plt.plot(m.W[10:-1])
plt.title("Households S&F")
plt.legend(["M", "W"])
plt.show()

plt.plot(m.CT[10:-1])
plt.plot(m.GT[10:-1])
plt.plot(m.YCT[10:-1])
plt.title("Demand")
plt.legend(["CT", "GT", "YT"])
plt.show()


plt.plot(m.P[10:-1])
plt.plot(m.PFC[10:-1])
plt.plot(m.PFK[10:-1])
plt.title("Profits")
plt.legend(["H", "FC", "FK"])
plt.show()


# %%
