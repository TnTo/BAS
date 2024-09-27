# %%
import numpy as np
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
        self.MF = np.full((TMAX,), np.nan)
        self.MG = np.full((TMAX,), np.nan)
        self.VH = np.full((TMAX,), np.nan)
        self.VF = np.full((TMAX,), np.nan)
        self.VG = np.full((TMAX,), np.nan)
        self.CT = np.full((TMAX,), np.nan)
        self.GT = np.full((TMAX,), np.nan)
        self.YT = np.full((TMAX,), np.nan)
        self.C = np.full((TMAX,), np.nan)
        self.G = np.full((TMAX,), np.nan)
        self.Y = np.full((TMAX,), np.nan)
        self.N = np.full((TMAX,), np.nan)
        self.W = np.full((TMAX,), np.nan)
        self.W0 = np.full((TMAX,), np.nan)
        self.P = np.full((TMAX,), np.nan)
        self.T = np.full((TMAX,), np.nan)
        self.p = np.full((TMAX,), np.nan)
        self.beta = np.full((TMAX,), np.nan)
        ###
        self.ay = 0.6
        self.av = 0.25
        self.d = 0.03
        self.aW = 0.05
        self.aBeta = 0.05
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
        self.CT[0] = 0.0
        self.GT[0] = 0.0
        self.YT[0] = 0.0
        self.C[0] = 0.0
        self.G[0] = 0.0
        self.Y[0] = 0.0
        self.N[0] = 0.0
        self.W[0] = 0.0
        self.W0[0] = 1.0
        self.P[0] = 0.0
        self.T[0] = 0.0
        self.p[0] = 1.0
        self.beta[0] = 1.0

    def set_ext(self, **kwargs):
        for var, value in kwargs.items():
            setattr(self, var, value)

    def set_init(self, **kwargs):
        for var, value in kwargs.items():
            getattr(self, var)[0] = value

    def step(self, t):
        self.p[t] = self.p[t - 1]
        for _ in range(50):
            self.MH[t] = self.MH[t - 1] + self.P[t] + self.W[t] - self.C[t] - self.T[t]
            self.MF[t] = self.MF[t - 1] + self.C[t] + self.G[t] - self.W[t] - self.P[t]
            self.MG[t] = self.MG[t - 1] - (self.T[t] - self.G[t])
            self.VH[t] = self.MH[t]
            self.VF[t] = self.MF[t]
            self.VG[t] = -self.MG[t]
            self.CT[t] = (
                self.ay * (1 - self.tW) * self.W[t - 1] + self.av * self.MH[t - 1]
            ) / (1 + self.tC)
            self.GT[t] = self.d * self.Y[t - 1] + self.T[t - 1]
            self.YT[t] = self.CT[t] + self.GT[t]
            self.W0[t] = self.W0[t - 1] * (1 + self.aW * self.N[t - 1])
            self.W[t] = self.W0[t] * self.N[t]
            self.N[t] = min(1, self.YT[t] / (self.beta[t - 1] * self.p[t - 1]))
            self.p[t] = (1 + self.mu) * self.W[t] / (self.N[t] * self.beta[t - 1])
            self.Y[t] = self.N[t] * self.beta[t - 1] * self.p[t]
            self.C[t] = self.CT[t] * self.Y[t] / self.YT[t]
            self.G[t] = self.Y[t] - self.C[t]
            self.P[t] = self.C[t] + self.G[t] - self.W[t]
            self.T[t] = self.tW * self.W[t] + self.tP * self.P[t] + self.tC * self.C[t]
            self.beta[t] = self.beta[t - 1] * (1 + self.aBeta * self.N[t])

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
# m.set_ext(**ext)
# m.set_init(**init)
m.run()
m.check()

plt.plot(m.N)
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
# ay av beta
bounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.2]]
bounds_step = [0.0001] * 3
# target = np.atleast_2d(np.full((T - bi,), 0.85)).T
target = np.zeros((T - bi, tgts))
target[:, :] = np.array(
    [
        [
            0.03,  # growth rate
            # 0.02,  # inflation rate
            # 0.35,  # profit share
            # 0.50,  # Gvt spending to GDP ratio
            0.80,  # APC out of income
            0.50,  # APC out of wealth
            0.85,  # N
        ]
    ]
)


def model(p, n, seed):

    m = MyModel(T)
    m.set_ext(ay=p[0], av=p[1], aBeta=p[2])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        m.run()

    res = np.zeros((T - bi, tgts))
    res[:, 0] = m.Y[bi:T] / m.Y[(bi - 1) : (T - 1)] - 1
    res[:, 1] = (m.C[bi:T] * (1 + m.tC)) / (m.W[bi:T] * (1 - m.tW))
    res[:, 2] = (m.C[bi:T] * (1 + m.tC)) / m.MH[bi:T]
    res[:, 3] = m.N[bi:T]

    return res


cal = MyCalibrator(bounds, bounds_step, target, "cals/02_3p_4t", model)

# %%
cal.run(50)

# %%
p = cal.best

print(f"BEST PARAMETERS: {p}")
# BEST PARAMETERS: [0.4351 0.4005 0.0 ]

m = MyModel(T)
m.set_ext(ay=p[0], av=p[1], aBeta=p[2])
m.run()
m.check()

# %%

plt.plot(((m.C * (1 + m.tC)) / (m.W * (1 - m.tW)))[50:-1])
plt.title("APC - Income")
plt.show()

plt.plot((m.C * (1 + m.tC) / m.MH)[50:-1])
plt.title("APC - Wealth")
plt.show()

plt.plot(m.N[50:-1])
plt.title("Employment share")
plt.show()

plt.plot((m.Y[1:-1] / m.Y[0:-2] - 1)[50:-1])
plt.title("Growth rate")
plt.show()

plt.plot(m.Y[50:-1])
plt.plot(m.C[50:-1])
plt.plot(m.G[50:-1])
plt.title("GDP")
plt.legend(["Y", "C", "G"])
plt.show()

plt.plot(m.MH[10:-1])
plt.plot(m.W[10:-1])
plt.title("Households S&F")
plt.legend(["M", "W"])
plt.show()

# %%
