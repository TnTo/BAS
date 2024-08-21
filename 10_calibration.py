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
        self.MFC = np.full((TMAX,), np.nan)
        self.MFK = np.full((TMAX,), np.nan)
        self.MG = np.full((TMAX,), np.nan)
        self.KFC = np.full((TMAX,), np.nan)
        self.KFK = np.full((TMAX,), np.nan)
        self.K = np.full((TMAX,), np.nan)
        self.VH = np.full((TMAX,), np.nan)
        self.VFC = np.full((TMAX,), np.nan)
        self.VFK = np.full((TMAX,), np.nan)
        self.VG = np.full((TMAX,), np.nan)
        self.CT = np.full((TMAX,), np.nan)
        self.GT = np.full((TMAX,), np.nan)
        self.YT = np.full((TMAX,), np.nan)
        self.IT = np.full((TMAX,), np.nan)
        self.IKT = np.full((TMAX,), np.nan)
        self.YKT = np.full((TMAX,), np.nan)
        self.NCT = np.full((TMAX,), np.nan)
        self.NKT = np.full((TMAX,), np.nan)
        self.NT = np.full((TMAX,), np.nan)
        self.N = np.full((TMAX,), np.nan)
        self.NC = np.full((TMAX,), np.nan)
        self.NK = np.full((TMAX,), np.nan)
        self.W0 = np.full((TMAX,), np.nan)
        self.W = np.full((TMAX,), np.nan)
        self.WFC = np.full((TMAX,), np.nan)
        self.WFK = np.full((TMAX,), np.nan)
        self.M = np.full((TMAX,), np.nan)
        self.Y = np.full((TMAX,), np.nan)
        self.C = np.full((TMAX,), np.nan)
        self.G = np.full((TMAX,), np.nan)
        self.YK = np.full((TMAX,), np.nan)
        self.I = np.full((TMAX,), np.nan)
        self.IK = np.full((TMAX,), np.nan)
        self.P = np.full((TMAX,), np.nan)
        self.PFC = np.full((TMAX,), np.nan)
        self.PFK = np.full((TMAX,), np.nan)
        self.T = np.full((TMAX,), np.nan)
        self.muC = np.full((TMAX,), np.nan)
        self.muK = np.full((TMAX,), np.nan)
        self.pC = np.full((TMAX,), np.nan)
        self.pK = np.full((TMAX,), np.nan)
        self.betaC = np.full((TMAX,), np.nan)
        self.betaK = np.full((TMAX,), np.nan)
        self.KFCu = np.full((TMAX,), np.nan)
        self.KFKu = np.full((TMAX,), np.nan)
        self.cuC = np.full((TMAX,), np.nan)
        self.cuK = np.full((TMAX,), np.nan)
        ###
        # self.d = 0
        # self.tW = 0
        # self.tP = 0
        # self.tC = 0
        # self.aW = 0
        # self.cuT = 0
        # self.dK = 0
        # self.phi = 0
        # self.ThetaMu = 0
        # self.Theta.I = 0
        # self.ay = 0
        # self.av = 0
        # self.aBetaC = 0
        # self.aBetaK = 0

    def set_ext(self, **kwargs):
        for var, value in kwargs.items():
            setattr(self, var, value)

    def set_init(self, **kwargs):
        for var, value in kwargs.items():
            getattr(self, var)[0] = value

    def step(self, t):
        self.CT[t] = np.fmax(
            0.0,
            (
                self.ay * (1 - self.tW) * (self.W[t - 1] + self.M[t - 1])
                + self.av * self.MH[t - 1]
            )
            / (1 + self.tC)
            * self.pC[t - 1],
        )
        self.GT[t] = np.fmax(
            0.0,
            (self.d * (self.Y[t - 1]) + self.T[t - 1] - self.M[t - 1]) / self.pC[t - 1],
        )
        self.YT[t] = self.CT[t] + self.GT[t]
        self.IT[t] = np.fmax(
            0.0,
            self.KFCu[t - 1]
            * self.ThetaI
            * np.where(self.cuC[t - 1] > 0.0, (1 / self.cuT - 1 / self.cuC[t - 1]), 0.0)
            + self.dK * self.KFC[t - 1],
        )
        self.IKT[t] = np.fmax(
            0.0,
            self.KFKu[t - 1]
            * self.ThetaI
            * np.where(self.cuK[t - 1] > 0.0, (1 / self.cuT - 1 / self.cuK[t - 1]), 0.0)
            + self.dK * self.KFK[t - 1],
        )
        self.YKT[t] = self.IT[t] + self.IKT[t]
        self.NCT[t] = np.fmin(
            1.0, np.fmin(self.KFC[t - 1], self.YT[t] / self.betaC[t - 1])
        )
        self.NKT[t] = np.fmin(
            1.0, np.fmin(self.KFK[t - 1], self.YKT[t] / self.betaK[t - 1])
        )
        self.NT[t] = self.NCT[t] + self.NKT[t]
        self.N[t] = np.fmin(1.0, self.NT[t])
        self.NC[t] = np.where(
            self.NCT[t] > 0.0, self.NCT[t] * self.N[t] / self.NT[t], 0.0
        )
        self.NK[t] = np.where(
            self.NKT[t] > 0.0, self.NKT[t] * self.N[t] / self.NT[t], 0.0
        )
        self.W0[t] = self.W0[t - 1] * (1.0 + self.aW * self.N[t - 1])
        self.WFC[t] = self.W0[t] * self.NC[t]
        self.WFK[t] = self.W0[t] * self.NK[t]
        self.W[t] = self.WFC[t] + self.WFK[t]
        self.M[t] = self.phi * self.W0[t] * (1.0 - self.N[t])
        self.Y[t] = self.NC[t] * self.betaC[t - 1]
        self.YK[t] = self.NK[t] * self.betaK[t - 1]
        self.I[t] = np.where(
            self.IT[t] > 0.0, self.IT[t] * self.YK[t] / self.YKT[t], 0.0
        )
        self.IK[t] = np.where(
            self.IKT[t] > 0.0, self.IKT[t] * self.YK[t] / self.YKT[t], 0.0
        )
        self.muC[t] = self.muC[t - 1] * (
            1.0 + self.ThetaMu * (self.cuC[t - 1] - self.cuT) / self.cuT
        )
        self.muK[t] = self.muK[t - 1] * (
            1.0 + self.ThetaMu * (self.cuK[t - 1] - self.cuT) / self.cuT
        )
        self.pC[t] = np.where(
            self.Y[t] > 0.0,
            (1.0 + self.muC[t]) * self.WFC[t] / self.Y[t],
            self.pC[t - 1],
        )
        self.pK[t] = np.where(
            self.I[t] > 0.0,
            (1.0 + self.muK[t]) * self.WFK[t] / self.I[t],
            self.pK[t - 1],
        )
        self.C[t] = np.fmin(
            np.where(self.CT[t] > 0.0, self.CT[t] * self.Y[t] / self.YT[t], 0.0),
            ((1.0 - self.tW) * (self.W[t] + self.M[t]) + np.fmax(0.0, self.MH[t - 1]))
            / ((1.0 + self.tC) * self.pC[t]),
        )
        self.G[t] = np.fmin(self.GT[t], self.Y[t] - self.C[t])
        self.KFC[t] = (1.0 - self.dK) * self.KFC[t - 1] + self.I[t]
        self.KFK[t] = (1.0 - self.dK) * self.KFK[t - 1] + self.IK[t]
        self.K[t] = self.KFC[t] + self.KFK[t]
        self.PFC[t] = np.fmax(
            0.0,
            self.MFC[t - 1]
            + self.pC[t] * (self.C[t] + self.G[t])
            - 2.0 * (self.pK[t] * self.I[t] + self.WFC[t]),
        )
        self.PFK[t] = np.fmax(
            0.0, self.MFK[t - 1] + self.pK[t] * self.I[t] - 2.0 * self.WFK[t]
        )
        self.P[t] = self.PFC[t] + self.PFK[t]
        self.T[t] = (
            self.tW * (self.W[t] + self.M[t])
            + self.tP * self.P[t]
            + self.tC * self.pC[t] * self.C[t]
        )
        self.MH[t] = (
            self.MH[t - 1]
            + self.P[t]
            + self.W[t]
            + self.M[t]
            - self.pC[t] * self.C[t]
            - self.T[t]
        )
        self.MFC[t] = (
            self.MFC[t - 1]
            + self.pC[t] * (self.C[t] + self.G[t])
            - self.pK[t] * self.I[t]
            - self.WFC[t]
            - self.PFC[t]
        )
        self.MFK[t] = (
            self.MFK[t - 1] + self.pK[t] * self.I[t] - self.WFK[t] - self.PFK[t]
        )
        self.MG[t] = self.MG[t - 1] - (self.T[t] - self.pC[t] * self.G[t] - self.M[t])
        self.VH[t] = self.MH[t]
        self.VFC[t] = self.MFC[t] + self.pK[t] * self.KFC[t]
        self.VFK[t] = self.MFK[t] + self.pK[t] * self.KFK[t]
        self.VG[t] = -self.MG[t]
        self.betaC[t] = self.betaC[t - 1] * (1.0 + self.aBetaC * self.NK[t])
        self.betaK[t] = self.betaK[t - 1] * (1.0 + self.aBetaK * self.NK[t])
        self.KFCu[t] = np.fmin(self.NC[t], self.KFC[t - 1])
        self.KFKu[t] = np.fmin(self.NK[t], self.KFK[t - 1])
        self.cuC[t] = self.KFCu[t] / self.KFC[t - 1]
        self.cuK[t] = self.KFKu[t] / self.KFK[t - 1]

    def run(self):
        for t in range(1, self.TMAX):
            self.step(t)

    def _bs(self):
        bs = np.zeros((self.TMAX, 3, 5))
        bs[:, 0, 0] = self.MH
        bs[:, 0, 1] = self.MFC
        bs[:, 0, 2] = self.MFK
        bs[:, 0, 3] = -self.MG
        bs[:, 1, 1] = self.pK * self.KFC
        bs[:, 1, 2] = self.pK * self.KFK
        bs[:, 1, 4] = self.pK * self.K
        bs[:, 2, 0] = -self.VH
        bs[:, 2, 1] = -self.VFC
        bs[:, 2, 2] = -self.VFK
        bs[:, 2, 3] = -self.VG
        bs[:, 2, 4] = -self.pK * self.K
        return bs

    def _tfm(self):
        tfm = np.zeros((self.TMAX - 1, 8, 5))
        tfm[:, 0, 0] = -self.pC[1 : self.TMAX] * self.C[1 : self.TMAX]
        tfm[:, 0, 1] = self.pC[1 : self.TMAX] * self.C[1 : self.TMAX]
        tfm[:, 1, 1] = self.pC[1 : self.TMAX] * self.G[1 : self.TMAX]
        tfm[:, 1, 3] = -self.pC[1 : self.TMAX] * self.G[1 : self.TMAX]
        tfm[:, 2, 1] = -self.pK[1 : self.TMAX] * self.I[1 : self.TMAX]
        tfm[:, 2, 2] = self.pK[1 : self.TMAX] * self.I[1 : self.TMAX]
        tfm[:, 3, 0] = self.W[1 : self.TMAX]
        tfm[:, 3, 1] = -self.WFC[1 : self.TMAX]
        tfm[:, 3, 2] = -self.WFK[1 : self.TMAX]
        tfm[:, 4, 0] = self.P[1 : self.TMAX]
        tfm[:, 4, 1] = -self.PFC[1 : self.TMAX]
        tfm[:, 4, 2] = -self.PFK[1 : self.TMAX]
        tfm[:, 5, 0] = self.M[1 : self.TMAX]
        tfm[:, 5, 3] = -self.M[1 : self.TMAX]
        tfm[:, 6, 0] = -self.T[1 : self.TMAX]
        tfm[:, 6, 3] = self.T[1 : self.TMAX]
        tfm[:, 7, 0] = -(self.MH[1 : self.TMAX] - self.MH[0 : self.TMAX - 1])
        tfm[:, 7, 1] = -(self.MFC[1 : self.TMAX] - self.MFC[0 : self.TMAX - 1])
        tfm[:, 7, 2] = -(self.MFK[1 : self.TMAX] - self.MFK[0 : self.TMAX - 1])
        tfm[:, 7, 3] = self.MG[1 : self.TMAX] - self.MG[0 : self.TMAX - 1]
        return tfm

    def check(self):
        # hidden eq
        hec = np.allclose(self.pK * self.K, self.VH + self.VFC + self.VFK + self.VG)
        print(f"Hidden equation check: {hec}")
        bs = self._bs()
        bsc = np.all(
            [
                np.allclose(bs[:, :, 0:4].sum(axis=2), bs[:, :, 4]),
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
ext = {
    "d": 0.03,
    "tW": 0.35,
    "tP": 0.2,
    "tC": 0.2,
    "aW": 0.1,
    "cuT": 0.8,
    "dK": 0.1,
    "phi": 0.7,
    "ThetaMu": 0.05,
    "ThetaI": 0.2,
    "ay": 0.6,
    "av": 0.2,
    "aBetaC": 0.05,
    "aBetaK": 0.0,
}

init = {
    "MH": 1.0,
    "MG": 1.0,
    "MFC": 0.0,
    "MFK": 0.0,
    "KFC": 0.3,
    "KFK": 0.3,
    "K": 0.6,
    "VH": 1.0,
    "VFC": 0.3,
    "VFK": 0.3,
    "VG": -1.0,
    "W0": 1.0,
    "pK": 1.0,
    "pC": 1.0,
    "betaC": 1.0,
    "betaK": 1.0,
    "cuC": 1.0,
    "cuK": 1.0,
    "muC": 0.2,
    "muK": 0.2,
    "N": 0.0,
}

# %%
m = MyModel(1000)
m.set_ext(**ext)
m.set_init(**init)
m.run()
m.check()

m.N[-1]


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
                loss_function=MethodOfMomentsLoss(),
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
bi = 100
# ay av aBetaC
bounds = [[0.4, 0.1, 0.0, 0.1], [1.0, 0.6, 1.0, 1.0]]
bounds_step = [0.01, 0.01, 0.01, 0.01]
# target = np.atleast_2d(np.full((T - bi,), 0.85)).T
target = np.zeros((T - bi, 3))
target[:, :] = np.array(
    [
        [
            # 0.03,  # growth rate
            # 0.02,  # inflation rate
            # 0.35,  # profit share
            # 0.10,  # unemployment rate
            # 0.50,  # Gvt spending to GDP ratio
            # 0.80,  # APC out of income
            # 0.50,  # APC out of wealth
            0.85,  # N
            0.10,  # NK
            0.0,  # Var(cuC)
        ]
    ]
)


def model(p, n, seed):

    m = MyModel(T)
    m.set_ext(**ext)
    m.set_ext(ay=p[0], av=p[1], aBetaC=p[2], ThetaI=p[3])
    m.set_init(**init)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        m.run()

    res = np.zeros((T - bi, 3))
    res[:, 0] = m.N[bi:T]
    res[:, 1] = m.NK[bi:T]
    res[:, 2] = pd.Series(m.cuC).rolling(window=bi).std().to_numpy()[bi:T]
    # res[0, :] = (
    #     (m.pC[bi:T] * m.Y[bi:T] + m.pK[bi:T] * m.I[bi:T])
    #     / (
    #         m.pC[bi - 1 : T - 1] * m.Y[bi - 1 : T - 1]
    #         + m.pK[bi - 1 : T - 1] * m.I[bi - 1 : T - 1]
    #     )
    # ) - 1
    # res[1, :] = (m.pC[bi:T] / m.pC[bi - 1 : T - 1]) - 1
    # res[2, :] = m.P[bi:T] / (m.P[bi:T] + m.W[bi:T])
    # res[:, 0] = 1.0 - (m.NC[bi:T] + m.NK[bi:T])
    # res[4, :] = (
    #     m.pC[bi:T] * m.G[bi:T] / (m.pC[bi:T] * m.Y[bi:T] + m.pK[bi:T] * m.I[bi:T])
    # )
    # res[:, 1] = m.pC[bi:T] * m.C[bi:T] / (m.W[bi:T] + m.M[bi:T])
    # res[:, 2] = m.pC[bi:T] * m.C[bi:T] / m.MH[bi:T]
    # res = np.atleast_2d(m.N[bi:T]).T

    return res


cal = MyCalibrator(bounds, bounds_step, target, "4_vars_3_tg", model)

# %%
cal.run(200)

# %%
p = cal.best
m = MyModel(1000)
m.set_ext(**ext)
# m.set_ext(ay=p[0], av=p[1], aBetaC=p[2], ThetaI=p[3])
m.set_ext(ay=0.6, av=0.15, aBetaC=0.797, ThetaI=0.1)
m.set_init(**init)
m.run()
m.check()

plt.plot(m.N)
plt.plot(m.NC)
plt.plot(m.NK)
plt.legend(["N", "NC", "NK"])
plt.show()

# plt.plot(m.YT)
# plt.plot(m.YKT)
# plt.plot(m.Y)
# plt.plot(m.YK)
# plt.legend(["YT", "YKT", "Y", "YK"])
# plt.show()

# plt.plot(m.cuC)
# plt.plot(m.cuK)
# plt.legend(["cuC", "cuK"])
# plt.show()

# plt.plot(m.pC)
# plt.plot(m.pK)
# plt.legend(["pC", "pK"])
# plt.show()


# %%
