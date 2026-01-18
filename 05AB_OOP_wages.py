# %%
import random
from random import seed, choice, sample
from statistics import mean, fmean, StatisticsError
from math import ceil, floor
from copy import deepcopy
import pickle

from tqdm import trange

# %%
# GLOBAL
seed(8686)
tol = 1e-4


def shuffle(x):
    x = x.copy()
    random.shuffle(x)
    return x


# %%
# CLASS DEFINITION


class Household:
    def __init__(h):
        # Stock
        h.M = 0

        # Flow
        h.C = dict()
        h.W = 0
        h.UB = 1
        h.P = dict()
        h.T = 0

        # Other
        h.employer = None
        h.CT = 0


class CapitalGood:
    def __init__(self, beta, p):
        self.age = 0
        self.beta = beta
        self.p = p


class ConsumptionFirm:
    def __init__(f):
        # Stock
        f.M = 0
        f.K = []
        f.L = 0

        # Flow
        f.C = dict()
        f.G = 0
        f.I = dict()
        f.W = dict()
        f.P = dict()
        f.intL = 0
        f.detL = 0

        # Other
        f.employees = []
        f.W0 = 1
        f.p = 1
        f.mu = 0.2
        f.beta = 2
        f.NT = 0
        f.cu = 1
        f.Ip = dict()
        f.Y = 0
        f.IT = 0


class CapitalFirm:
    def __init__(f):
        # Stock
        f.M = 0
        f.K = []
        f.L = 0

        # Flow
        f.I = dict()
        f.W = dict()
        f.P = dict()
        f.intL = 0
        f.detL = 0

        # Other
        f.employees = []
        f.W0 = 1
        f.p = 1
        f.mu = 0.2
        f.beta = 1
        f.NT = 0
        f.cu = 1
        f.Ip = dict()
        f.Y = 0
        f.IT = 0
        f.IK = 0


class Bank:
    def __init__(b):
        # Stock
        b.M = dict()
        b.L = dict()
        b.B = 0

        # Flow
        b.P = dict()
        b.intL = dict()
        b.intB = 0
        b.detL = dict()

        # Other
        b.rL = 0


class Government:
    def __init__(g):
        # Stock
        g.B = 0

        # Flow
        g.G = dict()
        g.UB = dict()
        g.T = dict()
        g.intB = 0

        # Other
        g.rB = 0
        g.GT = 0


class Model:
    def __init__(m):  # using m rather then self
        # Sim pars
        m.TMAX = 500
        m.NH = 1000
        m.NFC = 50
        m.NFK = 5

        # Pars
        m.thetaW = 0.01
        m.ay = 0.6
        m.av = 0.2
        m.dG = 0.03
        m.tW = 0.35
        m.tP = 0.2
        m.tC = 0.2
        m.phi = 0.7
        m.dK = 0.1
        m.rhoC = 0.05
        m.cuT = 0.8
        m.uT = 0.05
        m.iT = 0.02
        m.thetaMu = 0.1
        m.DrL = 0.05
        m.crT = 0.08

        # Others vars
        m.avgp = 1
        m.i = m.iT
        m.cu = m.cuT
        m.u = m.uT
        m.GDP = 0

        # Create agents
        m.H = [Household() for _ in range(m.NH)]
        m.FC = [ConsumptionFirm() for _ in range(m.NFC)]
        m.FK = [CapitalFirm() for _ in range(m.NFK)]
        m.B = Bank()
        m.G = Government()

        # init matrices
        for h in m.H:
            for f in m.FC:
                h.C[f] = 0
            for a in m.FC + m.FK + [m.B]:
                h.P[a] = 0

        for f in m.FC:
            for h in m.H:
                f.C[h] = 0
                f.W[h] = 0
                f.P[h] = 0
            for fk in m.FK:
                f.I[fk] = []
                f.Ip[fk] = 0

        for f in m.FK:
            for h in m.H:
                f.W[h] = 0
                f.P[h] = 0
            for fc in m.FC:
                f.I[fc] = []
                f.Ip[fc] = 0

        for f in m.FC + m.FK:
            m.B.L[f] = 0
            m.B.intL[f] = 0
            m.B.detL[f] = 0

        for h in m.H:
            m.B.P[h] = 0

        for f in m.FC:
            m.G.G[f] = 0

        for h in m.H:
            m.G.UB[h] = 0
            m.G.T[h] = 0

        # init stocks
        for f in m.FC + m.FK:
            f.K += [CapitalGood(1, 1)]

        for h in m.H:
            h.M = 1
            m.B.M[h] = 1

        for f in m.FC + m.FK:
            m.B.M[f] = 0

    def step(m):

        # Set Wage and Price level
        for f in m.FC + m.FK:
            try:
                f.W0 = f.W0 * (1 + m.thetaW * (f.NT - len(f.employees)) / f.NT)
            except ZeroDivisionError:
                f.W0 = f.W0

            f.mu = f.mu * (1 + m.thetaMu * (f.cu - m.cuT) / m.cuT)
            f.p = (1 + f.mu) * f.W0 / f.beta

        for h in m.H:
            if h.employer is not None:
                h.W = h.employer.W0
                h.UB = 0

        # Set interest rates
        m.G.rB = max(
            0,
            m.i + 0.5 * (m.i - m.iT) + 0.25 * (m.cu - m.cuT) - 0.25 * (m.u - m.uT),
        )
        m.B.rL = m.G.rB + m.DrL

        for h in m.H:
            try:
                p = (1 + m.tC) * fmean([f.p for f in h.C.keys()], h.C.values())
            except StatisticsError:
                p = (1 + m.tC) * mean([f.p for f in m.FC])
            h.CT = max(0, (m.ay * (h.UB + (1 - m.tW) * h.W) + m.av * h.M) / p)

        try:
            pG = fmean([f.p for f in m.FC], [f.G for f in m.FC])
        except StatisticsError:
            pG = mean([f.p for f in m.FC])
        m.G.GT = max(
            0,
            (m.dG * m.GDP + sum(m.G.T.values()) - sum(m.G.UB.values())) / pG,
        )

        for f in m.FC + m.FK:
            try:
                f.IT = ceil(
                    max(
                        0,
                        min(len(f.K), len(f.employees)) * max(0, 1 / m.cuT - 1 / f.cu)
                        + m.dK * min(len(f.K), len(f.employees)),
                    )
                )
            except ZeroDivisionError:
                f.IT = ceil(m.dK * min(len(f.K), len(f.employees)))

        # Capital Goods orders are based only on desired I

        while any([sum(fc.Ip.values()) > fc.IT for fc in m.FC]):
            fc = choice([fc for fc in m.FC if sum(fc.Ip.values()) > fc.IT])
            fk = choice([fk for fk in fc.Ip.keys() if fc.Ip[fk] > 0])
            d = min(fc.Ip[fk], sum(fc.Ip.values()) - fc.IT)
            fc.Ip[fk] -= d
            fk.Ip[fc] -= d

        while any([sum(fk.Ip.values()) > sum([k.beta for k in fk.K]) for fk in m.FK]):
            fk = choice(
                [fk for fk in m.FK if sum(fk.Ip.values()) > sum([k.beta for k in fk.K])]
            )
            fc = choice(fk.Ip[fc] > 0)
            d = min(fk.Ip[fc], sum(fk.Ip.values()) - sum([k.beta for k in fk.K]))
            fc.Ip[fk] -= d
            fk.Ip[fc] -= d

        while any([sum(fc.Ip.values()) < fc.IT for fc in m.FC]) and any(
            [sum(fk.Ip.values()) < sum([k.beta for k in fk.K]) for fk in m.FK]
        ):
            fc = choice([fc for fc in m.FC if sum(fc.Ip.values()) < fc.IT])
            fk = choice(
                [fk for fk in m.FK if sum(fk.Ip.values()) < sum([k.beta for k in fk.K])]
            )
            fc.Ip[fk] += 1
            fk.Ip[fc] += 1

        # Labour Market

        for fc in m.FC:
            fc.NT = min(
                len(fc.K), ceil((1 + m.rhoC) * (fc.G + sum(fc.C.values())) / fc.beta)
            )
        for fk in m.FK:
            fk.NT = min(len(fk.K), ceil((sum(fk.Ip.values()) + fk.IT) / fk.beta))

        while len([f for f in (m.FC + m.FK) if len(f.employees) > f.NT]) > 0:
            f = choice([f for f in (m.FC + m.FK) if len(f.employees) > f.NT])
            h = choice(f.employees)
            f.employees = [ah for ah in f.employees if ah != h]
            h.employer = None
            h.W = 0
            h.UB = m.phi * f.W0  ### EDIT from model !!!

        while (sum([len(f.employees) for f in (m.FC + m.FK)]) < m.NH) and (
            any([len(f.employees) < f.NT for f in (m.FC + m.FK)])
        ):
            f = choice([f for f in (m.FC + m.FK) if len(f.employees) < f.NT])
            h = choice([h for h in m.H if h.employer is None])
            f.employees += [h]
            h.employer = f
            h.W = f.W0
            h.UB = 0

        ### First Debt Emission and wage and ub payment (EDIT MOVED)
        for f in m.FC + m.FK:
            for h in f.employees:
                f.L += h.W
                m.B.L[f] += h.W
                f.M -= h.W
                m.B.M[f] -= h.W
                h.M += h.W
                m.B.M[h] += h.W
                h.T += m.tW * h.W
                m.G.T[h] += m.tW * h.W
                h.M -= h.T
                m.B.M[h] -= h.T
                m.B.B -= h.T
                m.G.B -= h.T

        for h in m.H:
            h.M += h.UB
            m.B.M[h] += h.UB
            m.B.B += h.UB
            m.G.B += h.UB

        # Production
        # NO WORKER - KC matching
        for f in m.FC + m.FK:
            f.Y = sum([k.beta for k in sample(f.K, k=min(len(f.employees), len(f.K)))])
            try:
                f.cu = min(len(f.employees), len(f.K)) / len(f.K)
            except ZeroDivisionError:
                f.cu = 0

        # Consumpion good market
        try:
            Hsh = sum([h.CT for h in m.H]) / (sum([h.CT for h in m.H]) + m.G.GT)
        except ZeroDivisionError:
            Hsh = 0

        # first set consumption
        while any([sum(f.C.values()) - f.Y * Hsh > tol for f in m.FC]):
            f = choice([f for f in m.FC if sum(f.C.values()) - f.Y * Hsh > tol])
            h = choice([h for h in f.C.keys() if f.C[h] > 0])
            d = min(sum(f.C.values()) - f.Y * Hsh, f.C[h])
            f.C[h] -= d
            h.C[f] -= d

        while any(
            [
                sum(h.C.values()) - h.CT > tol
                or sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) - h.M > tol
                and sum(h.C.values()) > tol
                for h in m.H
            ]
        ):  # M is already increased of DI
            h = choice(
                [
                    h
                    for h in m.H
                    if (sum(h.C.values()) > h.CT)
                    or (sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) > h.M)
                    and (sum(h.C.values()) > 0)
                ]
            )
            f = choice([f for f in h.C.keys() if h.C[f] > 0])
            d = min(
                h.C[f],
                max(
                    sum(h.C.values()) - h.CT,
                    sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) - h.M,
                ),
            )
            f.C[h] -= d
            h.C[f] -= d

        while any([sum(f.C.values()) - f.Y * Hsh < tol for f in m.FC]) and any(
            [
                sum(h.C.values()) < h.CT
                and sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) < h.M
                for h in m.H
            ]
        ):
            f = choice([f for f in m.FC if sum(f.C.values()) - f.Y * Hsh < tol])
            h = choice(
                [
                    h
                    for h in m.H
                    if (sum(h.C.values()) < h.CT)
                    and (sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) < h.M)
                ]
            )
            d = max(
                f.Y - sum(f.C.values()),
                h.CT - sum(h.C.values()),
                h.M - sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]),
            )
            f.C[h] += d
            h.C[f] += d

        # then move assets
        for h in m.H:
            for f in m.FC:
                f.M += f.C[h] * f.p
                m.B.M[f] += f.C[h] * f.p
                h.M -= h.C[f] * f.p
                m.B.M[h] -= h.C[f] * f.p
                h.T += h.C[f] * m.tC * f.p
                m.G.T[h] += h.T
                h.M -= h.T
                m.B.M[h] -= h.T
                m.B.B -= h.T
                m.G.B -= h.T

        # Gvt expenditure
        for f in m.FC:
            m.G.G[f] = 0
        for f in shuffle(m.FC):
            d = max(0, min(m.G.GT - sum(m.G.G.values()), f.Y - sum(f.C.values())))
            m.G.G[f] = d
            f.G = d
            f.M += d
            m.B.M[f] += d
            m.B.B += d
            m.G.B += d

        # Investment market

        # As before, first clean, later move asset
        for fk in m.FK:
            try:
                ICsh = fk.Y / (sum(fk.Ip.values()) + fk.IT)
            except ZeroDivisionError:
                ICsh = 0
            for fc in m.FC:
                fk.I[fc] = floor(fk.Ip[fc] * ICsh)
                fc.I[fk] = floor(fc.Ip[fk] * ICsh)

        while any([sum(fk.I.values()) > fk.Y for fk in m.FK]):
            fk = choice([fk for fk in m.FK if sum(fk.I.values()) > fk.Y])
            fc = choice([fc for fc in fk.I.keys() if fk.I[fc] > 0])
            fk.I[fc] -= 1
            fc.I[fk] -= 1

        for fk in m.FK:
            fk.IK = min(fk.IT, fk.Y - sum(fk.I.values()))

        while any([sum(fk.I.values()) + fk.IK < fk.Y for fk in m.FK]) and (
            [sum(fc.I.values()) < fc.IT for fc in m.FC]
        ):
            fk = choice([fk for fk in m.FK if sum(fk.I.values()) + fk.IK < fk.Y])
            fc = choice([fc for fc in m.FC if sum(fc.I.values()) < fc.IT])
            fk.I[fc] += 1
            fc.I[fk] += 1

        for fk in m.FK:
            fk.IK += fk.Y - sum(fk.I.values()) - fk.IT

        for fk in m.FK:
            fk.K += [CapitalGood(fk.beta, fk.p) for _ in range(fk.IK)]
            for fc in m.FC:
                fc.K += [CapitalGood(fk.beta, fk.p) for _ in range(fk.I[fc])]
                fc.M -= fk.p * fk.I[fc]
                m.B.M[fc] -= fk.p * fk.I[fc]
                fk.M += fk.p * fk.I[fc]
                m.B.M[fk] += fk.p * fk.I[fc]
                fc.L += fk.p * fk.I[fc]
                m.B.L[fc] += fk.p * fk.I[fc]

        # Chiusura del circuito
        for f in m.FC + m.FK:
            f.intL = f.L * m.B.rL
            m.B.intL[f] = f.intL
            f.M -= f.intL
            m.B.M[f] -= f.intL
            D = max(0, min(f.L, f.M))
            f.L -= D
            f.M -= D
            m.B.L[f] -= D
            m.B.M[f] -= D

        # Bond interest
        m.B.intB = m.G.rB * m.B.B
        m.G.intB = m.G.rB * m.G.B
        m.B.B += m.B.intB
        m.G.B += m.G.intB

        # Profits
        for f in m.FC + m.FK:
            for h in m.H:
                h.P[f] = f.M * h.M / sum([h.M for h in m.H])
                f.P[h] = h.P[f]

        PB = (m.crT - 1) * sum(m.B.L.values()) + m.B.B + sum(m.B.M.values())
        for f in m.H:
            h.P[m.B] = PB * h.M / sum([h.M for h in m.H])
            m.B.P[h] = PB * h.M / sum([h.M for h in m.H])

        for h in m.H:
            for f in m.FC + m.FK:
                f.M -= f.P[h]
                m.B.M[f] -= f.P[h]
                h.M += f.P[h]
                m.B.M[h] += f.P[h]
            h.M += h.P[m.B]
            m.B.M[h] += h.P[m.B]
            h.T += m.tP * sum(h.P.values())
            m.G.T[h] += h.T
            h.M -= h.T
            m.B.M[h] -= h.T
            m.B.B -= h.T
            m.G.B -= h.T

        # Capital depreciation
        for f in m.FC + m.FK:
            for k in f.K:
                k.age += 1
            f.K = [k for d in f.K if k.age < 1 / m.dK]

        for f in m.FC + m.FK:
            if f.M - f.L + sum([k.p * (1 - m.dK * k.age) for k in f.K]) < 0:
                f.detL = f.L
                f.L = 0
                m.B.detL[f] = f.detL
                m.B.L[f] = 0
        try:
            m.i = (
                1
                - fmean([f.p for f in m.FC], [sum(f.C.values()) for f in m.FC]) / m.avgp
            )
            m.avgp = fmean([f.p for f in m.FC], [sum(f.C.values()) for f in m.FC])
        except StatisticsError:
            m.i = 1 - mean([f.p for f in m.FC]) / m.avgp
            m.avgp = mean([f.p for f in m.FC])
        m.cu = fmean([f.cu for f in m.FC + m.FK], [len(f.K) for f in m.FC + m.FK])
        m.u = len([h for h in m.H if h.employer is None]) / m.NH
        m.GDP = sum([f.p * (sum(f.C.values()) + f.G) for f in m.FC]) + sum(
            [f.p * sum(f.I.values()) for f in m.FK]
        )


# %%
m = Model()
data = [deepcopy(m)]
for _ in trange(m.TMAX):
    m.step()
    data += [deepcopy(m)]
pickle.dump(data, open("05_data.pkl", "wb"))

# %%
data = pickle.load(open("05_data.pkl", "rb"))

# %%
# Consistency check
print("Consistency checks: they should be false")
# BS
# M
print(
    any(
        [
            abs(sum([a.M for a in m.H + m.FC + m.FK]) - sum(m.B.M.values())) > tol
            for m in data
        ]
    )
)
# L
print(
    any(
        [abs(sum([a.L for a in m.FC + m.FK]) - sum(m.B.L.values())) > tol for m in data]
    )
)
# B
print(any([abs(m.G.B - m.B.B) > tol for m in data]))
# FOF
# H
print(
    any(
        [
            any(
                [
                    abs(
                        -sum([data[t].H[i].C[f] * f.p for f in data[t].FC])
                        + data[t].H[i].UB
                        + data[t].H[i].W
                        + sum(data[t].H[i].P.values())
                        - data[t].H[i].T
                        - (data[t].H[i].M - data[t - 1].H[i].M)
                    )
                    > tol
                    for i in range(len(data[t].H))
                ]
            )
            for t in range(2, len(data))
        ]
    )
)
# %%
