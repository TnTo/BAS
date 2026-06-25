# %%
import random
from random import seed, choice, sample
from statistics import mean, fmean, StatisticsError
from math import ceil, floor, log
from copy import deepcopy
import pickle

from tqdm import trange
from matplotlib.pyplot import plot, legend, hist
import numpy as np
import pandas

# %%
# GLOBAL
seed(8686)
tol = 1e-4

# https://stackoverflow.com/a/39513799
def gini(x):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

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
        h.M0 = 0
        h.skill = 1
        h.machine = None


class CapitalGood:
    def __init__(self, beta, p):
        self.age = 0
        self.beta = beta
        self.p = p
        self.worker = None


class ConsumptionFirm:
    def __init__(f):
        # Stock
        f.M = 0
        f.K = []
        f.L = 0

        # Flow
        f.C = dict()
        f.G = 1
        f.I = dict()
        f.W = dict()
        f.P = dict()
        f.intL = 0
        f.detL = 0
        f.detM = 0

        # Other
        f.employees = []
        f.W0 = 1
        f.p = 1
        f.mu = 0.5
        f.NT = 0
        f.cu = 1
        f.Ip = dict()
        f.Y = 0
        f.IT = 0
        f.M0 = 0
        f.K0 = 0
        f.L0 = 0


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
        f.detM = 0

        # Other
        f.employees = []
        f.W0 = 1
        f.p = 1
        f.mu = 0.5
        f.beta = 1
        f.NT = 0
        f.cu = 1
        f.Ip = dict()
        f.Y = 0
        f.IT = 0
        f.IK = 0
        f.M0 = 0
        f.K0 = 0
        f.L0 = 0
        f.age = 0


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
        b.detM = dict()

        # Other
        b.rL = 0
        b.M0 = 0
        b.L0 = 0
        b.B0 = 0


class Government:
    def __init__(g):
        # Stock
        g.B = 0

        # Flow
        g.G = dict()
        g.UB = dict()
        g.T = dict()
        g.intB = 0
        g.deficit = 0

        # Other
        g.rB = 0
        g.GT = 0
        g.B0 = 0


class Model:
    def __init__(m):  # using m rather then self
        # Sim pars
        m.TMAX = 250
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
        m.ds = 0.005
        m.inn1 = 0.05
        m.inn2 = 0.05
        m.inn3 = 0.5

        # Others vars
        m.avgp = 1
        m.avgw = 1
        m.i = m.iT
        m.cu = m.cuT
        m.u = m.uT
        m.GDP = 0
        m.avgW = 1

        # Create agents
        m.H = [Household() for _ in range(m.NH)]
        m.FC = [ConsumptionFirm() for _ in range(m.NFC)]
        m.FK = [CapitalFirm() for _ in range(m.NFK)]
        m.B = Bank()
        m.G = Government()

        # init matrices
        for h in m.H:
            for f in m.FC:
                h.C[f] = 1
            for a in m.FC + m.FK + [m.B]:
                h.P[a] = 0

        for f in m.FC:
            for h in m.H:
                f.C[h] = 1
                f.W[h] = 0
                f.P[h] = 0
            for fk in m.FK:
                f.I[fk] = 0
                f.Ip[fk] = 0

        for f in m.FK:
            for h in m.H:
                f.W[h] = 0
                f.P[h] = 0
            for fc in m.FC:
                f.I[fc] = 0
                f.Ip[fc] = 0

        for f in m.FC + m.FK:
            m.B.L[f] = 0
            m.B.intL[f] = 0
            m.B.detL[f] = 0
            m.B.detM[f] = 0

        for h in m.H:
            m.B.P[h] = 0

        for f in m.FC:
            m.G.G[f] = m.NFC

        for h in m.H:
            m.G.UB[h] = 0
            m.G.T[h] = 0

        # init stocks
        for f in m.FC + m.FK:
            f.K += [CapitalGood(1, 1) for _ in range(1)]
            for i in range(len(f.K)):
                f.K[i].age = i

        for h in m.H:
            h.M = 10
            m.B.M[h] = 10

        for f in m.FC + m.FK:
            m.B.M[f] = 0

    def step(m):

        # Report Stocks for check purpose
        for h in m.H:
            h.M0 = h.M

        for f in m.FC + m.FK:
            f.M0 = f.M
            f.K0 = sum([k.p * (1 - m.dK * k.age) for k in f.K])
            f.L0 = f.L

        m.B.M0 = sum(m.B.M.values())
        m.B.L0 = sum(m.B.L.values())
        m.B.B0 = m.B.B

        m.G.B0 = m.G.B

        # Innovation and skill improvement
        for f in m.FK:
            f.age += 1
            try:
                f.beta = f.beta + max(
                    0,
                    m.inn1 * mean([h.skill for h in f.employees])
                    - m.inn2 * f.age
                    + m.inn3
                    * (f.p * sum(f.I.values()))
                    / (sum([f.p * sum(f.I.values()) for f in m.FK])),
                )
            except (StatisticsError, ZeroDivisionError):
                f.beta = f.beta

        for h in m.H:
            if h.employer is None:
                h.skill = max(1, h.skill / (1 + m.ds))
            else:
                h.skill = h.skill * (1 + m.ds / 10)

        # Set Wage and Price level
        for f in m.FC + m.FK:
            try:
                f.W0 = f.W0 * (1 + m.thetaW * (f.NT - len(f.employees)) / f.NT)
            except ZeroDivisionError:
                f.W0 = f.W0

            f.mu = max(0.5, f.mu * (1 + m.thetaMu * (f.cu - m.cuT) / m.cuT))
            try:
                f.p = (
                    (1 + f.mu)
                    * f.W0
                    * mean([h.skill for h in f.employees])
                    / mean([k.beta for k in f.K])
                )
            except StatisticsError:
                f.p = f.p

        for h in m.H:
            if h.employer is not None:
                h.W = h.employer.W0 * h.skill
            else:
                h.W = 0
            h.UB = m.phi * m.avgw

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

        # unit measure is now beta
        for f in m.FC + m.FK:
            try:
                f.IT = max(
                    0,
                    min(len(f.K), len(f.employees)) * max(0, 1 / m.cuT - 1 / f.cu)
                    + m.dK * min(len(f.K), len(f.employees)),
                )
            except ZeroDivisionError:
                f.IT = m.dK * min(len(f.K), len(f.employees))

        # Capital Goods orders are based only on desired I

        # Over buy
        while any([sum([fc.Ip[fk] * fk.beta for fk in m.FK]) > fc.IT for fc in m.FC]):
            fc = choice(
                [fc for fc in m.FC if sum([fc.Ip[fk] * fk.beta for fk in m.FK]) > fc.IT]
            )
            fk = choice([fk for fk in fc.Ip.keys() if fc.Ip[fk] > 0])
            d = min(
                fc.Ip[fk],
                ceil((sum([fc.Ip[fk] * fk.beta for fk in m.FK]) - fc.IT) / fk.beta),
            )
            fc.Ip[fk] -= d
            fk.Ip[fc] -= d

        # Over sell
        while any(
            [(sum(fk.Ip.values()) > floor(sum([k.beta for k in fk.K]))) for fk in m.FK]
        ):
            fk = choice(
                [
                    fk
                    for fk in m.FK
                    if sum(fk.Ip.values()) > floor(sum([k.beta for k in fk.K]))
                ]
            )
            fc = choice([fc for fc in m.FC if fk.Ip[fc] > 0])
            d = min(fk.Ip[fc], sum(fk.Ip.values()) - floor(sum([k.beta for k in fk.K])))
            fc.Ip[fk] -= d
            fk.Ip[fc] -= d

        # Under buy
        while any(
            [sum([fc.Ip[fk] * fk.beta for fk in m.FK]) < fc.IT for fc in m.FC]
        ) and any(
            [sum(fk.Ip.values()) < floor(sum([k.beta for k in fk.K])) for fk in m.FK]
        ):
            fc = choice(
                [fc for fc in m.FC if sum([fc.Ip[fk] * fk.beta for fk in m.FK]) < fc.IT]
            )
            fk = sorted(
                [
                    fk
                    for fk in m.FK
                    if sum(fk.Ip.values()) < floor(sum([k.beta for k in fk.K]))
                ],
                key=lambda fk: fk.p / fk.beta,
            )[0]
            fc.Ip[fk] += 1
            fk.Ip[fc] += 1

        # Labour Market

        for fc in m.FC:
            try:
                fc.NT = min(
                    len(fc.K),
                    ceil(
                        (1 + m.rhoC)
                        * (fc.G + sum(fc.C.values()))
                        / mean([k.beta for k in fc.K])
                    ),
                )
            except StatisticsError:
                fc.NT = 0
        for fk in m.FK:
            try:
                fk.NT = min(
                    len(fk.K),
                    ceil((sum(fk.Ip.values()) + fk.IT) / mean([k.beta for k in fk.K])),
                )
            except StatisticsError:
                fk.NT = 0

        while len([f for f in (m.FC + m.FK) if len(f.employees) > f.NT]) > 0:
            f = choice([f for f in (m.FC + m.FK) if len(f.employees) > f.NT])
            h = sorted(f.employees, key=lambda h: h.skill)[0]
            f.employees = [ah for ah in f.employees if ah != h]
            h.employer = None
            h.W = 0
            h.UB = m.phi * m.avgw

        while (sum([len(f.employees) for f in (m.FC + m.FK)]) < m.NH) and (
            any([len(f.employees) < f.NT for f in (m.FC + m.FK)])
        ):
            f = choice([f for f in (m.FC + m.FK) if len(f.employees) < f.NT])
            h = sorted([h for h in m.H if h.employer is None], key=lambda h: h.skill)[
                -1
            ]
            f.employees += [h]
            h.employer = f
            h.skill = h.skill * (1 + m.ds)
            h.W = f.W0 * h.skill
            h.UB = 0

        ### First Debt Emission and wage and ub payment (EDIT MOVED)
        for h in m.H:
            h.T = 0
            m.G.T[h] = 0
        for f in m.FC + m.FK:
            for h in m.H:
                f.W[h] = 0
            for h in f.employees:
                f.W[h] = h.W
                f.L += h.W
                m.B.L[f] += h.W
                f.M += h.W
                m.B.M[f] += h.W
                f.M -= h.W
                m.B.M[f] -= h.W
                h.M += h.W
                m.B.M[h] += h.W
                h.T += m.tW * h.W
                m.G.T[h] += m.tW * h.W
                h.M -= m.tW * h.W
                m.B.M[h] -= m.tW * h.W
                m.B.B -= m.tW * h.W
                m.G.B -= m.tW * h.W

        for h in m.H:
            m.G.UB[h] = h.UB
            h.M += h.UB
            m.B.M[h] += h.UB
            m.B.B += h.UB
            m.G.B += h.UB

        # Production
        # NO WORKER - KC matching
        for h in m.H:
            h.machine = None

        for f in m.FC + m.FK:
            for k in f.K:
                k.worker = None
            f.K = sorted(f.K, key=lambda k: k.beta, reverse=True)
            for k in f.K:
                hs = sorted(
                    [
                        h
                        for h in f.employees
                        if (h.machine is None) and (h.skill >= k.beta)
                    ],
                    key=lambda h: h.skill,
                )
                if len(hs) > 0:
                    h = hs[0]
                    k.worker = h
                    h.machine = k
        for f in m.FC:
            f.Y = sum([k.beta for k in f.K if k.worker is not None])
        for f in m.FK:
            f.Y = floor(sum([k.beta for k in f.K if k.worker is not None]))
        for f in m.FC + m.FK:
            try:
                f.cu = len([k for k in f.K if k.worker is not None]) / len(f.K)
            except ZeroDivisionError:
                f.cu = 0

        # Consumpion good market
        try:
            Hsh = sum([h.CT for h in m.H]) / (sum([h.CT for h in m.H]) + m.G.GT)
        except ZeroDivisionError:
            Hsh = 1

        # first set consumption

        # Over selling
        while any([sum(f.C.values()) - f.Y * Hsh > tol for f in m.FC]):
            f = choice([f for f in m.FC if sum(f.C.values()) - f.Y * Hsh > tol])
            h = choice([h for h in f.C.keys() if f.C[h] > 0])
            d = min(sum(f.C.values()) - f.Y * Hsh, f.C[h])
            f.C[h] -= d
            h.C[f] -= d

        # Over buying
        while any(
            [
                (
                    sum(h.C.values()) - h.CT > tol
                    or sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) - h.M > tol
                )
                and sum(h.C.values()) > tol
                for h in m.H
            ]
        ):  # M is already increased of DI
            h = choice(
                [
                    h
                    for h in m.H
                    if (
                        (sum(h.C.values()) - h.CT > tol)
                        or (sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) - h.M > tol)
                    )
                    and (sum(h.C.values()) > tol)
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

        # Sell Remaining
        while any([f.Y * Hsh - sum(f.C.values()) > tol for f in m.FC]) and any(
            [
                sum(h.C.values()) < h.CT
                and sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) < h.M
                for h in m.H
            ]
        ):
            h = choice(
                [
                    h
                    for h in m.H
                    if (sum(h.C.values()) < h.CT)
                    and (sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) < h.M)
                ]
            )
            f = sorted(
                [f for f in m.FC if f.Y * Hsh - sum(f.C.values()) > tol],
                key=lambda f: f.p,
            )[0]
            d = min(
                f.Y * Hsh - sum(f.C.values()),
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
                m.G.T[h] += h.C[f] * m.tC * f.p
                h.M -= h.C[f] * m.tC * f.p
                m.B.M[h] -= h.C[f] * m.tC * f.p
                m.B.B -= h.C[f] * m.tC * f.p
                m.G.B -= h.C[f] * m.tC * f.p

        # Gvt expenditure
        for f in m.FC:
            m.G.G[f] = 0
            f.G = 0
        while any([f.Y - (sum(f.C.values()) + f.G) > tol for f in m.FC]) and (
            (m.G.GT - sum(m.G.G.values()) > tol)
        ):
            f = choice([f for f in m.FC if f.Y - (sum(f.C.values()) + f.G) > tol])
            d = min(m.G.GT - sum(m.G.G.values()), f.Y - (sum(f.C.values()) + f.G))
            m.G.G[f] += d
            f.G += d
        for f in m.FC:
            f.M += f.G * f.p
            m.B.M[f] += f.G * f.p
            m.B.B += f.G * f.p
            m.G.B += f.G * f.p

        # Investment market

        # As before, first clean, later move asset
        for fk in m.FK:
            try:
                ICsh = fk.Y / (sum(fk.Ip.values()) + ceil(fk.IT / fk.beta))
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
            fk.IK = min(ceil(fk.IT / fk.beta), fk.Y - sum(fk.I.values()))

        while any([sum(fk.I.values()) + fk.IK < fk.Y for fk in m.FK]) and any(
            [sum([fc.I[fk] * fk.beta for fk in m.FK]) < fc.IT for fc in m.FC]
        ):
            fc = choice(
                [fc for fc in m.FC if sum([fc.I[fk] * fk.beta for fk in m.FK]) < fc.IT]
            )
            fk = sorted(
                [fk for fk in m.FK if sum(fk.I.values()) + fk.IK < fk.Y],
                key=lambda fk: fk.p / fk.beta,
            )[0]

            fk.I[fc] += 1
            fc.I[fk] += 1

        for fk in m.FK:
            fk.IK = fk.Y - sum(fk.I.values())

        for fk in m.FK:
            fk.K += [CapitalGood(fk.beta, fk.p) for _ in range(fk.IK)]
            for fc in m.FC:
                fc.K += [CapitalGood(fk.beta, fk.p) for _ in range(fk.I[fc])]
                fc.L += fk.p * fk.I[fc]
                m.B.L[fc] += fk.p * fk.I[fc]
                fc.M += fk.p * fk.I[fc]
                m.B.M[fc] += fk.p * fk.I[fc]
                fc.M -= fk.p * fk.I[fc]
                m.B.M[fc] -= fk.p * fk.I[fc]
                fk.M += fk.p * fk.I[fc]
                m.B.M[fk] += fk.p * fk.I[fc]

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

        # Failures
        for f in m.FC + m.FK:
            f.detL = 0
            f.detM = 0
            m.B.detL[f] = 0
            m.B.detM[f] = 0
            if f.M - f.L + sum([max(0, k.p * (1 - m.dK * k.age)) for k in f.K]) < 0:
                f.age = 0
                f.detL = f.L
                f.detM = f.M
                f.L = 0
                f.M = 0
                m.B.detL[f] = f.detL
                m.B.detM[f] = f.detM
                m.B.L[f] = 0
                m.B.M[f] = 0

        # Profits
        for f in m.FC + m.FK:
            for h in m.H:
                h.P[f] = max(0, f.M) * h.M / sum([h.M for h in m.H])
                f.P[h] = h.P[f]

        PB = max(0, (1 - m.crT) * sum(m.B.L.values()) + m.B.B - sum(m.B.M.values()))
        for h in m.H:
            h.P[m.B] = max(0, PB * h.M / sum([h.M for h in m.H]))
            m.B.P[h] = max(0, PB * h.M / sum([h.M for h in m.H]))

        for h in m.H:
            for f in m.FC + m.FK:
                f.M -= f.P[h]
                m.B.M[f] -= f.P[h]
                h.M += f.P[h]
                m.B.M[h] += f.P[h]
            h.M += h.P[m.B]
            m.B.M[h] += h.P[m.B]
            h.T += m.tP * sum(h.P.values())
            m.G.T[h] += m.tP * sum(h.P.values())
            h.M -= m.tP * sum(h.P.values())
            m.B.M[h] -= m.tP * sum(h.P.values())
            m.B.B -= m.tP * sum(h.P.values())
            m.G.B -= m.tP * sum(h.P.values())

        # Capital depreciation
        for f in m.FC + m.FK:
            for k in f.K:
                k.age += 1
            f.K = [k for k in f.K if k.age < (1 / m.dK)]

        try:
            m.i = (
                fmean([f.p for f in m.FC], [sum(f.C.values()) for f in m.FC]) / m.avgp - 1
            )
            m.avgp = fmean([f.p for f in m.FC], [sum(f.C.values()) for f in m.FC])
        except StatisticsError:
            m.i = mean([f.p for f in m.FC]) / m.avgp - 1
            m.avgp = mean([f.p for f in m.FC])

        try:
            m.cu = fmean([f.cu for f in m.FC + m.FK], [len(f.K) for f in m.FC + m.FK])
        except StatisticsError:
            m.cu = 0

        m.u = len([h for h in m.H if h.employer is None]) / m.NH

        try:
            m.avgw = fmean([f.W0 for f in m.FC], [len(f.employees) for f in m.FC])
        except StatisticsError:
            m.avgw = mean([f.W0 for f in m.FC])

        m.GDP = sum([f.p * (sum(f.C.values()) + f.G) for f in m.FC]) + sum(
            [f.p * sum(f.I.values()) for f in m.FK]
        )

        m.G.deficit = sum(m.G.T.values()) - sum([m.G.G[f] * f.p for f in m.G.G.keys()]) - sum(m.G.UB.values()) - m.G.intB


# %%
for s in range(1,6):
    seed(s)
    m = Model()
    data = []
    for _ in trange(m.TMAX):
        m.step()
        data += [deepcopy(m)]
    pickle.dump(data, open(f"08_data_{s}.pkl", "wb"))

# %%
rec = []
for s in range(1, 6):
    data = pickle.load(open(f"08_data_{s}.pkl", "rb"))
    for t in range(0, 250):
        rec += [("MGini", 'UB', s, t, gini([h.M for h in data[t].H]))]
        rec += [("WGini", 'UB', s, t, gini([h.W for h in data[t].H]))]
        rec += [
            (
                "PubExpShare",
                'UB',
                s,
                t,
                sum([f.G for f in data[t].FC])
                / (sum([sum(f.C.values()) for f in data[t].FC]) + sum([f.G for f in data[t].FC])),
            )
        ]
        rec += [("u", 'UB', s, t, data[t].u)]
        rec += [("i", 'UB', s, t, data[t].i)]
        rec += [("GDP", 'UB', s, t, data[t].GDP)]
        rec += [("GvtDebt", 'UB', s, t, data[t].G.B / data[t].GDP)]
        rec += [("GvtDeficit", 'UB', s, t, data[t].G.deficit / data[t].GDP)]

pandas.DataFrame(rec, columns=('Var', 'Model', 'seed','t', 'Val')).to_pickle("08_res.pkl")


# %%
data = pickle.load(open("08_data.pkl", "rb"))

# %%
# Consistency check
print("Consistency checks: they should be false")
# BS
# M
print(
    "M",
    any(
        [
            abs(sum([a.M for a in m.H + m.FC + m.FK]) - sum(m.B.M.values())) > tol
            for m in data
        ]
    ),
)
# L
print(
    "L",
    any(
        [abs(sum([a.L for a in m.FC + m.FK]) - sum(m.B.L.values())) > tol for m in data]
    ),
)
# B
print("B", any([abs(m.G.B - m.B.B) > tol for m in data]))
# FOF
# H
print(
    "H",
    any(
        [
            any(
                [
                    abs(
                        -sum([h.C[f] * f.p for f in m.FC])
                        + h.UB
                        + h.W
                        + sum(h.P.values())
                        - h.T
                        - (h.M - h.M0)
                    )
                    > tol
                    for h in m.H
                ]
            )
            for m in data[2:]
        ]
    ),
)
# FC
print(
    "FC",
    any(
        [
            any(
                [
                    abs(
                        +sum(f.C.values()) * f.p
                        + f.G * f.p
                        - sum([f.I[fk] * fk.p for fk in m.FK])
                        - sum(f.W.values())
                        - sum(f.P.values())
                        - f.intL
                        - (f.M - f.M0)
                        + (f.L - f.L0)
                        + f.detL
                        - f.detM
                    )
                    > tol
                    for f in m.FC
                ]
            )
            for m in data[2:]
        ]
    ),
)

# FK
print(
    "FK",
    any(
        [
            any(
                [
                    abs(
                        +sum(f.I.values()) * f.p
                        - sum(f.W.values())
                        - sum(f.P.values())
                        - f.intL
                        - (f.M - f.M0)
                        + (f.L - f.L0)
                        + f.detL
                        - f.detM
                    )
                    > tol
                    for f in m.FK
                ]
            )
            for m in data[2:]
        ]
    ),
)

# B
print(
    "B",
    any(
        [
            abs(
                -sum(m.B.P.values())
                + sum(m.B.intL.values())
                + m.B.intB
                + (sum(m.B.M.values()) - m.B.M0)
                - (sum(m.B.L.values()) - m.B.L0)
                - (m.B.B - m.B.B0)
                + sum(m.B.detM.values())
                - sum(m.B.detL.values())
            )
            > tol
            for m in data[2:]
        ]
    ),
)

# G
print(
    "G",
    any(
        [
            abs(
                -sum([m.G.G[f] * f.p for f in m.FC])
                - sum(m.G.UB.values())
                + sum(m.G.T.values())
                - m.G.intB
                + (m.G.B - m.G.B0)
            )
            > tol
            for m in data[2:]
        ]
    ),
)

# %%
plot([sum([sum(f.C.values()) for f in m.FC]) for m in data], label="C")
plot([sum([sum(f.C.values()) + f.G for f in m.FC]) for m in data], label="C+G")
plot([sum([f.G for f in m.FC]) for m in data], label="G")
plot([sum([(f.Y) for f in m.FC]) for m in data], label="Y")
# plot([m.G.GT for m in data], label="GT")
legend()
# %%
plot([sum([(f.IT) for f in m.FC]) for m in data], label="ITC")
plot([sum([(f.IT) for f in m.FK]) for m in data], label="ITK")
plot([sum([sum(f.I.values()) for f in m.FK]) for m in data], label="IC")
legend()
# %%
plot([sum([(f.NT) for f in m.FC]) for m in data], label="NTC")
plot([sum([(f.NT) for f in m.FK]) for m in data], label="NTK")
plot([sum([len(f.employees) for f in m.FC]) for m in data], label="NC")
plot([sum([len(f.employees) for f in m.FK]) for m in data], label="NK")
legend()
# %%
plot([sum([len(f.K) for f in m.FC]) for m in data], label="KC")
plot([sum([len(f.K) for f in m.FK]) for m in data], label="KK")
legend()
# %%
plot([sum([(h.CT) for h in m.H]) for m in data], label="CT")
plot([m.G.GT for m in data], label="GT")
legend()
# %%
plot([sum([len(f.employees) for f in m.FC]) for m in data], label="NC")
plot([sum([len(f.K) for f in m.FC]) for m in data], label="KC")
plot([sum([(f.Y) for f in m.FC]) for m in data], label="Y")
legend()
# %%
plot([m.cu for m in data], label="cu")
plot([m.i for m in data], label="i")
plot([m.u for m in data], label="u")
legend()
# %%
plot([sum([(h.W) for h in m.H]) for m in data], label="W")
plot([sum([(h.UB) for h in m.H]) for m in data], label="UB")
plot([sum([sum(h.P.values()) for h in m.H]) for m in data], label="P")
plot([sum([(h.M) for h in m.H]) for m in data], label="M")
legend()
# %%
plot([m.avgp for m in data], label="p")
plot([m.avgw for m in data], label="w")
legend()
# %%
plot([sum([-sum([h.C[f] * f.p for f in m.FC]) for h in m.H]) for m in data], label="pC")
plot([sum([+h.UB for h in m.H]) for m in data], label="UB")
plot([sum([+h.W for h in m.H]) for m in data], label="W")
plot([sum([+sum(h.P.values()) for h in m.H]) for m in data], label="P")
plot([sum([-h.T for h in m.H]) for m in data], label="T")
plot([sum([-(h.M - h.M0) for h in m.H]) for m in data], label="dM")
legend()
# %%
plot([sum(m.B.P.values()) for m in data], label="B")
plot([sum([sum(f.P.values()) for f in m.FC]) for m in data], label="FC")
plot([sum([sum(f.P.values()) for f in m.FK]) for m in data], label="FK")
legend()
# %%
plot([m.crT for m in data], label="tgt")
plot([m.B.B for m in data], label="B")
plot([sum(m.B.L.values()) for m in data], label="L")
plot([sum(m.B.M.values()) for m in data], label="M")
legend()
# %%
plot([sum([f.M for f in m.FC]) for m in data], label="MFC")
plot([sum([f.M for f in m.FK]) for m in data], label="MFK")
legend()
# %%
plot([-sum([m.G.G[f] * f.p for f in m.FC]) for m in data], label="pG")
plot([-sum(m.G.UB.values()) for m in data], label="UB")
plot([sum(m.G.T.values()) for m in data], label="T")
plot([m.G.intB for m in data], label="iB")
plot([m.G.B - m.G.B0 for m in data], label="dB")
plot(
    [
        -sum([m.G.G[f] * f.p for f in m.FC])
        - sum(m.G.UB.values())
        + sum(m.G.T.values())
        + m.G.intB
        + (m.G.B - m.G.B0)
        for m in data
    ],
    label="delta",
)
legend()
# %%
plot([-sum(m.B.P.values()) for m in data], label="P")
plot([sum(m.B.intL.values()) for m in data], label="iL")
plot([m.B.intB for m in data], label="iB")
plot([(sum(m.B.M.values()) - m.B.M0) for m in data], label="dM")
plot([-(sum(m.B.L.values()) - m.B.L0) for m in data], label="dL")
plot([-(m.B.B - m.B.B0) for m in data], label="dB")
plot([sum(m.B.detM.values()) for m in data], label="detM")
plot([-sum(m.B.detL.values()) for m in data], label="detL")
legend()

# %%
plot([sum([+sum(f.C.values()) * f.p for f in m.FC]) for m in data], label="pC")
plot([sum([+f.G * f.p for f in m.FC]) for m in data], label="pG")
plot(
    [sum([-sum([f.I[fk] * fk.p for fk in m.FK]) for f in m.FC]) for m in data],
    label="pI",
)
plot([sum([-sum(f.W.values()) for f in m.FC]) for m in data], label="W")
plot([sum([-sum(f.P.values()) for f in m.FC]) for m in data], label="P")
plot([sum([-f.intL for f in m.FC]) for m in data], label="iL")
plot([sum([-(f.M - f.M0) for f in m.FC]) for m in data], label="dM")
plot([sum([+(f.L - f.L0) for f in m.FC]) for m in data], label="dL")
plot([sum([+f.detL for f in m.FC]) for m in data], label="detL")
plot([sum([-f.detM for f in m.FC]) for m in data], label="detM")
plot(
    [
        sum(
            [
                +sum(f.C.values()) * f.p
                + f.G * f.p
                - sum([f.I[fk] * fk.p for fk in m.FK])
                - sum(f.W.values())
                - sum(f.P.values())
                - f.intL
                - (f.M - f.M0)
                + (f.L - f.L0)
                + f.detL
                - f.detM
                for f in m.FC
            ]
        )
        for m in data
    ],
    label="delta",
)
legend()
# %%
plot([mean([f.p for f in m.FC]) for m in data], label="pc")
plot([mean([f.p for f in m.FK]) for m in data], label="pk")
plot([mean([f.W0 for f in m.FC]) for m in data], label="wc")
plot([mean([f.W0 for f in m.FK]) for m in data], label="wk")
plot([mean([f.mu for f in m.FC]) for m in data], label="mc")
plot([mean([f.mu for f in m.FK]) for m in data], label="mk")
legend()
# %%
plot([sum([h.W for h in m.H]) for m in data], label="WH")
plot([sum([sum(f.W.values()) for f in m.FC + m.FK]) for m in data], label="WF")
plot([m.NH * (1 - m.u) for m in data], label="N")
legend()
# %%
plot(
    [sum([sum(f.I.values()) * f.p for f in m.FK]) for m in data],
    label="pI",
)
plot([sum([-sum(f.W.values()) for f in m.FK]) for m in data], label="W")
plot([sum([-sum(f.P.values()) for f in m.FK]) for m in data], label="P")
plot([sum([-f.intL for f in m.FK]) for m in data], label="iL")
plot([sum([-(f.M - f.M0) for f in m.FK]) for m in data], label="dM")
plot([sum([+(f.L - f.L0) for f in m.FK]) for m in data], label="dL")
plot([sum([+f.detL for f in m.FK]) for m in data], label="detL")
plot([sum([-f.detM for f in m.FK]) for m in data], label="detM")
plot(
    [
        sum(
            [
                +sum(f.I.values()) * f.p
                - sum(f.W.values())
                - sum(f.P.values())
                - f.intL
                - (f.M - f.M0)
                + (f.L - f.L0)
                + f.detL
                - f.detM
                for f in m.FK
            ]
        )
        for m in data
    ],
    label="delta",
)
legend()
# %%
plot([mean(f.beta for f in m.FK) for m in data], label="beta")
plot([log(mean(f.beta for f in m.FK)) for m in data], label="logbeta")
plot([mean(h.skill for h in m.H) for m in data], label="skill")
legend()
# %%
hist([h.M for h in data[-1].H])
# %%
plot([m.G.GT for m in data], label="GT")
plot([sum([h.CT for h in m.H]) for m in data], label="CT")
plot([sum([h.CT for h in m.H]) + m.G.GT for m in data], label="YT")
plot([sum([f.Y for f in m.FC]) for m in data], label="Y")
plot(
    [sum([sum(h.C.values()) for h in m.H]) + sum(m.G.G.values()) for m in data],
    label="S",
)
plot([sum(m.G.G.values()) for m in data], label="G")
plot([sum([sum(h.C.values()) for h in m.H]) for m in data], label="C")
legend()
# %%
plot([sum([sum(f.Ip.values()) for f in m.FK]) for m in data], label="Ip")
plot([sum([f.IT for f in m.FC]) for m in data], label="IT")
plot([sum([sum(f.I.values()) for f in m.FC]) for m in data], label="I")
plot([sum([len(f.employees) for f in m.FK]) for m in data], label="NK")
plot([sum([f.Y for f in m.FK]) for m in data], label="YK")
legend()
# %%
hist([f.p for f in data[-1].FC+data[-1].FK])
#%% 
hist([f.beta for f in data[-1].FK])
#%%
hist([f.age for f in data[-1].FK])

# %%