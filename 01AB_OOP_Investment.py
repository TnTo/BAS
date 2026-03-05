# %%
import random
from random import seed, choice, sample
from statistics import mean, fmean, StatisticsError
from math import ceil, floor
from copy import deepcopy
import pickle

from tqdm import trange
from matplotlib.pyplot import plot, legend, hist

# %%
# GLOBAL
seed(8686)
tol = 1e-4


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

        # Flow
        f.C = dict()
        f.G = 1
        f.I = dict()
        f.W = dict()
        f.P = dict()
        f.detM = 0

        # Other
        f.employees = []
        f.p = 1
        f.beta = 1
        f.NT = 0
        f.cu = 1
        f.Ip = dict()
        f.Y = 0
        f.IT = 0
        f.M0 = 0
        f.K0 = 0


class CapitalFirm:
    def __init__(f):
        # Stock
        f.M = 0

        # Flow
        f.I = dict()
        f.W = dict()
        f.P = dict()
        f.detM = 0

        # Other
        f.employees = []
        f.p = 1
        f.beta = 1
        f.NT = 0
        f.Ip = dict()
        f.Y = 0
        f.M0 = 0
        f.K0 = 0


class Government:
    def __init__(g):
        # Stock
        g.M = dict()

        # Flow
        g.G = dict()
        g.UB = dict()
        g.T = dict()

        # Other
        g.GT = 0


class Model:
    def __init__(m):  # using m rather then self
        # Sim pars
        m.TMAX = 250
        m.NH = 1000
        m.NFC = 50
        m.NFK = 5

        # Pars
        m.W0 = 1
        m.mu = 0.5
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

        # Others vars
        m.avgp = 1
        m.cu = m.cuT
        m.GDP = 0

        # Create agents
        m.H = [Household() for _ in range(m.NH)]
        m.FC = [ConsumptionFirm() for _ in range(m.NFC)]
        m.FK = [CapitalFirm() for _ in range(m.NFK)]
        m.G = Government()

        # init matrices
        for h in m.H:
            for f in m.FC:
                h.C[f] = 1
            for a in m.FC + m.FK:
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

        for f in m.FC:
            m.G.G[f] = m.NFC

        for h in m.H:
            m.G.UB[h] = 0
            m.G.T[h] = 0

        # init stocks
        for f in m.FC:
            f.K += [CapitalGood(1, 1) for _ in range(1)]
            for i in range(len(f.K)):
                f.K[i].age = i

        for h in m.H:
            h.M = 10
            m.G.M[h] = 10

        for f in m.FC + m.FK:
            m.G.M[f] = 0

    def step(m):

        # Report Stocks for check purpose
        for h in m.H:
            h.M0 = h.M

        for f in m.FC + m.FK:
            f.M0 = f.M

        for f in m.FC:
            f.K0 = sum([k.p * (1 - m.dK * k.age) for k in f.K])

        m.G.M0 = sum(m.G.M.values())

        # Set Wage and Price level
        for f in m.FC + m.FK:
            f.p = (1 + m.mu) * m.W0 / f.beta

        for h in m.H:
            if h.employer is not None:
                h.W = m.W0
                h.UB = 0
            else:
                h.W = 0
                h.UB = m.phi * m.W0

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

        for f in m.FC:
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

        while any([sum(fc.Ip.values()) < fc.IT for fc in m.FC]):
            fc = choice([fc for fc in m.FC if sum(fc.Ip.values()) < fc.IT])
            fk = choice(m.FK)
            fc.Ip[fk] += 1
            fk.Ip[fc] += 1

        # Labour Market

        for fc in m.FC:
            fc.NT = ceil(
                min(
                    len(fc.K),
                    ceil((1 + m.rhoC) * (fc.G + sum(fc.C.values())) / fc.beta),
                )
            )
        for fk in m.FK:
            fk.NT = ceil(sum(fk.Ip.values()) / fk.beta)

        while len([f for f in (m.FC + m.FK) if len(f.employees) > f.NT]) > 0:
            f = choice([f for f in (m.FC + m.FK) if len(f.employees) > f.NT])
            h = choice(f.employees)
            f.employees = [ah for ah in f.employees if ah != h]
            h.employer = None
            h.W = 0
            h.UB = m.phi * m.W0

        while (sum([len(f.employees) for f in (m.FC + m.FK)]) < m.NH) and (
            any([len(f.employees) < f.NT for f in (m.FC + m.FK)])
        ):
            f = choice([f for f in (m.FC + m.FK) if len(f.employees) < f.NT])
            h = choice([h for h in m.H if h.employer is None])
            f.employees += [h]
            h.employer = f
            h.W = m.W0
            h.UB = 0

        ### wage and ub payment (EDIT MOVED)
        for h in m.H:
            h.T = 0
            m.G.T[h] = 0
        for f in m.FC + m.FK:
            for h in m.H:
                f.W[h] = 0
            for h in f.employees:
                f.W[h] = h.W
                h.M += h.W
                f.M -= h.W
                h.T += m.tW * h.W
                m.G.T[h] += m.tW * h.W
                h.M -= m.tW * h.W
                m.G.M[h] -= m.tW * h.W

        for h in m.H:
            m.G.UB[h] = h.UB
            h.M += h.UB
            m.G.M[h] += h.UB

        # Production
        # NO WORKER - KC matching
        for f in m.FC:
            f.Y = sum([k.beta for k in sample(f.K, k=min(len(f.employees), len(f.K)))])
            try:
                f.cu = min(len(f.employees), len(f.K)) / len(f.K)
            except ZeroDivisionError:
                f.cu = 0

        for f in m.FK:
            f.Y = floor(len(f.employees) * f.beta)

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
            [(sum(h.C.values()) - h.CT > tol) and sum(h.C.values()) > tol for h in m.H]
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
            f = choice([f for f in m.FC if f.Y * Hsh - sum(f.C.values()) > tol])
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
                m.G.M[f] += f.C[h] * f.p
                h.M -= h.C[f] * f.p
                m.G.M[h] -= h.C[f] * f.p
                h.T += h.C[f] * m.tC * f.p
                m.G.T[h] += h.C[f] * m.tC * f.p
                h.M -= h.C[f] * m.tC * f.p
                m.G.M[h] -= h.C[f] * m.tC * f.p

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
            m.G.M[f] += f.G * f.p

        # Investment market

        # As before, first clean, later move asset
        while any([sum(fk.I.values()) > fk.Y for fk in m.FK]):
            fk = choice([fk for fk in m.FK if sum(fk.I.values()) > fk.Y])
            fc = choice([fc for fc in fk.I.keys() if fk.I[fc] > 0])
            fk.I[fc] -= 1
            fc.I[fk] -= 1

        while any([sum(fk.I.values()) < fk.Y for fk in m.FK]) and any(
            [sum(fc.I.values()) < fc.IT for fc in m.FC]
        ):
            fk = choice([fk for fk in m.FK if (sum(fk.I.values()) < fk.Y)])
            fc = choice([fc for fc in m.FC if (sum(fc.I.values()) < fc.IT)])
            fk.I[fc] += 1
            fc.I[fk] += 1

        for fk in m.FK:
            for fc in m.FC:
                fc.K += [CapitalGood(fk.beta, fk.p) for _ in range(fk.I[fc])]
                fc.M -= fk.p * fk.I[fc]
                m.G.M[fc] -= fk.p * fk.I[fc]
                fk.M += fk.p * fk.I[fc]
                m.G.M[fk] += fk.p * fk.I[fc]

        # Profits
        for f in m.FC + m.FK:
            for h in m.H:
                h.P[f] = f.M * h.M / sum([h.M for h in m.H])
                f.P[h] = h.P[f]

        for h in m.H:
            for f in m.FC + m.FK:
                f.M -= f.P[h]
                m.G.M[f] -= f.P[h]
                h.M += f.P[h]
                m.G.M[h] += f.P[h]
            h.T += m.tP * sum(h.P.values())
            m.G.T[h] += m.tP * sum(h.P.values())
            h.M -= m.tP * sum(h.P.values())
            m.G.M[h] -= m.tP * sum(h.P.values())

        # Capital depreciation
        for f in m.FC:
            for k in f.K:
                k.age += 1
            f.K = [k for k in f.K if k.age < (1 / m.dK)]

        try:
            m.avgp = fmean([f.p for f in m.FC], [sum(f.C.values()) for f in m.FC])
        except StatisticsError:
            m.avgp = mean([f.p for f in m.FC])

        try:
            m.cu = fmean([f.cu for f in m.FC], [len(f.K) for f in m.FC])
        except StatisticsError:
            m.cu = 0

        m.GDP = sum([f.p * (sum(f.C.values()) + f.G) for f in m.FC]) + sum(
            [f.p * sum(f.I.values()) for f in m.FK]
        )


# %%
m = Model()
data = []
for _ in trange(m.TMAX):
    m.step()
    data += [deepcopy(m)]
pickle.dump(data, open("01_data.pkl", "wb"))

# %%
data = pickle.load(open("01_data.pkl", "rb"))

# %%
# Consistency check
print("Consistency checks: they should be false")
# BS
# M
print(
    "M",
    any(
        [
            abs(sum([a.M for a in m.H + m.FC + m.FK]) - sum(m.G.M.values())) > tol
            for m in data
        ]
    ),
)
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
                        - (f.M - f.M0)
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
                        - (f.M - f.M0)
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

# G
print(
    "G",
    any(
        [
            abs(
                -sum([m.G.G[f] * f.p for f in m.FC])
                - sum(m.G.UB.values())
                + sum(m.G.T.values())
                + (sum(m.G.M.values()) - m.G.M0)
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
plot([sum([(f.IT) for f in m.FC]) for m in data], label="IT")
plot([sum([sum(f.Ip.values()) for f in m.FK]) for m in data], label="Ip")
plot([sum([sum(f.I.values()) for f in m.FK]) for m in data], label="I")
legend()
# %%
plot([sum([(f.NT) for f in m.FC]) for m in data], label="NTC")
plot([sum([(f.NT) for f in m.FK]) for m in data], label="NTK")
plot([sum([len(f.employees) for f in m.FC]) for m in data], label="NC")
plot([sum([len(f.employees) for f in m.FK]) for m in data], label="NK")
legend()
# %%
plot([sum([len(f.K) for f in m.FC]) for m in data], label="K")
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
legend()
# %%
plot([sum([(h.W) for h in m.H]) for m in data], label="W")
plot([sum([(h.UB) for h in m.H]) for m in data], label="UB")
plot([sum([sum(h.P.values()) for h in m.H]) for m in data], label="P")
plot([sum([(h.M) for h in m.H]) for m in data], label="M")
legend()
# %%
plot([m.avgp for m in data], label="p")
plot([m.W0 for m in data], label="w")
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
plot([sum([sum(f.P.values()) for f in m.FC]) for m in data], label="FC")
plot([sum([sum(f.P.values()) for f in m.FK]) for m in data], label="FK")
legend()
# %%
plot([sum([f.M for f in m.FC]) for m in data], label="MFC")
plot([sum([f.M for f in m.FK]) for m in data], label="MFK")
legend()
# %%
plot([-sum([m.G.G[f] * f.p for f in m.FC]) for m in data], label="pG")
plot([-sum(m.G.UB.values()) for m in data], label="UB")
plot([sum(m.G.T.values()) for m in data], label="T")
plot([sum(m.G.M.values()) - m.G.M0 for m in data], label="dM")
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
plot([sum([-(f.M - f.M0) for f in m.FC]) for m in data], label="dM")
plot([sum([-f.detM for f in m.FC]) for m in data], label="detM")
legend()
# %%
plot([mean([f.p for f in m.FC]) for m in data], label="pc")
plot([mean([f.p for f in m.FK]) for m in data], label="pk")
plot([mean([m.W0 for f in m.FC]) for m in data], label="w")
plot([m.mu for m in data], label="mu")
legend()
# %%
plot(
    [sum([sum(f.I.values()) * f.p for f in m.FK]) for m in data],
    label="pI",
)
plot([sum([-sum(f.W.values()) for f in m.FK]) for m in data], label="W")
plot([sum([-sum(f.P.values()) for f in m.FK]) for m in data], label="P")
plot([sum([-(f.M - f.M0) for f in m.FK]) for m in data], label="dM")
plot([sum([-f.detM for f in m.FK]) for m in data], label="detM")
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
