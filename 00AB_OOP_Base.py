# %%
import random
from random import seed, choice, sample
from statistics import mean, fmean, StatisticsError
from math import ceil, floor
from copy import deepcopy
import pickle

from tqdm import trange
from matplotlib.pyplot import plot, legend

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


class ConsumptionFirm:
    def __init__(f):
        # Stock
        f.M = 0

        # Flow
        f.C = dict()
        f.G = 1
        f.W = dict()
        f.P = dict()
        f.detM = 0

        # Other
        f.employees = []
        f.p = 1
        f.beta = 1
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

        # Others vars
        m.avgp = 1
        m.GDP = 0

        # Create agents
        m.H = [Household() for _ in range(m.NH)]
        m.FC = [ConsumptionFirm() for _ in range(m.NFC)]
        m.G = Government()

        # init matrices
        for h in m.H:
            for f in m.FC:
                h.C[f] = 1
            for a in m.FC:
                h.P[a] = 0

        for f in m.FC:
            for h in m.H:
                f.C[h] = 1
                f.W[h] = 0
                f.P[h] = 0

        for f in m.FC:
            m.G.G[f] = m.NFC

        for h in m.H:
            m.G.UB[h] = 0
            m.G.T[h] = 0

        # init stocks
        for h in m.H:
            h.M = 10
            m.G.M[h] = 10

        for f in m.FC:
            m.G.M[f] = 0

    def step(m):

        # Report Stocks for check purpose
        for h in m.H:
            h.M0 = h.M

        for f in m.FC:
            f.M0 = f.M

        m.G.M0 = sum(m.G.M.values())

        # Set Wage and Price level
        for f in m.FC:
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

        # Labour Market

        while sum([len(f.employees) for f in (m.FC)]) > (
            sum([h.CT for h in m.H]) + m.G.GT
        ) / mean([f.beta for f in m.FC]):
            f = choice(m.FC)
            h = choice(f.employees)
            f.employees = [ah for ah in f.employees if ah != h]
            h.employer = None
            h.W = 0
            h.UB = m.phi * m.W0

        while (sum([len(f.employees) for f in (m.FC)]) < m.NH) and (
            sum([len(f.employees) for f in (m.FC)])
            < (sum([h.CT for h in m.H]) + m.G.GT) / mean([f.beta for f in m.FC])
        ):
            f = choice(m.FC)
            h = choice([h for h in m.H if h.employer is None])
            f.employees += [h]
            h.employer = f
            h.W = m.W0
            h.UB = 0

        ### First Debt Emission and wage and ub payment (EDIT MOVED)
        for h in m.H:
            h.T = 0
            m.G.T[h] = 0
        for f in m.FC:
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
            f.Y = len(f.employees) * f.beta

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

        # Profits
        for f in m.FC:
            for h in m.H:
                h.P[f] = max(0, f.M) * h.M / sum([h.M for h in m.H])
                f.P[h] = h.P[f]

        for h in m.H:
            for f in m.FC:
                f.M -= f.P[h]
                m.G.M[f] -= f.P[h]
                h.M += f.P[h]
                m.G.M[h] += f.P[h]
            h.T += m.tP * sum(h.P.values())
            m.G.T[h] += m.tP * sum(h.P.values())
            h.M -= m.tP * sum(h.P.values())
            m.G.M[h] -= m.tP * sum(h.P.values())

        try:
            m.avgp = fmean([f.p for f in m.FC], [sum(f.C.values()) for f in m.FC])
        except StatisticsError:
            m.avgp = mean([f.p for f in m.FC])

        m.GDP = sum([f.p * (sum(f.C.values()) + f.G) for f in m.FC])


# %%
m = Model()
data = []
for _ in trange(m.TMAX):
    m.step()
    data += [deepcopy(m)]
pickle.dump(data, open("00_data.pkl", "wb"))

# %%
data = pickle.load(open("00_data.pkl", "rb"))

# %%
# Consistency check
print("Consistency checks: they should be false")
# BS
# M
print(
    "M",
    any(
        [abs(sum([a.M for a in m.H + m.FC]) - sum(m.G.M.values())) > tol for m in data]
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
plot([sum([len(f.employees) for f in m.FC]) for m in data], label="NC")
legend()
# %%
plot([sum([(h.CT) for h in m.H]) for m in data], label="CT")
plot([m.G.GT for m in data], label="GT")
legend()
# %%
plot([sum([len(f.employees) for f in m.FC]) for m in data], label="NC")
plot([sum([(f.Y) for f in m.FC]) for m in data], label="Y")
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
plot([sum([-sum(f.W.values()) for f in m.FC]) for m in data], label="W")
plot([sum([-sum(f.P.values()) for f in m.FC]) for m in data], label="P")
plot([sum([-(f.M - f.M0) for f in m.FC]) for m in data], label="dM")
plot([sum([-f.detM for f in m.FC]) for m in data], label="detM")
legend()
# %%
plot([mean([f.p for f in m.FC]) for m in data], label="pc")
plot([mean([m.W0 for f in m.FC]) for m in data], label="w")
plot([m.mu for m in data], label="mu")
legend()
# %%
