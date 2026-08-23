using Random
using Statistics
using Serialization
using ProgressMeter
using Plots

# ---------------------------------------------------------------------
# GLOBAL
# ---------------------------------------------------------------------
const RNG = MersenneTwister(8686)
const tol = 1e-4

choice(x) = rand(RNG, x)   # random.choice

# --- exceptions mirroring Python's statistics / arithmetic semantics ---
struct StatisticsError <: Exception end
struct ZeroDivisionError <: Exception end

# statistics.mean : raises on empty input
function smean(x)
    isempty(x) && throw(StatisticsError())
    return sum(x) / length(x)
end

# statistics.fmean(data, weights) : weighted mean; raises on empty data
# or non-positive total weight (matches CPython behaviour closely enough
# for the guards in this model).
function fmean(data, weights)
    isempty(data) && throw(StatisticsError())
    sw = sum(weights)
    sw == 0 && throw(StatisticsError())
    return sum(d * w for (d, w) in zip(data, weights)) / sw
end

# division that raises on a zero denominator (like Python int/float 1/0)
safediv(a, b) = b == 0 ? throw(ZeroDivisionError()) : a / b

# https://stackoverflow.com/a/39513799  (defined but unused, kept for parity)
function gini(x)
    mad = mean(abs.(x .- x'))          # mean absolute difference
    rmad = mad / mean(x)               # relative mean absolute difference
    return 0.5 * rmad
end

# ---------------------------------------------------------------------
# TYPE HIERARCHY
# (abstract supertypes are declared first so that mutually-referencing
#  struct fields can be annotated without forward-declaration issues)
# ---------------------------------------------------------------------
abstract type Agent end
abstract type Firm <: Agent end

mutable struct CapitalGood
    age::Int
    beta::Float64
    p::Float64
    worker::Union{Nothing,Agent}       # a Household, or nothing
    CapitalGood(beta, p) = new(0, beta, p, nothing)
end

mutable struct Household <: Agent
    # Stock
    M::Float64
    # Flow
    C::Dict{Firm,Float64}
    W::Float64
    UB::Float64
    P::Dict{Agent,Float64}
    T::Float64
    # Other
    employer::Union{Nothing,Firm}
    CT::Float64
    M0::Float64
    skill::Float64
    machine::Union{Nothing,CapitalGood}
    Household() = new(
        0.0,
        Dict{Firm,Float64}(),
        0.0,
        1.0,
        Dict{Agent,Float64}(),
        0.0,
        nothing,
        0.0,
        0.0,
        1.0,
        nothing,
    )
end

mutable struct ConsumptionFirm <: Firm
    # Stock
    M::Float64
    K::Vector{CapitalGood}
    L::Float64
    # Flow
    C::Dict{Household,Float64}
    G::Float64
    I::Dict{Firm,Float64}
    W::Dict{Household,Float64}
    P::Dict{Household,Float64}
    intL::Float64
    detL::Float64
    detM::Float64
    # Other
    employees::Vector{Household}
    W0::Float64
    p::Float64
    mu::Float64
    NT::Int
    cu::Float64
    Ip::Dict{Firm,Float64}
    Y::Float64
    IT::Float64
    M0::Float64
    K0::Float64
    L0::Float64
    age::Int
    ConsumptionFirm() = new(
        0.0,
        CapitalGood[],
        0.0,
        Dict{Household,Float64}(),
        1.0,
        Dict{Firm,Float64}(),
        Dict{Household,Float64}(),
        Dict{Household,Float64}(),
        0.0,
        0.0,
        0.0,
        Household[],
        1.0,
        1.0,
        0.5,
        0,
        1.0,
        Dict{Firm,Float64}(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

mutable struct CapitalFirm <: Firm
    # Stock
    M::Float64
    K::Vector{CapitalGood}
    L::Float64
    # Flow
    I::Dict{Firm,Float64}
    W::Dict{Household,Float64}
    P::Dict{Household,Float64}
    intL::Float64
    detL::Float64
    detM::Float64
    # Other
    employees::Vector{Household}
    W0::Float64
    p::Float64
    mu::Float64
    beta::Float64
    NT::Int
    cu::Float64
    Ip::Dict{Firm,Float64}
    Y::Float64
    IT::Float64
    IK::Int
    M0::Float64
    K0::Float64
    L0::Float64
    age::Int
    CapitalFirm() = new(
        0.0,
        CapitalGood[],
        0.0,
        Dict{Firm,Float64}(),
        Dict{Household,Float64}(),
        Dict{Household,Float64}(),
        0.0,
        0.0,
        0.0,
        Household[],
        1.0,
        1.0,
        0.5,
        1.0,
        0,
        1.0,
        Dict{Firm,Float64}(),
        0.0,
        0.0,
        0,
        0.0,
        0.0,
        0.0,
        0,
    )
end

mutable struct Bank <: Agent
    # Stock
    M::Dict{Agent,Float64}
    L::Dict{Firm,Float64}
    B::Float64
    # Flow
    P::Dict{Household,Float64}
    intL::Dict{Firm,Float64}
    intB::Float64
    detL::Dict{Firm,Float64}
    detM::Dict{Firm,Float64}
    # Other
    rL::Float64
    M0::Float64
    L0::Float64
    B0::Float64
    Bank() = new(
        Dict{Agent,Float64}(),
        Dict{Firm,Float64}(),
        0.0,
        Dict{Household,Float64}(),
        Dict{Firm,Float64}(),
        0.0,
        Dict{Firm,Float64}(),
        Dict{Firm,Float64}(),
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

mutable struct Government <: Agent
    # Stock
    B::Float64
    # Flow
    G::Dict{ConsumptionFirm,Float64}
    UB::Dict{Household,Float64}
    T::Dict{Household,Float64}
    intB::Float64
    deficit::Float64
    # Other
    rB::Float64
    GT::Float64
    B0::Float64
    Government() = new(
        0.0,
        Dict{ConsumptionFirm,Float64}(),
        Dict{Household,Float64}(),
        Dict{Household,Float64}(),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

mutable struct Model
    # Sim pars
    TMAX::Int
    NH::Int
    NFC::Int
    NFK::Int
    # Pars
    thetaW::Float64
    ay::Float64
    av::Float64
    dG::Float64
    tW::Float64
    tP::Float64
    tC::Float64
    tM::Float64
    phi::Float64
    dK::Float64
    rhoC::Float64
    cuT::Float64
    uT::Float64
    iT::Float64
    thetaMu::Float64
    DrL::Float64
    crT::Float64
    ds::Float64
    inn1::Float64
    inn2::Float64
    inn3::Float64
    rhoL::Float64
    # Other vars
    avgp::Float64
    avgw::Float64
    i::Float64
    cu::Float64
    u::Float64
    GDP::Float64
    avgW::Float64
    # Agents
    H::Vector{Household}
    FC::Vector{ConsumptionFirm}
    FK::Vector{CapitalFirm}
    B::Bank
    G::Government

    function Model()
        # Sim pars
        TMAX = 500
        NH = 1000
        NFC = 50
        NFK = 5
        # Pars
        thetaW = 0.01
        ay = 0.6
        av = 0.2
        dG = 0.03
        tW = 0.35
        tP = 0.2
        tC = 0.2
        tM = 0.02
        phi = 0.7
        dK = 0.1
        rhoC = 0.05
        cuT = 0.8
        uT = 0.05
        iT = 0.02
        thetaMu = 0.1
        DrL = 0.05
        crT = 0.08
        ds = 0.005
        inn1 = 0.05
        inn2 = 0.05
        inn3 = 0.5
        rhoL = 10
        # Other vars
        avgp = 1.0
        avgw = 1.0
        i = iT
        cu = cuT
        u = uT
        GDP = 0.0
        avgW = 1.0

        # Create agents
        H = [Household() for _ = 1:NH]
        FC = [ConsumptionFirm() for _ = 1:NFC]
        FK = [CapitalFirm() for _ = 1:NFK]
        B = Bank()
        G = Government()

        # init matrices
        for h in H
            for f in FC
                h.C[f] = 1.0
            end
            for a in vcat(FC, FK, Agent[B])
                h.P[a] = 0.0
            end
        end

        for f in FC
            for h in H
                f.C[h] = 1.0
                f.W[h] = 0.0
                f.P[h] = 0.0
            end
            for fk in FK
                f.I[fk] = 0.0
                f.Ip[fk] = 0.0
            end
        end

        for f in FK
            for h in H
                f.W[h] = 0.0
                f.P[h] = 0.0
            end
            for fc in FC
                f.I[fc] = 0.0
                f.Ip[fc] = 0.0
            end
        end

        for f in vcat(FC, FK)
            B.L[f] = 0.0
            B.intL[f] = 0.0
            B.detL[f] = 0.0
            B.detM[f] = 0.0
        end

        for h in H
            B.P[h] = 0.0
        end

        for f in FC
            G.G[f] = Float64(NFC)
        end

        for h in H
            G.UB[h] = 0.0
            G.T[h] = 0.0
        end

        # init stocks
        for f in vcat(FC, FK)
            append!(f.K, [CapitalGood(1.0, 1.0) for _ = 1:1])
            for i = 1:length(f.K)
                f.K[i].age = i - 1
            end
        end

        for h in H
            h.M = 10.0
            B.M[h] = 10.0
        end

        for f in vcat(FC, FK)
            B.M[f] = 0.0
        end

        return new(
            TMAX,
            NH,
            NFC,
            NFK,
            thetaW,
            ay,
            av,
            dG,
            tW,
            tP,
            tC,
            tM,
            phi,
            dK,
            rhoC,
            cuT,
            uT,
            iT,
            thetaMu,
            DrL,
            crT,
            ds,
            inn1,
            inn2,
            inn3,
            rhoL,
            avgp,
            avgw,
            i,
            cu,
            u,
            GDP,
            avgW,
            H,
            FC,
            FK,
            B,
            G,
        )
    end
end

# ---------------------------------------------------------------------
# STEP
# ---------------------------------------------------------------------
function step!(m::Model)

    # Report Stocks for check purpose
    for h in m.H
        h.M0 = h.M
    end

    for f in vcat(m.FC, m.FK)
        f.M0 = f.M
        f.K0 = sum([k.p * (1 - m.dK * k.age) for k in f.K])
        f.L0 = f.L
    end

    m.B.M0 = sum(values(m.B.M))
    m.B.L0 = sum(values(m.B.L))
    m.B.B0 = m.B.B

    m.G.B0 = m.G.B

    # Innovation and skill improvement
    for f in m.FK
        f.age += 1
        try
            f.beta =
                f.beta + max(
                    0,
                    m.inn1 * smean([h.skill for h in f.employees]) - m.inn2 * f.age +
                    m.inn3 * safediv(
                        f.p * sum(values(f.I)),
                        sum([g.p * sum(values(g.I)) for g in m.FK]),
                    ),
                )
        catch e
            (e isa StatisticsError || e isa ZeroDivisionError) || rethrow(e)
            f.beta = f.beta
        end
    end

    for h in m.H
        if h.employer === nothing
            h.skill = max(1, h.skill / (1 + m.ds))
        else
            h.skill = h.skill * (1 + m.ds / 10)
        end
    end

    # Set Wage and Price level
    for f in vcat(m.FC, m.FK)
        try
            f.W0 = f.W0 * (1 + m.thetaW * safediv(f.NT - length(f.employees), f.NT))
        catch e
            e isa ZeroDivisionError || rethrow(e)
            f.W0 = f.W0
        end

        f.mu = max(0.5, f.mu * (1 + m.thetaMu * (f.cu - m.cuT) / m.cuT))
        try
            f.p =
                (1 + f.mu) * f.W0 * smean([h.skill for h in f.employees]) / smean([k.beta for k in f.K])
        catch e
            e isa StatisticsError || rethrow(e)
            f.p = f.p
        end
    end

    for h in m.H
        if h.employer !== nothing
            h.W = h.employer.W0 * h.skill
            h.UB = 0
        else
            h.W = 0
            h.UB = m.phi * m.avgw
        end
    end

    # Set interest rates
    m.G.rB = max(0, m.i + 0.5 * (m.i - m.iT) + 0.25 * (m.cu - m.cuT) - 0.25 * (m.u - m.uT))
    m.B.rL = m.G.rB + m.DrL

    for h in m.H
        p = try
            ks = collect(keys(h.C))
            (1 + m.tC) * fmean([f.p for f in ks], [h.C[f] for f in ks])
        catch e
            e isa StatisticsError || rethrow(e)
            (1 + m.tC) * smean([f.p for f in m.FC])
        end
        h.CT = max(0, (m.ay * (h.UB + (1 - m.tW) * h.W) + m.av * h.M) / p)
    end

    pG = try
        fmean([f.p for f in m.FC], [f.G for f in m.FC])
    catch e
        e isa StatisticsError || rethrow(e)
        smean([f.p for f in m.FC])
    end
    m.G.GT = max(0, (m.dG * m.GDP + sum(values(m.G.T)) - sum(values(m.G.UB))) / pG)

    # unit measure is now beta
    for f in vcat(m.FC, m.FK)
        try
            f.IT = max(
                0,
                min(length(f.K), length(f.employees)) *
                max(0, 1 / m.cuT - safediv(1, f.cu)) +
                m.dK * min(length(f.K), length(f.employees)),
            )
        catch e
            e isa ZeroDivisionError || rethrow(e)
            f.IT = m.dK * min(length(f.K), length(f.employees))
        end
    end

    # Capital Goods orders are based only on desired I

    # Labour Market
    for fc in m.FC
        try
            fc.NT = min(
                length(fc.K),
                ceil(
                    Int,
                    (1 + m.rhoC) * (fc.G + sum(values(fc.C))) /
                    smean([k.beta for k in fc.K]),
                ),
            )
        catch e
            e isa StatisticsError || rethrow(e)
            fc.NT = 0
        end
    end
    for fk in m.FK
        try
            fk.NT = min(
                length(fk.K),
                ceil(Int, (sum(values(fk.Ip)) + fk.IT) / smean([k.beta for k in fk.K])),
            )
        catch e
            e isa StatisticsError || rethrow(e)
            fk.NT = 0
        end
    end

    for f in m.FC
        expW = try
            f.NT * f.W0 * smean([h.skill for h in f.employees])
        catch
            f.NT * f.W0
        end
        expI = f.IT * sum([k.p for k in f.K])
        maxexp = max(
            0,
            m.rhoL * sum([max(0, k.p * (1 - m.dK * k.age)) for k in f.K]) - f.L + f.M,
        )
        if expW + expI > maxexp
            f.NT = ceil(Int, f.NT * maxexp / (expW + expI))
            f.IT = ceil(f.IT * maxexp / (expW + expI))
        end
    end

    for f in m.FK
        expW = try
            f.NT * f.W0 * smean([h.skill for h in f.employees])
        catch
            f.NT * f.W0
        end
        maxexp = max(
            0,
            m.rhoL * sum([max(0, k.p * (1 - m.dK * k.age)) for k in f.K]) - f.L + f.M,
        )
        if expW > maxexp
            f.NT = ceil(Int, f.NT * maxexp / expW)
        end
    end

    # Over buy
    while any([sum([fc.Ip[fk] * fk.beta for fk in m.FK]) > fc.IT for fc in m.FC])
        fc =
            choice([fc for fc in m.FC if sum([fc.Ip[fk] * fk.beta for fk in m.FK]) > fc.IT])
        fk = choice([fk for fk in keys(fc.Ip) if fc.Ip[fk] > 0])
        d = min(
            fc.Ip[fk],
            ceil((sum([fc.Ip[fk] * fk.beta for fk in m.FK]) - fc.IT) / fk.beta),
        )
        fc.Ip[fk] -= d
        fk.Ip[fc] -= d
    end

    # Over sell
    while any([sum(values(fk.Ip)) > floor(sum([k.beta for k in fk.K])) for fk in m.FK])
        fk = choice([
            fk for fk in m.FK if sum(values(fk.Ip)) > floor(sum([k.beta for k in fk.K]))
        ])
        fc = choice([fc for fc in m.FC if fk.Ip[fc] > 0])
        d = min(fk.Ip[fc], sum(values(fk.Ip)) - floor(sum([k.beta for k in fk.K])))
        fc.Ip[fk] -= d
        fk.Ip[fc] -= d
    end

    # Under buy
    while any([sum([fc.Ip[fk] * fk.beta for fk in m.FK]) < fc.IT for fc in m.FC]) && any([sum(values(fk.Ip)) < floor(sum([k.beta for k in fk.K])) for fk in m.FK])
        fc =
            choice([fc for fc in m.FC if sum([fc.Ip[fk] * fk.beta for fk in m.FK]) < fc.IT])
        fk = sort(
            [fk for fk in m.FK if sum(values(fk.Ip)) < floor(sum([k.beta for k in fk.K]))];
            by = fk -> fk.p / fk.beta,
        )[1]
        fc.Ip[fk] += 1
        fk.Ip[fc] += 1
    end

    while length([f for f in vcat(m.FC, m.FK) if length(f.employees) > f.NT]) > 0
        f = choice([f for f in vcat(m.FC, m.FK) if length(f.employees) > f.NT])
        h = sort(f.employees; by = h -> h.skill)[1]
        f.employees = [ah for ah in f.employees if ah !== h]
        h.employer = nothing
        h.W = 0
        h.UB = m.phi * m.avgw
    end

    while (sum([length(f.employees) for f in vcat(m.FC, m.FK)]) < m.NH) && any([length(f.employees) < f.NT for f in vcat(m.FC, m.FK)])
        f = choice([f for f in vcat(m.FC, m.FK) if length(f.employees) < f.NT])
        h = sort([h for h in m.H if h.employer === nothing]; by = h -> h.skill)[end]
        push!(f.employees, h)
        h.employer = f
        h.skill = h.skill * (1 + m.ds)
        h.W = f.W0 * h.skill
        h.UB = 0
    end

    ### First Debt Emission and wage and ub payment (EDIT MOVED)
    for h in m.H
        h.T = 0
        m.G.T[h] = 0
    end
    for f in vcat(m.FC, m.FK)
        for h in m.H
            f.W[h] = 0
        end
        for h in f.employees
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
        end
    end

    for h in m.H
        m.G.UB[h] = h.UB
        h.M += h.UB
        m.B.M[h] += h.UB
        m.B.B += h.UB
        m.G.B += h.UB
    end

    # Production
    # NO WORKER - KC matching
    for h in m.H
        h.machine = nothing
    end

    for f in vcat(m.FC, m.FK)
        for k in f.K
            k.worker = nothing
        end
        f.K = sort(f.K; by = k -> k.beta, rev = true)
        for k in f.K
            hs = sort(
                [h for h in f.employees if (h.machine === nothing) && (h.skill >= k.beta)];
                by = h -> h.skill,
            )
            if length(hs) > 0
                h = hs[1]
                k.worker = h
                h.machine = k
            end
        end
    end
    for f in m.FC
        f.Y = sum([k.beta for k in f.K if k.worker !== nothing])
    end
    for f in m.FK
        f.Y = floor(sum([k.beta for k in f.K if k.worker !== nothing]))
    end
    for f in vcat(m.FC, m.FK)
        try
            f.cu = safediv(length([k for k in f.K if k.worker !== nothing]), length(f.K))
        catch e
            e isa ZeroDivisionError || rethrow(e)
            f.cu = 0
        end
    end

    # Consumpion good market
    Hsh = try
        safediv(sum([h.CT for h in m.H]), (sum([h.CT for h in m.H]) + m.G.GT))
    catch e
        e isa ZeroDivisionError || rethrow(e)
        1
    end

    # first set consumption

    # Over selling
    while any([sum(values(f.C)) - f.Y * Hsh > tol for f in m.FC])
        f = choice([f for f in m.FC if sum(values(f.C)) - f.Y * Hsh > tol])
        h = choice([h for h in keys(f.C) if f.C[h] > 0])
        d = min(sum(values(f.C)) - f.Y * Hsh, f.C[h])
        f.C[h] -= d
        h.C[f] -= d
    end

    # Over buying  (M is already increased of DI)
    while any([
        (
            sum(values(h.C)) - h.CT > tol ||
            sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) - h.M > tol
        ) && sum(values(h.C)) > tol for h in m.H
    ])
        h = choice([
            h for h in m.H if (
                (sum(values(h.C)) - h.CT > tol) ||
                (sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) - h.M > tol)
            ) && (sum(values(h.C)) > tol)
        ])
        f = choice([f for f in keys(h.C) if h.C[f] > 0])
        d = min(
            h.C[f],
            max(
                sum(values(h.C)) - h.CT,
                sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) - h.M,
            ),
        )
        f.C[h] -= d
        h.C[f] -= d
    end

    # Sell Remaining
    while any([f.Y * Hsh - sum(values(f.C)) > tol for f in m.FC]) && any([
        sum(values(h.C)) < h.CT && sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) < h.M for
        h in m.H
    ])
        h = choice([
            h for h in m.H if (sum(values(h.C)) < h.CT) &&
                (sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]) < h.M)
        ])
        f = sort([f for f in m.FC if f.Y * Hsh - sum(values(f.C)) > tol]; by = f -> f.p)[1]
        d = min(
            f.Y * Hsh - sum(values(f.C)),
            h.CT - sum(values(h.C)),
            h.M - sum([h.C[f] * (1 + m.tC) * f.p for f in m.FC]),
        )
        f.C[h] += d
        h.C[f] += d
    end

    # then move assets
    for h in m.H
        for f in m.FC
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
        end
    end

    # Gvt expenditure
    for f in m.FC
        m.G.G[f] = 0
        f.G = 0
    end
    while any([f.Y - (sum(values(f.C)) + f.G) > tol for f in m.FC]) && (m.G.GT - sum(values(m.G.G)) > tol)
        f = choice([f for f in m.FC if f.Y - (sum(values(f.C)) + f.G) > tol])
        d = min(m.G.GT - sum(values(m.G.G)), f.Y - (sum(values(f.C)) + f.G))
        m.G.G[f] += d
        f.G += d
    end
    for f in m.FC
        f.M += f.G * f.p
        m.B.M[f] += f.G * f.p
        m.B.B += f.G * f.p
        m.G.B += f.G * f.p
    end

    # Investment market
    # As before, first clean, later move asset
    for fk in m.FK
        ICsh = try
            safediv(fk.Y, (sum(values(fk.Ip)) + ceil(fk.IT / fk.beta)))
        catch e
            e isa ZeroDivisionError || rethrow(e)
            0
        end
        for fc in m.FC
            fk.I[fc] = floor(fk.Ip[fc] * ICsh)
            fc.I[fk] = floor(fc.Ip[fk] * ICsh)
        end
    end

    while any([sum(values(fk.I)) > fk.Y for fk in m.FK])
        fk = choice([fk for fk in m.FK if sum(values(fk.I)) > fk.Y])
        fc = choice([fc for fc in keys(fk.I) if fk.I[fc] > 0])
        fk.I[fc] -= 1
        fc.I[fk] -= 1
    end

    for fk in m.FK
        fk.IK = round(Int, min(ceil(fk.IT / fk.beta), fk.Y - sum(values(fk.I))))
    end

    while any([sum(values(fk.I)) + fk.IK < fk.Y for fk in m.FK]) && any([sum([fc.I[fk] * fk.beta for fk in m.FK]) < fc.IT for fc in m.FC])
        fc = choice([fc for fc in m.FC if sum([fc.I[fk] * fk.beta for fk in m.FK]) < fc.IT])
        fk = sort(
            [fk for fk in m.FK if sum(values(fk.I)) + fk.IK < fk.Y];
            by = fk -> fk.p / fk.beta,
        )[1]
        fk.I[fc] += 1
        fc.I[fk] += 1
    end

    for fk in m.FK
        fk.IK = round(Int, fk.Y - sum(values(fk.I)))
    end

    for fk in m.FK
        append!(fk.K, [CapitalGood(fk.beta, fk.p) for _ = 1:fk.IK])
        for fc in m.FC
            append!(fc.K, [CapitalGood(fk.beta, fk.p) for _ = 1:round(Int, fk.I[fc])])
            fc.L += fk.p * fk.I[fc]
            m.B.L[fc] += fk.p * fk.I[fc]
            fc.M += fk.p * fk.I[fc]
            m.B.M[fc] += fk.p * fk.I[fc]
            fc.M -= fk.p * fk.I[fc]
            m.B.M[fc] -= fk.p * fk.I[fc]
            fk.M += fk.p * fk.I[fc]
            m.B.M[fk] += fk.p * fk.I[fc]
        end
    end

    # Chiusura del circuito
    for f in vcat(m.FC, m.FK)
        f.intL = f.L * m.B.rL
        m.B.intL[f] = f.intL
        f.M -= f.intL
        m.B.M[f] -= f.intL
        D = max(0, min(f.L, f.M))
        f.L -= D
        f.M -= D
        m.B.L[f] -= D
        m.B.M[f] -= D
    end

    # Bond interest
    m.B.intB = m.G.rB * m.B.B
    m.G.intB = m.G.rB * m.G.B
    m.B.B += m.B.intB
    m.G.B += m.G.intB

    # Failures
    for f in vcat(m.FC, m.FK)
        f.detL = 0
        f.detM = 0
        m.B.detL[f] = 0
        m.B.detM[f] = 0
        if f.M - f.L + sum([max(0, k.p * (1 - m.dK * k.age)) for k in f.K]) < 0
            f.age = 0
            f.detL = f.L
            f.detM = f.M
            f.L = 0
            f.M = 0
            m.B.detL[f] = f.detL
            m.B.detM[f] = f.detM
            m.B.L[f] = 0
            m.B.M[f] = 0
        end
    end

    # Profits
    for f in vcat(m.FC, m.FK)
        V = f.M - f.L + sum([max(0, k.p * (1 - m.dK * k.age)) for k in f.K])
        for h in m.H
            h.P[f] = max(0, min(f.M, V)) * h.M / sum([hh.M for hh in m.H])
            f.P[h] = h.P[f]
        end
    end

    PB = max(0, (1 - m.crT) * sum(values(m.B.L)) + m.B.B - sum(values(m.B.M)))
    for h in m.H
        h.P[m.B] = max(0, PB * h.M / sum([hh.M for hh in m.H]))
        m.B.P[h] = max(0, PB * h.M / sum([hh.M for hh in m.H]))
    end

    for h in m.H
        for f in vcat(m.FC, m.FK)
            f.M -= f.P[h]
            m.B.M[f] -= f.P[h]
            h.M += f.P[h]
            m.B.M[h] += f.P[h]
        end
        h.M += h.P[m.B]
        m.B.M[h] += h.P[m.B]
        h.T += m.tP * sum(values(h.P))
        m.G.T[h] += m.tP * sum(values(h.P))
        h.M -= m.tP * sum(values(h.P))
        m.B.M[h] -= m.tP * sum(values(h.P))
        m.B.B -= m.tP * sum(values(h.P))
        m.G.B -= m.tP * sum(values(h.P))
    end

    for h in m.H
        WTaxes = max(0, m.tM * h.M)
        h.T += WTaxes
        m.G.T[h] += WTaxes
        h.M -= WTaxes
        m.B.M[h] -= WTaxes
        m.B.B -= WTaxes
        m.G.B -= WTaxes
    end

    # Capital depreciation
    for f in vcat(m.FC, m.FK)
        for k in f.K
            k.age += 1
        end
        f.K = [k for k in f.K if k.age < (1 / m.dK)]
    end

    try
        m.i = fmean([f.p for f in m.FC], [sum(values(f.C)) for f in m.FC]) / m.avgp - 1
        m.avgp = fmean([f.p for f in m.FC], [sum(values(f.C)) for f in m.FC])
    catch e
        e isa StatisticsError || rethrow(e)
        m.i = smean([f.p for f in m.FC]) / m.avgp - 1
        m.avgp = smean([f.p for f in m.FC])
    end

    m.cu = try
        fmean([f.cu for f in vcat(m.FC, m.FK)], [length(f.K) for f in vcat(m.FC, m.FK)])
    catch e
        e isa StatisticsError || rethrow(e)
        0
    end

    m.u = length([h for h in m.H if h.employer === nothing]) / m.NH

    m.avgw = try
        fmean([f.W0 for f in m.FC], [length(f.employees) for f in m.FC])
    catch e
        e isa StatisticsError || rethrow(e)
        smean([f.W0 for f in m.FC])
    end

    m.GDP =
        sum([f.p * (sum(values(f.C)) + f.G) for f in m.FC]) + sum([f.p * sum(values(f.I)) for f in m.FK])

    m.G.deficit =
        sum(values(m.G.T)) - sum([m.G.G[f] * f.p for f in keys(m.G.G)]) - sum(values(m.G.UB)) -
        m.G.intB

    return m
end

# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------
m = Model()
data = Model[]
@showprogress for _ = 1:m.TMAX
    step!(m)
    push!(data, deepcopy(m))
end
serialize("07_data.jls", data)          # pickle.dump equivalent

# ---------------------------------------------------------------------
data = deserialize("07_data.jls")       # pickle.load equivalent

# ---------------------------------------------------------------------
# Consistency check  (Python `data[2:]` -> Julia `data[3:end]`)
# ---------------------------------------------------------------------
println("Consistency checks: they should be false")

# BS / M
println(
    "M ",
    any([
        abs(sum([a.M for a in vcat(s.H, s.FC, s.FK)]) - sum(values(s.B.M))) > tol for
        s in data
    ]),
)
# L
println(
    "L ",
    any([
        abs(sum([a.L for a in vcat(s.FC, s.FK)]) - sum(values(s.B.L))) > tol for s in data
    ]),
)
# B
println("B ", any([abs(s.G.B - s.B.B) > tol for s in data]))

# FOF / H
println(
    "H ",
    any([
        any([
            abs(
                -sum([h.C[f] * f.p for f in s.FC]) + h.UB + h.W + sum(values(h.P)) - h.T - (h.M - h.M0),
            ) > tol for h in s.H
        ]) for s in data[3:end]
    ]),
)

# FC
println(
    "FC ",
    any([
        any([
            abs(
                +sum(values(f.C)) * f.p + f.G * f.p - sum([f.I[fk] * fk.p for fk in s.FK]) - sum(values(f.W)) -
                sum(values(f.P)) - f.intL - (f.M - f.M0) +
                (f.L - f.L0) +
                f.detL - f.detM,
            ) > tol for f in s.FC
        ]) for s in data[3:end]
    ]),
)

# FK
println(
    "FK ",
    any([
        any([
            abs(
                +sum(values(f.I)) * f.p - sum(values(f.W)) - sum(values(f.P)) - f.intL -
                (f.M - f.M0) +
                (f.L - f.L0) +
                f.detL - f.detM,
            ) > tol for f in s.FK
        ]) for s in data[3:end]
    ]),
)

# B
println(
    "B ",
    any([
        abs(
            -sum(values(s.B.P)) +
            sum(values(s.B.intL)) +
            s.B.intB +
            (sum(values(s.B.M)) - s.B.M0) - (sum(values(s.B.L)) - s.B.L0) -
            (s.B.B - s.B.B0) + sum(values(s.B.detM)) - sum(values(s.B.detL)),
        ) > tol for s in data[3:end]
    ]),
)

# G
println(
    "G ",
    any([
        abs(
            -sum([s.G.G[f] * f.p for f in s.FC]) - sum(values(s.G.UB)) + sum(values(s.G.T)) -
            s.G.intB + (s.G.B - s.G.B0),
        ) > tol for s in data[3:end]
    ]),
)

# ---------------------------------------------------------------------
# PLOTS
# Each `let ... display(plt) end` block corresponds to one `# %%` cell.
# For headless runs, replace `display(plt)` with `savefig(plt, "name.png")`.
# ---------------------------------------------------------------------
let
    plt = plot([sum([sum(values(f.C)) for f in s.FC]) for s in data], label = "C")
    plot!(plt, [sum([sum(values(f.C)) + f.G for f in s.FC]) for s in data], label = "C+G")
    plot!(plt, [sum([f.G for f in s.FC]) for s in data], label = "G")
    plot!(plt, [sum([f.Y for f in s.FC]) for s in data], label = "Y")
    display(plt)
end

let
    plt = plot([sum([f.IT for f in s.FC]) for s in data], label = "ITC")
    plot!(plt, [sum([f.IT for f in s.FK]) for s in data], label = "ITK")
    plot!(plt, [sum([sum(values(f.I)) for f in s.FK]) for s in data], label = "IC")
    display(plt)
end

let
    plt = plot([sum([f.NT for f in s.FC]) for s in data], label = "NTC")
    plot!(plt, [sum([f.NT for f in s.FK]) for s in data], label = "NTK")
    plot!(plt, [sum([length(f.employees) for f in s.FC]) for s in data], label = "NC")
    plot!(plt, [sum([length(f.employees) for f in s.FK]) for s in data], label = "NK")
    display(plt)
end

let
    plt = plot([sum([length(f.K) for f in s.FC]) for s in data], label = "KC")
    plot!(plt, [sum([length(f.K) for f in s.FK]) for s in data], label = "KK")
    display(plt)
end

let
    plt = plot([sum([h.CT for h in s.H]) for s in data], label = "CT")
    plot!(plt, [s.G.GT for s in data], label = "GT")
    display(plt)
end

let
    plt = plot([sum([length(f.employees) for f in s.FC]) for s in data], label = "NC")
    plot!(plt, [sum([length(f.K) for f in s.FC]) for s in data], label = "KC")
    plot!(plt, [sum([f.Y for f in s.FC]) for s in data], label = "Y")
    display(plt)
end

let
    plt = plot([s.cu for s in data], label = "cu")
    plot!(plt, [s.i for s in data], label = "i")
    plot!(plt, [s.u for s in data], label = "u")
    display(plt)
end

let
    plt = plot([sum([h.W for h in s.H]) for s in data], label = "W")
    plot!(plt, [sum([h.UB for h in s.H]) for s in data], label = "UB")
    plot!(plt, [sum([sum(values(h.P)) for h in s.H]) for s in data], label = "P")
    plot!(plt, [sum([h.M for h in s.H]) for s in data], label = "M")
    display(plt)
end

let
    plt = plot([s.avgp for s in data], label = "p")
    plot!(plt, [s.avgw for s in data], label = "w")
    display(plt)
end

let
    plt = plot(
        [sum([-sum([h.C[f] * f.p for f in s.FC]) for h in s.H]) for s in data],
        label = "pC",
    )
    plot!(plt, [sum([h.UB for h in s.H]) for s in data], label = "UB")
    plot!(plt, [sum([h.W for h in s.H]) for s in data], label = "W")
    plot!(plt, [sum([sum(values(h.P)) for h in s.H]) for s in data], label = "P")
    plot!(plt, [sum([-h.T for h in s.H]) for s in data], label = "T")
    plot!(plt, [sum([-(h.M - h.M0) for h in s.H]) for s in data], label = "dM")
    display(plt)
end

let
    plt = plot([sum(values(s.B.P)) for s in data], label = "B")
    plot!(plt, [sum([sum(values(f.P)) for f in s.FC]) for s in data], label = "FC")
    plot!(plt, [sum([sum(values(f.P)) for f in s.FK]) for s in data], label = "FK")
    display(plt)
end

let
    plt = plot([s.crT for s in data], label = "tgt")
    plot!(plt, [s.B.B for s in data], label = "B")
    plot!(plt, [sum(values(s.B.L)) for s in data], label = "L")
    plot!(plt, [sum(values(s.B.M)) for s in data], label = "M")
    display(plt)
end

let
    plt = plot([sum([f.M for f in s.FC]) for s in data], label = "MFC")
    plot!(plt, [sum([f.M for f in s.FK]) for s in data], label = "MFK")
    display(plt)
end

let
    plt = plot([-sum([s.G.G[f] * f.p for f in s.FC]) for s in data], label = "pG")
    plot!(plt, [-sum(values(s.G.UB)) for s in data], label = "UB")
    plot!(plt, [sum(values(s.G.T)) for s in data], label = "T")
    plot!(plt, [s.G.intB for s in data], label = "iB")
    plot!(plt, [s.G.B - s.G.B0 for s in data], label = "dB")
    plot!(
        plt,
        [
            -sum([s.G.G[f] * f.p for f in s.FC]) - sum(values(s.G.UB)) +
            sum(values(s.G.T)) +
            s.G.intB +
            (s.G.B - s.G.B0) for s in data
        ],
        label = "delta",
    )
    display(plt)
end

let
    plt = plot([-sum(values(s.B.P)) for s in data], label = "P")
    plot!(plt, [sum(values(s.B.intL)) for s in data], label = "iL")
    plot!(plt, [s.B.intB for s in data], label = "iB")
    plot!(plt, [(sum(values(s.B.M)) - s.B.M0) for s in data], label = "dM")
    plot!(plt, [-(sum(values(s.B.L)) - s.B.L0) for s in data], label = "dL")
    plot!(plt, [-(s.B.B - s.B.B0) for s in data], label = "dB")
    plot!(plt, [sum(values(s.B.detM)) for s in data], label = "detM")
    plot!(plt, [-sum(values(s.B.detL)) for s in data], label = "detL")
    display(plt)
end

let
    plt = plot([sum([sum(values(f.C)) * f.p for f in s.FC]) for s in data], label = "pC")
    plot!(plt, [sum([f.G * f.p for f in s.FC]) for s in data], label = "pG")
    plot!(
        plt,
        [sum([-sum([f.I[fk] * fk.p for fk in s.FK]) for f in s.FC]) for s in data],
        label = "pI",
    )
    plot!(plt, [sum([-sum(values(f.W)) for f in s.FC]) for s in data], label = "W")
    plot!(plt, [sum([-sum(values(f.P)) for f in s.FC]) for s in data], label = "P")
    plot!(plt, [sum([-f.intL for f in s.FC]) for s in data], label = "iL")
    plot!(plt, [sum([-(f.M - f.M0) for f in s.FC]) for s in data], label = "dM")
    plot!(plt, [sum([(f.L - f.L0) for f in s.FC]) for s in data], label = "dL")
    plot!(plt, [sum([f.detL for f in s.FC]) for s in data], label = "detL")
    plot!(plt, [sum([-f.detM for f in s.FC]) for s in data], label = "detM")
    plot!(
        plt,
        [
            sum([
                +sum(values(f.C)) * f.p + f.G * f.p - sum([f.I[fk] * fk.p for fk in s.FK]) - sum(values(f.W)) -
                sum(values(f.P)) - f.intL - (f.M - f.M0) +
                (f.L - f.L0) +
                f.detL - f.detM for f in s.FC
            ]) for s in data
        ],
        label = "delta",
    )
    display(plt)
end

let
    plt = plot([mean([f.p for f in s.FC]) for s in data], label = "pc")
    plot!(plt, [mean([f.p for f in s.FK]) for s in data], label = "pk")
    plot!(plt, [mean([f.W0 for f in s.FC]) for s in data], label = "wc")
    plot!(plt, [mean([f.W0 for f in s.FK]) for s in data], label = "wk")
    plot!(plt, [mean([f.mu for f in s.FC]) for s in data], label = "mc")
    plot!(plt, [mean([f.mu for f in s.FK]) for s in data], label = "mk")
    display(plt)
end

let
    plt = plot([sum([h.W for h in s.H]) for s in data], label = "WH")
    plot!(
        plt,
        [sum([sum(values(f.W)) for f in vcat(s.FC, s.FK)]) for s in data],
        label = "WF",
    )
    plot!(plt, [s.NH * (1 - s.u) for s in data], label = "N")
    display(plt)
end

let
    plt = plot([sum([sum(values(f.I)) * f.p for f in s.FK]) for s in data], label = "pI")
    plot!(plt, [sum([-sum(values(f.W)) for f in s.FK]) for s in data], label = "W")
    plot!(plt, [sum([-sum(values(f.P)) for f in s.FK]) for s in data], label = "P")
    plot!(plt, [sum([-f.intL for f in s.FK]) for s in data], label = "iL")
    plot!(plt, [sum([-(f.M - f.M0) for f in s.FK]) for s in data], label = "dM")
    plot!(plt, [sum([(f.L - f.L0) for f in s.FK]) for s in data], label = "dL")
    plot!(plt, [sum([f.detL for f in s.FK]) for s in data], label = "detL")
    plot!(plt, [sum([-f.detM for f in s.FK]) for s in data], label = "detM")
    plot!(
        plt,
        [
            sum([
                +sum(values(f.I)) * f.p - sum(values(f.W)) - sum(values(f.P)) - f.intL -
                (f.M - f.M0) +
                (f.L - f.L0) +
                f.detL - f.detM for f in s.FK
            ]) for s in data
        ],
        label = "delta",
    )
    display(plt)
end

let
    plt = plot([mean([f.beta for f in s.FK]) for s in data], label = "beta")
    plot!(plt, [log(mean([f.beta for f in s.FK])) for s in data], label = "logbeta")
    plot!(plt, [mean([h.skill for h in s.H]) for s in data], label = "skill")
    display(plt)
end

let
    display(histogram([h.M for h in data[end].H], legend = false))
end

let
    plt = plot([s.G.GT for s in data], label = "GT")
    plot!(plt, [sum([h.CT for h in s.H]) for s in data], label = "CT")
    plot!(plt, [sum([h.CT for h in s.H]) + s.G.GT for s in data], label = "YT")
    plot!(plt, [sum([f.Y for f in s.FC]) for s in data], label = "Y")
    plot!(
        plt,
        [sum([sum(values(h.C)) for h in s.H]) + sum(values(s.G.G)) for s in data],
        label = "S",
    )
    plot!(plt, [sum(values(s.G.G)) for s in data], label = "G")
    plot!(plt, [sum([sum(values(h.C)) for h in s.H]) for s in data], label = "C")
    display(plt)
end

let
    plt = plot([sum([sum(values(f.Ip)) for f in s.FK]) for s in data], label = "Ip")
    plot!(plt, [sum([f.IT for f in s.FC]) for s in data], label = "IT")
    plot!(plt, [sum([sum(values(f.I)) for f in s.FC]) for s in data], label = "I")
    plot!(plt, [sum([length(f.employees) for f in s.FK]) for s in data], label = "NK")
    plot!(plt, [sum([f.Y for f in s.FK]) for s in data], label = "YK")
    display(plt)
end

let
    display(histogram([f.p for f in vcat(data[end].FC, data[end].FK)], legend = false))
end

let
    display(histogram([f.beta for f in data[end].FK], legend = false))
end

let
    display(histogram([f.age for f in data[end].FK], legend = false))
end
