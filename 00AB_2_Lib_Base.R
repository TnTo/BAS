### BAS: Building towards Artificial Societies
# Model 0:
#   no Fixed Capital
#   It appears to need to be a full-employment model
#   In this form is completely a-cyclical
###
# Scenario Eq:
# Flat initial wealth
# Proportional profit distribution
###
# renv::init()
# renv::restore()
# renv::install("sfcr", prompt = FALSE)
# renv::install("devtools", prompt = FALSE)
devtools::install_github("TnTo/sfcr", ref = "sankey")
renv::install("tidyverse", prompt = FALSE)
# renv::install("networkD3", prompt = FALSE)
# renv::install("ggraph", prompt = FALSE)
renv::install("moments", prompt = FALSE)
renv::install("progress", prompt = FALSE)
# renv::install("profvis", prompt = FALSE)
renv::snapshot()
renv::status()

library(tidyverse)
library(moments)
library(progress)
library(profvis)

sample.vec <- function(x, ...) x[sample.int(length(x), ...)]

{
    ## prelude
    # eps <- 1e-25
    {
        tol <- 1e-2
        set.seed(8686)
    }

    ## Size
    {
        TMAX <- 250
        NH <- 1000
        NF <- 50
    }

    ## Constants
    {
        W0 <- 1 # Wage level
        ay <- 0.375 # Desired share of consumption out of income
        av <- 0.75 # Desired share of consumptio out of wealth
        beta <- 1.0 # Output per worker in units of goods
        mu <- 0.2 # Firms' mark-up
        d <- 0.03 # Target Gvt deficit
        tW <- 0.35 # Tax rate on wages
        tP <- 0.2 # Tax rate on profits
        tC <- 0.2 # Tax rate on Hs consumption
        phi <- 0.7 # Unemployment benefit amount (in percentage of wage level)
    }

    ## Initialization
    {
        MH <- array(NA, c(TMAX, NH))
        MF <- array(NA, c(TMAX, NF))
        M <- array(NA, c(TMAX))
        VH <- array(NA, c(TMAX, NH))
        VF <- array(NA, c(TMAX, NF))
        VG <- array(NA, c(TMAX))
        NetW <- array(NA, c(TMAX, NH))
        DI <- array(NA, c(TMAX, NH))
        CT <- array(NA, c(TMAX, NH))
        GT <- array(NA, c(TMAX))
        YT <- array(NA, c(TMAX))
        W <- array(NA, c(TMAX, NH, NF))
        UB <- array(NA, c(TMAX, NH))
        N <- array(NA, c(TMAX, NH, NF))
        Y <- array(NA, c(TMAX, NF))
        C <- array(NA, c(TMAX, NH, NF))
        G <- array(NA, c(TMAX, NF))
        P <- array(NA, c(TMAX, NH, NF))
        T <- array(NA, c(TMAX, NH))
        p <- array(NA, c(TMAX, NF))
        Hp <- array(NA, c(TMAX, NF))
        V <- array(NA, c(TMAX))
        GDP <- array(NA, c(TMAX))
        PI <- array(NA, c(TMAX))
        CPI <- array(NA, c(TMAX))
    }

    # Initiaal values
    {
        MH[1, ] <- 1 # Initialize with flat wealth distribution
        MF[1, ] <- 0
        M[1] <- sum(MH[1, ])
        DI[1, ] <- 0
        UB[1, ] <- 0
        T[1, ] <- 0
        GDP[1] <- 0
        p[1, ] <- 1
        PI[1] <- 1
        CPI[1] <- 1 * (1 + tC)
        N[1, , ] <- 0
        C[1, , ] <- 0
    }

    ## A model
    {
        pb <- progress_bar$new(total = TMAX)
        pb$tick()
        for (t in 2:TMAX) {
            CT[t, ] <- max(0, (ay * DI[t - 1, ] + av * MH[t - 1, ]) / CPI[t - 1]) # Desired Demand for Hs, in units of goods
            GT[t] <- max(0, (d * GDP[t - 1] + sum(T[t - 1, ]) - sum(UB[t - 1, ])) / PI[t - 1]) # Desired Demand for Gvt, in units of goods
            YT[t] <- sum(CT[t, ]) + GT[t] # Desired Demand

            # Job market
            N[t, , ] <- N[t - 1, , ]
            repeat {
                if ((sum(N[t, , ]) > NH) || ((sum(N[t, , ]) - 1) * beta >= YT[t])) {
                    # Fire
                    Fid <- sample.vec(which(colSums(N[t, , ]) != 0), 1)
                    Hid <- sample.vec(which(N[t, , Fid] != 0), 1)
                    N[t, Hid, Fid] <- 0
                } else if ((sum(N[t, , ]) < NH) && (sum(N[t, , ]) * beta < YT[t])) {
                    # Hire
                    Fid <- sample.int(NF, 1) # Randomly choose Firm for hiring
                    Hid <- sample.vec(which(rowSums(N[t, , ]) == 0))
                    N[t, Hid, Fid] <- 1
                } else {
                    break
                }
            }

            Y[t, ] <- beta * colSums(N[t, , ]) # Output in units of goods
            W[t, , ] <- W0 * N[t, , ] # Wages
            UB[t, ] <- 0
            UB[t, which(rowSums(N[t, , ]) == 0)] <- phi * W0 # Unemployment benefits
            NetW[t, ] <- (1 - tW) * rowSums(W[t, , ]) # Net Wages
            DI[t, ] <- NetW[t, ] + UB[t, ] # Disposable Income for Hs
            p[t, ] <- (1 + mu) * colSums(W[t, , ]) / Y[t, ] # price
            p[t, which(is.na(p[t, ]))] <- p[t - 1, which(is.na(p[t, ]))]
            Hp[t, ] <- (1 + tC) * p[t, ] # price after VAT (Hs price)

            # Consumption Goods market
            # Since there is no innovation, no wage dynamics and homogeneus mark-up the prices are uniform among Firms
            C[t, , ] <- C[t - 1, , ]
            HC <- CT[t, ] * sum(Y[t, ]) / YT[t]
            # Over-selling
            Fids <- which(colSums(C[t, , ]) - Y[t, ] > tol)
            if (length(Fids) > 0) {
                for (Fid in Fids) {
                    while (sum(C[t, , Fid] - Y[t, Fid] > tol)) {
                        Hid <- sample.vec(which(C[t, , Fid] > 0), 1)
                        d <- min(C[t, Hid, Fid], sum(C[t, , Fid]) - Y[t, Fid])
                        C[t, Hid, Fid] <- C[t, Hid, Fid] - d
                    }
                }
            }
            # Over-buying
            Hids <- which(rowSums(C[t, , ]) - HC > tol)
            if (length(Hids) > 0) {
                for (Hid in Hids) {
                    while (sum(C[t, Hid, ]) - HC[Hid] > tol) {
                        Fid <- sample.vec(which(C[t, Hid, ] > 0), 1)
                        d <- min(C[t, Hid, Fid], sum(C[t, Hid, ]) - HC[Hid])
                        C[t, Hid, Fid] <- C[t, Hid, Fid] - d
                    }
                }
            }
            # Fill unsatisfied demand
            repeat {
                Hids <- which(HC - rowSums(C[t, , ]) > tol)
                if (length(Hids > 0) && (sum(Y[t, ]) - sum(C[t, , ]) > tol)) {
                    for (Hid in Hids) {
                        Fid <- sample.vec(which(Y[t, ] - colSums(C[t, , ]) > 0), 1)
                        d <- min(HC[Hid] - sum(C[t, Hid, ]), Y[t, Fid] - sum(C[t, , Fid]))
                        C[t, Hid, Fid] <- C[t, Hid, Fid] + d
                    }
                } else {
                    break
                }
            }

            G[t, ] <- Y[t, ] - colSums(C[t, , ]) # Gvt Consumption, in units of goods

            P[t, , ] <- sweep(t(array((p[t, ] * Y[t, ] - colSums(W[t, , ])), c(NF, NH))), 1, MH[t - 1, ], "*") / sum(MH[t - 1, ]) # Firms' Profits (all distributed) -- proportional to MH
            T[t, ] <- tW * rowSums(W[t, , ]) + tP * rowSums(P[t, , ]) + tC * rowSums(sweep(C[t, , ], 2, p[t, ], "*")) # Taxes
            MH[t, ] <- MH[t - 1, ] + rowSums(P[t, , ]) + rowSums(W[t, , ]) + UB[t, ] - rowSums(sweep(C[t, , ], 2, p[t, ], "*")) - T[t, ] # Households' Money
            MF[t, ] <- MF[t - 1, ] + p[t, ] * Y[t, ] - colSums(W[t, , ]) - colSums(P[t, , ]) # Firms' Money
            M[t] <- M[t - 1] - (sum(T[t, ]) - sum(p[t, ] * G[t, ]) - sum(UB[t, ])) # Gvt's Money
            VH[t, ] <- MH[t, ] # Households' Net Wealth
            VF[t, ] <- MF[t, ] # Firms' Net Wealth
            VG[t] <- -M[t] # Gvt' Net Wealth
            V[t] <- sum(VH[t, ]) + sum(VF[t, ]) + VG[t] # System Total Wealth
            GDP[t] <- sum(p[t, ] * Y[t, ])
            PI[t] <- weighted.mean(p[t, ], Y[t, ]) # Price index
            CPI[t] <- (1 + tC) * PI[t] # Consumers price index

            pb$tick()
        }
    }

    {
        if (any(rowSums(MH[2:TMAX, ]) + rowSums(MF[2:TMAX, ]) - M[2:TMAX] > tol)) {
            print("M row in BS not consistent")
        }
        if (any(rowSums(VH[2:TMAX, ]) + rowSums(VF[2:TMAX, ]) + VG[2:TMAX] - V[2:TMAX] > tol)) {
            print("V row in BS not consistent")
        }
        if (any(MH[2:TMAX, ] - VH[2:TMAX, ] > tol)) {
            print("H column in BS not consistent (checked at agent level)")
        }
        if (any(MF[2:TMAX, ] - VF[2:TMAX, ] > tol)) {
            print("F column in BS not consistent (checked at agent level)")
        }
        if (any(-M[2:TMAX] - VG[2:TMAX] > tol)) {
            print("G column in BS not consistent")
        }
    }

    {
        if (any(
            -rowSums(sweep(C[2:TMAX, , ], c(1, 3), p[2:TMAX, ], "*"), dims = 2)
            + UB[2:TMAX, ]
                + rowSums(W[2:TMAX, , ], dims = 2)
                + rowSums(P[2:TMAX, , ], dims = 2)
                - T[2:TMAX, ]
                - (MH[2:TMAX, ] - MH[1:TMAX - 1, ])
            > tol
        )) {
            print("H column in TFM not consistent (checked ad agent level)")
        }
        if (any(
            p[2:TMAX, ] * rowSums(aperm(C[2:TMAX, , ], c(1, 3, 2)), dims = 2)
                + p[2:TMAX, ] * G[2:TMAX, ]
                - rowSums(aperm(W[2:TMAX, , ], c(1, 3, 2)), dims = 2)
                - rowSums(aperm(P[2:TMAX, , ], c(1, 3, 2)), dims = 2)
                - (MF[2:TMAX, ] - MF[1:TMAX - 1, ])
            > tol
        )) {
            print("F column in TFM not consistent (checked ad agent level)")
        }
        if (any(
            -rowSums(p[2:TMAX, ] * G[2:TMAX, ])
            - rowSums(UB[2:TMAX, ])
                + rowSums(T[2:TMAX, ])
                + (M[2:TMAX] - M[1:TMAX - 1])
            > tol
        )) {
            print("G column in TFM not consistent")
        }
    }
}

{{
    par(mfrow = c(2, 3))
    plot(rowSums(MH), type = "l", main = "MH")
    plot(rowSums(MF), type = "l", main = "MF")
    plot(M, type = "l", main = "MG")
    plot(rowSums(VH), type = "l", main = "VH")
    plot(rowSums(VF), type = "l", main = "VF")
    plot(VG, type = "l", main = "VG")
}

{
    par(mfrow = c(2, 3))
    plot(rowSums(p * rowSums(aperm(C, c(1, 3, 2)), dims = 2)), type = "l", main = "C")
    plot(rowSums(p * G), type = "l", main = "G")
    plot(rowSums(UB), type = "l", main = "UB")
    plot(rowSums(W), type = "l", main = "W")
    plot(rowSums(P), type = "l", main = "P")
    plot(rowSums(T), type = "l", main = "T")
}

{
    par(mfrow = c(1, 2))
    plot(PI, type = "l", main = "PI")
    plot(rowSums(N) / NH, type = "l", main = "N")
}

{
    par(mfrow = c(1, 3))
    plot(rowSums(C), type = "l", main = "C", ylim = c(0, max(rowSums(CT), na.rm = TRUE)))
    lines(rowSums(CT), lty = "dashed")
    plot(rowSums(G), type = "l", main = "G", ylim = c(0, max(GT, na.rm = TRUE)))
    lines(GT, lty = "dashed")
    plot(rowSums(Y), type = "l", main = "Y", ylim = c(0, max(YT, na.rm = TRUE)))
    lines(YT, lty = "dashed")
}

{
    par(mfrow = c(2, 2))
    plot(apply(MH, 1, mean), type = "l", main = "MH mean")
    plot(apply(MH, 1, var), type = "l", main = "MH var")
    plot(apply(MH, 1, skewness), type = "l", main = "MH skewness")
    plot(apply(MH, 1, kurtosis), type = "l", main = "MH kurtosis")
}

{
    par(mfrow = c(2, 2))
    boxplot(MH[TMAX, ], main = "MH")
    boxplot(rowSums(W[TMAX, , ]), main = "WH")
    boxplot(rowSums(C[TMAX, , ]), main = "CH")
    boxplot(rowSums(P[TMAX, , ]), main = "PH")
}}
