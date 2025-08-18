### BAS: Building towards Artificial Societies
# Model 4:
#   Fixed Capital in both consumption and capital firms
#   It appears to need to be a full-employment model
#   In this form is substantially a-cyclical
#   Credit introduced with a circuit model
###
# install.packages(c("devtools", "tidyverse", "moments", "progress", "profvis"))

library(tidyverse)
library(tidyr)
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

        # SET SCENARIO
        scenario <- "Lib" # "Eq" "Hist" "Lib" "Con" "Fat"
        M0 <- switch(scenario,
            "Eq" = "Flat",
            "Hist" = "Exp",
            "Lib" = "Flat",
            "Con" = "Exp",
            "Fat" = "LN"
        )
        PDist <- switch(scenario,
            "Eq" = "Flat",
            "Hist" = "Flat",
            "Lib" = "Prop",
            "Con" = "Prop",
            "Fat" = "Prop"
        )
    }

    ## Size
    {
        TMAX <- 500
        NH <- 1000
        NFC <- 50
        NFK <- 5
    }

    ## Constants
    {
        W0 <- 1 # Wage level
        thetaW <- 0.01 # wage update rate
        ay <- 0.6 # Desired share of consumption out of income
        av <- 0.2 # Desired share of consumptio out of wealth
        betaC <- 2.0 # Output per worker in units of goods in Consumption Goods Firms
        betaK <- 1.0 # Output per worker in units of goods in Capital Goods Firms
        dG <- 0.03 # Target Gvt deficit
        tW <- 0.35 # Tax rate on wages
        tP <- 0.2 # Tax rate on profits
        tC <- 0.2 # Tax rate on Hs consumption
        phi <- 0.7 # Unemployment benefit amount (in percentage of wage level)
        dK <- 0.1 # Capital depreciation/decay ratio
        rhoC <- 0.05 # Target production increase for Consumption firms
        cuT <- 0.8 # Target capacity utilization
        uT <- 0.05 # target unemployment rate
        iT <- 0.02 # target inflation rate
        thetaMu <- 0.1 # markup update rate
        mu0 <- 0.5 # initial markup
        DrL <- 0.05 # Bank premium on loans
        crT <- 0.08 # target capital rate
    }

    ## Initialization
    {
        MH <- array(NA, c(TMAX, NH))
        MFC <- array(NA, c(TMAX, NFC))
        MFK <- array(NA, c(TMAX, NFK))
        M <- array(NA, c(TMAX))
        VH <- array(NA, c(TMAX, NH))
        VFC <- array(NA, c(TMAX, NFC))
        VFK <- array(NA, c(TMAX, NFK))
        VB <- array(NA, c(TMAX))
        VG <- array(NA, c(TMAX))
        LFC <- array(NA, c(TMAX, NFC))
        LFK <- array(NA, c(TMAX, NFK))
        L <- array(NA, c(TMAX))
        DLFC <- array(NA, c(TMAX, NFC))
        DLFK <- array(NA, c(TMAX, NFK))
        DL <- array(NA, c(TMAX))
        RLFC <- array(NA, c(TMAX, NFC))
        RLFK <- array(NA, c(TMAX, NFK))
        RL <- array(NA, c(TMAX))
        B <- array(NA, c(TMAX))
        KC <- array(NA, c(TMAX, NFC))
        KK <- array(NA, c(TMAX, NFK))
        K <- array(NA, c(TMAX))
        NetW <- array(NA, c(TMAX, NH))
        DI <- array(NA, c(TMAX, NH))
        CT <- array(NA, c(TMAX, NH))
        GT <- array(NA, c(TMAX))
        YT <- array(NA, c(TMAX))
        ICT <- array(NA, c(TMAX, NFC))
        IKT <- array(NA, c(TMAX, NFK))
        IT <- array(NA, c(TMAX))
        Ip <- array(NA, c(TMAX, NFC, NFK))
        WFC <- array(NA, c(TMAX, NH, NFC))
        WFK <- array(NA, c(TMAX, NH, NFK))
        W <- array(NA, c(TMAX, NH))
        W0FC <- array(NA, c(TMAX, NFC))
        W0FK <- array(NA, c(TMAX, NFK))
        UB <- array(NA, c(TMAX, NH))
        NCT <- array(NA, c(TMAX, NFC))
        NKT <- array(NA, c(TMAX, NFK))
        NC <- array(NA, c(TMAX, NH, NFC))
        NK <- array(NA, c(TMAX, NH, NFK))
        Y <- array(NA, c(TMAX, NFC))
        C <- array(NA, c(TMAX, NH, NFC))
        G <- array(NA, c(TMAX, NFC))
        S <- array(NA, c(TMAX, NFC))
        YK <- array(NA, c(TMAX, NFK))
        I <- array(NA, c(TMAX))
        IC <- array(NA, c(TMAX, NFC, NFK))
        IK <- array(NA, c(TMAX, NFK))
        PFC <- array(NA, c(TMAX, NH, NFC))
        PFK <- array(NA, c(TMAX, NH, NFK))
        PB <- array(NA, c(TMAX, NH))
        P <- array(NA, c(TMAX, NH))
        T <- array(NA, c(TMAX, NH))
        pC <- array(NA, c(TMAX, NFC))
        pK <- array(NA, c(TMAX, NFK))
        pKC <- array(NA, c(TMAX, NFC))
        muC <- array(NA, c(TMAX, NFC))
        muK <- array(NA, c(TMAX, NFK))
        HpC <- array(NA, c(TMAX, NFC))
        V <- array(NA, c(TMAX))
        GDP <- array(NA, c(TMAX))
        KCu <- array(NA, c(TMAX, NFC))
        cuC <- array(NA, c(TMAX, NFC))
        KKu <- array(NA, c(TMAX, NFK))
        cuK <- array(NA, c(TMAX, NFK))
        Ku <- array(NA, c(TMAX))
        cu <- array(NA, c(TMAX))
        i <- array(NA, c(TMAX))
        u <- array(NA, c(TMAX))
        rB <- array(NA, c(TMAX))
        rL <- array(NA, c(TMAX))
        cr <- array(NA, c(TMAX))
    }

    # Initial values
    {
        MH[1, ] <- switch(M0,
            "Flat" = 1,
            "Exp" = rexp(NH),
            "LN" = rlnorm(NH)
        )
        MFC[1, ] <- 0
        MFK[1, ] <- 0
        M[1] <- sum(MH[1, ])
        LFC[1, ] <- 0
        LFK[1, ] <- 0
        L[1] <- 0
        B[1] <- 0
        VB[1] <- 0
        DI[1, ] <- 0
        KC[1, ] <- 10
        KK[1, ] <- 10
        K[1] <- sum(KC[1, ]) + sum(KK[1, ])
        IC[1, , ] <- 0
        KCu[1, ] <- 0
        KKu[1, ] <- 0
        cuC[1, ] <- cuT
        cuK[1, ] <- cuT
        UB[1, ] <- 0
        T[1, ] <- 0
        GDP[1] <- 0
        S[1, ] <- 0.01 * NH / NFC
        muC[1, ] <- mu0
        muK[1, ] <- mu0
        pC[1, ] <- (1 + muC[1, ]) * W0 / betaC
        HpC[1, ] <- (1 + tC) * pC[1, ]
        pK[1, ] <- (1 + muK[1, ]) * W0 / betaK
        pKC[1, ] <- (1 + mean(muK[1, ])) * W0 / betaK
        W0FC[1, ] <- W0
        W0FK[1, ] <- W0
        NC[1, , ] <- 0
        NK[1, , ] <- 0
        C[1, , ] <- 0.01
        G[1, ] <- 0.01
        i[1] <- iT
        cu[1] <- cuT
        u[1] <- uT
    }

    ## A model
    {
        pb <- progress_bar$new(total = TMAX)
        pb$tick()
        for (t in 2:TMAX) {
            W0FC[t, ] <- W0FC[t - 1, ] * (1 + thetaW * (NCT[t - 1, ] - colSums(NC[t - 1, , ])) / NCT[t - 1, ])
            W0FK[t, ] <- W0FK[t - 1, ] * (1 + thetaW * (NKT[t - 1, ] - colSums(NK[t - 1, , ])) / NKT[t - 1, ])
            W0FC[t, ] <- ifelse(is.na(W0FC[t, ]), W0FC[t - 1, ], W0FC[t, ])
            W0FK[t, ] <- ifelse(is.na(W0FK[t, ]), W0FK[t - 1, ], W0FK[t, ])

            muC[t, ] <- muC[t - 1, ] * (1 + thetaMu * (cuC[t - 1, ] - cuT) / cuT)
            muK[t, ] <- muK[t - 1, ] * (1 + thetaMu * (cuK[t - 1, ] - cuT) / cuT)
            pC[t, ] <- (1 + muC[t, ]) * W0FC[t, ] / betaC # price
            pK[t, ] <- (1 + muK[t, ]) * W0FK[t, ] / betaK # price
            pC[t, ] <- ifelse(is.na(pC[t, ]), pC[t - 1, ], pC[t, ])
            HpC[t, ] <- (1 + tC) * pC[t, ] # price after VAT (Hs price)

            rB[t] <- max(0, i[t - 1] + 0.5 * (i[t - 1] - iT) + 0.25 * (cu[t - 1] - cuT) - 0.25 * (u[t - 1] - uT)) # Gvt Bonds' interest rate
            rL[t] <- max(0, rB[t] + DrL) # Loans' interest rate

            CT[t, ] <- pmax((ay * DI[t - 1, ] + av * MH[t - 1, ]) / replace_na(rowSums(C[t - 1, , ] * HpC[t, ]) / rowSums(C[t - 1, , ]), mean(HpC[t, ])), 0) # Desired Demand for Hs, in units of goods
            GT[t] <- max((dG * GDP[t - 1] + sum(T[t - 1, ]) - sum(UB[t - 1, ])) / ifelse(sum(G[t - 1, ]) == 0, mean(pC[t, ]), sum(G[t - 1, ] * pC[t, ]) / sum(G[t - 1, ])), 0) # Desired Demand for Gvt, in units of goods
            YT[t] <- sum(CT[t, ]) + GT[t] # Desired Demand
            ICT[t, ] <- ceiling(pmax(KCu[t - 1, ] * pmax(1 / cuT - 1 / cuC[t - 1, ], 0, na.rm = TRUE) + dK * KCu[t - 1, ], 0)) # Desired investments in units of capital goods for C Firms
            IKT[t, ] <- ceiling(pmax(KKu[t - 1, ] * pmax(1 / cuT - 1 / cuK[t - 1, ], 0, na.rm = TRUE) + dK * KKu[t - 1, ], 0)) # Desired investments in units of capital goods for K Firms
            IT[t] <- sum(ICT[t, ]) + sum(IKT[t, ])

            ### To normalize by expected production
            # Capital Goods Orders K->C
            Ip[t, , ] <- IC[t - 1, , ] # Promised investment
            # Over ordered
            while (any(rowSums(Ip[t, , ]) > ICT[t, ])) {
                FCids <- which(rowSums(Ip[t, , ]) > ICT[t, ])
                for (FCid in FCids) {
                    FKid <- sample.vec(which(Ip[t, FCid, ] > 0), 1)
                    d <- min(Ip[t, FCid, FKid], sum(Ip[t, FCid, ]) - ICT[t, FCid])
                    Ip[t, FCid, FKid] <- Ip[t, FCid, FKid] - d
                }
            }
            # Over promised
            while (any(colSums(Ip[t, , ]) > KK[t - 1, ] * betaK)) {
                FKids <- which(colSums(Ip[t, , ]) > KK[t - 1, ] * betaK)
                for (FKid in FKids) {
                    FCid <- sample.vec(which(Ip[t, , FKid] > 0), 1)
                    d <- min(Ip[t, FCid, FKid], sum(Ip[t, , FKid]) - KK[t - 1, FKid] * betaK)
                    Ip[t, FCid, FKid] <- Ip[t, FCid, FKid] - d
                }
            }

            # Under ordered
            while (any(rowSums(Ip[t, , ]) < ICT[t, ]) && any(colSums(Ip[t, , ]) < KK[t - 1, ] * betaK)) {
                FCid <- sample.vec(which(rowSums(Ip[t, , ]) < ICT[t, ]), 1)
                FKids <- which(colSums(Ip[t, , ]) < KK[t - 1, ] * betaK)
                FKid <- FKids[which.min(pK[t, FKids])]
                Ip[t, FCid, FKid] <- Ip[t, FCid, FKid] + 1
            }

            # Job market
            # Demand
            NCT[t, ] <- pmin(KC[t - 1, ], ceiling((1 + rhoC) * S[t - 1, ] / betaC))
            NKT[t, ] <- pmin(KK[t - 1, ], ceiling((colSums(Ip[t, , ]) + IKT[t, ]) / betaK))

            NC[t, , ] <- NC[t - 1, , ]
            NK[t, , ] <- NK[t - 1, , ]

            # Fire
            for (Fid in 1:NFC) {
                while (sum(NC[t, , Fid]) > NCT[t, Fid]) {
                    Hid <- sample.vec(which(NC[t, , Fid] != 0), 1)
                    NC[t, Hid, Fid] <- 0
                }
            }
            for (Fid in 1:NFK) {
                while (sum(NK[t, , Fid]) > NKT[t, Fid]) {
                    Hid <- sample.vec(which(NK[t, , Fid] != 0), 1)
                    NK[t, Hid, Fid] <- 0
                }
            }

            # Hiring
            while (
                (sum(NC[t, , ]) + sum(NK[t, , ]) < NH) &&
                    (any(colSums(NC[t, , ]) < NCT[t, ]) || any(colSums(NK[t, , ]) < NKT[t, ]))
            ) {
                FCids <- which(colSums(NC[t, , ]) < NCT[t, ])
                FKids <- which(colSums(NK[t, , ]) < NKT[t, ])
                # it could be written as a function of the number of vacancies in each firm rather then the number of firms with vacancies...
                if (runif(1) < (length(FCids) / (length(FCids) + length(FKids)))) {
                    # FC
                    Fid <- sample.vec(FCids, 1)
                    Hid <- sample.vec(which(rowSums(NC[t, , ]) + rowSums(NK[t, , ]) == 0), 1)
                    NC[t, Hid, Fid] <- 1
                } else {
                    # FK
                    Fid <- sample.vec(FKids, 1)
                    Hid <- sample.vec(which(rowSums(NC[t, , ]) + rowSums(NK[t, , ]) == 0), 1)
                    NK[t, Hid, Fid] <- 1
                }
            }

            Y[t, ] <- betaC * pmin(colSums(NC[t, , ]), KC[t - 1, ]) # Consumption Goods output in units of goods
            YK[t, ] <- floor(betaK * pmin(colSums(NK[t, , ]), KK[t - 1, ])) # Capital Goods output in units of goods
            WFC[t, , ] <- sweep(NC[t, , ], 2, W0FC[t, ], "*") # Wages
            WFK[t, , ] <- sweep(NK[t, , ], 2, W0FK[t, ], "*") # Wages
            W[t, ] <- rowSums(WFC[t, , ]) + rowSums(WFK[t, , ])
            UB[t, ] <- 0
            UB[t, which(W[t, ] == 0)] <- ifelse(sum(NC[t - 1, , ]) + sum(NK[t - 1, , ]) > 0, phi * sum(W[t - 1, ]) / (sum(NC[t - 1, , ]) + sum(NK[t - 1, , ])), phi * W0) # Unemployment benefits
            NetW[t, ] <- (1 - tW) * W[t, ] # Net Wages
            DI[t, ] <- NetW[t, ] + UB[t, ] # Disposable Income for Hs

            # Consumption Goods market
            C[t, , ] <- C[t - 1, , ]
            # Over-selling
            Fids <- which(colSums(C[t, , ]) - Y[t, ] * sum(CT[t, ]) / YT[t] > tol)
            if (length(Fids) > 0) {
                for (Fid in Fids) {
                    while (sum(C[t, , Fid]) - Y[t, Fid] * sum(CT[t, ]) / YT[t] > tol) {
                        Hid <- sample.vec(which(C[t, , Fid] > 0), 1)
                        d <- min(C[t, Hid, Fid], sum(C[t, , Fid]) - Y[t, Fid] * sum(CT[t, ]) / YT[t])
                        C[t, Hid, Fid] <- C[t, Hid, Fid] - d
                    }
                }
            }
            # Over-buying
            # Sincerely decentralized market ignoring G
            Hids <- which((rowSums(C[t, , ]) - CT[t, ] > tol) | (rowSums(sweep(C[t, , ], 2, pC[t, ], "*")) - pmax(0, DI[t, ] + MH[t - 1, ]) > tol))
            if (length(Hids) > 0) {
                for (Hid in Hids) {
                    while ((sum(C[t, Hid, ]) - CT[t, Hid] > tol) || (sum(C[t, Hid, ] * pC[t, ]) - max(0, DI[t, Hid] + MH[t - 1, Hid]) > tol)) {
                        Fid <- sample.vec(which(C[t, Hid, ] > 0), 1)
                        d <- min(C[t, Hid, Fid], max(sum(C[t, Hid, ]) - CT[t, Hid], (sum(C[t, Hid, ] * pC[t, ]) - (DI[t, Hid] + MH[t - 1, Hid])) / pC[t, Fid]))
                        C[t, Hid, Fid] <- C[t, Hid, Fid] - d
                    }
                }
            }
            # Fill unsatisfied demand
            repeat {
                Hids <- which((CT[t, ] - rowSums(C[t, , ]) > tol) & ((DI[t, ] + MH[t - 1, ]) - rowSums(sweep(C[t, , ], 2, pC[t, ], "*")) > tol))
                if (length(Hids > 0) && (sum(Y[t, ]) * sum(CT[t, ]) / YT[t] - sum(C[t, , ]) > tol)) {
                    Hid <- sample.vec(Hids, 1)
                    Fids <- which(Y[t, ] * sum(CT[t, ]) / YT[t] - colSums(C[t, , ]) > 0)
                    Fid <- Fids[which.min(pC[t, Fids])]
                    d <- min(CT[t, Hid] - sum(C[t, Hid, ]), ((DI[t, Hid] + MH[t - 1, Hid]) - sum(C[t, Hid, ] * pC[t, ])) / pC[t, Fid], Y[t, Fid] * sum(CT[t, ]) / YT[t] - sum(C[t, , Fid]))
                    C[t, Hid, Fid] <- C[t, Hid, Fid] + d
                } else {
                    break
                }
            }

            # If G buys before H, C goes -> 0
            S[t, ] <- colSums(C[t, , ])
            if (sum(Y[t, ] - S[t, ]) > 0) {
                G[t, ] <- min(GT[t], sum(Y[t, ] - S[t, ])) / sum(Y[t, ] - S[t, ]) * (Y[t, ] - S[t, ]) # Gvt Consumption, in units of goods
            } else {
                G[t, ] <- 0
            }
            S[t, ] <- S[t, ] + G[t, ]
            # Residual production is lost


            # Capital Goods Market

            IC[t, , ] <- floor(Ip[t, , ] * replace_na(YK[t, ] / (colSums(Ip[t, , ]) + IKT[t, ]), 0))

            # over promised
            while (any(colSums(IC[t, , ]) > YK[t, ])) {
                FKid <- sample.vec(which(colSums(IC[t, , ]) > YK[t, ]), 1)
                FCid <- sample.vec(which(IC[t, , FKid] > 0), 1)
                IC[t, FCid, FKid] <- IC[t, FCid, FKid] - 1
            }

            # unable to pay
            # while (any(((MFC[t - 1, ] + pC[t, ] * S[t, ] - rowSums(sweep(IC[t, , ], 2, pK[t, ], "*")) - colSums(WFC[t, , ])) < -tol) & (rowSums(IC[t, , ]) > 0))) {
            #    FCid <- sample.vec(which(((MFC[t - 1, ] + pC[t, ] * S[t, ] - rowSums(sweep(IC[t, , ], 2, pK[t, ], "*")) - colSums(WFC[t, , ])) < -tol) & (rowSums(IC[t, , ]) > 0)), 1)
            #    FKid <- sample.vec(which(IC[t, FCid, ] > 0), 1)
            #    IC[t, FCid, FKid] <- IC[t, FCid, FKid] - 1
            # }

            IK[t, ] <- YK[t, ] - colSums(IC[t, , ])

            # sell remaining
            repeat {
                FKids <- which(IK[t, ] > IKT[t, ])
                FCids <- which(rowSums(IC[t, , ]) < ICT[t, ])
                if (length(FKids > 0) && length(FCids > 0)) {
                    FCid <- sample.vec(FCids, 1)
                    FKid <- sample.vec(FKids, 1)
                    IC[t, FCid, FKid] <- IC[t, FCid, FKid] + 1
                    IK[t, FKid] <- IK[t, FKid] - 1
                } else {
                    break
                }
            }

            I[t] <- sum(IC[t, , ]) + sum(IK[t, ])

            DLFC[t, ] <- rowSums(sweep(IC[t, , ], 2, pK[t, ], "*")) + colSums(WFC[t, , ])
            DLFK[t, ] <- colSums(WFK[t, , ])
            DL[t] <- sum(DLFC[t, ]) + sum(DLFK[t, ])

            RLFC[t, ] <- pmax(0, pmin(LFC[t - 1, ] + DLFC[t, ], MFC[t - 1, ] + DLFC[t, ] + pC[t, ] * S[t, ] - rowSums(sweep(IC[t, , ], 2, pK[t, ], "*")) - colSums(WFC[t, , ]) - rL[t] * (LFC[t - 1, ] + DLFC[t, ])))
            RLFK[t, ] <- pmax(0, pmin(LFK[t - 1, ] + DLFK[t, ], MFK[t - 1, ] + DLFK[t, ] + pK[t, ] * colSums(IC[t, , ]) - colSums(WFK[t, , ]) - rL[t] * (LFK[t - 1, ] + DLFK[t, ])))
            RL[t] <- sum(RLFC[t, ]) + sum(RLFK[t, ])

            LFC[t, ] <- LFC[t - 1, ] + DLFC[t, ] - RLFC[t, ]
            LFK[t, ] <- LFK[t - 1, ] + DLFK[t, ] - RLFK[t, ]
            L[t] <- sum(LFC[t, ]) + sum(LFK[t, ])

            # Profits
            PFK[t, , ] <- switch(PDist,
                "Flat" = t(array(pmax(
                    0,
                    MFK[t - 1, ] + DLFK[t, ] + pK[t, ] * colSums(IC[t, , ]) - colSums(WFK[t, , ]) - rL[t] * (DLFK[t, ] + LFK[t - 1, ]) - RLFK[t, ]
                ) / NH, c(NFK, NH))),
                "Prop" = sweep(t(array(pmax(
                    0,
                    MFK[t - 1, ] + DLFK[t, ] + pK[t, ] * colSums(IC[t, , ]) - colSums(WFK[t, , ]) - rL[t] * (DLFK[t, ] + LFK[t - 1, ]) - RLFK[t, ]
                ), c(NFK, NH))), 1, pmax(0, MH[t - 1, ]), "*") / sum(pmax(0, MH[t - 1, ]))
            )
            PFC[t, , ] <- switch(PDist,
                "Flat" = t(array(pmax(
                    0,
                    MFC[t - 1, ] + DLFC[t, ] + pC[t, ] * S[t, ] - rowSums(sweep(IC[t, , ], 2, pK[t, ], "*")) - colSums(WFC[t, , ]) - rL[t] * (DLFC[t, ] + LFC[t - 1, ]) - RLFC[t, ]
                ) / NH, c(NFC, NH))),
                "Prop" = sweep(t(array(pmax(
                    0,
                    MFC[t - 1, ] + DLFC[t, ] + pC[t, ] * S[t, ] - rowSums(sweep(IC[t, , ], 2, pK[t, ], "*")) - colSums(WFC[t, , ]) - rL[t] * (DLFC[t, ] + LFC[t - 1, ]) - RLFC[t, ]
                ), c(NFC, NH))), 1, pmax(0, MH[t - 1, ]), "*") / sum(pmax(0, MH[t - 1, ]))
            )
            PB[t, ] <- switch(PDist,
                "Flat" = array(max(
                    0,
                    (1 - rB[t]) / (1 - rB[t] - rB[t] * tP) * (VB[t - 1] + rL[t] * (DL[t] + L[t - 1]) - crT * L[t] + rB[t] / (1 - rB[t]) * (B[t - 1] + sum(pC[t, ] * G[t, ]) + sum(UB[t, ]) - (tW * sum(W[t, ]) + tP * (sum(PFC[t, , ]) + sum(PFK[t, , ])) + tC * sum(sweep(C[t, , ], 2, pC[t, ], "*")))))
                ) / NH, c(NH)),
                "Prop" = array(max(
                    0,
                    (1 - rB[t]) / (1 - rB[t] - rB[t] * tP) * (VB[t - 1] + rL[t] * (DL[t] + L[t - 1]) - crT * L[t] + rB[t] / (1 - rB[t]) * (B[t - 1] + sum(pC[t, ] * G[t, ]) + sum(UB[t, ]) - (tW * sum(W[t, ]) + tP * (sum(PFC[t, , ]) + sum(PFK[t, , ])) + tC * sum(sweep(C[t, , ], 2, pC[t, ], "*")))))
                ) * pmax(0, MH[t - 1, ]) / sum(pmax(0, MH[t - 1, ])), c(NH))
            )
            P[t, ] <- rowSums(PFK[t, , ]) + rowSums(PFC[t, , ]) + PB[t, ]

            T[t, ] <- tW * W[t, ] + tP * P[t, ] + tC * rowSums(sweep(C[t, , ], 2, pC[t, ], "*")) # Taxes
            MH[t, ] <- MH[t - 1, ] + P[t, ] + W[t, ] + UB[t, ] - rowSums(sweep(C[t, , ], 2, pC[t, ], "*")) - T[t, ] # Households' Money
            MFK[t, ] <- MFK[t - 1, ] + pK[t, ] * colSums(IC[t, , ]) - colSums(WFK[t, , ]) - colSums(PFK[t, , ]) - rL[t] * (DLFK[t, ] + LFK[t - 1, ]) + (DLFK[t, ] - RLFK[t, ]) # Capital Firms' Money
            MFC[t, ] <- MFC[t - 1, ] + pC[t, ] * S[t, ] - rowSums(sweep(IC[t, , ], 2, pK[t, ], "*")) - colSums(WFC[t, , ]) - colSums(PFC[t, , ]) - rL[t] * (DLFC[t, ] + LFC[t - 1, ]) + (DLFC[t, ] - RLFC[t, ]) # Consumption Firms' Money
            M[t] <- sum(MH[t, ]) + sum(MFC[t, ]) + sum(MFK[t, ]) # Bank's Money

            B[t] <- (B[t - 1] - (sum(T[t, ]) - sum(pC[t, ] * G[t, ]) - sum(UB[t, ]))) / (1 - rB[t]) # Gvt Bonds


            # Update Variables for next cycle
            KC[t, ] <- floor(KC[t - 1, ] * (1 - dK)) + rowSums(IC[t, , ])
            KK[t, ] <- floor(KK[t - 1, ] * (1 - dK)) + IK[t, ]
            K[t] <- sum(KC[t, ]) + sum(KK[t, ])
            KCu[t, ] <- pmin(KC[t - 1, ], colSums(NC[t, , ]))
            KKu[t, ] <- pmin(KK[t - 1, ], colSums(NK[t, , ]))
            Ku[t] <- pmin(K[t - 1], sum(NC[t, , ]) + sum(NK[t, , ]))
            cuC[t, ] <- replace_na(KCu[t, ] / KC[t - 1, ], 0)
            cuK[t, ] <- replace_na(KKu[t, ] / KK[t - 1, ], 0) # but are dead firms if KK == 0...
            cu[t] <- Ku[t] / K[t - 1]
            pKC[t, ] <- rowSums(sweep(IC[t, , ], 2, pK[t, ], "*")) / rowSums(IC[t, , ])
            pKC[t, ] <- ifelse(is.na(pKC[t, ]), pKC[t - 1, ], pKC[t, ])

            u[t] <- 1 - ((sum(NC[t, , ]) + sum(NK[t, , ])) / NH)
            i[t] <- (sum(pC[t, ] * S[t, ]) / sum(S[t, ])) / (sum(pC[t - 1, ] * S[t - 1, ]) / sum(S[t - 1, ])) - 1
            cr[t] <- ifelse(L[t] == 0, crT, VB[t] / L[t])

            VH[t, ] <- MH[t, ] # Households' Net Wealth
            VFC[t, ] <- MFC[t, ] + pKC[t, ] * KC[t, ] - LFC[t, ] # Capital Firms' Net Wealth
            VFK[t, ] <- MFK[t, ] + pK[t, ] * KK[t, ] - LFK[t, ] # Consumption Firms' Net Wealth
            VB[t] <- -M[t] + L[t] + B[t]
            VG[t] <- -B[t] # Gvt' Net Wealth
            V[t] <- sum(VH[t, ]) + sum(VFK[t, ]) + sum(VFC[t, ]) + VB[t] + VG[t] # System Total Wealth

            # Defaulting
            LFC[t, which(VFC[t, ] < -tol)] <- 0
            MFC[t, which(VFC[t, ] < -tol)] <- 0
            muC[t, which(VFC[t, ] < -tol)] <- mu0
            VFC[t, which(VFC[t, ] < -tol)] <- pKC[t, which(VFC[t, ] < -tol)] * KC[t, which(VFC[t, ] < -tol)]
            LFK[t, which(VFK[t, ] < -tol)] <- 0
            MFK[t, which(VFK[t, ] < -tol)] <- 0
            muK[t, which(VFK[t, ] < -tol)] <- mu0
            VFK[t, which(VFK[t, ] < -tol)] <- pK[t, which(VFK[t, ] < -tol)] * KK[t, which(VFK[t, ] < -tol)]

            M[t] <- sum(MH[t, ]) + sum(MFC[t, ]) + sum(MFK[t, ]) # Bank's Money
            L[t] <- sum(LFC[t, ]) + sum(LFK[t, ])
            V[t] <- sum(VH[t, ]) + sum(VFK[t, ]) + sum(VFC[t, ]) + VB[t] + VG[t] # System Total Wealth

            GDP[t] <- sum(pC[t] * S[t, ]) + sum(pK[t, ] * colSums(IC[t, , ]))

            pb$tick()
        }
    }
}

{{        if (any(rowSums(MH[2:TMAX, ]) + rowSums(MFC[2:TMAX, ]) + rowSums(MFK[2:TMAX, ]) - M[2:TMAX] > tol)) {
    print("M row in BS not consistent")
}
if (any(rowSums(KC[2:TMAX, ] * pKC[2:TMAX, ]) + rowSums(KK[2:TMAX, ] * pK[2:TMAX, ]) - (rowSums(KK[2:TMAX, ] * pK[2:TMAX, ]) + rowSums(KC[2:TMAX, ] * pKC[2:TMAX, ])) > tol)) {
    print("K row in BS not consistent")
} # Tautology, but keeping track of the value of each K Good is not suitable in this setting
if (any(B[2:TMAX] - B[2:TMAX] > tol)) {
    print("B row in BS not consistent")
} # Ops, Tautology again
if (any(rowSums(VH[2:TMAX, ]) + rowSums(VFC[2:TMAX, ]) + rowSums(VFK[2:TMAX, ]) + VB[2:TMAX] + VG[2:TMAX] - V[2:TMAX] > tol)) {
    print("V row in BS not consistent")
}
if (any(MH[2:TMAX, ] - VH[2:TMAX, ] > tol)) {
    print("H column in BS not consistent (checked at agent level)")
}
if (any(MFC[2:TMAX, ] + (KC[2:TMAX, ] * pKC[2:TMAX, ]) - LFC[2:TMAX, ] - VFC[2:TMAX, ] > tol)) {
    print("FC column in BS not consistent (checked at agent level)")
}
if (any(MFK[2:TMAX, ] + (KK[2:TMAX, ] * pK[2:TMAX, ]) - LFK[2:TMAX, ] - VFK[2:TMAX, ] > tol)) {
    print("FK column in BS not consistent (checked at agent level)")
}
if (any(-M[2:TMAX] + L[2:TMAX] + B[2:TMAX] - VB[2:TMAX] > tol)) {
    print("B column in BS not consistent")
}
if (any(-B[2:TMAX] - VG[2:TMAX] > tol)) {
    print("G column in BS not consistent")
}
if (any(rowSums(KK[2:TMAX, ] * pK[2:TMAX, ]) + rowSums(KC[2:TMAX, ] * pKC[2:TMAX, ]) - V[2:TMAX] > tol)) {
    print("Total column in BS not consistent")
}    }

{
    if (any(
        -rowSums(sweep(C[2:TMAX, , ], c(1, 3), pC[2:TMAX, ], "*"), dims = 2)
        + UB[2:TMAX, ]
            + W[2:TMAX, ]
            + P[2:TMAX, ]
            - T[2:TMAX, ]
            - (MH[2:TMAX, ] - MH[1:TMAX - 1, ])
        > tol
    )) {
        print("H column in TFM not consistent (checked ad agent level)")
    }
    if (any(
        (rowSums(aperm(C[2:TMAX, , ], c(1, 3, 2)), dim = 2) * pC[2:TMAX, ])
        + (G[2:TMAX, ] * pC[2:TMAX, ])
            - rowSums(sweep(IC[2:TMAX, , ], c(1, 3), pK[2:TMAX, ], "*"), dims = 2)
            - rowSums(aperm(WFC[2:TMAX, , ], c(1, 3, 2)), dims = 2)
            - rowSums(aperm(PFC[2:TMAX, , ], c(1, 3, 2)), dims = 2)
            - sweep(DLFC[2:TMAX, ] + LFC[1:TMAX - 1, ], 1, rL[2:TMAX], "*")
            - (MFC[2:TMAX, ] - MFC[1:TMAX - 1, ])
            + (LFC[2:TMAX, ] - LFC[1:TMAX - 1, ])
        > tol
    )) {
        print("FC column in TFM not consistent (checked ad agent level)")
    }
    if (any(
        rowSums(aperm(IC[2:TMAX, , ], c(1, 3, 2)), dims = 2) * pK[2:TMAX, ]
            - rowSums(aperm(WFK[2:TMAX, , ], c(1, 3, 2)), dims = 2)
            - rowSums(aperm(PFK[2:TMAX, , ], c(1, 3, 2)), dims = 2)
            - sweep(DLFK[2:TMAX, ] + LFK[1:TMAX - 1, ], 1, rL[2:TMAX], "*")
            - (MFK[2:TMAX, ] - MFK[1:TMAX - 1, ])
            + (LFK[2:TMAX, ] - LFK[1:TMAX - 1, ])
        > tol
    )) {
        print("FK column in TFM not consistent (checked ad agent level)")
    }
    if (any(
        -rowSums(PB[2:TMAX, ])
        + rL[2:TMAX] * (DL[2:TMAX] + L[1:TMAX - 1])
            + rB[2:TMAX] * B[2:TMAX]
            + (M[2:TMAX] - M[1:TMAX - 1])
            - (L[2:TMAX] - L[1:TMAX - 1])
            - (B[2:TMAX] - B[1:TMAX - 1])
        > tol
    )) {
        print("B column in TFM not consistent")
    }
    if (any(
        -rowSums(G[2:TMAX, ] * pC[2:TMAX, ])
        - rowSums(UB[2:TMAX, ])
            + rowSums(T[2:TMAX, ])
            - rB[2:TMAX] * B[2:TMAX]
            + (B[2:TMAX] - B[1:TMAX - 1])
        > tol
    )) {
        print("G column in TFM not consistent")
    }
}}

{{
    par(mfrow = c(2, 5))
    plot(rowSums(MH), type = "l", main = "MH")
    plot(rowSums(MFC), type = "l", main = "MFC")
    plot(rowSums(MFK), type = "l", main = "MFK")
    plot(M, type = "l", main = "MB")
    plot(B, type = "l", main = "B")
    plot(rowSums(VH), type = "l", main = "VH")
    plot(rowSums(VFC), type = "l", main = "VFC")
    plot(rowSums(VFK), type = "l", main = "VFK")
    plot(VB, type = "l", main = "VB")
    plot(VG, type = "l", main = "VG")
}

{
    par(mfrow = c(2, 4))
    plot(rowSums(rowSums(aperm(C, c(1, 3, 2)), dims = 2) * pC), type = "l", main = "C")
    plot(rowSums(G * pC), type = "l", main = "G")
    plot(rowSums(rowSums(aperm(IC, c(1, 3, 2)), dims = 2) * pK), type = "l", main = "IC")
    plot(rowSums(KC * pKC), type = "l", main = "KC")
    plot(rowSums(UB), type = "l", main = "UB")
    plot(rowSums(W), type = "l", main = "W")
    plot(rowSums(P), type = "l", main = "P")
    plot(rowSums(T), type = "l", main = "T")
}

{
    par(mfrow = c(2, 4))
    plot((rowSums(NC) + rowSums(NK)) / NH, type = "l", main = "N", ylim = c(0, 1))
    plot(rowSums(NC) / NH, type = "l", main = "NC", ylim = c(0, 1))
    lines(rowSums(NCT) / NH, lty = "dashed")
    plot(rowSums(NK) / NH, type = "l", main = "NK", ylim = c(0, 1))
    lines(rowSums(NKT) / NH, lty = "dashed")
    plot(pmin(rowSums(KC)[1:TMAX - 1], rowSums(NC)[2:TMAX]) / rowSums(KC)[1:TMAX - 1], type = "l", main = "cu", ylim = c(0, 1))
    plot(apply(pC, 1, mean), type = "l", main = "pC")
    plot(apply(pK, 1, mean), type = "l", main = "pK")
    plot(apply(muC, 1, mean), type = "l", main = "muC")
    plot(apply(muK, 1, mean), type = "l", main = "muK")
}

{
    par(mfrow = c(2, 3))
    plot(rowSums(C), type = "l", main = "C", ylim = c(0, max(rowSums(CT), rowSums(C), na.rm = TRUE)))
    lines(rowSums(CT), lty = "dashed")
    plot(rowSums(G), type = "l", main = "G", ylim = c(0, max(GT, rowSums(G), na.rm = TRUE)))
    lines(GT, lty = "dashed")
    plot(rowSums(Y), type = "l", main = "Y", ylim = c(0, max(YT, rowSums(Y), na.rm = TRUE)))
    lines(YT, lty = "dashed")
    plot(rowSums(IC), type = "l", main = "IC", ylim = c(0, max(rowSums(ICT), na.rm = TRUE)))
    lines(rowSums(ICT), lty = "dashed")
    plot(rowSums(IK), type = "l", main = "IK", ylim = c(0, max(rowSums(IKT), rowSums(IK), na.rm = TRUE)))
    lines(rowSums(IKT), lty = "dashed")
    plot(rowSums(YK), type = "l", main = "YK")
}

{
    par(mfrow = c(2, 4))
    plot(rowSums(S), type = "l", main = "S")
    plot(rowSums(IC), type = "l", main = "IC")
    plot(GDP, type = "l", main = "GDP")
    plot(-VG, type = "l", main = "Public Debt")
    plot(apply(pC, 1, mean), type = "l", main = "pC")
    plot(apply(pK, 1, mean), type = "l", main = "pK")
    plot(rowSums(W), type = "l", main = "W")
    plot(rowSums(P), type = "l", main = "P")
}

{
    par(mfrow = c(3, 4))
    plot(L, type = "l", main = "L")
    plot(DL, type = "l", main = "DL")
    plot(M, type = "l", main = "M")
    plot(B, type = "l", main = "B")
    plot(rowSums(LFC), type = "l", main = "LFC")
    plot(rowSums(DLFC), type = "l", main = "DLFC")
    plot(rowSums(RLFC), type = "l", main = "RLFC")
    plot((rowSums(DLFC) - rowSums(RLFC)), type = "l", main = "Delta LFC")
    plot(rowSums(LFK), type = "l", main = "LFK")
    plot(rowSums(DLFK), type = "l", main = "DLFK")
    plot(rowSums(RLFK), type = "l", main = "RLFK")
    plot((rowSums(DLFK) - rowSums(RLFK)), type = "l", main = "Delta LFK")
}

{
    par(mfrow = c(4, 3))
    plot(rowSums(PFC), type = "l", main = "PFC")
    plot(rowSums(PFK), type = "l", main = "PFK")
    plot(rowSums(PB), type = "l", main = "PB")
    plot(rowSums(MFC), type = "l", main = "MFC")
    plot(rowSums(MFK), type = "l", main = "MFK")
    plot(M, type = "l", main = "M")
    plot(rowSums(LFC), type = "l", main = "LFC")
    plot(rowSums(LFK), type = "l", main = "LFK")
    plot(rL, type = "l", main = "rL")
    plot(rowSums(VFC), type = "l", main = "VFC")
    plot(rowSums(VFK), type = "l", main = "VFK")
    plot(VB, type = "l", main = "VB")
}

{
    par(mfrow = c(2, 3))
    plot(u, type = "l", main = "u")
    plot(i, type = "l", main = "i")
    plot(cu, type = "l", main = "cu")
    plot(rB, type = "l", main = "rB")
    plot(rL, type = "l", main = "rL")
}

{
    par(mfrow = c(2, 3))
    plot(apply(pC, 1, mean), type = "l", main = "pC")
    plot(apply(muC, 1, mean), type = "l", main = "muC")
    plot(apply(cuC, 1, mean), type = "l", main = "cuC")
    plot(apply(pK, 1, mean), type = "l", main = "pK")
    plot(apply(muK, 1, mean), type = "l", main = "muK")
    plot(apply(cuK, 1, mean), type = "l", main = "cuK")
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
    boxplot(W[TMAX, ], main = "WH")
    boxplot(rowSums(C[TMAX, , ]), main = "CH")
    boxplot(P[TMAX, ], main = "PH")
}

{
    par(mfrow = c(3, 1))
    hist(MH[1, ], main = "MH T=1")
    hist(MH[floor(TMAX / 10), ], main = "MH T=TMAX/10")
    hist(MH[TMAX, ], main = "MH T=TMAX")
}

{
    par(mfrow = c(2, 3))
    plot(apply(VH, 1, function(x) sum(x < -tol)), main = "any VH < 0")
    plot(apply(VFC, 1, function(x) sum(x < -tol)), main = "any VFC < 0")
    plot(apply(VFK, 1, function(x) sum(x < -tol)), main = "any VFK < 0")
    plot(VB < -tol, main = " VB < 0")
    plot(VG > tol, main = " VG > 0")
}}
