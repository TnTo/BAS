# install.packages(c("devtools", "tidyverse", "networkD3", "ggraph", "ggplot2"))
# devtools::install_github("TnTo/sfcr", ref = "sankey")

library(sfcr)
library(tidyverse)

dKs <- seq(from = 0, to = 1, length.out = 41)
cols <- heat.colors(41)

plot(c(0, 500), c(0, 1))

fN <- array(0, 41)

for (i in 1:41) {
    ## A modelCon Dream
    model_eqs <- sfcr_set(
        MH ~ MH[-1] + P + W + UB - pC * C - T,
        MFC ~ MFC[-1] + pC * Y - pK * IC - WFC - PFC,
        MFK ~ MFK[-1] + pK * IC - WFK - PFK,
        M ~ M[-1] - (T - pC * G - UB),
        KC ~ (1 - dK) * KC[-1] + IC,
        KK ~ (1 - dK) * KK[-1] + IK,
        K ~ KC + KK,
        VH ~ MH,
        VFC ~ MFC + pK * KC,
        VFK ~ MFK + pK * KK,
        VG ~ -M,
        V ~ VH + VFC + VFK + VG,
        pKK ~ pK * K,
        NetW ~ (1 - tW) * W,
        DI ~ NetW + UB,
        CT ~ max(0, (ay * DI[-1] + av * MH[-1]) / HpC[-1]),
        GT ~ max(0, (d * GDP[-1] + T[-1] + UB[-1]) / pC[-1]),
        YT ~ CT + GT,
        ICT ~ max(0, KCu[-1] * (1 / cuT - 1 / cuC[-1]) + dK * KC[-1], na.rm = TRUE),
        IKT ~ max(0, KKu[-1] * (1 / cuT - 1 / cuK[-1]) + dK * KK[-1], na.rm = TRUE),
        IT ~ ICT + IKT,
        NCT ~ min(1, KC[-1], YT / betaC),
        NKT ~ min(1, KK[-1], IT / betaK),
        NT ~ NCT + NKT,
        N ~ min(1, NT),
        NC ~ NCT * N / NT,
        NK ~ NKT * N / NT,
        W ~ WFC + WFK,
        WFC ~ W0 * NC,
        WFK ~ W0 * NK,
        Y ~ NC * betaC,
        UB ~ W0 * phi * (1 - N),
        C ~ CT * Y / YT,
        G ~ Y - C,
        I ~ NK * betaK,
        IC ~ ifelse(ICT > 0, ICT * I / IT, 0),
        IK ~ I - IC,
        P ~ PFC + PFK,
        PFC ~ MFC[-1] + pC * C + pC * G - WFC - pK * IC,
        PFK ~ MFC[-1] + pK * IC - WFK,
        T ~ tW * W + tP * P + tC * pC * C,
        muC ~ muC[-1] * (1 + Thetha * (cuC[-1] - cuT) / cuT), # FCs' mark-up
        muK ~ muK[-1] * (1 + Thetha * (cuK[-1] - cuT) / cuT), # FKs' mark-up
        HpC ~ (1 + tC) * pC,
        pC ~ (1 + muC) * (WFC / Y),
        pK ~ ifelse(NK == 0, pK[-1], (1 + muK) * WFK / I),
        KCu ~ min(NC, KC[-1]),
        KKu ~ min(NK, KK[-1]),
        Ku ~ KCu + KKu,
        cuC ~ KCu / KC[-1],
        cuK ~ KKu / KK[-1],
        cu ~ Ku / K[-1],
        GDP ~ Y * pC + I * pK,
        deficit ~ (pC * G + UB - T) / GDP,
        debt ~ M / GDP,
        wshare ~ W / (W + P),
        pshare ~ P / (W + P),
        ayR ~ HpC * C / DI,
        avR ~ HpC * C / MH
    )

    model_ext <- sfcr_set(
        ay ~ 0.6,
        av ~ 0.2,
        d ~ 0.03,
        tW ~ 0.35,
        tP ~ 0.2,
        tC ~ 0.2,
        W0 ~ 1.0,
        cuT ~ 0.8,
        dK ~ dKs[i],
        betaC ~ 2.0,
        betaK ~ 1.0,
        phi ~ 0.7,
        Thetha ~ 0.1 # mark-up adjastment speed
    )

    model_init <- sfcr_set(
        MH ~ 1,
        VH ~ 1,
        M ~ 1,
        VG ~ -1,
        W0 ~ 1,
        KC ~ 0.1,
        KK ~ 0.1,
        K ~ 0.2,
        cuC ~ 1,
        cuK ~ 1,
        HpC ~ 1.2,
        pC ~ 1,
        pK ~ 1,
        muC ~ 0.2,
        muK ~ 0.2
    )

    model <- sfcr_baseline(
        equations = model_eqs,
        external = model_ext,
        init = model_init,
        periods = 500,
        tol = 1e-7,
        hidden = c("V" = "pKK"),
        hidden_tol = 1e-7,
        method = "Broyden"
    )

    lines(model$N, col = cols[i], type = "l")
    fN[i] <- model$N[500]
}

legend(450, 0.75, dKs[c(1, 6, 11, 16, 21, 26, 31, 36, 41)], cols[c(1, 6, 11, 16, 21, 26, 31, 36, 41)])

dev.print(pdf, "plot/chaos_trace.pdf")

plot(dKs, fN)

dev.print(pdf, "plot/chaos_end.pdf")
