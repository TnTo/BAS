### BAS: Building towards Artificial Societies
# Model 3:
#   Fixed Capital only in C sector
#   Endogenous mark-up
#   It appears to need to be a full-employment model
#   In this form is completely a-cyclical
#   Profit inflation, without catastropheses at t=5000
###
# renv::init()
# renv::restore()
# renv::install("sfcr", prompt = FALSE)
# renv::install("devtools", prompt = FALSE)
# devtools::install_github("TnTo/sfcr", ref = "sankey")
# renv::install("tidyverse", prompt = FALSE)
# renv::install("networkD3", prompt = FALSE)
# renv::install("ggraph", prompt = FALSE)
# renv::install("ggpubr", prompt = FALSE)
# renv::snapshot()
# renv::status()

install.packages(c("devtools", "tidyverse", "networkD3", "ggraph", "ggplot2"))
devtools::install_github("TnTo/sfcr", ref = "sankey")

library(sfcr)
library(tidyverse)

## A modelCon Dream
model_eqs <- sfcr_set(
    MH ~ MH[-1] + P + W + UB - pC * C - T,
    MFC ~ MFC[-1] + pC * Y - pK * IC - WFC - PFC - rL * LFC + (LFC - LFC[-1]),
    MFK ~ MFK[-1] + pK * IC - WFK - PFK - rL * LFK + (LFK - LFK[-1]),
    M ~ MH + MFC + MFK, # Bank's Money
    LFC ~ LFC[-1] + max(0, WFC + pK * IC - pC * (C + G) - MFC[-1]), # FCs' Loans stock
    LFK ~ LFK[-1] + max(0, WFK - pK * IC - MFK[-1]), # FKs' Loans stock (!!!)
    L ~ LFC + LFK, # Bank's Loans stock
    B ~ (B[-1] - (T - pC * G - UB)) / (1 - rB), # Gvt Bonds
    KC ~ (1 - dK) * KC[-1] + IC,
    KK ~ (1 - dK) * KK[-1] + IK,
    K ~ KC + KK,
    VH ~ MH,
    VFC ~ MFC + pK * KC - LFC,
    VFK ~ MFK + pK * KK - LFK,
    VB ~ -M + L + B, # Bank's Net Wealth
    VG ~ -B, # Gvt Net Wealth
    V ~ VH + VFC + VFK + VB + VG,
    pKK ~ pK * K,
    NetW ~ (1 - tW) * W,
    DI ~ NetW + UB,
    CT ~ max(0, (ay * DI[-1] + av * MH[-1]) / HpC[-1]),
    GT ~ max(0, (d * GDP[-1] + T[-1]) / pC[-1]),
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
    UB ~ W0 * phi * u,
    C ~ CT * Y / YT,
    G ~ Y - C,
    I ~ NK * betaK,
    IC ~ ifelse(ICT > 0, ICT * I / IT, 0),
    IK ~ I - IC,
    P ~ PFC + PFK + PB,
    PFC ~ max(0, pC * C + pC * G - WFC - pK * IC - rL * LFC),
    PFK ~ max(0, pK * IC - WFK - rL * LFK),
    PB ~ max(0, (1 - crT) * L[-1] + B[-1] - M[-1]), # Bank's Profits (!!!)
    T ~ tW * W + tP * P + tC * pC * C,
    muC ~ muC[-1] * (1 + ThethaMu * (cuC[-1] - cuT) / cuT),
    muK ~ muK[-1] * (1 + ThethaMu * (cuK[-1] - cuT) / cuT),
    HpC ~ (1 + tC) * pC,
    pC ~ (1 + muC) * (Y / WFC + dK * pK / betaC),
    pK ~ ifelse(NK == 0, pK[-1], (1 + muK) * (I / WFK + dK * W0 / betaK)),
    KCu ~ min(NC, KC[-1]),
    KKu ~ min(NK, KK[-1]),
    Ku ~ KCu + KKu,
    cuC ~ KCu / KC[-1],
    cuK ~ KKu / KK[-1],
    cu ~ Ku / K[-1],
    i ~ pC / pC[-1] - 1, # consumption goods inflation rate
    u ~ 1 - N, # unemployment rate
    rB ~ i + 0.5 * (i - iT) + 0.25 * (cu - cuT) - 0.25 * (u - uT), # Gvt Bonds' interest rate
    rL ~ rB + DrL, # Loans' interest rate
    GDP ~ Y * pC + I * pK,
    cr ~ VB / L # capital ratio
)

sfcr_dag_cycles_plot(model_eqs, size = 6)

model_bs <- sfcr_matrix(
    columns = c("Households", "Consumption Firms", "Capital Firms", "Bank", "Government", "Sum"),
    codes = c("H", "FC", "FK", "B", "G", "sum"),
    c("Money", H = "+MH", FC = "+MFC", FK = "+MFK", B = "-M"),
    c("Capital", FC = "+pK * KC", FK = "+pK * KK", sum = "+pK * K"),
    c("Loan", FC = "-LFC", FK = "-LFK", B = "+L"),
    c("Bond", B = "+B", G = "-B"),
    c("Balance", H = "-VH", FC = "-VFC", FK = "-VFK", B = "-VB", G = "-VG", sum = "-pK * K")
)
sfcr_matrix_display(model_bs, "bs")

model_tfm <- sfcr_matrix(
    columns = c("Households", "Consumption Firms", "Capital Firms", "Bank", "Government"),
    codes = c("H", "FC", "FK", "B", "G"),
    c("Consumption", H = "-pC * C", FC = "pC * C"),
    c("Gvt Spending", FC = "pC * G", G = "-pC * G"),
    c("Investment", FC = "-pK * IC", FK = "+pK * IC"),
    c("Benefits", H = "UB", G = "-UB"),
    c("Wages", H = "W", FC = "-WFC", FK = "-WFK"),
    c("Profits", H = "P", FC = "-PFC", FK = "-PFK", B = "-PB"),
    c("Taxes", H = "-T", G = "T"),
    c("Loan int.", FC = "-rL * LFC", FK = "-rL * LFK", B = "+rL * L"),
    c("Bond int.", B = "+rB * B", G = "-rB * B"),
    c("D Money", H = "-(MH - MH[-1])", FC = "-(MFC - MFC[-1])", FK = "-(MFK - MFK[-1])", B = "(M - M[-1])"),
    c("D Loan", FC = "(LFC - LFC[-1])", FK = "(LFK - LFK[-1])", B = "-(L - L[-1])"),
    c("D Bond", B = "-(B - B[-1])", G = "(B - B[-1])"),
)
sfcr_matrix_display(model_tfm, "tfm")

model_ext <- sfcr_set(
    ay ~ 0.6,
    av ~ 0.2,
    d ~ 0.03,
    tW ~ 0.35,
    tP ~ 0.2,
    tC ~ 0.2,
    W0 ~ 1.0,
    cuT ~ 0.8,
    uT ~ 0.05,
    iT ~ 0.02,
    dK ~ 0.1,
    betaC ~ 2.0,
    betaK ~ 1.0,
    phi ~ 0.7,
    ThethaMu ~ 0.1,
    DrL ~ 0.05, # Bank premium for Loans interest rate
    crT ~ 0.08 # target capital ratio
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
    betaC ~ 2.0,
    betaK ~ 1.0,
    cuC ~ 1,
    cuK ~ 1,
    HpC ~ 1.2,
    pC ~ 1,
    pK ~ 1,
    muC ~ 0.2,
    muK ~ 1.0
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

sfcr_validate(model_bs, model, "bs", tol = 1e-7, rtol = TRUE)

sfcr_validate(model_tfm, model, "tfm", tol = 1e-7, rtol = TRUE)

sfcr_sankey(model_tfm, model, when = "end")

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("MH", "MFC", "MFK", "M", "VH", "VFC", "VFK", "VG")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("C", "G", "Y", "W", "P")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("pC", "pK")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("muC", "muK")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("N", "NC", "NK")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("Y", "YT")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("I", "IT")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("KC", "KK", "IC", "IK")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("cuC", "cuK", "cuT")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("rB", "rL")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("L", "LFC", "LFK")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("i", "iT", "cu", "cuT", "u", "uT")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("P", "PFC", "PFK", "PB")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("cr", "crT")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("B", "L", "M")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))
