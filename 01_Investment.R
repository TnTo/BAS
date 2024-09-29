### BAS: Building towards Artificial Societies
# Model 1:
#   Fixed Capital only in C sector
#   It appears to need to be a full-employment model
#   In this form is completely a-cyclical
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
    MFC ~ MFC[-1] + pC * Y - pK * I - WFC - PFC, # FC -> Consumption goods firms
    MFK ~ MFK[-1] + pK * I - WFK - PFK, # FK -> Capital goods firms
    M ~ M[-1] - (T - pC * G - UB),
    K ~ (1 - dK) * K[-1] + I, # Fixed Capital in consumption goods firms, 1 unit of capital is used by 1 unit of labour
    VH ~ MH,
    VFC ~ MFC + pK * K,
    VFK ~ MFK,
    VG ~ -M,
    V ~ VH + VFC + VFK + VG,
    pKK ~ pK * K, # Value of fixed capital
    NetW ~ (1 - tW) * W,
    DI ~ NetW + UB,
    CT ~ max(0, (ay * DI[-1] + av * MH[-1]) / HpC[-1]),
    GT ~ max(0, (d * GDP[-1] + T[-1] - UB[-1]) / pC[-1]),
    YT ~ CT + GT,
    IT ~ max(0, Ku[-1] * (1 / cuT - 1 / cu[-1]) + dK * K[-1]), # Desired investments in units of capital goods
    NCT ~ min(1, K[-1], YT / betaC), # Labour demand for Consumptio Firms
    NKT ~ min(1, IT / betaK), # Labour demand for Capital Firms
    NT ~ NCT + NKT, # Labour demand
    N ~ min(1, NT), # Employment share
    NC ~ NCT * N / NT, # Employment share in FCs
    NK ~ NKT * N / NT, # Employment sfare in FKs
    W ~ WFC + WFK,
    WFC ~ W0 * NC, # FCs' wages
    WFK ~ W0 * NK, # FKs' wages
    Y ~ NC * betaC,
    UB ~ W0 * phi * (1 - N),
    C ~ CT * Y / YT,
    G ~ Y - C,
    I ~ NK * betaK,
    P ~ PFC + PFK,
    PFC ~ pC * (C + G) - WFC - pK * I, # FCs' profits (all distributed)
    PFK ~ pK * I - WFK, # FK's profits (all distributed)
    T ~ tW * W + tP * P + tC * pC * C,
    HpC ~ (1 + tC) * pC,
    pC ~ (1 + mu) * (Y / WFC), # Consumption goods price
    pK ~ ifelse(NK == 0, pK[-1], (1 + mu) * I / WFK), # Capital goods price
    Ku ~ min(NC, K[-1]), # Used Capital
    cu ~ Ku / K[-1], # Capacity utilization
    GDP ~ Y * pC + I * pK,
)

sfcr_dag_cycles_plot(model_eqs, size = 6)

model_bs <- sfcr_matrix(
    columns = c("Households", "Consumption Firms", "Capital Firms", "Government", "Sum"),
    codes = c("H", "FC", "FK", "G", "sum"),
    c("Money", H = "+MH", FC = "+MFC", FK = "+MFK", G = "-M"),
    c("Capital", FC = "+pK * K", sum = "+pK * K"),
    c("Balance", H = "-VH", FC = "-VFC", FK = "-VFK", G = "-VG", sum = "-pK * K")
)
sfcr_matrix_display(model_bs, "bs")

model_tfm <- sfcr_matrix(
    columns = c("Households", "Consumption Firms", "Capital Firms", "Government"),
    codes = c("H", "FC", "FK", "G"),
    c("Consumption", H = "-pC * C", FC = "pC * C"),
    c("Gvt Spending", FC = "pC * G", G = "-pC * G"),
    c("Investment", FC = "-pK * I", FK = "+pK * I"),
    c("Benefits", H = "UB", G = "-UB"),
    c("Wages", H = "W", FC = "-WFC", FK = "-WFK"),
    c("Profits", H = "P", FC = "-PFC", FK = "-PFK"),
    c("Taxes", H = "-T", G = "T"),
    c("D Money", H = "-(MH - MH[-1])", FC = "-(MFC - MFC[-1])", FK = "-(MFK - MFK[-1])", G = "(M - M[-1])"),
)
sfcr_matrix_display(model_tfm, "tfm")

model_ext <- sfcr_set(
    ay ~ 0.6,
    av ~ 0.2,
    mu ~ 0.2,
    d ~ 0.03,
    tW ~ 0.35,
    tP ~ 0.2,
    tC ~ 0.2,
    W0 ~ 1.0,
    cuT ~ 0.8, # target capacity utilization
    dK ~ 0.1, # capital depreciation rate
    betaC ~ 2.0, # FCs' productivity (units per worker)
    betaK ~ 1.0, # FKs' productivity (units per worker)
    phi ~ 0.7
)

model_init <- sfcr_set(
    MH ~ 1,
    VH ~ 1,
    M ~ 1,
    VG ~ -1,
    W0 ~ 1,
    K ~ 0.1,
    HpC ~ 1.2,
    pC ~ 1,
    pK ~ 1
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
