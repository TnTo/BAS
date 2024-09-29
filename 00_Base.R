### BAS: Building towards Artificial Societies
# Model 0:
#   no Fixed Capital
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
# renv::snapshot()
# renv::status()

install.packages(c("devtools", "tidyverse", "networkD3", "ggraph", "ggplot2"))
devtools::install_github("TnTo/sfcr", ref = "sankey")

library(sfcr)
library(tidyverse)

## A modelCon Dream
model_eqs <- sfcr_set(
    MH ~ MH[-1] + P + W + UB - p * C - T, # Households' Money
    MF ~ MF[-1] + p * Y - W - P, # Firms' Money
    M ~ M[-1] - (T - p * G - UB), # Gvt's Money
    VH ~ MH, # Households' Net Wealth
    VF ~ MF, # Firms' Net Wealth
    VG ~ -M, # Gvt' Net Wealth
    V ~ VH + VF + VG, # System Total Wealth
    NetW ~ (1 - tW) * W, # Net Wages
    DI ~ NetW + UB, # Disposable Income for Hs
    CT ~ max(0, (ay * DI[-1] + av * MH[-1]) / Hp[-1]), # Desired Demand for Hs, in units of goods
    GT ~ max(0, (d * GDP[-1] + T[-1] - UB[-1]) / p[-1]), # Desired Demand for Gvt, in units of goods
    YT ~ CT + GT, # Desired Demand
    W ~ W0 * N, # Wages
    UB ~ W0 * phi * (1 - N), # Unemployment benefits
    N ~ min(1, YT / beta), # Employment share (normalize in [0,1])
    Y ~ beta * N, # Output in units of goods
    C ~ CT * Y / YT, # Hs Consumption, in units of goods
    G ~ Y - C, # Gvt Consumption, in units of goods
    P ~ p * (C + G) - W, # Firms' Profits (all distributed)
    T ~ tW * W + tP * P + tC * p * C, # Taxes
    p ~ (1 + mu) * W / Y, # price
    Hp ~ (1 + tC) * p, # price after VAT (Hs price)
    GDP ~ p * Y
)

sfcr_dag_cycles_plot(model_eqs, size = 6)

model_bs <- sfcr_matrix(
    columns = c("Households", "Firms", "Government", "Sum"),
    codes = c("H", "F", "G", "sum"),
    c("Money", H = "+MH", F = "+MF", G = "-M"),
    c("Balance", H = "-VH", F = "-VF", G = "-VG")
)
sfcr_matrix_display(model_bs, "bs")

model_tfm <- sfcr_matrix(
    columns = c("Households", "Firms", "Government"),
    codes = c("H", "F", "G"),
    c("Consumption", H = "-(p*C)", F = "(p*C)"),
    c("Gvt Spending", F = "(p*G)", G = "-(p*G)"),
    c("Benefits", H = "UB", G = "-UB"),
    c("Wages", H = "W", F = "-W"),
    c("Profits", H = "P", F = "-P"),
    c("Taxes", H = "-T", G = "T"),
    c("D Money", H = "-(MH - MH[-1])", F = "-(MF - MF[-1])", G = "(M - M[-1])")
)
sfcr_matrix_display(model_tfm, "tfm")

model_ext <- sfcr_set(
    W0 ~ 1, # Wage level
    ay ~ 0.5, # Desired share of consumption out of income
    av ~ 0.1, # Desired share of consumptio out of wealth
    beta ~ 1.0, # Output per worker in units of goods
    mu ~ 0.2, # Firms' mark-up
    d ~ 0.03, # Target Gvt deficit
    tW ~ 0.35, # Tax rate on wages
    tP ~ 0.2, # Tax rate on profits
    tC ~ 0.2, # Tax rate on Hs consumption
    phi ~ 0.7 # Unemployment benefit amount (in percentage of wage level)
)

model_init <- sfcr_set(
    MH ~ 1,
    VH ~ 1,
    M ~ 1,
    VG ~ -1
)

model <- sfcr_baseline(
    equations = model_eqs,
    external = model_ext,
    init = model_init,
    periods = 500,
    max_iter = 1000,
    tol = 1e-7,
    hidden = c("V" = 0),
    hidden_tol = 1e-7,
    method = "Broyden"
)

sfcr_validate(model_bs, model, "bs", tol = 1e-7, rtol = TRUE)

sfcr_validate(model_tfm, model, "tfm", tol = 1e-7, rtol = TRUE)

sfcr_sankey(model_tfm, model, when = "end")

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("MH", "MF", "M", "VH", "VF", "VG")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("C", "G", "Y", "W", "P")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("p")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("N")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("Y", "YT")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))
