### BAS: Building towards Artificial Societies
# Model 0:
#   no Fixed Capital
#   no Constraints on Labour Offer
#   no Innovation
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
library(sfcr)
library(tidyverse)

## A modelCon Dream
model_eqs <- sfcr_set(
    MH ~ MH[-1] + P + W - C - T,
    MF ~ MF[-1] + C + G - W - P,
    MG ~ MG[-1] - (T - G),
    VH ~ MH,
    VF ~ MF,
    VG ~ VH + VF,
    C ~ ay * W[-1] + av * MH[-1],
    G ~ d * (Y[-1]) + T[-1],
    Y ~ C + G,
    W ~ W0 * N,
    N ~ Y / (beta * p),
    P ~ C + G - W,
    T ~ tW * W + tP * P + tC * C,
    # p ~ (1 + mu) * W / (Y / p)
    p ~ (1 + mu) * W0 / beta,
    DebtGDP ~ MG / Y,
    GGDP ~ G / Y
)

sfcr_dag_cycles_plot(model_eqs, size = 6)

model_bs <- sfcr_matrix(
    columns = c("Households", "Firms", "Government", "Sum"),
    codes = c("H", "F", "G", "sum"),
    c("Money", H = "+MH", F = "+MF", G = "-MG"),
    c("Balance", H = "-VH", F = "-VF", G = "+VG")
)
sfcr_matrix_display(model_bs, "bs")

model_tfm <- sfcr_matrix(
    columns = c("Households", "Firms", "Government"),
    codes = c("H", "F", "G"),
    c("Consumption", H = "-C", F = "C"),
    c("Gvt Spending", F = "G", G = "-G"),
    c("Wages", H = "W", F = "-W"),
    c("Profits", H = "P", F = "-P"),
    c("Taxes", H = "-T", G = "T"),
    c("D Money", H = "-(MH - MH[-1])", F = "-(MF - MF[-1])", G = "(MG - MG[-1])")
)
sfcr_matrix_display(model_tfm, "tfm")

model_ext <- sfcr_set(
    W0 ~ 1,
    ay ~ 0.6,
    av ~ 0.4,
    beta ~ 1.2,
    mu ~ 0.2,
    d ~ 0.03,
    tW ~ 0.35,
    tP ~ 0.2,
    tC ~ 0.2
)

model_init <- sfcr_set(
    MH ~ 1,
    VH ~ 1,
    MG ~ 1,
    VG ~ 1
)

model <- sfcr_baseline(
    equations = model_eqs,
    external = model_ext,
    init = model_init,
    periods = 100,
    tol = 1e-7,
    hidden = c("MG" = "VG"),
    hidden_tol = 1e-7,
    method = "Broyden"
)

sfcr_validate(model_bs, model, "bs", tol = 1e-7, rtol = TRUE)

sfcr_validate(model_tfm, model, "tfm", tol = 1e-7, rtol = TRUE)

sfcr_sankey(model_tfm, model, when = "end")

model %>%
    pivot_longer(cols = -period) %>%
    filter(name %in% c("MH", "MF", "MG", "VH", "VF", "VG")) %>%
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
    filter(name %in% c("DebtGDP", "GGDP")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))
