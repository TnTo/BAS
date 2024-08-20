### BAS: Building towards Artificial Societies
# Model 2:
#   no Fixed Capital
#   X  Constraints on Labour Offer
#   X  Innovation
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
renv::snapshot()
renv::status()
library(sfcr)
library(tidyverse)
library(purrr)
library(ggpubr)

## A modelCon Dream
model_eqs <- sfcr_set(
    MH ~ MH[-1] + P + W + M - pC * C - T,
    MFC ~ MFC[-1] + pC * (C + G) - pK * I - WFC - PFC,
    MFK ~ MFK[-1] + pK * I - WFK - PFK,
    MG ~ MG[-1] - (T - pC * G - M),
    KFC ~ (1 - dK) * KFC[-1] + I,
    KFK ~ (1 - dK) * KFK[-1] + IK,
    K ~ KFC + KFK,
    VH ~ MH,
    VFC ~ MFC + pK * KFC,
    VFK ~ MFK + pK * KFK,
    VG ~ -MG,
    CT ~ max(0, (ay * (1 - tW) * (W[-1] + M[-1]) + av * MH[-1]) / ((1 + tC) * pC[-1])),
    GT ~ max(0, (d * (Y[-1]) + T[-1] - M[-1]) / pC[-1]),
    YT ~ CT + GT,
    IT ~ max(0, KFCu[-1] * ThetaI * (1 / cuT - 1 / cuC[-1]) + dK * KFC[-1], na.rm = TRUE),
    IKT ~ max(0, KFKu[-1] * ThetaI * (1 / cuT - 1 / cuK[-1]) + dK * KFK[-1], na.rm = TRUE),
    YKT ~ IT + IKT,
    NCT ~ min(1, KFC[-1], YT / betaC[-1]),
    NKT ~ min(1, KFK[-1], YKT / betaK[-1]),
    NT ~ NCT + NKT,
    N ~ min(1, NT),
    NC ~ NCT * N / NT,
    NK ~ NKT * N / NT,
    W0 ~ W0[-1] * (1 + aW * N[-1]),
    W ~ WFC + WFK,
    WFC ~ W0 * NC,
    WFK ~ W0 * NK,
    M ~ phi * W0 * (1 - N),
    Y ~ NC * betaC[-1],
    C ~ min(CT * Y / YT, ((1 - tW) * (W + M) + max(0, MH[-1])) / ((1 + tC) * pC)),
    G ~ min(GT, Y - C),
    YK ~ NK * betaK[-1],
    I ~ ifelse(IT > 0, IT * YK / YKT, 0),
    IK ~ ifelse(IKT > 0, IKT * YK / YKT, 0),
    P ~ PFC + PFK,
    PFC ~ max(0, MFC[-1] + pC * (C + G) - 2 * (pK * I + WFC)),
    PFK ~ max(0, MFK[-1] + pK * I - 2 * WFK),
    T ~ tW * (W + M) + tP * P + tC * pC * C,
    muC ~ muC[-1] * (1 + ThetaMu * (cuC[-1] - cuT) / cuT),
    muK ~ muK[-1] * (1 + ThetaMu * (cuK[-1] - cuT) / cuT),
    pC ~ (1 + muC) * WFC / Y,
    pK ~ (1 + muK) * WFK / I,
    betaC ~ betaC[-1] * (1 + aBetaC * N),
    betaK ~ betaK[-1] * (1 + aBetaK * N),
    KFCu ~ min(NC, KFC[-1]),
    KFKu ~ min(NK, KFK[-1]),
    cuC ~ KFCu / KFC[-1],
    cuK ~ KFKu / KFK[-1],
    #
    Ku ~ KFCu + KFKu,
    cu ~ Ku / K[-1],
    g ~ GDP / GDP[-1] - 1,
    rg ~ Y / Y[-1] - 1,
    GDP ~ Y * pC + I * pK,
    V ~ VH + VFC + VFK + VG,
    pKK ~ pK * K,
    rEFC ~ PFC / VFC,
    rEFK ~ PFK / VFK,
    i ~ pC / pC[-1] - 1
)

sfcr_dag_cycles_plot(model_eqs, size = 6)

model_bs <- sfcr_matrix(
    columns = c("Households", "Consumption Firms", "Capital Firms", "Government", "Sum"),
    codes = c("H", "FC", "FK", "G", "sum"),
    c("Money", H = "+MH", FC = "+MFC", FK = "+MFK", G = "-MG"),
    c("Capital", FC = "+pK * KFC", FK = "+pK * KFK", sum = "+pK * K"),
    c("Balance", H = "-VH", FC = "-VFC", FK = "-VFK", G = "-VG", sum = "-pK * K")
)
sfcr_matrix_display(model_bs, "bs")

model_tfm <- sfcr_matrix(
    columns = c("Households", "Consumption Firms", "Capital Firms", "Government"),
    codes = c("H", "FC", "FK", "G"),
    c("Consumption", H = "-pC * C", FC = "pC * C"),
    c("Gvt Spending", FC = "pC * G", G = "-pC * G"),
    c("Investment", FC = "-pK * I", FK = "+pK * I"),
    c("Wages", H = "W", FC = "-WFC", FK = "-WFK"),
    c("Profits", H = "P", FC = "-PFC", FK = "-PFK"),
    c("Transferts", H = "M", G = "-M"),
    c("Taxes", H = "-T", G = "T"),
    # c("K Revaluation", FC = "+(pK - pK[-1]) * KFC[-1], FK = "+(pK - pK[-1]) * KFK[-1]"),
    c("D Money", H = "-(MH - MH[-1])", FC = "-(MFC - MFC[-1])", FK = "-(MFK - MFK[-1])", G = "(MG - MG[-1])"),
    # c("D Capital", FC = "+pK * (KFC - KFC[-1])", FC = "+pK * (KFK - KFK[-1])"),
)
sfcr_matrix_display(model_tfm, "tfm")

model_ext <- sfcr_set(
    d ~ 0.03,
    tW ~ 0.35,
    tP ~ 0.2,
    tC ~ 0.2,
    aW ~ 0.05,
    cuT ~ 0.8,
    dK ~ 0.1,
    phi ~ 0.7,
    ###
    ThetaMu ~ 0.05,
    ThetaI ~ 0.1,
    ###
    ay ~ 0.74,
    av ~ 0.2,
    aBetaC ~ 0.05,
    aBetaK ~ 0
)

model_init <- sfcr_set(
    MH ~ 1,
    MG ~ 1,
    KFC ~ 0.3,
    KFK ~ 0.3,
    K ~ 0.6,
    VH ~ 1,
    VFC ~ 0.3,
    VFK ~ 0.3,
    VG ~ -1,
    W0 ~ 1,
    pK ~ 1,
    betaC ~ 1.0,
    betaK ~ 1.0,
    cuC ~ 1,
    cuK ~ 1,
    muC ~ 0.2,
    muK ~ 0.2
)

model <- sfcr_baseline(
    equations = model_eqs,
    external = model_ext,
    init = model_init,
    periods = 200,
    tol = 1e-7,
    hidden = c("V" = "pKK"),
    hidden_tol = 1e-7,
    # hidden = NULL,
    method = "Broyden"
)

sfcr_validate(model_bs, model, "bs", tol = 1e-7, rtol = TRUE)

sfcr_validate(model_tfm, model, "tfm", tol = 1e-7, rtol = TRUE)

sfcr_sankey(model_tfm, model, when = "end")

data <- model %>%
    pivot_longer(cols = -period)

suffixes <- c("")

plot_vars_multi <- function(data, vars, suffixes = c(""), start = 1) {
    ggarrange(
        plotlist = map(
            suffixes,
            (\(s) {
                data %>%
                    filter(name %in% map(
                        vars,
                        (\(x) paste(x, s, sep = ""))
                    )) %>%
                    filter(period >= start) %>%
                    ggplot(aes(x = period, y = value)) +
                    geom_line(aes(linetype = name, color = name))
            })
        ),
        nrow = length(suffixes)
    )
}

plot_vars_multi(
    data,
    c("MH", "VG", "Y"),
    suffixes
)

plot_vars_multi(
    data,
    c("MH", "W", "M", "T", "C", "P"),
    suffixes
)

plot_vars_multi(
    data,
    c("N", "NC", "NK"),
    suffixes
)

plot_vars_multi(
    data,
    c("muC", "muK", "rEFC", "rEFK"),
    suffixes
)

plot_vars_multi(
    data,
    c("pC"),
    suffixes
)

plot_vars_multi(
    data,
    c("pK"),
    suffixes
)


###
#
###
