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
library(ggpubr)

## A modelCon Dream
model_eqs <- sfcr_set(
    MH ~ MH[-1] + P + W - pC * C - T,
    MFC ~ MFC[-1] + pC * C + pC * G - pK * I - WFC - PFC,
    MFK ~ MFK[-1] + pK * I - WFK - PFK,
    MG ~ MG[-1] - (T - pC * G),
    KC ~ (1 - dK) * KC[-1] + I,
    KK ~ (1 - dK) * KK[-1] + IK,
    K ~ KC + KK,
    VH ~ MH,
    VFC ~ MFC + pK * KC,
    VFK ~ MFK + pK * KK,
    VG ~ -MG,
    CT ~ (ay * W[-1] + av * MH[-1]) / pC[-1],
    GT ~ (d * (Y[-1]) + T[-1]) / pC[-1],
    YT ~ CT + GT,
    IT ~ max(0, KCu[-1] * (1 / cuT - 1 / cuC[-1]) + dK * KC[-1], na.rm = TRUE),
    IKT ~ max(0, KKu[-1] * (1 / cuT - 1 / cuK[-1]) + dK * KK[-1], na.rm = TRUE),
    YKT ~ IT + IKT,
    NCT ~ min(1, KC[-1], YT / betaC[-1]),
    NKT ~ min(1, KK[-1], YKT / betaK[-1]),
    NT ~ NCT + NKT,
    N ~ min(1, NT),
    NC ~ NCT * N / NT,
    NK ~ NKT * N / NT,
    W0 ~ W0[-1] * (1 + aW * N[-1]),
    W ~ WFC + WFK,
    WFC ~ W0 * NC,
    WFK ~ W0 * NK,
    Y ~ NC * betaC[-1],
    C ~ CT * Y / YT,
    G ~ GT * Y / YT,
    YK ~ NK * betaK[-1],
    I ~ ifelse(IT > 0, IT * YK / YKT, 0),
    IK ~ ifelse(IKT > 0, IKT * YK / YKT, 0),
    P ~ PFC + PFK,
    PFC ~ pC * C + pC * G - WFC - pK * I,
    PFK ~ pK * I - WFK,
    T ~ tW * W + tP * P + tC * pC * C,
    pC ~ (1 + mu) * W0 / betaC[-1],
    pK ~ (1 + mu) * W0 / betaK[-1],
    betaC ~ betaC0 * betaC[-1] * (1 + aBeta * N),
    betaK ~ betaK0 * betaK[-1] * (1 + aBeta * N),
    KCu ~ min(NC, KC[-1]),
    KKu ~ min(NK, KK[-1]),
    Ku ~ KCu + KKu,
    cuC ~ KCu / KC[-1],
    cuK ~ KKu / KK[-1],
    cu ~ Ku / K[-1],
    g ~ GDP / GDP[-1] - 1,
    rg ~ Y / Y[-1] - 1,
    GDP ~ Y * pC + I * pK,
    V ~ VH + VFC + VFK + VG,
    pKK ~ pK * K
)

sfcr_dag_cycles_plot(model_eqs, size = 6)

model_bs <- sfcr_matrix(
    columns = c("Households", "Consumption Firms", "Capital Firms", "Government", "Sum"),
    codes = c("H", "FC", "FK", "G", "sum"),
    c("Money", H = "+MH", FC = "+MFC", FK = "+MFK", G = "-MG"),
    c("Capital", FC = "+pK * KC", FK = "+pK * KK", sum = "+pK * K"),
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
    c("Taxes", H = "-T", G = "T"),
    # c("K Revaluation", FC = "+(pK - pK[-1]) * KC[-1], FK = "+(pK - pK[-1]) * KK[-1]"),
    c("D Money", H = "-(MH - MH[-1])", FC = "-(MFC - MFC[-1])", FK = "-(MFK - MFK[-1])", G = "(MG - MG[-1])"),
    # c("D Capital", FC = "+pK * (KC - KC[-1])", FC = "+pK * (KK - KK[-1])"),
)
sfcr_matrix_display(model_tfm, "tfm")

model_ext <- sfcr_set(
    ay ~ 0.6,
    av ~ 0.4,
    mu ~ 0.2,
    d ~ 0.03,
    tW ~ 0.35,
    tP ~ 0.2,
    tC ~ 0.2,
    aW ~ 0.05,
    aBeta ~ 0.05,
    cuT ~ 0.8,
    dK ~ 0.1,
    betaC0 ~ 1,
    betaK0 ~ 1
)

model_init <- sfcr_set(
    MH ~ 1,
    VH ~ 1,
    MG ~ 1,
    VG ~ 1,
    W0 ~ 1,
    KC ~ 0.1,
    KK ~ 0.1,
    K ~ 0.2,
    betaC ~ 1.1,
    betaK ~ 1.1,
    cuC ~ 1,
    cuK ~ 1
)

model_base <- sfcr_baseline(
    equations = model_eqs,
    external = model_ext,
    init = model_init,
    periods = 50,
    tol = 1e-7,
    hidden = c("V" = "pKK"),
    hidden_tol = 1e-7,
    method = "Broyden"
)

sfcr_validate(model_bs, model_base, "bs", tol = 1e-7, rtol = TRUE)

sfcr_validate(model_tfm, model_base, "tfm", tol = 1e-7, rtol = TRUE)

sfcr_sankey(model_tfm, model_base, when = "end")

C_shock <- sfcr_shock(
    variables = sfcr_set(betaC0 ~ 10.0),
    start = 5,
    end = 5
)

K_shock <- sfcr_shock(
    variables = sfcr_set(betaK0 ~ 0.001),
    start = 5,
    end = 5
)

model <- sfcr_scenario(
    model_base,
    NULL,
    periods = 100
)

C_model <- sfcr_scenario(
    model_base,
    C_shock,
    periods = 100
)

K_model <- sfcr_scenario(
    model_base,
    K_shock,
    periods = 100
)

data <- model %>%
    full_join(C_model, by = "period", suffix = c("", ".C")) %>%
    full_join(K_model, by = "period", suffix = c("", ".K")) %>%
    pivot_longer(cols = -period)

ggarrange(
    data %>%
        filter(name %in% c("MH", "MFC", "MFK", "MG", "VH", "VF", "VG")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("MH.C", "MFC.C", "MFK.C", "MG.C", "VH.C", "VF.C", "VG.C")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("MH.K", "MFC.K", "MFK.K", "MG.K", "VH.K", "VF.K", "VG.K")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    nrow = 3
)

ggarrange(
    data %>%
        filter(name %in% c("C", "G", "Y", "YK", "I", "IK", "W", "P", "GDP")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("C.C", "G.C", "Y.C", "YK.C", "I.C", "IK.C", "W.C", "P.C", "GDP.C")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("C.K", "G.K", "Y.K", "YK.K", "I.K", "IK.K", "W.K", "P.K", "GDP.K")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    nrow = 3
)

ggarrange(
    data %>%
        filter(name %in% c("PFC", "PFK")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("PFC.C", "PFK.C")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("PFC.K", "PFK.K")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    nrow = 3
)

data %>%
    filter(name %in% c("I", "I.C", "I.K")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

ggarrange(
    data %>%
        filter(name %in% c("pC", "pK", "NC", "NK")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("pC.C", "pK.C", "NC.C", "NK.C")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("pC.K", "pK.K", "NC.K", "NK.K")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    nrow = 3
)

ggarrange(
    data %>%
        filter(name %in% c("N", "Y")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("N.C", "Y.C")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    data %>%
        filter(name %in% c("N.K", "Y.K")) %>%
        ggplot(aes(x = period, y = value)) +
        geom_line(aes(linetype = name, color = name)),
    nrow = 3
)

data %>%
    filter(name %in% c("g", "rg", "g.C", "rg.C", "g.K", "rg.K")) %>%
    filter(period > 6) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

data %>%
    filter(name %in% c("N", "NC", "NK", "N.C", "NC.C", "NK.C", "N.K", "NC.K", "NK.K")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

data %>%
    filter(name %in% c("I", "IK", "I.C", "IK.C", "I.K", "IK.K")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

data %>%
    filter(name %in% c("betaC", "betaK", "betaC.C", "betaK.C", "betaC.K", "betaK.K")) %>%
    ggplot(aes(x = period, y = value)) +
    geom_line(aes(linetype = name, color = name))

###
# if betaK is too low economy collapse
###
