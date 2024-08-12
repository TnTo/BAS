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
    muC ~ muC[-1] * (1 + Thetha * (cuC[-1] - cuT) / cuT),
    muK ~ muK[-1] * (1 + Thetha * (cuK[-1] - cuT) / cuT),
    pC ~ (1 + muC) * W0 / betaC[-1],
    pK ~ (1 + muK) * W0 / betaK[-1],
    betaC ~ betaC[-1] * (1 + aBetaC * N),
    betaK ~ betaK[-1] * (1 + aBetaK * N),
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
    d ~ 0.03,
    tW ~ 0.35,
    tP ~ 0.2,
    tC ~ 0.2,
    aW ~ 0.05,
    aBetaC ~ 0.05,
    aBetaK ~ 0.05,
    cuT ~ 0.8,
    dK ~ 0.1,
    Thetha ~ 0.1
)

model_init <- sfcr_set(
    MH ~ 1,
    VH ~ 1,
    MG ~ 1,
    VG ~ 1,
    W0 ~ 1,
    KC ~ 0.3,
    KK ~ 0.3,
    K ~ 0.6,
    betaC ~ 1.1,
    betaK ~ 1.1,
    cuC ~ 1,
    cuK ~ 1,
    muC ~ 0.2,
    muK ~ 0.2
)

model_base <- sfcr_baseline(
    equations = model_eqs,
    external = model_ext,
    init = model_init,
    periods = 2,
    tol = 1e-7,
    hidden = c("V" = "pKK"),
    hidden_tol = 1e-7,
    method = "Broyden"
)

sfcr_validate(model_bs, model_base, "bs", tol = 1e-7, rtol = TRUE)

sfcr_validate(model_tfm, model_base, "tfm", tol = 1e-7, rtol = TRUE)

L <- 250

m0 <- sfcr_scenario(
    model_base,
    NULL,
    periods = L
)

sh2 <- sfcr_shock(
    variables = sfcr_set(aBetaC ~ 0, aBetaK ~ 0.05),
    start = 1,
    end = L
)

m2 <- sfcr_scenario(
    model_base,
    sh2,
    periods = L
)

sh3 <- sfcr_shock(
    variables = sfcr_set(aBetaC ~ 0.05, aBetaK ~ 0),
    start = 1,
    end = L
)

m3 <- sfcr_scenario(
    model_base,
    sh3,
    periods = L
)

data <- m0 %>%
    full_join(m2, by = "period", suffix = c("", ".2")) %>%
    full_join(m3, by = "period", suffix = c("", ".3")) %>%
    pivot_longer(cols = -period)

suffixes <- c("", ".2", ".3")

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
    c("MH", "MFC", "MFK", "MG", "VH", "VFC", "VFK", "VG"),
    suffixes
)

plot_vars_multi(
    data,
    c("C", "G", "Y", "YK", "I", "IK", "W", "P", "GDP"),
    suffixes
)

plot_vars_multi(
    data,
    c("C", "G", "Y", "YK", "I", "IK"),
    suffixes
)

plot_vars_multi(
    data,
    c("I", "IK"),
    suffixes
)

plot_vars_multi(
    data,
    c("P", "PFC", "PFK"),
    suffixes
)

plot_vars_multi(
    data,
    c("pC", "pK", "muC", "muK", "NC", "NK"),
    suffixes
)

plot_vars_multi(
    data,
    c("cu", "cuC", "cuK"),
    suffixes
)

plot_vars_multi(
    data,
    c("g", "rg"),
    suffixes,
    start = 5
)

plot_vars_multi(
    data,
    c("N", "NC", "NK"),
    suffixes,
    start = 5
)

ms <- list(m0, m1, m2, m3)
ggarrange(
    plotlist = unlist(map(
        ms,
        (\(m) {
            d <- data.frame(
                betaK_betaC = m$betaK / m$betaC,
                NC_NK = m$NC / m$NK,
                KC_KK = m$KC / m$KK,
                period = m$period
            )
            map(
                Filter((\(s) s != "NC_NK"), names(d)),
                (\(v) {
                    g <- ggplot(d) +
                        geom_line((aes(x = .data[[v]], y = NC_NK)))
                })
            )
        })
    ), recursive = FALSE),
    nrow = length(ms),
    ncol = 3
)

###
# No K innovation stabilize the employment share in FK sector BUT create negative profits
###
