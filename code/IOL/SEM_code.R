library(lavaan)
library(semTools)

# --- Load data ---
mydata <- read.csv("C:/Users/USER/Desktop/לימודים/תואר שני/תזה/דאטה/all_data_to_sem (4).csv",
                   header = TRUE, sep = ","
)

# --------- ordered items (only ordinal) ---------
ordered_items <- c(
  "iol1","iol2","iol3","iol4",
  "fatig1","fatig2","fatig3","fatig5",
  "satf1","satf2","satf3","satf4","satf6"
)

# ---------------- CFA (including new understanding decay component) ----------------
modelCFA <- '

Fpatrn_m =~ patrn2 + patrn3
Fpatrn_d =~ patrn7 + patrn8 + patrn10 + patrn11

Fiol     =~ iol1 + iol2 + iol3 + iol4
FfatigA  =~ fatig1 + fatig2 + fatig3 + fatig5

Fsat_sys =~ satf1 + satf2 + satf3
Ftrust_peer =~ satf4 + satf6

Fundrst_m =~ understand_project_patrn2 + understand_project_patrn3
Fundrst_d =~ understand_project_patrn7 + understand_project_patrn8 + understand_project_patrn10 + understand_project_patrn11

fatig3 ~~ fatig5
patrn7 ~~ patrn10
patrn8 ~~ patrn11
understand_project_patrn7 ~~ understand_project_patrn10
understand_project_patrn8 ~~ understand_project_patrn11
'

fit_cfa <- cfa(
  modelCFA,
  data = mydata,
  estimator = "WLSMV",
  ordered = ordered_items,
  parameterization = "theta",
  std.lv = TRUE,
  missing = "listwise"
)

summary(fit_cfa, fit.measures = TRUE, standardized = TRUE)
lavInspect(fit_cfa, "converged")
lavInspect(fit_cfa, "post.check")


modelSEM <- '

# Measurement
Fpatrn_m =~ patrn2 + patrn3
Fpatrn_d =~ patrn7 + patrn8 + patrn10 + patrn11

Fiol     =~ iol1 + iol2 + iol3 + iol4
FfatigA  =~ fatig1 + fatig2 + fatig3 + fatig5

Fsat_sys =~ satf1 + satf2 + satf3
Ftrust_peer =~ satf4 + satf6

Fundrst_m =~ understand_project_patrn2 + understand_project_patrn3
Fundrst_d =~ understand_project_patrn7 + understand_project_patrn8 + understand_project_patrn10 + understand_project_patrn11

# Structural (keeping strong paths + adding satisfaction)
FfatigA   ~ Fiol + Fsat_sys
Fsat_sys  ~ Fiol
Ftrust_peer ~ Fiol   # can remain; remove if it causes instability

# Critical part:
# Instead of Fsat_sys -> Fpatrn_d, let it influence cognition/understanding
Fundrst_d ~ FfatigA
Fundrst_m ~ FfatigA #+ Fsat_sys

# Pattern outcomes (as previously worked well)
Fpatrn_d ~ Fundrst_d  + Fiol 
Fpatrn_m ~ Fundrst_m + Ftrust_peer

# Residual covariances (same adjustments that previously worked)
fatig3 ~~ fatig5
patrn7 ~~ patrn10
patrn8 ~~ patrn11
understand_project_patrn7 ~~ understand_project_patrn10
understand_project_patrn8 ~~ understand_project_patrn11
'

fit_sem <- sem(
  modelSEM, data = mydata,
  estimator = "WLSMV", ordered = ordered_items,
  parameterization = "theta",
  std.lv = TRUE, missing = "listwise"
)

summary(fit_sem, fit.measures = TRUE, standardized = TRUE)


modelSEM_indirect <- '

Fpatrn_m =~ patrn2 + patrn3
Fpatrn_d =~ patrn7 + patrn8 + patrn10 + patrn11

Fiol     =~ iol1 + iol2 + iol3 + iol4
FfatigA  =~ fatig1 + fatig2 + fatig3 + fatig5

Fsat_sys =~ satf1 + satf2 + satf3
Ftrust_peer =~ satf4 + satf6

Fundrst_m =~ understand_project_patrn2 + understand_project_patrn3
Fundrst_d =~ understand_project_patrn7 + understand_project_patrn8 + understand_project_patrn10 + understand_project_patrn11

FfatigA ~ a1*Fiol + a2*Fsat_sys
Fsat_sys ~ a3*Fiol
Ftrust_peer ~ a4*Fiol

Fundrst_d ~ b1*FfatigA
Fundrst_m ~ b2*FfatigA

Fpatrn_d ~ c1*Fundrst_d + c2*Fiol
Fpatrn_m ~ c3*Fundrst_m + c4*Ftrust_peer

fatig3 ~~ fatig5
patrn7 ~~ patrn10
patrn8 ~~ patrn11
understand_project_patrn7 ~~ understand_project_patrn10
understand_project_patrn8 ~~ understand_project_patrn11

ind_Fiol_FfatigA_Fundrst_d := a1*b1
ind_Fiol_FfatigA_Fundrst_m := a1*b2
ind_Fiol_Ftrustpeer_Fpatrn_m := a4*c4
ind_Fiol_FfatigA_Fundrst_d_Fpatrn_d := a1*b1*c1
ind_Fiol_FfatigA_Fundrst_m_Fpatrn_m := a1*b2*c3
ind_Fsat_FfatigA_Fundrst_d_Fpatrn_d := a2*b1*c1
ind_Fsat_FfatigA_Fundrst_m_Fpatrn_m := a2*b2*c3

total_Fiol_on_Fpatrn_d := c2 + (a1*b1*c1)
total_Fiol_on_Fpatrn_m := (a4*c4) + (a1*b2*c3)
'

fit_sem_indirect <- sem(
  modelSEM_indirect,
  data = mydata,
  estimator = "WLSMV",
  ordered = ordered_items,
  parameterization = "theta",
  std.lv = TRUE,
  missing = "listwise"
)

summary(fit_sem_indirect, standardized = TRUE, fit.measures = TRUE, rsquare = TRUE)

pe <- parameterEstimates(fit_sem_indirect, standardized = TRUE, ci = TRUE)
pe[pe$op == ":=", ]

indirect_table <- pe[pe$op == ":=", c("lhs", "op", "rhs", "est", "se", "z", "pvalue", "ci.lower", "ci.upper", "std.all")]
indirect_table

# Reliability analysis
library(semTools)
reliability(fit_sem_indirect)

# Latent correlations
cor_lv <- lavInspect(fit_sem_indirect, "cor.lv")

# AVE values
rel <- reliability(fit_sem_indirect)
AVE_vals <- rel["avevar", ]

# Fornell–Larcker table
FL <- cor_lv
diag(FL) <- sqrt(AVE_vals[names(diag(FL))])

round(FL, 3)

parameterEstimates(fit_sem_indirect, standardized = TRUE)
lavInspect(fit_sem_indirect, "r2")