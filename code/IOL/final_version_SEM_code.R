library(lavaan)
library(semTools)

# read data
mydata <- read.csv(
  "all_data_to_sem.csv",
  header = TRUE, sep = ","
)

# --------- ordered items (ordinal)------
ordered_items <- c(
  "iol1","iol2","iol3","iol4",
  "fatig1","fatig2","fatig3","fatig5",
  "satf1","satf2","satf3","satf4","satf6"
)

# ---------------- CFA  ----------------
modelCFA_new <- '

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

fit_cfa_new <- cfa(
  modelCFA_new,
  data = mydata,
  estimator = "WLSMV",
  ordered = ordered_items,
  parameterization = "theta",
  std.lv = TRUE,
  missing = "listwise"
)

summary(fit_cfa_new, fit.measures = TRUE, standardized = TRUE)
lavInspect(fit_cfa_new, "converged")
lavInspect(fit_cfa_new, "post.check")


modelSEM_v9 <- '

# Measurement
Fpatrn_m =~ patrn2 + patrn3
Fpatrn_d =~ patrn7 + patrn8 + patrn10 + patrn11

Fiol     =~ iol1 + iol2 + iol3 + iol4
FfatigA  =~ fatig1 + fatig2 + fatig3 + fatig5

Fsat_sys =~ satf1 + satf2 + satf3
Ftrust_peer =~ satf4 + satf6

Fundrst_m =~ understand_project_patrn2 + understand_project_patrn3
Fundrst_d =~ understand_project_patrn7 + understand_project_patrn8 + understand_project_patrn10 + understand_project_patrn11

# Structural (שימור החזקים + שילוב satisfaction)
FfatigA   ~ Fiol + Fsat_sys
Fsat_sys  ~ Fiol
Ftrust_peer ~ Fiol   # אפשר להשאיר; אם יבלגן - נוציא

# כאן החלק הקריטי:
# במקום Fsat_sys -> Fpatrn_d, תני לו להשפיע על הבנה/קוגניציה
Fundrst_d ~ FfatigA
Fundrst_m ~ FfatigA #+ Fsat_sys

# Pattern בסוף כמו שעבד לך
Fpatrn_d ~ Fundrst_d  + Fiol 
Fpatrn_m ~ Fundrst_m + Ftrust_peer

# Residual covariances (אותן התאמות שעבדו)
fatig3 ~~ fatig5
patrn7 ~~ patrn10
patrn8 ~~ patrn11
understand_project_patrn7 ~~ understand_project_patrn10
understand_project_patrn8 ~~ understand_project_patrn11
'
fit_v9 <- sem(
  modelSEM_v9, data = mydata,
  estimator = "WLSMV", ordered = ordered_items,
  parameterization = "theta",
  std.lv = TRUE, missing = "listwise"
)

summary(fit_v9, fit.measures = TRUE, standardized = TRUE)


modelSEM_v9_indirect <- '

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

fit_v9_indirect <- sem(
  modelSEM_v9_indirect,
  data = mydata,
  estimator = "WLSMV",
  ordered = ordered_items,
  parameterization = "theta",
  std.lv = TRUE,
  missing = "listwise"
)

summary(fit_v9_indirect, standardized = TRUE, fit.measures = TRUE, rsquare = TRUE)

pe <- parameterEstimates(fit_v9_indirect, standardized = TRUE, ci = TRUE)
pe[pe$op == ":=", ]

indirect_table <- pe[pe$op == ":=", c("lhs", "op", "rhs", "est", "se", "z", "pvalue", "ci.lower", "ci.upper", "std.all")]
indirect_table

#install.packages("semTools")
library(semTools)

reliability(fit_v9_indirect)   # fit = אובייקט ה-SEM שלך

cor_lv <- lavInspect(fit_v9_indirect, "cor.lv")

# 2. AVE לכל לטנט, מתוך reliability()
rel <- reliability(fit_v9_indirect)
AVE_vals <- rel["avevar", ]  # וקטור AVE לפי שמות הלטנטים

# 3. בניית טבלת Fornell–Larcker
FL <- cor_lv
diag(FL) <- sqrt(AVE_vals[names(diag(FL))])

round(FL, 3)
parameterEstimates(fit_v9_indirect, standardized = TRUE)
lavInspect(fit_v9_indirect, "r2")

