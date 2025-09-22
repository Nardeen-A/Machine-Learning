# packages
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# convert stata to csv
data = pd.read_stata("E:\\UofT\\03_ML\\Project\\project_file\\data\\AEJApp-20090168_data.dta")
data = pd.DataFrame(data)

# removing NA values and unnecessary columns
data = data.dropna(subset = ["salary_06"])
data = data.drop(columns = ["lsalary_06", "lprofit_06", "ldays_06", "lhours_06", "ldays_04", "lhours_04", "tenure_04"])
data = data.dropna()

data.to_csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\full_data.csv", index = False)
data = pd.read_csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\full_data.csv")

# OLS fixed effects regression #
# Create coursefixe (Stata: egen group(city codigo_ecap codigo_curs))
data["coursefixe"] = data.groupby(["city", "codigo_ecap", "codigo_curs"]).ngroup()
data["select"] = data["select"].map({"control": 0, "selected": 1}).astype(int)

# Treatment variable: training offer dummy
treatment = "select"   

# Baseline controls (Panel B)
controls = [
    "age_lb", "educ_lb", "dmarried_lb", 
    "empl_04", "pempl_04", "salary_04", "profit_04",
    "dformal_04", "contract_04", "days_04", "hours_04", "city"
]

# Fixed effects: course
fe = "coursefixe"

# --- STEP 1: Define "no_benefit" outcome for treated individuals ---
data["no_benefit"] = ((data["select"] == 1) & (data["empl_06"] == 0)).astype(int)
logit_formula = "no_benefit ~ age_lb + educ_lb + empl_04 + salary_04 + dmarried_lb + profit_04 + dformal_04 + contract_04 + days_04 + hours_04 + C(city)"

# --- STEP 2: Split regressions by gender ---
# Women only
data_women = data[data["dwomen"] == 1].copy()
if not data_women.empty:
    print("\n--- Logistic regression for WOMEN ---")
    model_w = smf.logit(formula=logit_formula, data=data_women).fit()
    print(model_w.summary())
    mfx_w = model_w.get_margeff(at="mean")
    print(mfx_w.summary())
else:
    print("[Warning] No female observations found.")

# Men only
data_men = data[data["dwomen"] == 0].copy()
if not data_men.empty:
    print("\n--- Logistic regression for MEN ---")
    model_m = smf.logit(formula=logit_formula, data=data_men).fit()
    print(model_m.summary())
    mfx_m = model_m.get_margeff(at="mean")
    print(mfx_m.summary())
else:
    print("[Warning] No male observations found.")

