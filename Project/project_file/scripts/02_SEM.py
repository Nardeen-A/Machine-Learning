import pandas as pd
import numpy as np
from semopy import Model
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          DecisionTreeRegressor as DTR,
                          plot_tree,
                          export_text)
from sklearn.metrics import (accuracy_score,
                             log_loss)
from sklearn.ensemble import \
     (RandomForestRegressor as RF,
      GradientBoostingRegressor as GBR)
from ISLP.bart import BART
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools import add_constant
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Read the Stata file
data = pd.read_stata("E:\\UofT\\03_ML\\Project\\project_file\\data\\AEJApp-20090168_data.dta")

# Drop columns by index (Python is 0-based, so adjust indices)
cols_to_drop = [5, 11, 12, 13, 14, 21, 27, 28]  # R's -c(6,12,13,14,15,22,28,29)
data_clean = data.drop(data.columns[cols_to_drop], axis=1)

# Drop missing values
data = data_clean.dropna()

data.to_csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\clean_data.csv", index=False)
data = pd.read_csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\clean_data.csv")

# Convert 'select' to numeric 0/1
data["select"] = data["select"].map({"control": 0, "selected": 1}).astype(int)
data["select_dwomen"] = data["select"] * data["dwomen"]

data = pd.get_dummies(data, columns=["city", "educ_lb"])
dummy_vars = [col for col in data.columns if col.startswith("city_")]
for col in dummy_vars:
    data[col] = data[col].astype(int)
dummy_vars = [col for col in data.columns if col.startswith("educ_lb_")]
for col in dummy_vars:
    data[col] = data[col].astype(int)
for col in ["salary_06"]:
    data[col] = data[col] / 1000
    
scaler = MinMaxScaler(feature_range=(0, 1))
cont = ["salary_04", "hours_04", "days_04", "profit_04", "age_lb"]
data[cont] = scaler.fit_transform(data[cont])
data.columns = data.columns.str.replace('.', '_', regex=False)

data.to_csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\SEM_scaled_clean_data.csv", index=False)

# SEM model string (lavaan style, works in semopy)
m3e = """
empl_06 ~ select + dwomen + select_dwomen + age_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04 + dformal_04 + pempl_04 + dmarried_lb + city_2_0 + city_3_0 + city_4_0 + city_5_0 + city_6_0 + city_1_0 + educ_lb_2_0 + educ_lb_3_0 + educ_lb_4_0 + educ_lb_5_0 + educ_lb_6_0 + educ_lb_7_0 + educ_lb_8_0 + educ_lb_9_0 + educ_lb_10_0 + educ_lb_0_0 + educ_lb_12_0 + educ_lb_13_0 + educ_lb_14_0 + educ_lb_16_0
pempl_06 ~ select + dwomen + select_dwomen + age_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04 + dformal_04 + pempl_04 + dmarried_lb + city_2_0 + city_3_0 + city_4_0 + city_5_0 + city_6_0 + city_1_0 + educ_lb_2_0 + educ_lb_3_0 + educ_lb_4_0 + educ_lb_5_0 + educ_lb_6_0 + educ_lb_7_0 + educ_lb_8_0 + educ_lb_9_0 + educ_lb_10_0 + educ_lb_0_0 + educ_lb_12_0 + educ_lb_13_0 + educ_lb_14_0 + educ_lb_16_0
salary_06 ~ select + dwomen + select_dwomen + age_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04 + dformal_04 + pempl_04 + dmarried_lb + city_2_0 + city_3_0 + city_4_0 + city_5_0 + city_6_0 + city_1_0 + educ_lb_2_0 + educ_lb_3_0 + educ_lb_4_0 + educ_lb_5_0 + educ_lb_6_0 + educ_lb_7_0 + educ_lb_8_0 + educ_lb_9_0 + educ_lb_10_0 + educ_lb_0_0 + educ_lb_12_0 + educ_lb_13_0 + educ_lb_14_0 + educ_lb_16_0
"""

# Estimate SEM
model = Model(m3e)
res = model.fit(data)
estimates = model.inspect()
pd.set_option("display.max_rows", None)    # show all rows
pd.set_option("display.max_columns", None)  # show all columns
print(estimates)

#################### OLS #############################
# Define dependent and independent variables
y_1 = data['empl_06']
y_2 = data['pempl_06']
y_3 = data['salary_06']

X = data[['select', 'dwomen', 'select_dwomen', 'age_lb', 'empl_04', 'salary_04',
        'hours_04', 'days_04', 'contract_04', 'dformal_04', 'pempl_04',
        'dmarried_lb', 'city_2_0', 'city_3_0', 'city_4_0', 'city_5_0', 'city_6_0',
        'city_1_0', 'educ_lb_2_0', 'educ_lb_3_0', 'educ_lb_4_0', 'educ_lb_5_0',
        'educ_lb_6_0', 'educ_lb_7_0', 'educ_lb_8_0', 'educ_lb_9_0', 'educ_lb_10_0',
        'educ_lb_0_0', 'educ_lb_12_0', 'educ_lb_13_0', 'educ_lb_14_0', 'educ_lb_16_0']]

# Add constant (intercept)
X = sm.add_constant(X)

# Run OLS
OLS_1 = sm.OLS(y_1, X).fit(cov_type='HC1')   # <-- Robust standard errors
OLS_2 = sm.OLS(y_2, X).fit(cov_type='HC1')   # <-- Robust standard errors
OLS_3 = sm.OLS(y_3, X).fit(cov_type='HC1')   # <-- Robust standard errors

# Print results
print(OLS_1.summary())
print(OLS_2.summary())
print(OLS_3.summary())

logit_1 = sm.Logit(y_1, X).fit(disp=0)  # disp=0 suppresses convergence messages
print("\n===== LOGIT Regression: Employment =====")
print(logit_1.summary())

# Paid Employment
logit_2 = sm.Logit(y_2, X).fit(disp=0)
print("\n===== LOGIT Regression: Paid Employment =====")
print(logit_2.summary())
