import pandas as pd
import numpy as np
from semopy import Model

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

# coursefixe: unique identifier for each course within each city
data["coursefixe"] = data.groupby(["city", "codigo_ecap", "codigo_curs"]).ngroup()

# Scale numeric columns (mean 0, sd 1)
numeric_cols = data.select_dtypes(include=[np.number]).columns
data_scaled = data.copy()
data_scaled[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()

data_scaled["select_dwomen"] = data_scaled["select"] * data_scaled["dwomen"]

# SEM model string (lavaan style, works in semopy)
m3e = """
empl_06 ~ select + dwomen + select_dwomen + age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb
pempl_06 ~ select + dwomen + select_dwomen + age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb
salary_06 ~ select + dwomen + select_dwomen + age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb
"""

# Estimate SEM
model = Model(m3e)
res = model.fit(data_scaled)

# Summarize results
estimates = model.inspect()
print(estimates)
