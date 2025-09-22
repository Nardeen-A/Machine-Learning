import pandas as pd
from pandasgui import show

# Load the Stata dataset
data = pd.read_stata("E:\\UofT\\03_ML\\Project\\project_file\\data\\AEJApp-20090168_data.dta")

# define pre/post variable groups
var_map = {
    "age": ["age_lb", "age_s"],
    "education": ["educ_lb", "educ_s"],
    "tenure": ["tenure_04", "tenure_06"],
    "salary": ["salary_04", "salary_06"],
    "profit": ["profit_04", "profit_06"],
    "days_worked": ["days_04", "days_06"],
    "hours_worked": ["hours_04", "hours_06"]
}

def make_summary(df):
    frames = []
    for label, (pre, post) in var_map.items():
        temp = df[[pre, post]].agg(["mean", "median", "std", "min", "max"])
        temp = temp.T.reset_index().rename(columns={"index": "Variable"})
        temp["Variable"] = label
        temp["Period"] = ["Pre", "Post"]
        frames.append(temp)
    result = pd.concat(frames, ignore_index=True)
    result = result[["Variable", "Period", "mean", "median", "std", "min", "max"]]
    return result.round(2)

# Split by treatment
control_summary = make_summary(data[data["select"] == "control"])
treated_summary = make_summary(data[data["select"] == "selected"])

# Print results
print("\n=== Control Group (treatment = control) ===")
print(control_summary.to_string(index=False))

print("\n=== Treated Group (treatment = selected) ===")
print(treated_summary.to_string(index=False))

# Export control summary
control_summary.to_csv("E:\\UofT\\03_ML\\Project\\project_file\\outputs\\control_summary.csv", index=False)

# Export treated summary
treated_summary.to_csv("E:\\UofT\\03_ML\\Project\\project_file\\outputs\\treated_summary.csv", index=False)

print("CSV files saved: control_summary.csv and treated_summary.csv") 