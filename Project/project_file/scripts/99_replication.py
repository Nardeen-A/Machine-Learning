# =====================================================

# 1. Packages
# =====================================================
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =====================================================
# 2. Load and clean data
# =====================================================
# Convert Stata to CSV (one-time export)
data = pd.read_stata("E:\\UofT\\03_ML\\Project\\project_file\\data\\AEJApp-20090168_data.dta")
data = pd.DataFrame(data)

# Drop NA values and unnecessary columns
data = data.dropna(subset=["salary_06"])
data = data.drop(columns=[
    "lsalary_06", "lprofit_06", "ldays_06", "lhours_06",
    "ldays_04", "lhours_04", "tenure_04"
])
data = data.dropna()

# Save and reload as CSV (optional)
data.to_csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\full_data.csv", index=False)
data = pd.read_csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\full_data.csv")

# =====================================================
# 3. Variable construction
# =====================================================
# Create course fixed effect (Stata: egen group(city codigo_ecap codigo_curs))
data["coursefixe"] = data.groupby(["city", "codigo_ecap", "codigo_curs"]).ngroup()

# Treatment variable
data["select"] = data["select"].map({"control": 0, "selected": 1}).astype(int)

# Baseline controls (Panel B)
controls = [
    "age_lb", "educ_lb", "dmarried_lb",
    "empl_04", "pempl_04", "salary_04", "profit_04",
    "dformal_04", "contract_04", "days_04", "hours_04", "city"
]

# Fixed effects: course
fe = "coursefixe"

# Outcomes
outcomes = ["empl_06", "pempl_06", "salary_06"]

# =====================================================
# 4. Split dataset: Men vs Women
# =====================================================
df_men = data[data["dwomen"] == 0].copy()
df_women = data[data["dwomen"] == 1].copy()

# =====================================================
# 5. Run regressions and collect results
# =====================================================
def run_and_collect(subdf, label):
    results = []
    for y in outcomes:
        # -------- Panel A --------
        formula_a = f"{y} ~ select + C({fe})"
        model_a = smf.ols(formula=formula_a, data=subdf).fit(
            cov_type="cluster", cov_kwds={"groups": subdf[fe]}
        )
        coef, se, pval = model_a.params["select"], model_a.bse["select"], model_a.pvalues["select"]
        results.append({
            "outcome": y, "panel": "Panel A", "group": label,
            "coef": coef, "se": se, "pval": pval, "nobs": int(model_a.nobs)
        })

        # -------- Panel B --------
        formula_b = f"{y} ~ select + " + " + ".join(controls) + f" + C({fe})"
        model_b = smf.ols(formula=formula_b, data=subdf).fit(
            cov_type="cluster", cov_kwds={"groups": subdf[fe]}
        )
        coef, se, pval = model_b.params["select"], model_b.bse["select"], model_b.pvalues["select"]
        results.append({
            "outcome": y, "panel": "Panel B", "group": label,
            "coef": coef, "se": se, "pval": pval, "nobs": int(model_b.nobs)
        })
    return results

results = []
results.extend(run_and_collect(df_men, "Men"))
results.extend(run_and_collect(df_women, "Women"))

table = pd.DataFrame(results)

# =====================================================
# 6. Formatter with significance stars
# =====================================================
def add_stars(pval):
    if pval < 0.01: return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.1: return "*"
    return ""

def format_results_table_like_paper(df):
    df["coef"] = df["coef"].round(3)
    df["se"] = df["se"].round(3)
    df["stars"] = df["pval"].apply(add_stars)

    # Format: coef + stars + (se) + N
    df["formatted"] = df.apply(
        lambda r: f"{r['coef']}{r['stars']} ({r['se']}, N={r['nobs']})", axis=1
    )

    # Pivot wide: outcome as index, panels × groups as columns
    pivot = df.pivot_table(
        index="outcome", columns=["panel","group"],
        values="formatted", aggfunc="first"
    )

    # Flatten columns: "Panel A_Men", etc.
    pivot.columns = [f"{panel}_{group}" for panel, group in pivot.columns]
    pivot = pivot.reset_index()
    return pivot

# =====================================================
# 7. Final table
# =====================================================
paper_table = format_results_table_like_paper(table)

print("\n=== Table 3 Style Results ===")
print(paper_table.to_string(index=False))
print(table)
print(results)
table.to_csv("E:\\UofT\\03_ML\\Project\\project_file\\outputs\\table.csv", index=False)