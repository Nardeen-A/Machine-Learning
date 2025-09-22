import pandas as pd
import statsmodels.formula.api as smf

# =====================================================
# Load & Prepare Data
# =====================================================
data = pd.read_csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\full_data.csv")

# Create coursefixe (Stata: egen group(city codigo_ecap codigo_curs))
data["coursefixe"] = data.groupby(["city", "codigo_ecap", "codigo_curs"]).ngroup()
data["select"] = data["select"].map({"control": 0, "selected": 1}).astype(int)

# Baseline controls
controls = [
    "age_lb", "educ_lb", "dmarried_lb", 
    "empl_04", "pempl_04", "salary_04", "profit_04",
    "dformal_04", "contract_04", "days_04", "hours_04", "city"
]

# Create outcome: treated but not employed
data["no_benefit"] = ((data["select"] == 1) & (data["empl_06"] == 0)).astype(int)

# Restrict to treated group only
treated = data[data["select"] == 1].copy()

# =====================================================
# Helper: Run logit + marginal effects for a subgroup
# =====================================================
def run_no_benefit_regression(df, group_label):
    results_nb = []
    try:
        # Drop groups with no variation in the outcome
        df = df.groupby("coursefixe").filter(lambda g: g["no_benefit"].nunique() > 1)
        df = df.dropna(subset=controls + ["no_benefit", "coursefixe"])

        # --- Diagnostics ---
        print(f"{group_label}: {df.shape[0]} observations after filtering")

        # Keep only controls that vary in this subgroup
        varying_controls = [c for c in controls if df[c].nunique() > 1]
        print(f"{group_label}: {len(varying_controls)} varying controls kept -> {varying_controls}")

        # Build formula
        formula = "no_benefit ~ " + " + ".join(varying_controls) + " + C(coursefixe)"

        # Pure logit with robust SEs (HC1)
        logit_model = smf.logit(formula, data=df).fit(maxiter=200, disp=0, cov_type="HC1")

        # Collect coefficients
        for var in logit_model.params.index:
            results_nb.append({
                "group": group_label,
                "variable": var,
                "coef": round(logit_model.params[var], 3),
                "se": round(logit_model.bse[var], 3),
                "pval": logit_model.pvalues[var],
                "nobs": int(logit_model.nobs),
                "type": "Coefficient"
            })

        # Collect marginal effects
        try:
            mfx = logit_model.get_margeff(at="overall").summary_frame()
            possible_pval_cols = [c for c in mfx.columns if "P>" in c or "p" in c.lower()]
            if possible_pval_cols:
                pval_col = possible_pval_cols[0]
                for var in mfx.index:
                    results_nb.append({
                        "group": group_label,
                        "variable": f"ME_{var}",
                        "coef": round(mfx.loc[var, "dy/dx"], 3),
                        "se": round(mfx.loc[var, "Std. Err."], 3),
                        "pval": mfx.loc[var, pval_col],
                        "nobs": int(logit_model.nobs),
                        "type": "MarginalEffect"
                    })
            else:
                print(f"[Warning] Could not detect p-values in marginal effects for {group_label}.")
        except Exception as e:
            print(f"[Warning] Marginal effects failed for {group_label}: {e}")

    except Exception as e:
        print(f"[Error] Logit failed for {group_label}: {e}")

    return results_nb


# =====================================================
# Run for All, Men, Women
# =====================================================
all_results = []
all_results += run_no_benefit_regression(treated, "All")
all_results += run_no_benefit_regression(treated[treated["dwomen"] == 1], "Women")
all_results += run_no_benefit_regression(treated[treated["dwomen"] == 0], "Men")

# =====================================================
# Clean and display results (drop coursefixe dummies)
# =====================================================
def stars(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.1: return "*"
    return ""

results_df = pd.DataFrame(all_results)

if not results_df.empty:
    # Drop coursefixe dummy variables
    results_df = results_df[~results_df["variable"].str.contains("coursefixe")]

    # Add formatted coef with SE and stars
    results_df["Coef (SE)"] = results_df.apply(
        lambda r: f"{r['coef']}{stars(r['pval'])} ({r['se']})", axis=1
    )

    # Keep only neat columns
    table = results_df[["group", "variable", "Coef (SE)", "nobs", "type"]]

    # Sort for readability
    table = table.sort_values(["group", "type", "variable"]).reset_index(drop=True)

    from IPython.display import display
    display(table)
else:
    print("No results available.")
