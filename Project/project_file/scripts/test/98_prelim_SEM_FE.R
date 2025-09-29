library(haven)
library(lavaan)
library(plm)

# Read Stata file
data <- read_dta("E:\\UofT\\03_ML\\Project\\project_file\\data\\AEJApp-20090168_data.dta")

data_clean <- data[, -c(6, 12, 13, 14, 15, 22, 28, 29)]
data <- na.omit(data_clean)
data$select <- as.numeric(as_factor(data$select)) 
data$select <- ifelse(data$select == 2, 1, 0)

data$coursefixe <- interaction(data$city, data$codigo_ecap, data$codigo_curs, drop = TRUE)

r_1 <- plm(
  empl_06 ~ select + dwomen + dwomen*select +
    age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04
  + dformal_04 + profit_04 + pempl_04 + dmarried_lb, 
  index = "coursefixe", data = data, model = "within"
)

r_2 <- plm(
  pempl_06 ~ select + dwomen + dwomen*select +
    age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04
  + dformal_04 + profit_04 + pempl_04 + dmarried_lb, 
  index = "coursefixe", data = data, model = "within"
)

r_3 <- plm(
  salary_06 ~ select + dwomen + dwomen*select +
    age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + contract_04
  + dformal_04 + profit_04 + pempl_04 + dmarried_lb, 
  index = "coursefixe", data = data, model = "within"
)

stargazer::stargazer(r_1, r_2, r_3, type = "latex", 
                     keep = c("select", "dwomen", "dwomen:select", "educ_lb"),
                     covariate.labels = c("Treatment", "Female", "Treatment × Female", "Pre Treatment Education"),
                     add.lines = list(c("Controls", "Yes", "Yes", "Yes")),
                     dep.var.labels = c("Employed", "Paid Employment", "Salary (1,000s)"),
                     title = "OLS Estimate of Treatment on Employment, Paid Employment, and Salary",
                     single.row = TRUE,
                     model.numbers = FALSE)

m3e  <- '
  empl_06 ~ 1 + select + dwomen + dwomen*select +
  age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + 
  contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb

  pempl_06 ~ 1 + select + dwomen + dwomen*select +
  age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + 
  contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb

  salary_06 ~ 1 + select + dwomen + dwomen*select +
  age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + 
  contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb
  
'

data_scaled <- data %>%
  mutate(across(where(is.numeric), scale))

# Estimate with SUR
SEM <- sem(m3e, data = data_scaled)

