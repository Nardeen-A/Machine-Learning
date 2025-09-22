library(haven)
library(lavaan)
library(plm)

# Read Stata file
data <- read_dta("C:\\Users\\12898\\Desktop\\ML\\data\\AEJApp-20090168_data.dta")

data_clean <- data[, -c(6, 12, 13, 14, 15, 22, 28, 29)]
data <- na.omit(data_clean)

data$coursefixe <- interaction(data$city, data$codigo_ecap, data$codigo_curs, drop = TRUE)
data$salary_04 <- data$salary_04 / 10000
data$salary_06 <- data$salary_06 / 10000
data$profit_04 <- data$profit_04 / 10000

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


m4b <- '
  empl_06 ~ select + dwomen + dwomen*select +
  age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + 
  contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb

  pempl_06 ~ select + dwomen + dwomen*select +
  age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + 
  contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb

  salary_06 ~ select + dwomen + dwomen*select +
  age_lb + educ_lb + empl_04 + salary_04 + hours_04 + days_04 + 
  contract_04 + dformal_04 + profit_04 + pempl_04 + dmarried_lb
  
'

# Estimate with SUR
SEM <- sem(m4b, data = data)
summary(SEM)

plot <- stargazer(SEM, type = "latex")




write.csv(data_clean, "C:\\Users\\12898\\Desktop\\ML\\data\\data.csv", row.names = FALSE)
