data <- read.csv("E:\\UofT\\03_ML\\Project\\project_file\\outputs\\table.csv")
data <- data[,-c(9)]

data$outcome[data$outcome == "empl_06"] <- "Employment"
data$outcome[data$outcome == "pempl_06"] <- "Paid Employment"
data$outcome[data$outcome == "salary_06"] <- "Salary"

data$panel[data$panel == "Panel A"] <- "Fixed Effects"
data$panel[data$panel == "Panel B"] <- "FE + Control"

colnames(data)[1] <- "Dep_Variable"
colnames(data)[2] <- "Model"
colnames(data)[3] <- "Sex"
colnames(data)[4] <- "Treat Effect"
colnames(data)[5] <- "Error"
colnames(data)[6] <- "P-Value"
colnames(data)[7] <- "Obs"
colnames(data)[8] <- "Sig."


# Combine effect, p-value, and sig into one column
data <- data %>%
  mutate(Result = paste0(round(`Treat Effect`, 3),
                         " (", round(`P-Value`, 4), ") ",
                         Sig.))

table_wide <- data %>%
  dplyr::select(Model, Sex, Obs, Dep_Variable, Result) %>%
  tidyr::pivot_wider(names_from = Dep_Variable, values_from = Result) %>%
  group_by(Sex) %>%
  mutate(
    Sex   = ifelse(duplicated(Sex), "", Sex),
    Obs   = ifelse(duplicated(Obs), "", Obs)) %>%
  dplyr::select(Sex, Obs, everything())



stargazer(table_wide, type = "latex", summary = FALSE, rownames = FALSE, 
          title = "Impact of Training on Employment Outcomes",  style = "qje")


data_2 <- read.csv("E:\\UofT\\03_ML\\Project\\project_file\\data\\full_data.csv")
data_2$coursefixe <- interaction(data_2$city, data_2$codigo_ecap, data_2$codigo_curs, drop = TRUE)

logit_women <- clogit(empl_06 ~ factor(select) + age_lb + educ_lb + dmarried_lb + 
       empl_04 + pempl_04 + salary_04 + profit_04 + dformal_04 + contract_04 + 
       days_04 + hours_04 + strata(coursefixe), data = subset(data_2, dwomen == 1))


logit_men <- clogit(empl_06 ~ select + age_lb + educ_lb + dmarried_lb + 
       empl_04 + pempl_04 + salary_04 + profit_04 + dformal_04 + contract_04 + 
       days_04 + hours_04 + strata(coursefixe), data = subset(data_2, dwomen == 0))


logit_women <- glm(
  empl_06 ~ factor(select) + age_lb + educ_lb + dmarried_lb + 
    empl_04 + pempl_04 + salary_04 + profit_04 + dformal_04 + contract_04 + 
    days_04 + hours_04 + factor(coursefixe), data = subset(data_2, dwomen == 1),
  family = binomial(link = "logit"))

summary(logit_women)