# Sample Project: Predicting Baseball Wins by Number of Home Runs via Binary Logistic Regression 
# load libraries
library("caret")
library("magrittr")
library("dplyr")
library("tidyr")
library("lmtest")
library("popbio")
library("e1071")

# load data (using sample baseball dataset)
baseball <- read.csv("C:/Users/Rania/OneDrive/Documents/Project Portfolio/R Projects/baseball/baseball.csv")
View(baseball)

# Question: Are home runs a good predictor of game winners? 
## Predictor (IV) = number of home runs (continuous)
## Response (DV) = Whether team wins or loses (categorical)

# Data Wrangling - recode DV to numeric values (0=loss, 1=win)

baseball$WinsR <- NA
baseball$WinsR[baseball$W.L=='W'] <- 1
baseball$WinsR[baseball$W.L=='L'] <- 0
View(baseball)

# Test Assumptions
# 1. Sample Size
## run base logistic model
mylogit <- glm(WinsR ~ HR.Count, data=baseball, family="binomial")
## predict wins and losses
probabilities <- predict(mylogit, type = "response")
## convert to positive and negative predictions (anything above .5 is positive, anything below .5 is negative)

probabilities <- predict(mylogit, type = "response")
baseball$Predicted <- ifelse(probabilities > .5, "pos", "neg")

## recode predicted variable
baseball$PredictedR <- NA
baseball$PredictedR[baseball$Predicted=='pos'] <- 1
baseball$PredictedR[baseball$Predicted=='neg'] <- 0

## convert variables to factors in order to create confusion matrix
baseball$PredictedR <- as.factor(baseball$PredictedR)
baseball$WinsR <- as.factor(baseball$WinsR)

## create confusion matrix to test sample size and provide model accuracy
baseballMatrix <- caret::confusionMatrix(baseball$PredictedR, baseball$WinsR)
baseballMatrix

## each cell is greater than 5, so this dataset passes the sample size assumption for binary logit regression. 
# Test Assumption #2 - Logit Linearity 

## Data Wrangling to test for linearity
baseball1 <- baseball %>%
  dplyr::select_if(is.numeric)

## rename column names to be fed into predictors
predictors <- colnames(baseball1)

## create logit 

baseball1 <- baseball1 %>%
  mutate(logit=log(probabilities/(1-probabilities))) %>%
  gather(key= "predictors", value="predictor.value", -logit)

## graph to assess for linearity
ggplot(baseball1, aes(logit, predictor.value))+
  geom_point(size=.5, alpha=.5)+
  geom_smooth(method= "loess")+
  theme_bw()+
  facet_wrap(~predictors, scales="free_y")

## Homerun count (HR.Count) shows strong linear relationship, passing the assumption of logit linearity

# Test Assumption # 3 - Multicollinearity (independent variables should not be too closely related to each other; however, we are only testing for one IV at this time so we can skip)
# Test Assumption #4 - Independent Errors
## graph residuals
plot(mylogit$residuals)
## distribution is pretty even across the x-axis. test further using durbin-watson test

dwtest(mylogit, alternative="two.sided")
## p-value is significant at >0.01; however, durbin-watson statistic is 2.08 i.e. not under 1 or greater than 3. based on graph and DW value, I would say it is safe to say that we have passed the assumption of independent errors

# Test Assumption #5 - Outliers
infl <- influence.measures(mylogit)
summary(infl)
## here, i used chatgpt to help me write a function that would print only rows that violate assumption of outliers. 

filter_influence <- function(infl, dfb_threshold = 1, dffit_threshold = 1, hat_threshold = 0.3) {
  infl_mat <- as.data.frame(infl$infmat)  # Convert influence matrix to a dataframe
  
  # Identify columns that start with "dfb."
  dfb_cols <- grep("^dfb\\.", names(infl_mat), value = TRUE)
  
  # Create a logical filter
  filter_rows <- apply(infl_mat[, dfb_cols], 1, function(x) any(abs(x) > dfb_threshold)) |
    abs(infl_mat$dffit) > dffit_threshold |
    infl_mat$hat > hat_threshold
  
  # Print only the filtered rows
  return(infl_mat[filter_rows, ])
}

# Usage example
filtered_data <- filter_influence(infl)
print(filtered_data)
## assumption is met. 
# All Assumptions have been met; run the logistic regression and interpret results

## we created a logit when testing sample size; interpret results
summary(mylogit)

## p-value of coefficients table under HR.Count is significant at p < .001, indicating that the number of home runs is a significant predictor of the number of wins and losses a team has.
## For non-technical audience:
## The model suggests a strong positive relationship between home runs and winning. When a team has zero home runs, their odds of winning are about 45% of the odds of losing. 
## Each additional home run nearly doubles the odds of winning.

logi.hist.plot(baseball$HR.Count,baseball$WinsR, boxp=FALSE, type="hist", col="gray")
