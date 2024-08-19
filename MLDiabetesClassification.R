# load data and libraries -------------------------------------------------

library(DataExplorer)
library(ggplot2)
library(tidyverse)
library(caret)
library(randomForest)
library(vioplot)
library(corrplot)
library(DMwR2)
library(ROSE)

#data
data <- read.csv("C:/Users/Cole Jones/OneDrive/Code/R/diabetes.csv")

# data cleaning -----------------------------------------------------------

#descriptive summary
summary(data)

#NAs
any(is.na(data))

#variable types
str(data)

#rename data$Outcome
colnames(data)[9] ="Diabetes"

#convert Diabetes to factor
data$Diabetes <- as.factor(data$Diabetes)

#duplicate rows
n_unique_rows <- nrow(unique(data))

# univariate EDA and impute ---------------------------------------------------------------------

'#automated EDA
create_report(data)'

#testing normality
#model<-aov(MPG~Country, data=cars2)
#ggqqplot(residuals(model))

#univariate histograms numeric columns
plot_histogram <- function(column, title, xlab) {
  ggplot(data, aes_string(x = column)) +
    geom_histogram(binwidth = 1) +
    labs(title = title, x = xlab, y = "Frequency") +
    theme_minimal()
}

numeric_columns <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")
lapply(numeric_columns, function(col) plot_histogram(col, paste("Histogram of", col), col))

#diabetes bar plot
ggplot(data, aes(x = Diabetes)) +
  geom_bar() +
  labs(title = "Barplot of Diabetes", x = "Diabetes", y = "Frequency") +
  theme_minimal()

#identify 0 values that represent missing values
na_bp <- sum(data$BloodPressure == 0)
na_bmi <- sum(data$BMI == 0)
na_st <- sum(data$SkinThickness == 0)
na_g <- sum(data$Glucose == 0)

#convert to NA
data$BloodPressure[data$BloodPressure == 0] <- NA
data$BMI[data$BMI == 0] <- NA
data$SkinThickness[data$SkinThickness == 0] <- NA
data$Glucose[data$Glucose == 0] <- NA

#impute values for inconsistent columns using rf
set.seed(5)
data_impute <- rfImpute(Diabetes ~ ., data = data, iter = 5)

# bivariate EDA -----------------------------------------------------------

#bivaraite box plots by diabetes
lapply(numeric_columns, function(col) {
  ggplot(data, aes_string(x = "Diabetes", y = col)) +
    geom_boxplot() +
    labs(title = paste("Boxplot of Diabetes and", col), x = "Diabetes", y = col) +
    theme_minimal()
})

#scatter plots
scatter_plot <- function(x, y, color = factor(data$Diabetes)) {
  ggplot(data, aes_string(x = x, y = y, color = factor(data$Diabetes))) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    scale_color_manual(values = c("black", "blue"), labels = c("No Diabetes", "Diabetes")) +
    labs(x = x, y = y) +
    theme_minimal()
}

scatter_plot("Age", "BMI")
scatter_plot("Age", "Insulin")
scatter_plot("Age", "Glucose")
scatter_plot("Age", "BloodPressure")
scatter_plot("Age", "SkinThickness")
scatter_plot("Age", "DiabetesPedigreeFunction")

'#violin plots
ggplot(data, aes(x = Diabetes, y = Age)) +
  geom_violin() +
  labs(x = "Diabetes", y = "Age") +
  scale_fill_manual(values = c("Female" = "black", "Male" = "blue")) +
  theme_minimal()'

'#data preparation for correlation
data$Diabetes <- as.numeric(data$Diabetes)

#correlation table
cor_mat <- cor(data)
cor_mat

#correlation plots
plot(data)

#color correlation matrix
corrplot(cor_mat, method = "color", tl.cex = .6)

#convert back to factor
data$Diabetes <- as.factor(data$Diabetes)'

# preprocess data -----------------------------------------------------

#split data
set.seed(5)
train_index <- sample(1:nrow(data_impute), size = 0.7 * nrow(data_impute))
train <- data_impute[train_index, ]
test <- data_impute[-train_index, ]


# modeling ----------------------------------------------------------------

#model 1
set.seed(5)
rf_model_1 <- randomForest(Diabetes ~ ., data = train, proximity = TRUE)
print(rf_model_1)

plot(rf_model_1)

#model 2 ntree = 1000
set.seed(5)
rf_model_2 <- randomForest(Diabetes ~ ., data = train, ntree = 1000, proximity = TRUE)
print(rf_model_2)

#model 3 ntree = 1500
set.seed(5)
rf_model_3 <- randomForest(Diabetes ~ ., data = train, ntree = 1500, proximity = TRUE)
print(rf_model_3)

#model 4 ntree = 2000
#did not decrease OOB error
set.seed(5)
rf_model_4 <- randomForest(Diabetes ~ ., data = train, ntree = 2000, proximity = TRUE)
print(rf_model_4)

#given model 3 produced the best OOB rate with ntrees = 1500
#find optimal number of variables at each split
var_op <- vector(length=10)

for (i in 1:10) {
  temp_model <- randomForest(Diabetes ~ ., data = train, mtry = i, ntree = 1500)
  var_op[i] <- temp_model$err.rate[nrow(temp_model$err.rate), 1]
}

var_op
#min(i) = 1

#plot var_op
plot(var_op, type = "b", pch = 19, xlab = "mtry", ylab = "OOB Error Rate", 
     main = "OOB Error Rates for Number of Variables at Each Split (mtry)")

'#could also use tuneRF or a nested for loop

var_op <- matrix(NA, nrow = 10, ncol = 4)
colnames(var_op) <- c("ntree_500", "ntree_1000", "ntree_1500", "ntree_2000")

for (i in 1:10) {
  for (j in seq_along(c(500, 1000, 1500, 2000))) {
    set.seed(4)
    temp_model <- randomForest(Diabetes ~ ., data = train, mtry = i, ntree = c(500, 1000, 1500, 2000)[j])
    var_op[i, j] <- temp_model$err.rate[nrow(temp_model$err.rate), 1]
  }
}

print(min(var_op))'

#model 5 ntree = 1500, mtry = 1
set.seed(5)
rf_model_5 <- randomForest(Diabetes ~ ., data = train, ntree = 1500, mtry = 1, proximity = TRUE)
print(rf_model_5)
#OOB estimate of  error rate: 23.46%

#try oversampling to reduce type II error rate

#oversampling for better sensitivity
table(train$Diabetes)
set.seed(5)
over_sample_train <- ovun.sample(Diabetes ~ ., data = train, method = "over", N = 687)$data
table(over_sample_train$Diabetes)

#model 6
set.seed(5)
rf_model_6 <- randomForest(Diabetes ~ ., data = over_sample_train, mtry = 1, ntree = 1500, importance = TRUE)
rf_model_6
#likely overfitted


# validation --------------------------------------------------------------

#evaluate model 6 (oversampled, mtry = 1, ntree = 1500)
result <- predict(rf_model_6, test)

confusionMatrix(result, test$Diabetes)
#accuracy : 0.7316

#evaluate model 5 (mtry = 1, ntree = 1500)
result_2 <- predict(rf_model_5, test)

confusionMatrix(result_2, test$Diabetes)
#accuracy : 0.7532

#evaluate original model
result_3 <- predict(rf_model_1, test)

confusionMatrix(result_3, test$Diabetes)
#accuracy : 0.7403

#feature importance
feature_importance <- importance(rf_model_5)
feature_importance
