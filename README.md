<h1>Diabetes Prediction Using Random Forest</h1>

This project focuses on exploring the application of hyperparameter tuning and preprocessing techniques in conjunction with random forest classification models for the diagnosis of Type 2 Diabetes. The data for this project originates from a 1988 joint study conducted by the National Institute of Diabetes and Digestive and Kidney Diseases. Several constraints were placed on the selection criteria of this dataset, particularly, all patients are at least 21 years old females of Pima Indian heritage.

<br>

<h2>Exploratory Data Analysis</h2>

<h5>Data Overview</h5>
The dataset used in this project includes 8 numeric predictor variables and a binary outcome indicating the presence of diabetes. Below is a preview of the data alongside the summary statistics before data cleaning.

<br>

<table>
  <tr>
    <th>Pregnancies</th>
    <th>Glucose</th>
    <th>BloodPressure</th>
    <th>SkinThickness</th>
    <th>Insulin</th>
    <th>BMI</th>
    <th>DiabetesPedigreeFunction</th>
    <th>Age</th>
    <th>Diabetes</th>
  </tr>
  <tr>
    <td>6</td>
    <td>148</td>
    <td>72</td>
    <td>35</td>
    <td>0</td>
    <td>33.6</td>
    <td>0.627</td>
    <td>50</td>
    <td>1</td>
  </tr>
  <tr>
    <td>1</td>
    <td>85</td>
    <td>66</td>
    <td>29</td>
    <td>0</td>
    <td>26.6</td>
    <td>0.351</td>
    <td>31</td>
    <td>0</td>
  </tr>
  <tr>
    <td>8</td>
    <td>183</td>
    <td>64</td>
    <td>0</td>
    <td>0</td>
    <td>23.3</td>
    <td>0.672</td>
    <td>32</td>
    <td>1</td>
  </tr>
  <tr>
    <td>1</td>
    <td>89</td>
    <td>66</td>
    <td>23</td>
    <td>94</td>
    <td>28.1</td>
    <td>0.167</td>
    <td>21</td>
    <td>0</td>
  </tr>
  <tr>
    <td>0</td>
    <td>137</td>
    <td>40</td>
    <td>35</td>
    <td>168</td>
    <td>43.1</td>
    <td>2.288</td>
    <td>33</td>
    <td>1</td>
  </tr>
</table>

<br>

<table>
  <thead>
    <tr>
      <th><strong>Variable</strong></th>
      <th><strong>Min.</strong></th>
      <th><strong>1st Qu.</strong></th>
      <th><strong>Median</strong></th>
      <th><strong>Mean</strong></th>
      <th><strong>3rd Qu.</strong></th>
      <th><strong>Max.</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Pregnancies</strong></td>
      <td>0.000</td>
      <td>1.000</td>
      <td>3.000</td>
      <td>3.845</td>
      <td>6.000</td>
      <td>17.000</td>
    </tr>
    <tr>
      <td><strong>Glucose</strong></td>
      <td>0.0</td>
      <td>99.0</td>
      <td>117.0</td>
      <td>120.9</td>
      <td>140.2</td>
      <td>199.0</td>
    </tr>
    <tr>
      <td><strong>BloodPressure</strong></td>
      <td>0.00</td>
      <td>62.00</td>
      <td>72.00</td>
      <td>69.11</td>
      <td>80.00</td>
      <td>122.00</td>
    </tr>
    <tr>
      <td><strong>SkinThickness</strong></td>
      <td>0.00</td>
      <td>0.00</td>
      <td>23.00</td>
      <td>20.54</td>
      <td>32.00</td>
      <td>99.00</td>
    </tr>
    <tr>
      <td><strong>Insulin</strong></td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.5</td>
      <td>79.8</td>
      <td>127.2</td>
      <td>846.0</td>
    </tr>
    <tr>
      <td><strong>BMI</strong></td>
      <td>0.00</td>
      <td>27.30</td>
      <td>32.00</td>
      <td>31.99</td>
      <td>36.60</td>
      <td>67.10</td>
    </tr>
    <tr>
      <td><strong>DiabetesPedigreeFunction</strong></td>
      <td>0.0780</td>
      <td>0.2437</td>
      <td>0.3725</td>
      <td>0.4719</td>
      <td>0.6262</td>
      <td>2.4200</td>
    </tr>
    <tr>
      <td><strong>Age</strong></td>
      <td>21.00</td>
      <td>24.00</td>
      <td>29.00</td>
      <td>33.24</td>
      <td>41.00</td>
      <td>81.00</td>
    </tr>
    <tr>
      <td><strong>Outcome</strong></td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.349</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>

<br>

```r
summary(data)
any(is.na(data))
str(data)
create_report(data)
```

<br>

<h5>Summary of Data Quality</h5>
Several continuous variables in the dataset were found to have abnormal values of zero. These variables were imputed using the vector median. Given the already limited size of the datase, this approach was deemed more appropriate than omitting the affected rows. The absence of any null values ensured that all observations could be retained. The outcome variable, `Diabetes`, was renamed and converted to a factor for analysis.

<br>

```r
#identify 0 values that represent missing values
na_bp <- sum(data$BloodPressure == 0)
na_bmi <- sum(data$BMI == 0)
na_st <- sum(data$SkinThickness == 0)
na_g <- sum(data$Glucose == 0)

data$BloodPressure[data$BloodPressure == 0] <- NA
data$BMI[data$BMI == 0] <- NA
data$SkinThickness[data$SkinThickness == 0] <- NA
data$Glucose[data$Glucose == 0] <- NA

set.seed(5)
data_impute <- rfImpute(Diabetes ~ ., data = data, iter = 5)
```

<br>

<h5>Visualizations</h5>
Histograms, bar plots, scatter plots, correlation plots, and violin plots were used to better understand the distributions and relationships between variables.

<br>

```r
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
```

<br>

<h5>Data Partitioning</h5>
The dataset was split into training and test sets, with 70% of the data used for training and 30% for testing.

<br>

```r
set.seed(5)
train_index <- sample(1:nrow(data_impute), size = 0.7 * nrow(data_impute))
train <- data_impute[train_index, ]
test <- data_impute[-train_index, ]
```

<br>

<h2>Modeling with Random Forest</h2>
<h5>Model Development</h5>
Several Random Forest models were developed with varying hyperparameters. The optimal model was selected based on Out-of-Bag (OOB) error rates.

<br>

```r

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

#model 5 ntree = 1500, mtry = 1
set.seed(5)
rf_model_5 <- randomForest(Diabetes ~ ., data = train, ntree = 1500, mtry = 1, proximity = TRUE)
print(rf_model_5)
#OOB estimate of  error rate: 23.46%
```

<br>

<h5>Oversampling</h5>
To reduce the Type II error rate, an oversampling technique was applied, which resulted in a reduced error rate of 13.1%. However, this is likely due to overfitting. Additional sampling techniques could likely be deployed for better results.

<br>

```r
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
```

<br>

<h5>Model Evaluation</h5>
The models were evaluated using accuracy on the testing set. The original random forest model when applied to the testing dataset returned an accuracy of 74.03%. As expected, the oversampled model proves the existence of overfitting to the training set. The final model returned an accuracy of 75.32%.

<br>

```r

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
```

<br>

<h5>Feature Importance</h5>
Feature importance was analyzed to identify the most significant predictors according to the results of the final model.

<br>

```r
#feature importance
feature_importance <- importance(rf_model_5)
feature_importance
```

<br>

<h2>Conclusion</h2>
This project demonstrates the application of Random Forest models in predicting diabetes, highlighting the importance of preprocessing and hyperparameter tuning. While Model 5 achieved an accuracy of 75.32%, further improvements can be explored through additional feature engineering, sampling techniques, and model optimization. 

<br>

<h2>References</h2>
- Dataset: Kaggle: https://www.kaggle.com/ <br>
- Original Study: ResearchGate: https://www.researchgate.net/publication/248284447_Using_the_ADAP_Learning_Algorithm_to_Forcast_the_Onset_of_Diabetes_Mellitus <br>
- Similar Research: PubMed: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8943493/
