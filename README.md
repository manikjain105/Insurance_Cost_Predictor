# Insurance Cost Predictor

This project predicts insurance costs based on several factors such as age, sex, BMI, number of children, smoker status, and region. The main goal is to create a machine learning model that accurately estimates insurance charges using these features.

## Overview
The project involves preprocessing the data, implementing machine learning models, and selecting the best-performing model. After trying multiple models, **Random Forest** was selected due to its superior performance in terms of R² score and overall accuracy.

## Dataset
The dataset contains the following features:
- **age**: Age of the individual
- **sex**: Gender of the individual (male/female)
- **bmi**: Body Mass Index of the individual
- **children**: Number of children the individual has
- **smoker**: Whether the individual is a smoker (yes/no)
- **region**: The region in which the individual lives
- **charges**: Insurance charges (this is the target variable to predict)

## Preprocessing
1. **Outlier Removal**: Outliers were removed using the Z-score method.
2. **Feature Scaling**: The features were normalized using **Standard Scaler** to ensure consistent scaling across all models.

## Modeling
The following machine learning models were trained and evaluated:
- **Linear Regression**: A baseline regression model.
- **Random Forest**: An ensemble method using decision trees.
- **XGBoost**: A powerful gradient boosting algorithm.

### Model Selection
After evaluating the models, **Random Forest** was chosen as the final model due to its best performance in terms of the R² score.

## Results
- **Best Model**: Random Forest
- **R² Score**: The Random Forest model provided the best R² score, outperforming Linear Regression and XGBoost.

## Requirements
The following Python libraries are required to run the code:
```bash
numpy
pandas
scikit-learn
xgboost
