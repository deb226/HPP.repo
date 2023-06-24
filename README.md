# HPP-ML
This project is on House price Prediction Using Machine Learning
# California Housing Prices Prediction
This project aims to predict the median house prices in California based on various features using machine learning algorithms. The dataset used for this project is the "California Housing Prices" dataset obtained from Kaggle. The goal is to build a regression model that can accurately predict the median house prices given a set of input features.

## Project Overview

The project involves the following steps:

1. Data Exploration and Preprocessing: The dataset is loaded and examined to understand the available features and their distributions. Data preprocessing techniques such as handling missing values, scaling, and feature engineering are applied to prepare the data for model training.

2. Model Training: Two regression models, Linear Regression and Random Forest Regression, are implemented and trained on the preprocessed data. Hyperparameter tuning is performed to optimize the models' performance.

3. Model Evaluation: The trained models are evaluated using the test dataset to assess their predictive accuracy. Mean Squared Error (MSE) is used as the evaluation metric to measure the models' performance.

4. Model Comparison: The performance of the Linear Regression and Random Forest Regression models is compared based on their MSE scores. The model with the lower MSE is considered better at predicting the median house prices.

## Dependencies

The following libraries and packages are required to run the project:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

To run the project, follow these steps:

1. Install the required dependencies mentioned above.

2. Clone the repository:git clone <repository-url>

3. Navigate to the project directory:cd california-housing-prices-prediction.
    
4. Run the Jupyter notebook or Python script to execute the project.

## Results

The trained models achieve a certain level of predictive accuracy, which is reflected in the calculated MSE scores. The lower the MSE, the better the model's performance in predicting the median house prices.

## Future Enhancements

Some potential enhancements for this project include:

- Exploring additional regression models such as Support Vector Regression or Gradient Boosting Regression to further improve predictive accuracy.
- Performing more feature engineering to create new informative features that might enhance the models' performance.
- Conducting in-depth feature selection techniques to identify the most significant features for predicting median house prices.
