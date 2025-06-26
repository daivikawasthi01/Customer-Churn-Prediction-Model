# Credit Card Churn Prediction Project

This project is a practical implementation of a machine learning model designed to predict customer churn for credit card users.

## How I Built This Project

I started by training the model on Kaggle, using a dataset containing over 10,000 entries of customer data related to credit card churn. The dataset provided rich information about customer demographics, account details, and behavior patterns.

You can find the original Kaggle notebook and dataset here:  
[Customer Churn Prediction Model on Kaggle](https://www.kaggle.com/code/daivikawasthi/customer-churn-prediction-model)

After training the model on Kaggle, I exported the trained model as `c_model.h5` and also saved the scaler used for data preprocessing. Both files were imported into this project folder.

## Running the Project

To run this model locally or deploy it, you need to install the dependencies listed in the `requirements.txt` file. The project is built using Streamlit, which provides an easy-to-use web interface for interacting with the model.

The Streamlit environment allows for quick deployment and testing of the model's predictions based on user input.

Feel free to explore the project, test different inputs, and see how the model predicts the likelihood of customer churn.