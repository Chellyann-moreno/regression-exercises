## IMPORTS:
import pandas as pd 
import env as env
import os
# data visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# stats data 
import scipy.stats as stats
import statsmodels.formula.api as smf
# scaling and modeling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


## FUNCTIONS:

def plot_residuals(y, yhat):
    """
    Creates a residual plot given the true values `y` and the predicted values `yhat`.
    """
    # calculate residuals
    residuals = y - yhat
    
    # create a scatter plot of residuals against yhat
    plt.scatter(yhat, residuals)
    
    # add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle='-')
    
    # set axis labels and title
    plt.xlabel("Predicted values (yhat)")
    plt.ylabel("Residuals")
    plt.title("Residual plot")
    
    # show the plot
    plt.show()



    def regression_errors(y, yhat):
        """Calculates regression error metrics given the true values `y` and the predicted values `yhat`
            Returns a Pandas DataFrame with SSE, ESS, TSS, MSE, and RMSE."""
    # calculate SSE
    SSE = np.sum((y - yhat)**2)
    
    # calculate ESS
    ESS = np.sum((yhat - np.mean(y))**2)
    
    # calculate TSS
    TSS = np.sum((y - np.mean(y))**2)
    
    # calculate MSE
    MSE = SSE / len(y)
    
    # calculate RMSE
    RMSE = np.sqrt(MSE)
    
    # create a DataFrame to store the regression error metrics
    error_df = pd.DataFrame({'Metric': ['SSE', 'ESS', 'TSS', 'MSE', 'RMSE'],
                             'Value': [SSE, ESS, TSS, MSE, RMSE]})
    
    # set the 'Metric' column as the index of the DataFrame
    error_df.set_index('Metric', inplace=True)
    
    # return the DataFrame
    return error_df

def baseline_mean_errors(y):
    """
    Calculates regression error metrics for the baseline model given the true values `y`.
    Returns SSE, MSE, and RMSE.
    """
    # calculate the mean of y
    y_mean = np.mean(y)
    
    # calculate the SSE
    SSE = np.sum((y - y_mean)**2)
    
    # calculate the MSE
    MSE = SSE / len(y)
    
    # calculate the RMSE
    RMSE = np.sqrt(MSE)
    
    # return the calculated values as a tuple
    return SSE, MSE, RMSE

# function that combines both yhat and baseline calculations and are add it to a data frame
def regression_errors(y, yhat):
    """
    Calculates regression error metrics given the true values `y` and the predicted values `yhat`.
    Returns a Pandas DataFrame with SSE, ESS, TSS, MSE, RMSE, and baseline error metrics.
    """
    # calculate SSE
    SSE = np.sum((y - yhat)**2)
    
    # calculate ESS
    ESS = np.sum((yhat - np.mean(y))**2)
    
    # calculate TSS
    TSS = np.sum((y - np.mean(y))**2)
    
    # calculate MSE
    MSE = SSE / len(y)
    
    # calculate RMSE
    RMSE = np.sqrt(MSE)
    
    # calculate baseline error metrics
    baseline_SSE, baseline_MSE, baseline_RMSE = baseline_mean_errors(y)
    
    # create a DataFrame to store the regression error metrics
    error_df = pd.DataFrame({'Metric': ['SSE', 'ESS', 'TSS', 'MSE', 'RMSE', 'Baseline SSE', 'Baseline MSE', 'Baseline RMSE'],
                             'Value': [SSE, ESS, TSS, MSE, RMSE, baseline_SSE, baseline_MSE, baseline_RMSE]})
    
    # set the 'Metric' column as the index of the DataFrame
    error_df.set_index('Metric', inplace=True)
    
    # return the DataFrame
    return error_df


def baseline_mean_errors(y):
    """
    Calculates regression error metrics for the baseline model given the true values `y`.
    Returns SSE, MSE, and RMSE.
    """
    # calculate the mean of y
    y_mean = np.mean(y)
    
    # calculate the SSE
    SSE = np.sum((y - y_mean)**2)
    
    # calculate the MSE
    MSE = SSE / len(y)
    
    # calculate the RMSE
    RMSE = np.sqrt(MSE)
    
    # return the calculated values as a tuple
    return SSE, MSE, RMSE

### END OF FUNCTION ########


def better_than_baseline(y, yhat):
    """
    Determines whether the model with predicted values `yhat` performs better than the baseline model
    with mean values of `y`. Returns True if the model is better than the baseline, otherwise False.
    """
    # calculate the SSE for the baseline model
    baseline_SSE = baseline_mean_errors(y)[0]
    
    # calculate the SSE for the model with predicted values yhat
    model_SSE = np.sum((y - yhat)**2)
    
    # compare the SSEs and return True if the model's SSE is less than the baseline's SSE
    return model_SSE < baseline_SSE




