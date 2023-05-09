# IMPORTS
import pandas as pd 
import env as env
import os
import wrangle as w

# data visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# stats data 
import scipy.stats as stats
import statsmodels.formula.api as smf
from scipy.stats import ttest_1samp, ttest_ind,f_oneway

cat_var = 'county'
cont_var = ['bedrooms', 'bathrooms', 'area', 'taxvalue', 'yearbuilt', 'taxamount']

# FUNCTIONS:

# Plot variables pairs with specific columns
def plot_variable_pairs(df):
    cols = ['bedrooms', 'bathrooms', 'area', 'taxvalue','yearbuilt','taxamount']
    sns.pairplot(df[cols], kind='reg')
    plt.show()


# Plot cat vs cont variables:
def plot_categorical_and_continuous_vars(df, cat_var, cont_var):
    for var in cont_var:
        # Create a box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=cat_var, y=var, data=df)
        plt.axhline(y=df[var].mean(), color='r', linestyle='--')
        plt.xlabel(cat_var)
        plt.ylabel(var)
        plt.title(f'{cat_var} vs. {var}')
        plt.show()

        # Create a violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=cat_var, y=var, data=df)
        plt.axhline(y=df[var].mean(), color='r', linestyle='--')
        plt.xlabel(cat_var)
        plt.ylabel(var)
        plt.title(f'{cat_var} vs. {var}')
        plt.show()

        # Create a swarm plot
        plt.figure(figsize=(12, 6))
        sns.swarmplot(x=cat_var, y=var, data=df)
        plt.axhline(y=df[var].mean(), color='r', linestyle='--')
        plt.xlabel(cat_var)
        plt.ylabel(var)
        plt.title(f'{cat_var} vs. {var}')
        plt.show()

# Plot correlations:
target_var = 'taxvalue'
feat_vars = ['bedrooms', 'bathrooms', 'area', 'yearbuilt', 'taxamount','county']
def plot_correlations(df, target_var, feat_vars):
    # Calculate correlations between feature variables and target variable
    correlations = df[feat_vars].corrwith(df[target_var]).sort_values()
    target_var = 'taxvalue'
    feat_vars = ['bedrooms', 'bathrooms', 'area', 'yearbuilt', 'taxamount','county']
    # Create a bar chart to visualize correlations
    plt.figure(figsize=(10,6))
    plt.barh(correlations.index, correlations.values)
    plt.xlabel('Correlation with Target Variable')
    plt.title('Feature Variable Correlations with Target Variable')
    plt.show()

# Function to calculate different stats test:
def test_relationships(df, cat_var, cont_vars):
    # One-sample t-test for each continuous variable against the population mean
    for var in cont_vars:
        t_stat, p_value = ttest_1samp(df[var], df[var].mean())
        print(f'One-sample t-test for {var}: t={t_stat:.4f}, p={p_value:.4f}')

    # Independent t-test for each continuous variable between the two categories in the categorical variable
    for var in cont_vars:
        for cat in df[cat_var].unique():
            group1 = df[df[cat_var] == cat][var]
            group2 = df[df[cat_var] != cat][var]
            t_stat, p_value = ttest_ind(group1, group2)
            print(f'Independent t-test for {var} between {cat} and not {cat}: t={t_stat:.4f}, p={p_value:.4f}')

  # One-way ANOVA 
    for var in cont_vars:
        groups = [df[df[cat_var] == cat][var] for cat in df[cat_var].unique()]
        f_stat, p_value = f_oneway(*groups)
        print(f'One-way ANOVA for {var} by {cat_var}: F={f_stat:.4f}, p={p_value:.4f}')




