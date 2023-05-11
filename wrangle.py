### IMPORTS
import pandas as pd 
import env
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
def get_db_url(database):
    return f'mysql+pymysql://{env.username}:{env.password}@{env.host}/{database}'

### FUNCTIONS:

#functions to get data set

def data_set():
    
    '''This funcion will:
    -take in a SQL Query
    - create a connect_url to my SQL
    return a df of the given query from the zillow databse'''
    
    directory='/Users/chellyannmoreno/codeup-data-science/regression-exercises/'
    url=get_db_url('zillow')
    
    SQL_query= """select id, parcelid,bathroomcnt, bedroomcnt,calculatedfinishedsquarefeet
        ,taxvaluedollarcnt,yearbuilt,taxamount, fips
         from properties_2017
         where propertylandusetypeid=261"""
    
    filename=('zillow.csv')
    
    if os.path.exists(directory+filename):
        df=pd.read_csv(filename)
        return df
    else:
        df=pd.read_sql(SQL_query,url)
        df.to_csv(filename,index=False)
        return df
    


# function to wrangle zillow with nulls

def wrangle_zillow_nulls(df):
    #rename columns 
    df = df.rename(columns={
        'bedroomcnt': 'bedrooms',
        'bathroomcnt': 'bathrooms',
        'calculatedfinishedsquarefeet': 'area',
        'taxvaluedollarcnt': 'taxvalue',
        'fips': 'county'
    })
    
    ## fill nulls 
    df.bedrooms=df.bedrooms.fillna(3)
    df.bathrooms=df.bathrooms.fillna(df.bathrooms.mode())
    df.area=df.area.fillna(1200.00)
    df.taxvalue=df.taxvalue.fillna( 450000.0)
    df.yearbuilt=df.yearbuilt.fillna(1960)
    df.taxamount=df.taxamount.fillna(5634.87)
    
    ## change to int and change county into object
    df[['bedrooms', 'area', 'taxvalue', 'yearbuilt']] = df[['bedrooms', 'area', 'taxvalue', 'yearbuilt']].astype(int)
    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    
    return df




# functions to get data set for the no nulls wrangle:

def data_set():
    
    '''This funcion will:
    -take in a SQL Query
    - create a connect_url to my SQL
    return a df of the given query from the zillow databse'''
    
    directory='/Users/chellyannmoreno/codeup-data-science/regression-exercises/'
    url=get_db_url('zillow')
    
    SQL_query= """select id, parcelid,bathroomcnt, bedroomcnt,calculatedfinishedsquarefeet
        ,taxvaluedollarcnt,yearbuilt,taxamount, fips
         from properties_2017
         where propertylandusetypeid=261"""
    
    filename=('zillow.csv')
    
    if os.path.exists(directory+filename):
        df=pd.read_csv(filename)
        return df
    else:
        df=pd.read_sql(SQL_query,url)
        df.to_csv(filename)
        return df
    
## wrangle data to drop nulls, rename columns, remove outliers:
def wrangle_data(df):
    df = df.rename(columns={
        'bedroomcnt': 'bedrooms',
        'bathroomcnt': 'bathrooms',
        'calculatedfinishedsquarefeet': 'area',
        'taxvaluedollarcnt': 'taxvalue',
        'fips': 'county'
    })
 # Filter out rows with large area and filter out places with zero bathrooms and baths, and with more than 15.
    df = df[df.area < 25_000]
    df = df[df.yearbuilt > 1890]
    df = df[(df.bathrooms > 0) & (df.bathrooms < 15) & (df.bedrooms > 0) & (df.bedrooms < 15)]


    # Drop rows with missing values
    df = df.dropna()

    # Filter out rows with high tax value
    taxvalue_threshold = df.taxvalue.quantile(.95)
    df = df[df.taxvalue < taxvalue_threshold].copy()

    # Convert data types
    df[['bedrooms', 'area', 'taxvalue', 'yearbuilt']] = df[['bedrooms', 'area', 'taxvalue', 'yearbuilt']].astype(int)

    # Map county codes to names
    county_map = {6037: 'LA', 6059: 'Orange', 6111: 'Ventura'}
    df.county = df.county.map(county_map)
    return df


 # Function to split data:


def split_data(df):
    # Split into train_validate and test sets
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # Split into train and validate sets
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123)

    return train, validate, test


# Function to scale my data using minmaxscaler:
def minmax_scale_data(X_train, X_validate):
    # Initialize MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Fit scaler object to training data
    scaler.fit(X_train)
    
    # Transform training and validation data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    
    # Return scaled data
    return X_train_scaled, X_validate_scaled

# function to scale my data using robust scaler (good for data with outliers):

def robust_scale_data(X_train, X_validate):
    # Initialize RobustScaler object
    scaler = RobustScaler()
    
    # Fit scaler object to training data
    scaler.fit(X_train)
    
    # Transform training and validation data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    
    # Return scaled data
    return X_train_scaled, X_validate_scaled

#function to scale my data using standard scaler:

def standard_scale_data(X_train, X_validate):
    # Initialize StandardScaler object
    scaler = StandardScaler()
    
    # Fit scaler object to training data
    scaler.fit(X_train)
    
    # Transform training and validation data
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    
    # Return scaled data
    return X_train_scaled, X_validate_scaled


#encode county:
#dummy_df = pd.get_dummies(df['county'],
#                                 )
#df = pd.concat( [df,dummy_df], axis=1 )

