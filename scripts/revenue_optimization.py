import pandas as pd
from datetime import datetime
from pytz import timezone
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


# User defined module
from data_cleaning import clean_data

def get_analysis_data(file_path):

    # Import data and clean
    df = clean_data(file_path)

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    # Remove columns with same value
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)

    return df

def generate_new_columns(revenue_df, revenue_cols):
    # Define columns with monetary data
    money_cols = ['Total Cost Price','Total Price','gross_revenue']

    # Convert to Kenyan currency
    revenue_df['conversion'] = 1
    revenue_df['conversion'].loc[revenue_df['country_code'] == '234'] = 0.27
    # Get gross margin 
    revenue_df['gross_revenue'] = revenue_df['Total Price'] - revenue_df['Total Cost Price']
    revenue_df[money_cols] = revenue_df[money_cols].multiply(revenue_df["conversion"], axis="index")
    revenue_df = revenue_df[revenue_cols].drop_duplicates()

    # generating time data
    revenue_df['Order Day'] = revenue_df['Order Time'].dt.dayofweek
    revenue_df['Order Hour'] = revenue_df['Order Time'].dt.hour

    # convert catergorical data
    revenue_df['country_code'].replace(['254', '234'],
                            [0, 1], inplace=True)
    return revenue_df

def generate_customer_dataset(df):

    # Define relevant columns 
    revenue_cols = ['Order ID','Latitude', 'Longitude','Category Name','Loyalty Points',
                'Order Time','Total Cost Price','Total Price','country_code','gross_revenue']

    x_cols=['Latitude', 'Longitude','Loyalty Points','Order Day','Order Hour',
                'country_code']

    revenue_df = generate_new_columns(df, revenue_cols).dropna()
    
    y=revenue_df['gross_revenue'].values

    revenue_df = pd.get_dummies(revenue_df[x_cols],columns=['Order Day'])

    X = revenue_df.values
    return X,y

def fit_model(X,y):
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        print(r2_score(y_test, y_pred))

def main():
    df = get_analysis_data("data/formatted_data.csv")
    X,y = generate_customer_dataset(df)
    fit_model(X,y)

if __name__ == '__main__':
    main()

