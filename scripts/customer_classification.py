import pandas as pd
from datetime import datetime
from pytz import timezone
import numpy as np
from sklearn.cluster import KMeans


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

def generate_new_columns(customer_df_before, money_cols):
    # Extract interval between orders
    customer_df_before['interval'] = (customer_df_before.groupby('Customer ID')['Order Time'].diff().dt.total_seconds()/86400)*-1
    customer_df_before['interval'] = customer_df_before['interval'].replace(0,np.nan)

    #  Get the cost of orders
    order_totals = customer_df_before.groupby('Order ID',as_index=False)['Total Cost Price'].sum()
    order_totals = order_totals.rename(columns={'Total Cost Price':'Order Cost Total'})

    # Merge calculated  columns
    customer_df_before = pd.merge(customer_df_before,order_totals,how='left')

    # Get gross margin 
    customer_df_before['gross_revenue'] = customer_df_before['Order Total'] - customer_df_before['Order Cost Total']

    # Convert to Kenyan currency
    customer_df_before['conversion'] = 1
    customer_df_before['conversion'].loc[customer_df_before['country_code'] == '234'] = 0.27
    customer_df_before[money_cols] = customer_df_before[money_cols].multiply(customer_df_before["conversion"], axis="index")

    return customer_df_before

def aggregate_features(customer_df_before):
    customer_order_profile = customer_df_before.groupby(
       'Customer ID').agg({'Order Total':['sum','mean'], 'gross_revenue':'sum','Order Time':['max','min'],
       'Order ID':'count','Redeemed Loyalty Points':'sum', 'Consumed Loyalty Points':'sum',
       "Task_ID":"count",'interval':'mean','Number of Employees':'min','Latitude':'mean','Longitude':'mean',
       'Outstanding Amount':'min','Loyalty Points':'min','country_code':'min'
       })

    customer_order_profile.reset_index(inplace=True)
    customer_order_profile.columns = ['Customer ID', 'Order Total','Avg Order Total','gross_revenue',
                                        'Order Time Max','Order Time Min','Order ID','Redeemed Loyalty Points',
                                        'Consumed Loyalty Points','Task_ID','interval','Number of Employees','Latitude','Longitude',
                                    'Outstanding Amount', 'Loyalty Points','country_code']

    order_time_interval = customer_order_profile['Order Time Max']-customer_order_profile['Order Time Min']
    order_time_interval = order_time_interval.dt.total_seconds()/86400
    customer_order_profile['Order Interval'] = order_time_interval
    customer_order_profile['Avg Order Interval'] = order_time_interval/customer_order_profile['Order ID']

    return customer_order_profile


def generate_customer_dataset(df):

    # Define relevant columns 
    customer_cols = ['Customer ID','Total Cost Price','Order Total', 'Sub Total', 'Tip', 'Discount',
            'Order ID', 'Redeemed Loyalty Points', 'Consumed Loyalty Points','Subtotal','Latitude','Longitude',
        'Number of Employees','Outstanding Amount','Loyalty Points','Rating',
        'Task_ID','country_code','Order Time']

    # Define columns with monetary data
    money_cols = ['Total Cost Price','Order Cost Total','Order Total',
                    'Sub Total', 'Tip', 'Discount']

    # Columns with generated columns
    new_customer_cols = ['Customer ID','Order Cost Total','Order Total', 'Sub Total', 'Tip', 'Discount',
            'Order ID', 'Redeemed Loyalty Points', 'Consumed Loyalty Points','Subtotal','Latitude','Longitude',
        'Number of Employees','Outstanding Amount','Loyalty Points','Rating',
        'Task_ID','country_code','Order Time','gross_revenue','interval']

    model_cols = ['Order Total', 'Avg Order Total', 'gross_revenue',
                'Order ID','Redeemed Loyalty Points', 'Consumed Loyalty Points',
                'interval', 'Outstanding Amount',
                'Loyalty Points', 'Order Interval',
                'Avg Order Interval']


    df = generate_new_columns(df[customer_cols], money_cols)

    df = df[new_customer_cols].drop_duplicates()

    customer_order_profile = aggregate_features(df)

    customer_purchase_history = customer_order_profile[customer_order_profile['Order ID']>0]

    X_df = customer_purchase_history[model_cols].dropna(how='any').sample(frac=1)

    X = X_df.values

    return X

def fit_model(X):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    y=kmeans.predict(X)
    return y

def main():
    df = get_analysis_data("../data/merged_data.csv")
    X = generate_customer_dataset(df)
    y = fit_model(X)

if __name__ == '__main__':
    main()

