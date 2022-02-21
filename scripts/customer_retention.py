import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression

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

def split_data(df, customer_cols):
    date_time_str = '01/02/22 00:00:00'

    date_time_obj = datetime.strptime(date_time_str, '%d/%m/%y %H:%M:%S')
    date_time_obj = date_time_obj.replace(tzinfo=timezone('UTC'))

    # Extract data before February 2022
    customer_df_before = df[customer_cols][~(df['Order Time']>=date_time_obj)].drop_duplicates()

    # Extract data after February 2022
    after= df[customer_cols][df['Order Time']>=date_time_obj].drop_duplicates()['Customer ID']

    return customer_df_before, after

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

def get_repeat_customers_column(purchasing_customers, after):
    purchasing_customers['repeat_customer'] = np.nan
    repeat_mask = purchasing_customers['Customer ID'].isin(after)
    purchasing_customers.loc[repeat_mask,'repeat_customer'] = 0
    purchasing_customers.loc[~repeat_mask,'repeat_customer'] = 1
    purchasing_customers = purchasing_customers.dropna(how='any')
    return purchasing_customers


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

    model_cols = ['Order Total', 'Avg Order Total', 'Order ID',
       'Redeemed Loyalty Points', 'Consumed Loyalty Points','Latitude','Longitude',
       'Avg Order Interval','Order Interval','repeat_customer']

    x_cols = ['Order Total', 'Avg Order Total', 'Order ID',
        'Redeemed Loyalty Points', 'Consumed Loyalty Points',
        'Avg Order Interval','Order Interval']

    january_df, after = split_data(df, customer_cols)

    january_df = generate_new_columns(january_df, money_cols)

    january_df = january_df[new_customer_cols].drop_duplicates()

    customer_order_profile = aggregate_features(january_df)

    purchasing_customers=customer_order_profile[customer_order_profile['Order ID']>0]

    purchasing_customers = get_repeat_customers_column(purchasing_customers,after)

    purchasing_customers_model = purchasing_customers[model_cols].sample(frac=1)

    X = purchasing_customers_model[x_cols].values
    y = purchasing_customers_model['repeat_customer'].values

    return X, y

def fit_model(X,y):
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(accuracy_score(y_test, y_pred))

def main():
    df = get_analysis_data("data/merged_data.csv")
    X,y = generate_customer_dataset(df)
    fit_model(X,y)

if __name__ == '__main__':
    main()