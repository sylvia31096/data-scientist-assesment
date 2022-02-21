# import required libraries
import pandas as pd
import numpy as np

def merge_data():
    # read all the data into dataframes
    ken_cust_df = pd.read_csv("../data/Kenya Customers.csv") # Kenya customer data
    ken_del_df = pd.read_csv("../data/Kenya Deliveries.csv") # Kenya deliveries data
    ken_ord_df = pd.read_csv("../data/Kenya Orders.csv") # Kenya orders data
    nig_cust_df = pd.read_csv("../data/Nigeria Customers.csv") # Nigeria customer data
    nig_del_df = pd.read_csv("../data/Nigeria Deliveries.csv") # Nigeria deliveries data
    nig_ord_df = pd.read_csv("../data/Nigeria Orders.csv") # Nigeria orders data

    # Import data with removed comma typos 
    ken_del_df = pd.read_csv("../data/Kenya Deliveries Cleaner.csv") 

    # Spell 'Number of Employees' in the customer data the same.
    ken_cust_df = ken_cust_df.rename(columns={'Number of employees':'Number of Employees'})

    # Add country codes 
    nig_cust_df["country_code"]="234"
    nig_del_df["country_code"]="234"
    nig_ord_df["country_code"]="234"

    ken_cust_df["country_code"]="254"
    ken_del_df["country_code"]="254"
    ken_ord_df["country_code"]="254"

    # Append datasets by countries
    cust_df = ken_cust_df.append(nig_cust_df)
    del_df = ken_del_df.append(nig_del_df)
    ord_df = ken_ord_df.append(nig_ord_df)

    # Rename columns to match columns in other dataframes
    del_df = del_df.rename(columns={"Order_ID": "Order ID"})

    # Match pattern of "Order ID" in orders and deliveries data
    pat=r'YR-(\d+),0'
    del_df["Order ID"] = del_df["Order ID"].str.replace(pat, r'\1', regex=True)

    # Type cast to string all common columns
    del_df["Order ID"]=del_df["Order ID"].astype(str)
    ord_df["Order ID"]=ord_df["Order ID"].astype(str)
    cust_df["Customer ID"]=cust_df["Customer ID"].astype(str)
    ord_df["Customer ID"]=ord_df["Customer ID"].astype(str)

    ord_ord_set = set(ord_df["Order ID"])
    del_ord_set = set(del_df["Order ID"])
    ord_cust_set = set(ord_df["Customer ID"])
    cust_cust_set = set(cust_df["Customer ID"])

    # Initialize the common columns to merge on
    cust_cols = ['Customer ID','country_code']
    ord_cols = ['Order ID','country_code','Discount','Tip']

    # Merge orders data and customer data on 'Customer ID','country_code'
    ord_cust_df = pd.merge(cust_df, ord_df, left_on=cust_cols, right_on = cust_cols, how="outer")

    # Merge the new merged data frame and deliveries data on 'Order ID','country_code','Discount','Tip'
    df = pd.merge(ord_cust_df, del_df, left_on=ord_cols, right_on = ord_cols, how="outer")

    df.replace(to_replace="None", value=np.nan, inplace=True)

    df.to_csv("../data/merged_data.csv", na_rep=None,index=False)

    return df

def main():
    merge_data()    

if __name__ == '__main__':
    main()




