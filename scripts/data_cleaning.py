
import pandas as pd
import numpy as np

def money_formatting(col):
    """
    This functions formats monetary data.

    Args:
        col:dataframe column

    Returns:
        col: a formatted dataframe column 
    """
    curr_pattern = r'[\w|₦\s]+(\d.\d+)[\s\w��]*'
    col = col.astype(str)
    col = col.str.replace(curr_pattern,r'\1')
    col = col.replace(to_replace="-", value=np.nan)
    return col.astype(float)

def clean_amounts(df):
    """
    This functions formats amounts data in passed dataframe.

    Args:
        df:dataframe    
    """
    df['Outstanding Amount'] = df['Outstanding Amount'].astype("float")
    df['Outstanding Amount'] = df['Loyalty Points'].astype("float")
    df['Number of Employees'] = df['Number of Employees'].astype('float32').apply(lambda x: '%.0f' % x)
    df['Quantity'] = df['Quantity'].astype(float)
    df['Unit Price'] = df['Quantity'].astype(float)
    df['Cost Price'] = df['Cost Price'].astype(float)
    df['Total Cost Price'] = df['Total Cost Price'].astype(float)
    df['Total Price'] = df['Total Price'].astype(float)
    df['Order Total'] = df['Order Total'].astype(float)
    df['Sub Total'] = df['Sub Total'].astype(float)
    df['Tax'] = df['Tax'].replace(to_replace="-", value=np.nan).astype(float)
    df['Delivery Charge'] = df['Delivery Charge'].astype(float)
    df['Remaining Balance'] = df['Remaining Balance'].astype(float)
    df['Additional Charge'] = df['Additional Charge'].astype(float)
    df['Taxable Amount'] = df['Taxable Amount'].astype(float)
    df['Distance (in km)'] = df['Distance (in km)'].astype(float)
    df['Ratings'] = df['Ratings'].astype(float)
    df['Order Preparation Time'] = df['Order Preparation Time'].astype(float)
    df['Flat Discount'] = df['Flat Discount'].astype(float)
    df['Total_Time_Taken(min)'] = df['Total_Time_Taken(min)'].astype(float)
    df['Rating'] = df['Rating'].astype(float)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)


    df['Debt Amount'] = df['Debt Amount'].astype(float)
    df['Redeemed Loyalty Points'] = df['Redeemed Loyalty Points'].astype(float)
    df['Consumed Loyalty Points'] = df['Consumed Loyalty Points'].astype(float)
    df['Checkout Template Name'] = df['Checkout Template Name'].astype(float)
    df['Checkout Template Value'] = df['Checkout Template Value'].astype(float)
    df['Distance(m)'] = df['Distance(m)'].astype(float)
    df['Ref_Images'] = df['Ref_Images'].astype(float)
    df['Tags'] = df['Tags'].astype(float)
    df['Task_Details_QTY'] = df['Task_Details_QTY'].astype(float)
    #df['Subtotal'] = df['Subtotal'].astype(float)
    df['Earning'] = df['Earning'].astype(float)
    df['Pricing'] = df['Pricing'].astype(float)

def clean_money_amount(df):
    """
    This functions formats money amounts data in passed dataframe.

    Args:
        df:dataframe    
    """
    df['Tip'] = money_formatting(df['Tip'])
    df['Discount'] = money_formatting(df['Discount'])
    df['Task_Details_AMOUNT'] = money_formatting(df['Task_Details_AMOUNT'])
    df['Delivery_Charges'] = money_formatting(df['Delivery_Charges'])
    df['Task_Details_AMOUNT'] = money_formatting(df['Task_Details_AMOUNT'])
    #df['Subtotal'] = df['Subtotal'].str.replace(curr_pattern,r'\1').replace(to_replace="-", value=np.nan).astype(float)

def clean_datetime_data(df):
    """
    This functions formats datetime data in passed dataframe.

    Args:
        df:dataframe    
    """
    df['Order Time'] = pd.to_datetime(df['Order Time'])
    df['Pickup Time'] = pd.to_datetime(df['Pickup Time'])
    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Start_Before'] = pd.to_datetime(df['Start_Before'])
    df['Complete_Before'] = pd.to_datetime(df['Complete_Before'])
    df['Completion_Time'] = pd.to_datetime(df['Completion_Time'])

def clean_data(file_path):
    """
    This cleans the data from the merged dataset.

    Args:
        file_path:file path of merged data
    Returns:
        df: dataframe of cleaned data
    """

    # Format Unique identifiers to string
    dtypes = {'Customer ID':'str', 'Custom_Template_ID':'str', 'Order ID':'str',
    'Transaction ID':'str','Merchant ID':'str','Task_ID':'str','Agent_ID':'str',
    'country_code':'str'
    }

    # Import data from CSV
    df = pd.read_csv(file_path, dtype=dtypes)

    # Replace missing data symbols like "-" or "None" to NaN
    df.replace(to_replace="None", value=np.nan, inplace=True)
    df.replace(to_replace="-", value=np.nan, inplace=True)

    # clean amount
    clean_amounts(df)

    # clean money amount columns
    clean_money_amount(df)

    # Format timestamps
    clean_datetime_data(df)

    return df

def main():
    df = clean_data("../data/merged_data.csv")
    df.to_csv("../data/formatted_data.csv", na_rep=None,index=False)
    

if __name__ == '__main__':
    main()


import pandas as pd
import numpy as np

def money_formatting(col):
    """
    This functions formats monetary data.

    Args:
        col:dataframe column

    Returns:
        col: a formatted dataframe column 
    """
    curr_pattern = r'[\w|₦\s]+(\d.\d+)[\s\w��]*'
    col = col.astype(str)
    col = col.str.replace(curr_pattern,r'\1')
    col = col.replace(to_replace="-", value=np.nan)
    return col.astype(float)

def clean_amounts(df):
    """
    This functions formats amounts data in passed dataframe.

    Args:
        df:dataframe    
    """
    df['Outstanding Amount'] = df['Outstanding Amount'].astype("float")
    df['Outstanding Amount'] = df['Loyalty Points'].astype("float")
    df['Number of Employees'] = df['Number of Employees'].astype('float32').apply(lambda x: '%.0f' % x)
    df['Quantity'] = df['Quantity'].astype(float)
    df['Unit Price'] = df['Quantity'].astype(float)
    df['Cost Price'] = df['Cost Price'].astype(float)
    df['Total Cost Price'] = df['Total Cost Price'].astype(float)
    df['Total Price'] = df['Total Price'].astype(float)
    df['Order Total'] = df['Order Total'].astype(float)
    df['Sub Total'] = df['Sub Total'].astype(float)
    df['Tax'] = df['Tax'].replace(to_replace="-", value=np.nan).astype(float)
    df['Delivery Charge'] = df['Delivery Charge'].astype(float)
    df['Remaining Balance'] = df['Remaining Balance'].astype(float)
    df['Additional Charge'] = df['Additional Charge'].astype(float)
    df['Taxable Amount'] = df['Taxable Amount'].astype(float)
    df['Distance (in km)'] = df['Distance (in km)'].astype(float)
    df['Ratings'] = df['Ratings'].astype(float)
    df['Order Preparation Time'] = df['Order Preparation Time'].astype(float)
    df['Flat Discount'] = df['Flat Discount'].astype(float)
    df['Total_Time_Taken(min)'] = df['Total_Time_Taken(min)'].astype(float)
    df['Rating'] = df['Rating'].astype(float)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)


    df['Debt Amount'] = df['Debt Amount'].astype(float)
    df['Redeemed Loyalty Points'] = df['Redeemed Loyalty Points'].astype(float)
    df['Consumed Loyalty Points'] = df['Consumed Loyalty Points'].astype(float)
    df['Checkout Template Name'] = df['Checkout Template Name'].astype(float)
    df['Checkout Template Value'] = df['Checkout Template Value'].astype(float)
    df['Distance(m)'] = df['Distance(m)'].astype(float)
    df['Ref_Images'] = df['Ref_Images'].astype(float)
    df['Tags'] = df['Tags'].astype(float)
    df['Task_Details_QTY'] = df['Task_Details_QTY'].astype(float)
    #df['Subtotal'] = df['Subtotal'].astype(float)
    df['Earning'] = df['Earning'].astype(float)
    df['Pricing'] = df['Pricing'].astype(float)

def clean_money_amount(df):
    """
    This functions formats money amounts data in passed dataframe.

    Args:
        df:dataframe    
    """
    df['Tip'] = money_formatting(df['Tip'])
    df['Discount'] = money_formatting(df['Discount'])
    df['Task_Details_AMOUNT'] = money_formatting(df['Task_Details_AMOUNT'])
    df['Delivery_Charges'] = money_formatting(df['Delivery_Charges'])
    df['Task_Details_AMOUNT'] = money_formatting(df['Task_Details_AMOUNT'])
    #df['Subtotal'] = df['Subtotal'].str.replace(curr_pattern,r'\1').replace(to_replace="-", value=np.nan).astype(float)

def clean_datetime_data(df):
    """
    This functions formats datetime data in passed dataframe.

    Args:
        df:dataframe    
    """
    df['Order Time'] = pd.to_datetime(df['Order Time'])
    df['Pickup Time'] = pd.to_datetime(df['Pickup Time'])
    df['Delivery Time'] = pd.to_datetime(df['Delivery Time'])
    df['Start_Before'] = pd.to_datetime(df['Start_Before'])
    df['Complete_Before'] = pd.to_datetime(df['Complete_Before'])
    df['Completion_Time'] = pd.to_datetime(df['Completion_Time'])

def clean_data(file_path):
    """
    This cleans the data from the merged dataset.

    Args:
        file_path:file path of merged data
    Returns:
        df: dataframe of cleaned data
    """

    # Format Unique identifiers to string
    dtypes = {'Customer ID':'str', 'Custom_Template_ID':'str', 'Order ID':'str',
    'Transaction ID':'str','Merchant ID':'str','Task_ID':'str','Agent_ID':'str',
    'country_code':'str'
    }

    # Import data from CSV
    df = pd.read_csv(file_path, dtype=dtypes)

    # Replace missing data symbols like "-" or "None" to NaN
    df.replace(to_replace="None", value=np.nan, inplace=True)
    df.replace(to_replace="-", value=np.nan, inplace=True)

    # clean amount
    clean_amounts(df)

    # clean money amount columns
    clean_money_amount(df)

    # Format timestamps
    clean_datetime_data(df)

    return df

def main():
    df = clean_data("../data/merged_data.csv")
    df.to_csv("../data/formatted_data.csv", na_rep=None,index=False)
    

if __name__ == '__main__':
    main()

