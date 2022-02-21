import pandas as pd
from datetime import datetime
from pytz import timezone
import numpy as np
from surprise import NMF
from surprise import Reader,Dataset

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

def generate_product_dataset(df):

    recommendation_cols = ['Customer ID','Category Name']
    recommendation_df = df[recommendation_cols].dropna()
    recommendation_pivot = pd.pivot_table(data=recommendation_df,index='Customer ID',columns='Category Name',aggfunc=len,fill_value=0)
    recommendation_pivot = (recommendation_pivot-recommendation_pivot.min())/(recommendation_pivot.max()-recommendation_pivot.min())+1
    recommendation_data = recommendation_pivot.reset_index().melt(id_vars=['Customer ID'], var_name='Category Name', value_name='count')
    recommendation_data.loc[recommendation_data['count']>1,'count']=2
    X=recommendation_data.sample(frac=1)
    X_train=X[:9000]
    X_test = X[9000:]
    return X_train, X_test

def fit_model(X_train, X_test):
    algo = NMF()
    data = Dataset.load_from_df(X_train,reader=Reader(rating_scale=(1, 2)))

    algo.fit(data.build_full_trainset())
    my_recs = []

    for uid,iid,rec in X_test.values:
        my_recs.append([uid,iid, algo.predict(uid=uid,iid=iid).est,rec])
    
    pred_df = pd.DataFrame(my_recs,columns=['Customer ID','Category','predictions','true'])
    pred_df['true_thresh']=1
    pred_df['pred_thresh']=1
    pred_df.loc[pred_df['true']>1,'true_thresh']=2
    pred_df.loc[pred_df['predictions']>1.5,'pred_thresh']=2
    return pred_df

def main(): 
    df = get_analysis_data("data/formatted_data.csv")
    X_train, X_test = generate_product_dataset(df)
    pred_df = fit_model(X_train, X_test)

if __name__ == '__main__':
    main()

