import pandas as pd
import numpy as np

def fill_missing_category_data(df):
    """
    If 'categories' is missing, fill it with 'parent_categories' so we don't lose info.
    """
    def fill_missing_category(row):
        if pd.isna(row['categories']):
            return row['parent_categories']
        return row['categories']
    
    df["categories"] = df.apply(fill_missing_category, axis=1)
    return df

def engineer_features(df):
    """
    Perform feature engineering on the merged DataFrame.
    - Convert purchase_date to datetime
    - Create days_since_purchase
    - Create total_purchases & unique_products
    - Calculate recency_div_product
    - Add 'last_purchase'
    """
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    latest_date = df['purchase_date'].max()

    # days_since_purchase
    df['days_since_purchase'] = (latest_date - df['purchase_date']).dt.days

    # total_purchases per customer
    df['total_purchases'] = df.groupby('customer_id')['customer_id'].transform('count')

    # unique_products per customer
    df['unique_products'] = df.groupby('customer_id')['product_id'].transform('nunique')

    # Sort for grouping operations
    df_sorted = df.sort_values(by=['customer_id', 'product_id'])
    
    # total_unique_product_sold => total quantity of that product for each customer
    df_sorted["total_unique_product_sold"] = df_sorted.groupby(
        ['customer_id','product_id']
    )['quantity'].transform('sum')
    
    # last_purchase => min days_since_purchase for that product
    df_sorted['last_purchase'] = df_sorted.groupby(
        ['customer_id', 'product_id']
    )['days_since_purchase'].transform('min').reset_index(drop=True)
    
    # recency_div_product => (max_date - min_date) / total_unique_product_sold
    recency_diff = df_sorted.groupby(['customer_id', 'product_id']).agg(
        max_date=('days_since_purchase', 'max'),
        min_date=('days_since_purchase', 'min'),
        total_unique_product_sold=('total_unique_product_sold', 'first')
    )
    recency_diff['recency_div_product'] = (
        recency_diff['max_date'] - recency_diff['min_date']
    ) / recency_diff['total_unique_product_sold']

    # Merge back
    df_sorted = df_sorted.merge(
        recency_diff['recency_div_product'].reset_index(),
        on=['customer_id', 'product_id'],
        how='left'
    )

    # If recency_div_product == 0, fill with days_since_purchase
    df_sorted['recency_div_product'] = df_sorted.apply(
        lambda row: int(row['days_since_purchase']) if row['recency_div_product'] == 0 else int(row['recency_div_product']), 
        axis=1
    )
    return df_sorted

def prepare_test_data(test, engineered_df, mlb):
    """
    Prepare the test set so that it has the same columns as the training set.
      - Merge with the final engineered df to get features like 'last_purchase'
      - Apply the same MultiLabelBinarizer to 'parent_categories'
      - Drop unused columns
    """
    # Drop duplicates from engineered_df
    final_unique = engineered_df.drop_duplicates(subset=['customer_id', 'product_id']).copy()
    
    # Only keep rows that appear in test by (customer_id, product_id) pairs
    test_data_filled = final_unique[
        final_unique[['customer_id', 'product_id']].apply(tuple, axis=1).isin(
            test[['customer_id', 'product_id']].apply(tuple, axis=1)
        )
    ]

    # Replace days_since_purchase with the last_purchase
    test_data_filled['days_since_purchase'] = test_data_filled['last_purchase']

    # If 'next_purchase_weeks' is in the df, drop it
    if 'next_purchase_weeks' in test_data_filled.columns:
        test_data_filled.drop(columns=['next_purchase_weeks'], inplace=True)

    test_data_filled.reset_index(drop=True, inplace=True)

    # MultiLabelBinarize parent_categories
    test_cat_enc = mlb.transform(test_data_filled['parent_categories'])
    cat_enc_df = pd.DataFrame(test_cat_enc, columns=mlb.classes_)

    test_data_prepared = pd.concat([
        test_data_filled.drop('parent_categories', axis=1),
        cat_enc_df
    ], axis=1)

    # Drop columns not needed for inference
    drop_cols = ["purchase_date", "categories", "next_purchase_date", "next_purchase_days"]
    test_data_prepared.drop(
        columns=[col for col in drop_cols if col in test_data_prepared.columns],
        inplace=True,
        errors='ignore'
    )
    return test_data_prepared
