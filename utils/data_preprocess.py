import pandas as pd
from sklearn.preprocessing import LabelEncoder


def isCategorical(df, col, threshold):
    """
    This function decides if a column is categorical or not
    :param df: dataframe
    :param col: column name
    :param threshold: threshold to decide if a column is categorical or not
    :return: True if the column is categorical, False otherwise
    """

    if df[col].nunique() <= threshold:
        return True
    else:
        return False
    
def isNumerical(df, col):
    """
    This function decides if a column is numerical or not
    :param df: dataframe
    :param col: column name
    :param threshold: threshold to decide if a column is numerical or not
    :return: True if the column is numerical, False otherwise
    """
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    if df[col].dtype in numeric_types:
        return True
    else:
        return False