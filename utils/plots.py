import matplotlib.pyplot as plt
import seaborn as sns

def plotdist(df, col, mx):
    """
    This function plots the distribution of a column
    :param df: dataframe
    :param col: column name
    :return: None
    """
    plt.figure(figsize=(10, 6))
    sns.distplot(df[col], kde=True, label='skewness : %.2f'%(df[col].skew()))
    plt.vlines(df[col].mean(), 0, mx, colors='r', label='mean')
    plt.text(df[col].mean(), mx*1.01, "mean : "+ str(round(df[col].mean(),2)), fontsize=12)
    plt.title(col.upper() + " DISTRIBUTION")
    plt.legend()
    plt.show()

def plotbox(df, col):
    """
    This function plots the boxplot of a column
    :param df: dataframe
    :param col: column name
    :return: None
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(df[col])
    plt.show()

def plotbar(df, col):
    """
    This function plots the barplot of a column
    :param df: dataframe
    :param col: column name
    :return: None
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(df[col])
    plt.title(col.upper() + " Bar Plot")
    plt.xticks(rotation=90)
    plt.show()