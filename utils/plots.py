"""
Authors: 1) Hasan Taha Bagci - 150210330
         2) Selman Turan Toker - 150220330
File: plots.py
"""


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


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

def plot_distribution(df, col):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

def plot_confusion_matrix(cm, model_classes, title="Confusion Matrix"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_classes)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()
