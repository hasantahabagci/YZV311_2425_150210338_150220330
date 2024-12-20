from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier



def gridSearch(model, param_grid, X_train, y_train, cv=2, scoring='accuracy', model_name= 'model_name'):
    """
    This function performs grid search
    :param model: model
    :param param_grid: parameter grid
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :return: best estimator
    """

    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid.fit(X_train, y_train)

    print("Best parameters for ", model_name, ": ", grid.best_params_)
    print("Best score for ", model_name, ": ", grid.best_score_)
    print("Best estimator for ", model_name, ": ", grid.best_estimator_)


    return grid

def randomForest(X_train, y_train, n_estimators=100, max_depth=10, max_features='auto', min_samples_split=2, min_samples_leaf=1, bootstrap=True):
    """
    This function performs random forest
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param n_estimators: number of trees
    :param max_depth: max depth of the tree
    :param max_features: max number of features
    :param min_samples_split: min number of samples to split
    :param min_samples_leaf: min number of samples in a leaf
    :param bootstrap: if True, bootstrap samples are used
    :return: predicted labels
    """

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=bootstrap)
    rf.fit(X_train, y_train)


    return rf


def xgboost(X_train, y_train, max_depth=10, learning_rate=0.1, n_estimators=100, objective='binary:logistic', booster='gbtree', n_jobs=1):
    """
    This function performs XGBoost
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param max_depth: max depth of the tree
    :param learning_rate: learning rate
    :param n_estimators: number of trees
    :param objective: objective function
    :param booster: booster type
    :param n_jobs: number of jobs
    :return: predicted labels
    """

    xgb = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, objective=objective, booster=booster, n_jobs=n_jobs)
    xgb.fit(X_train, y_train)


    return xgb


def adaBoost(X_train, y_train, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R'):
    """
    This function performs AdaBoost
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param n_estimators: number of trees
    :param learning_rate: learning rate
    :param algorithm: algorithm type
    :return: predicted labels
    """

    ada = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
    ada.fit(X_train, y_train)


    return ada