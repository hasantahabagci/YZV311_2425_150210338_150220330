from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def gridSearch(model, param_grid, X_train, y_train, cv=2, scoring='accuracy', model_name='model_name'):
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid.fit(X_train, y_train)
    print("Best parameters for", model_name, ":", grid.best_params_)
    print("Best score for", model_name, ":", grid.best_score_)
    print("Best estimator for", model_name, ":", grid.best_estimator_)
    return grid

def randomForest(X_train, y_train,
                 n_estimators=100, max_depth=10, max_features='auto', 
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        max_features=max_features,
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        bootstrap=bootstrap
    )
    rf.fit(X_train, y_train)
    return rf

def xgboost_model(X_train, y_train,
                  n_estimators=100, max_depth=3, learning_rate=0.1,
                  objective='multi:softmax', random_state=42):
    xgb_clf = XGBClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate, 
        objective=objective, 
        random_state=random_state
    )
    xgb_clf.fit(X_train, y_train)
    return xgb_clf
