"""
This file contains machine models
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from joblib import dump
from joblib import load


def fit_random_forest(train_features: pd.DataFrame,
                      train_results: pd.DataFrame) -> RandomForestClassifier:
    """
        This function creates and trains default RandomForestClassifier
        from sklearn
    :param train_features: a dataFrame of training objects
    :param train_results: a species of training objects
    :return: the trained RandomForestClassifier model
    """
    model = RandomForestClassifier()
    model.fit(train_features, train_results)
    return model


def fit_gradient_boosting(train_features: pd.DataFrame,
                          train_results: pd.DataFrame) -> \
        GradientBoostingClassifier:
    """
        This function creates and trains default GradientBoostingClassifier
        from sklearn
    :param train_features: a dataFrame of training objects
    :param train_results: a species of training objects
    :return: the trained GradientBoostingClassifier model
    """
    model = GradientBoostingClassifier()
    model.fit(train_features, train_results)
    return model


def fit_ada_boost(train_features: pd.DataFrame, 
                  train_results: pd.DataFrame) -> AdaBoostClassifier:
    """
        This function creates and trains default AdaBoostClassifier
        from sklearn
    :param train_features: a dataFrame of training objects
    :param train_results: a species of training objects
    :return: the trained AdaBoostClassifier model
    """
    model = AdaBoostClassifier()
    model.fit(train_features, train_results)
    return model


def fit_svc(train_features: pd.DataFrame,
            train_results: pd.DataFrame) -> SVC:
    """
        This function creates and trains default SVC from sklearn
    :param train_features: a dataFrame of training objects
    :param train_results: a species of training objects
    :return: the trained SVC model
    """
    model = SVC(probability=True)
    model.fit(train_features, train_results)
    return model


def fit_gauss(train_features: pd.DataFrame,
              train_results: pd.DataFrame) -> GaussianNB:
    """
        This function creates and trains default GaussianNB from sklearn
    :param train_features: a dataFrame of training objects
    :param train_results: a species of training objects
    :return: the trained GaussianNB model
    """
    model = GaussianNB()
    model.fit(train_features, train_results)
    return model


def fit_knn(train_features: pd.DataFrame,
            train_results: pd.DataFrame) -> KNeighborsClassifier:
    """
        This function creates and trains default KNeighborsClassifier
        from sklearn
    :param train_features: a dataFrame of training objects
    :param train_results: a species of training objects
    :return: the trained KNeighborsClassifier model
    """
    model = KNeighborsClassifier()
    model.fit(train_features, train_results)
    return model


def save_model(model, filepath: str) -> None:
    """
        This function saves the model to the specified file using joblib
    :param model: the model to ve saved
    :param filepath: the file saving path
    :return: the file in the specified path
    """
    dump(model, filepath)


def load_model(filepath: str):
    """
        This function loads and returns the model from the specified file
    :param filepath: the file where the model is
    :return: uploaded model
    """
    return load(filepath)
