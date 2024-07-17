"""
This file contains the main steps of the workflow
link to the competition:
https://www.kaggle.com/competitions/leaf-classification/overview
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import models


def fit_and_save_models(train_data: pd.DataFrame, label: str) -> None:
    """
        This function fits and saves models
    :param train_data: DataFrame of a train data
    :param label: the name of the target label
    :return: None
    """
    x_train = train_data.drop(label, axis=1)
    y_train = train_data[label]
    models.save_model(models.fit_random_forest(x_train, y_train),
                      r'models/random_forest.joblib')
    models.save_model(models.fit_gradient_boosting(x_train, y_train),
                      r'models/gradient_boosting.joblib')
    models.save_model(models.fit_ada_boost(x_train, y_train),
                      r'models/ada_boost.joblib')
    models.save_model(models.fit_gauss(x_train, y_train),
                      r'models/gauss.joblib')
    models.save_model(models.fit_knn(x_train, y_train),
                      r'models/knn.joblib')
    models.save_model(models.fit_svc(x_train, y_train),
                      r'models/svc.joblib')


def pred_and_save_result(model, test_data: pd.DataFrame, species: list,
                         filepath: str) -> None:
    """
        This function makes predictions and saves them to the species file
        in the correct form
    :param model: the machine model
    :param test_data: data whose labels need to be predicted and saved
    :param species: list of result DataFrame species
    :param filepath: the file saving path
    :return: None
    """
    result_df = pd.DataFrame()
    predictions = model.predict_proba(test_data.drop('id', axis=1)).T
    result_df['id'] = test_data['id']
    for ind in range(len(species)):
        result_df[species[ind]] = predictions[ind]
    result_df.to_csv(filepath, index=False)


train_df = pd.read_csv(r'datasets/train.csv')
test_df = pd.read_csv(r'datasets/test.csv')

# fit and save models
# fit_and_save_models(train_df.drop('id', axis=1), 'species')

# load models
model_rand_forest = models.load_model(r'models/random_forest.joblib')
model_grad_boost = models.load_model(r'models/gradient_boosting.joblib')
model_ada_boost = models.load_model(r'models/ada_boost.joblib')
model_gauss = models.load_model(r'models/gauss.joblib')
model_knn = models.load_model(r'models/knn.joblib')
model_svc = models.load_model(r'models/svc.joblib')

# predict and save results
pred_and_save_result(model_rand_forest, test_df,
                     sorted(train_df['species'].unique()),
                     r'results/pred_rand_forest.csv')
pred_and_save_result(model_grad_boost, test_df,
                     sorted(train_df['species'].unique()),
                     r'results/pred_grad_boost.csv')
pred_and_save_result(model_ada_boost, test_df,
                     sorted(train_df['species'].unique()),
                     r'results/pred_ada_boost.csv')
pred_and_save_result(model_gauss, test_df,
                     sorted(train_df['species'].unique()),
                     r'results/pred_gauss.csv')
pred_and_save_result(model_knn, test_df,
                     sorted(train_df['species'].unique()),
                     r'results/pred_knn.csv')
pred_and_save_result(model_svc, test_df,
                     sorted(train_df['species'].unique()),
                     r'results/pred_svc.csv')


#test commit