#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File : Model.py


# #coding=utf8
# import sys
# import importlib
# importlib.reload(sys)

import os
import time
import logging
import argparse

#from cluster import StrictClustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from xgboost import XGBRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor, AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, make_scorer
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV

# import torch
# from torch.autograd import Variable
# import torch.utils.data as Data
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# torch.manual_seed(1)
import Get_Features as gf
import warnings
warnings.filterwarnings("ignore")
import sys


def get_parser():
    parser = argparse.ArgumentParser(description="Logging Demo")
    parser.add_argument('--not-save', default=False, action='store_true',
                        help='If yes, only output log to terminal')
    parser.add_argument('--work-dir', default='./work_dir',
                        help='the work folder for storing results')
    return parser

def loadLogger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    if not args.not_save:
        work_dir = os.path.join(args.work_dir)

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)
    return logger


def feature_process(Test_size):#描述符
    pubchem = gf.Calculate_PubChem()
    exECFP = gf.Calculate_ExtECFP()
    esECFP = gf.Calculate_EStECFP()
    FP = gf.Calculate_FP()
    GOFP = gf.Calculate_GOFP()
    KRFP = gf.Calculate_KRFP()
    KRFPC = gf.Calculate_KRFPC()
    MACCSFP = gf.Calculate_MACCSFP()
    SBFP = gf.Calculate_SBFP()
    SBFPC = gf.Calculate_SBFPC()
    Q1D_2D = gf.Calculate_1D_2D()
    AP2DFP = gf.Calculate_AP2DFP()
    AP2DFPC = gf.Calculate_AP2DFPC()
    KRSBFPC = gf.Calculate_KRSBFPC()
    FPSBFPC = gf.Calculate_FPSBFPC()
    FPKRFPC = gf.Calculate_FPKRFPC()
    FPKRSBFPC = gf.Calculate_FPKRSBFPC()
    Q1D_2DSBFPC = gf.Calculate_1D_2DSBFPC()
    Q1D_2DKRFPC = gf.Calculate_1D_2DKRFPC()
    KRFPCKRFP = gf.Calculate_KRFPCKRFP()
    KRFPCKRFPSBFPC = gf.Calculate_KRFPCKRFPSBFPC()
    Q1D_2DKRFP = gf.Calculate_1D_2DKRFP()
    Q1D_2DKRFPCKRFP = gf.Calculate_1D_2DKRFPCKRFP()

    # test_Des_4a5f = gf.Calculate_Des_4a5f()     # 实验化合物的特征

    # test_prediction = gf.Calculate_test_prediction()
    # RFtest_prediction = gf.Calculate_RFtest_prediction()
    A232_prediction = gf.Calculate_232_prediction()

    _, value = gf.get_value()

    X, y, test_experment = np.array(FPSBFPC), np.array(value), np.array(A232_prediction)
    # X, y = np.array(FPSBFPC), np.array(value)

    #y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_size, random_state=33)

    fs = VarianceThreshold(.1)#特征选择
    X_train = fs.fit_transform(X_train)
    X_test = fs.transform(X_test)

    # test_Des_4a5f_fs = fs.transform(test_experment)   # 实验化合物的特征选择
    # test_prediction_fs = fs.transform(test_experment)
    # RFtest_prediction_fs = fs.transform(RFtest_prediction)
    A232_prediction_fs = fs.transform(A232_prediction)

    #ft = fs.transform(ft)
    print(f"Features selection of Train:{len(X_train[0])}, and test set: {len(A232_prediction[0])}")
    return X_train, X_test, y_train, y_test, A232_prediction_fs


def train_SVR_model(CV):
    X_train, _, y_train, _, _ = feature_process(Test_size=0.2)
    Reg = SVR()
    scoring = {'R2_Score': 'r2', 'MAE': make_scorer(mean_absolute_error),
               'MSE': make_scorer(mean_squared_error), "MedAE": make_scorer(median_absolute_error)}

    params_search = {"C": [0.1, 0.2, 0.3, 0.4, 0.5],
                   "kernel": ['linear','rbf','sigmoid'],
                   'gamma': [0.01,0.02, 0.03, 0.005, 'auto']}

    grid_search_params = {'estimator': Reg,
                           'param_grid': params_search,
                           'cv': CV,
                           'n_jobs': 8,
                           'verbose': 3}

    grsearch = GridSearchCV(**grid_search_params,scoring=scoring,refit='R2_Score')
    grsearch.fit(X_train, y_train)
    return grsearch


def train_RF_model(CV):
    X_train, _, y_train, _, _ = feature_process(Test_size=0.2)
    reg2 = RandomForestRegressor(random_state=44)
    scoring = {'R2_Score': 'r2', 'MAE': make_scorer(mean_absolute_error),
               'MSE': make_scorer(mean_squared_error), "MedAE": make_scorer(median_absolute_error)}

    params_search = {'n_estimators': [100, 300, 400, 600, 700], 'max_depth': [11, 13, 17, 21, None],
                     'max_features': ['auto', 'sqrt', 'log2'], 'criterion': ['mse', 'mae']}

    grid_search_params = {'estimator': reg2,
                          'param_grid': params_search,
                          'cv': CV,
                          'n_jobs': 30,
                          'verbose': 3,
                          }
    grsearch = GridSearchCV(**grid_search_params,scoring=scoring,refit='R2_Score')
    grsearch.fit(X_train, y_train)
    return grsearch

def train_GB_model(CV):
    X_train, _, y_train, _, _= feature_process(Test_size=0.2)
    reg3 = GradientBoostingRegressor(random_state=44)
    scoring = {'R2_Score': 'r2', 'MAE': make_scorer(mean_absolute_error),
               'MSE': make_scorer(mean_squared_error), "MedAE": make_scorer(median_absolute_error)}
    params_search = {'n_estimators': [150,200],
                     'max_depth': [2],
                     'learning_rate': [0.1, 0.05,0.15],
                     'loss': ['ls','lad''huber'],
                     'criterion': ['friedman_mse','mae','mse']}
    grid_search_params = {'estimator': reg3,
                          'param_grid': params_search,
                          'cv': CV,
                          'n_jobs': 8}
    grsearch = GridSearchCV(**grid_search_params,scoring=scoring,refit='R2_Score')
    grsearch.fit(X_train, y_train)
    return grsearch

def train_AdaBoost_model(CV):
    X_train, _, y_train, _, _ = feature_process(Test_size=0.2)
    reg4 = AdaBoostRegressor(random_state=44)
    scoring = {'R2_Score': 'r2', 'MAE': make_scorer(mean_absolute_error),
               'MSE': make_scorer(mean_squared_error), "MedAE": make_scorer(median_absolute_error)}
    paras_search = {'n_estimators': [400,800,1100,1200,1300,1400],
                    'learning_rate':[0.01,0.15,0.3,0.5,0.7,0.9,0.95],
                    'loss': ['linear', 'square', 'exponential']}
    grid_search_params = {'estimator': reg4,
                          'param_grid':paras_search,
                          'cv': CV,
                          'n_jobs': 8}
    grsearch = GridSearchCV(**grid_search_params,scoring=scoring,refit='R2_Score')
    grsearch.fit(X_train, y_train)
    return grsearch

def train_XGBoost_model(CV):
    X_train, _, y_train, _, _ = feature_process(Test_size=0.2)
    reg5 = XGBRegressor(random_state=44)
    scoring = {'R2_Score': 'r2', 'MAE': make_scorer(mean_absolute_error, greater_is_better=True),
               'MSE': make_scorer(mean_squared_error, greater_is_better=True),
               "MedAE": make_scorer(median_absolute_error, greater_is_better=True)}
    params_search = {"max_depth": [1,3,6,10,50,300,500],
                     "lambda": [1,2,3,4],
                     "learning_rate": [0.02,0.03,0.01],
                     "n_estimators": [900,700,800],
                     "subsample": [0.6,0.7,0.8,0.9]
                     }

    grid_search_params = {'estimator': reg5,
                          'param_grid': params_search,
                          'cv': CV,
                          'n_jobs': 8,
                          'verbose': 3}

    grsearch = GridSearchCV(**grid_search_params,scoring=scoring,refit='R2_Score')
    grsearch.fit(X_train, y_train)
    return grsearch

def estimate_model(CV,logger):
    grsearch1 = train_SVR_model(CV)
    # grsearch2 = train_RF_model(CV)
    # grsearch3 = train_GB_model(CV)
    # grsearch4 = train_AdaBoost_model(CV)
    # grsearch5 = train_XGBoost_model(CV)

    num = [grsearch]
    # num =[grsearch1,grsearch2,grsearch3,grsearch4,grsearch5]
    for grsearch in num:
        Data = []
        index = grsearch.best_index_

        results = grsearch.cv_results_
        logger.info(f"Best parameters: {grsearch.best_params_}, Best Estimator: {grsearch.best_estimator_}, "
                    f"Numbers of CV: {grsearch.n_splits_}, CV_R2_Best: {grsearch.best_score_}, CV_R2_mean: {results['mean_test_R2_Score'][index]}, "
                    f"CV_MAE:{results['mean_test_MAE'][index]}, CV_MSE:{results['mean_test_MSE'][index]}"
                    f"CV_MedAE:{results['mean_test_MedAE'][index]}, Index: {index}, "
                    f"CV_R2_Std: {results['std_test_R2_Score'][index]}, CV_MAE_Std: {results['std_test_MAE'][index]}"
                    f"CV_MSE_Std: {results['std_test_MSE'][index]}")

        X_train, X_test, y_train, y_test, A232_prediction_fs = feature_process(Test_size=0.2) #输入实验值
        best = grsearch.best_estimator_

        train_mse = mean_squared_error(y_train, best.predict(X_train))
        train_R2 = r2_score(y_train, best.predict(X_train))
        train_mae = mean_absolute_error(y_train, best.predict(X_train))
        train_medAE = median_absolute_error(y_train, best.predict(X_train))

        y_pred = best.predict(X_test)

        y_pred_experment = best.predict(A232_prediction_fs)  # 预测实验保留时间

        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_medAE = median_absolute_error(y_test, y_pred)
        test_R2 = r2_score(y_test, y_pred)
        logger.info(f"Train MAE: {train_mae},Train MSE:{train_mse},"
                    f" Train RMSE: {np.sqrt(train_mse)},Train R2:{train_R2} "
                    f"Train MedAE: {train_medAE}"
                    f"  Test MAE: {test_mae},Test MSE:{test_mse},"
                    f" Test RMSE: {np.sqrt(test_mse)},Test R2:{test_R2},"
                    f"Test MedAE:{test_medAE},"
                    f"Prediction of Rention Time: {y_pred_experment}")

        data = {'number of CV': [grsearch.n_splits_], 'CV_R2_Best': [grsearch.best_score_],
                'CV_MAE': [results['mean_test_MAE'][index]],
                'CV_MSE': [results['mean_test_MSE'][index]], 'CV_MedAE': [results['mean_test_MedAE'][index]],
                'CV_Mean_test_R2_score': [results['mean_test_R2_Score'][index]],
                'CV_R2_Std': [results['std_test_R2_Score'][index]],
                'CV_MAE_std': [results['std_test_MAE'][index]], 'CV_MSE_std': [results['std_test_MSE'][index]],
                'CV_MedAE_std': [results['std_test_MedAE'][index]], 'Train MAE': [train_mae], 'Train MSE': [train_mse],
                'Train RMSE': [np.sqrt(train_mse)], 'Train MedAE': [train_medAE], 'Train R2': [train_R2],
                'Test MAE': [test_mae], 'Test MSE': [test_mse],
                'Test RMSE': [np.sqrt(test_mse)], 'Test MedAE': [train_medAE], 'Test R2': [test_R2]
                }


def main():
    parser = get_parser()
    args = parser.parse_args()
    logger = loadLogger(args)

    estimate_model(10, logger=logger)

if __name__ == "__main__":
    main()