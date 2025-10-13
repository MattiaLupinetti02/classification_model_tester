import pandas as pd
import re
from typing import Dict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV, cross_val_score
from typing import Dict, List
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score,f1_score
from functools import partial
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import re
import seaborn as sns
from classification_model_tester import ModelTester
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier


dt = pd.read_csv("../dati_dott_fedele/dati_m2_incolonnati_v2.csv")

models = {}

models = {
    XGBClassifier(eval_metric="logloss"): { 
        'n_estimators': [50,75],
        'max_depth': [5,6],  
        'learning_rate': [ 0.3,0.4],
        'subsample': [0.8,0.9],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_lambda': [0, 0.1, 1, 10],   # L2 regularization (lambda)
        'reg_alpha': [0, 0.1, 1, 10],    # L1 regularization (alpha)
        'gamma': [0, 0.1, 0.5, 1]   
    },    
   

    # GradientBoostingClassifier
    GradientBoostingClassifier(): { 
        'n_estimators': [50, 100, 200],
        'n_estimators': [25,50],
        'max_depth': [8,10], 
        'max_depth': [3, 5, 10], 
        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],
        'learning_rate': [ 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'subsample': [0.7],
        'criterion': ['friedman_mse'],  
        'criterion': ['friedman_mse', 'mse'],
        'min_samples_split': [2, 5, 10],  
        'min_samples_split': [2,3],  
        'min_samples_leaf': [1, 2, 4,5],
        'min_samples_leaf': [ 4,5]
    },

    # LogisticRegression
    LogisticRegression(max_iter=1000000): { 
        'penalty': ['none', 'l1', 'l2', 'elasticnet'],
        'C': np.logspace(-3, 9, 7),
        'max_iter': np.arange(50, 100, 10),
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
        'tol': np.logspace(-5, -1, 5),
        'multi_class': ['ovr', 'multinomial']
    },

    # SVC
    SVC(): { 
        'kernel': ['rbf', 'poly', 'linear'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4, 5]  
    },

    # DecisionTreeClassifier
    DecisionTreeClassifier(): { 
        'max_depth': [5, 10, 15], 
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },

    # RandomForestClassifier
    RandomForestClassifier(n_estimators=100): { 
        'n_estimators': [50, 100, 200], 
        'max_depth': [5, 10, 20], 
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },

    # ExtraTreesClassifier
    ExtraTreesClassifier(n_estimators=100): { 
        'n_estimators': [50, 100, 200], 
        'max_depth': [5, 10, 20], 
        'min_samples_split': [2, 5, 10], 
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
}
resampling_methods = [
    SMOTE(random_state=42, sampling_strategy='auto'),
    ADASYN(random_state=42, sampling_strategy='minority')
]

metrics = {"accuracy":accuracy_score,"precision":precision_score,"recall":recall_score,"f1-score":f1_score}
tester = ModelTester(models,metrics,dt.copy(),"alert",ensambleModelList=[StackingClassifier(
        estimators=None),VotingClassifier(estimators=None)],resamplingMethods=resampling_methods)


tester.best_param_calculator(cv=10,avg="macro")
tester.best_param_calculator_by_label(cv=10)
tester.best_param_calculator_from_augmented_data(cv=10)
tester.best_param_calculator_from_augmented_data_by_label(cv=10)
tester.best_param_calculator_ensamble(cv=10)
tester.best_param_calculator_ensamble_by_label(cv=10)
tester.best_param_calculator_ensamble_from_augmented_data(cv=10)
tester.best_param_calculator_ensamble_from_augmented_data_by_label(cv=10)
tester.make_dataframe_performances(tester.performances).to_csv("performances_ensamble_dt2.csv",index=False)