from model_tester import ModelTester

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
from data_handler import DataHandler
#from classification_model_tester import ModelTester
import seaborn as sns
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier
 

dt = pd.read_csv("../dati_dott_fedele/dati_m2_incolonnati.csv")

#data_handler = DataHandler(dt,"alert")
#print(data_handler.encode_features())
#print(data_handler.encode_target())
#print(data_handler.concat_encoded_dataset(data_handler.mapping,data_handler.categorical_features,data_handler.encoded_data))


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

models = {

        # XGBClassifier
     XGBClassifier(eval_metric="logloss"): { 
        'n_estimators': [50,75],
        'max_depth': [5,6],  
    }    
   
}

from imblearn.over_sampling import SMOTE, ADASYN

resampling_methods = [
    SMOTE(random_state=42, sampling_strategy='auto'),
    ADASYN(random_state=42, sampling_strategy='minority')
]

metrics = {"accuracy":accuracy_score,"precision":precision_score,"recall":recall_score,"f1-score":f1_score}
tester = ModelTester(models,metrics,dt.copy(),"alert",resamplingMethods = resampling_methods ,ensambleModelList=[StackingClassifier(
        estimators=None)])
#print(tester.data_handler.resampled_data_dict)
tester.best_param_calculator_ensamble(cv=2)
tester.best_param_calculator_ensamble_by_label(cv=2)
tester.best_param_calculator_ensamble_from_augmented_data(cv=2)
tester.best_param_calculator_ensamble_from_augmented_data_by_label(cv=2)
tester.best_param_calculator_by_label(cv=2)
print(tester.make_dataframe_performances())
#print(tester.data_handler.encoded_data)

