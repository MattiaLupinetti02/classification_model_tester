import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score,f1_score, classification_report
warnings.filterwarnings('ignore')
import re
import json 

class ModelFactory:
    """Factory per creare modelli con iperparametri specifici"""
    
    @staticmethod
    def create_model(model_obj, hyperparams):
        """
        Crea un modello sklearn con gli iperparametri specificati
        """
        
        
            
        return model_obj.set_params(**hyperparams)

class DataFrameParser:
    """Parser per estrarre informazioni dal dataframe"""
    
    @staticmethod
    def parse_hyperparameters(hyperparams_str):
        """
        Converte la stringa di iperparametri in dict
        """
        try:
            # Rimuovi spazi e converti in dict
            hyperparams_str = hyperparams_str.replace('""', '"')
            return ast.literal_eval(hyperparams_str)
        except:
            raise ValueError(f"Impossibile parsare gli iperparametri: {hyperparams_str}")
    
    @staticmethod
    def extract_model_info(df, target_models, target_metric, exp_type = 'overall_base_dt'):
        """
        Estrae informazioni sui modelli dal dataframe
        """
        model_info = {}
        
        for model_name in target_models:
            # Filtra per modello e metrica
            model_data = df[
                (df['Model'] == re.split(r'\(', f'{model_name}')[0]) & 
                (df['Metric/Class'] == target_metric) &
                (df['Experiment'] == exp_type)
            ]
            
            if model_data.empty:
                print(f"Warning: Nessun dato trovato per {model_name} con metrica {target_metric}")
                continue

            # Prendi la prima occorrenza (dovrebbero essere tutte uguali per lo stesso modello)
            #hyperparams = DataFrameParser.parse_hyperparameters(model_data['Hyperparameters'].to_string(index=False) )
            hyperparams = model_data['Hyperparameters'].iloc[0]
            model_info[model_name] = {
                'hyperparameters': json.loads(hyperparams.replace("'", "\"")),
                'experiment': model_data['Experiment'],
                'performance': model_data['Performance']
            }
            
        return model_info

class EnsembleOptimizer:
    """Classe per ottimizzare ensemble models"""
    
    def __init__(self, scoring_metric='accuracy', cv_folds=10, random_state=42):
        self.scoring_metric = scoring_metric
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_ensemble_ = None
        self.best_score_ = None
        self.best_params_ = None
    
    def create_base_models(self, model_info):
        """
        Crea i modelli base per l'ensemble
        """
        base_models = []
        for model_obj, info in model_info.items():
            #try:
                # Estrai il nome della classe dal model_name
                # Assumendo che model_name sia il nome della classe
                
            hyperparams = info['hyperparameters']
            #print(type(hyperparams))
            #hyperparams = json.loads(hyperparams.replace("'", "\""))
            model = ModelFactory.create_model(model_obj, hyperparams)
            base_models.append((re.split(r'\(', f'{model_obj}')[0], model))
            """   
            except Exception as e:
                print(f"Errore nella creazione del modello {re.split(r'\(', f'{model_obj}')[0]}: {e}")
                continue
            """    
        return base_models
    
    def create_ensemble_model(self, ensemble_type, base_models, final_estimator=None, **kwargs):
        """
        Crea un modello ensemble del tipo specificato
        """
        if ensemble_type == 'voting':
            return VotingClassifier(estimators=base_models, voting='soft', **kwargs)
        elif ensemble_type == 'stacking':
            if final_estimator is None:
                final_estimator = LogisticRegression(random_state=self.random_state)
            return StackingClassifier(
                estimators=base_models, 
                final_estimator=final_estimator,
                **kwargs
            )
        else:
            raise ValueError("Tipo ensemble non supportato. Usa 'voting' o 'stacking'")
    
    def optimize_ensemble(self, ensemble_type, base_models, X, y, param_grid,scorers):
        """
        Ottimizza i parametri dell'ensemble
        """
        # Crea grid search parameters per ensemble
        search_params = {}
        
        for param_name, param_values in param_grid.items():
            if param_name == 'final_estimator':
                # Per stacking classifier
                if ensemble_type == 'stacking':
                    search_params['final_estimator'] = param_values
            else:
                search_params[param_name] = param_values
        
        # Crea ensemble base
        base_ensemble = self.create_ensemble_model(ensemble_type, base_models)
        
        # Grid Search
        grid_search = GridSearchCV(
            base_ensemble,
            search_params,
            scoring=scorers,
            cv=10,
            refit = 'f1-score_YELLOW',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        self.best_ensemble_ = grid_search.best_estimator_
        self.best_score_ = grid_search.best_score_
        self.best_params_ = grid_search.best_params_
        
        return self.best_ensemble_

def modular_ensemble_optimization(
    df, 
    target_models, 
    target_metric, 
    ensemble_type, 
    ensemble_param_grid, 
    X, y,
    scoring_metric='accuracy',
    cv=10,
    exp_type = 'overall_base_dt',
    scorers = [make_scorer(f1_score, average='macro')]
):
    """
    Funzione principale completamente modulare per l'ottimizzazione di ensemble models
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame con i risultati dei modelli
    target_models : list
        Lista di oggetti di modelli da includere nell'ensemble
    target_metric : str
        Metrica su cui sono ottimizzati i modelli base
    ensemble_type : str
        Tipo di ensemble ('voting' o 'stacking')
    ensemble_param_grid : dict
        Grid di parametri per l'ensemble
    X, y : array-like
        Dati di training
    scoring_metric : str
        Metrica per l'ottimizzazione dell'ensemble
    cv_folds : int
        Numero di fold per cross-validation
    
    Returns:
    --------
    best_ensemble : sklearn estimator
        Migliore ensemble ottimizzato
    optimization_results : dict
        Risultati dell'ottimizzazione
    """
    
    # Step 1: Parsing del dataframe
    print("Step 1: Estrazione informazioni modelli base...")
    model_info = DataFrameParser.extract_model_info(df, target_models, target_metric,exp_type=exp_type)
    if not model_info:
        raise ValueError("Nessun modello valido trovato nel dataframe")
    
    print(f"Trovati {len(model_info)} modelli validi")
    
    # Step 2: Creazione modelli base
    print("Step 2: Creazione modelli base...")
    base_models = EnsembleOptimizer().create_base_models(model_info)
    
    if not base_models:
        raise ValueError("Nessun modello base creato correttamente")
    
    print(f"Creati {len(base_models)} modelli base")
    
    # Step 3: Ottimizzazione ensemble
    print("Step 3: Ottimizzazione ensemble...")
    optimizer = EnsembleOptimizer(
        scoring_metric=scoring_metric, 
        cv_folds=cv
    )
    
    best_ensemble = optimizer.optimize_ensemble(
        ensemble_type, 
        base_models, 
        X, y, 
        ensemble_param_grid,
        scorers
    )
    
    # Step 4: Risultati
    optimization_results = {
        'best_score': optimizer.best_score_,
        'best_params': optimizer.best_params_,
        'base_models_used': [name for name, _ in base_models],
        'model_info': model_info
    }
    
    print(f"Ottimizzazione completata. Best score: {optimizer.best_score_:.4f}")
    
    return best_ensemble, optimization_results

