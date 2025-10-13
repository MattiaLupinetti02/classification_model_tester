import json
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV, HalvingGridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from data_handler import DataHandler
from sklearn.preprocessing import StandardScaler


class CustomBestParamCalculator:
    def __init__(self, models:dict, metrics:dict,label_mapping:dict, cv=5, searcher_class=None):
        self.models = models
        self.metrics = metrics
        self.cv = cv
        self.searcher_class = searcher_class if searcher_class else GridSearchCV
        self.best_params_ = None   
        self.label_mapping = label_mapping
        self.scaler = StandardScaler()
        

    def make_metrics_by_labels(self,avg='binary'):
        scorers = {}
        for name_score, obj_score in self.metrics.items():
            for label, index in self.label_mapping.items():
                avg = "macro"
                if name_score not in ['accuracy']: #è stato inserito questo if che fa creare la metrica su una classe specifica alle metriche che non sono l'accuracy
                    scorer_key = f"{name_score}_{label}"
                    scorers[scorer_key] = make_scorer(custom_scorer(index, obj_score, avg))
        return scorers
    def make_metrics(self,avg='binary'):
        scorers = {}
        for name_score, obj_score in self.metrics.items():
                if name_score not in ['accuracy']:
                    scorers[name_score] = make_scorer(obj_score, average=avg, zero_division=0)
                else:
                    scorers[name_score] = make_scorer(obj_score)
        return scorers

    def best_param_calculator(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        avg: str = 'binary',
        by_target_label: bool | None = None,
        searcher_class = GridSearchCV,
        searcher_kwargs: dict | None = None
    ):
        scorers = {}

        # Decidi il dataset
        
        
        # Configurazione scorers
        if by_target_label:
            scorers = self.make_metrics_by_labels(avg=avg)
        else:
            scorers = self.make_metrics(avg=avg)

        cv_strategy = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        # Parametri extra (se vuoti, inizializza a dict)
        if searcher_kwargs is None:
            searcher_kwargs = {}



     
        # Loop sui modelli
        perf = {}
        for model, params in self.models.items():
            # inizializza dizionario per il modello se non esiste
            model_name = f"{model}".split("(")[0]
            if model_name not in perf:
                perf[model_name] = {}
    
            print(f"Testing model: {model} with {searcher_class.__name__}")

            # Inizializza il "searcher" scelto
            if searcher_class == RandomizedSearchCV:
                searcher = searcher_class(
                    estimator=model,
                    param_distributions=params,
                    scoring=scorers,
                    refit=list(scorers.keys())[0],
                    cv=cv_strategy,
                    verbose=1,
                    **searcher_kwargs
                )

            else:
                searcher = searcher_class(
                    estimator=model,
                    param_grid=params,         
                    scoring='f1' if searcher_class == HalvingGridSearchCV else scorers,
                    refit=list(scorers.keys())[0],
                    cv=cv_strategy,
                    verbose=1,
                    **searcher_kwargs
                )


            # Se è RandomizedSearchCV, serve param_distributions
            if searcher_class == RandomizedSearchCV:
                searcher.param_distributions = params

            # Fit
            searcher.fit(self.scale_data(X), y)

            # Estraggo i risultati
            
            for scorer_name in scorers.keys():
                best_params = searcher.best_params_
                best_score = searcher.cv_results_[f'mean_test_{scorer_name}'][searcher.best_index_]
                print(f"\tBest parameters for {scorer_name}: {best_params}")
                print(f"\tBest score for {scorer_name}: {best_score}")
                perf[model_name][scorer_name] = {
                    json.dumps({
                        k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                        for k, v in best_params.items()
                    }): best_score
                }
        return perf
            
    def scale_data(self, X: pd.DataFrame) -> np.ndarray:
        """Applica StandardScaler alle feature."""
        return self.scaler.fit_transform(X)
def custom_scorer(target_class,metric,avg = 'macro'):
        
        def score_for_class(y_true, y_pred):
            if metric == accuracy_score:
                return metric(y_true, y_pred)
            else:
                return metric(y_true, y_pred, labels=[target_class], average=avg, zero_division=0)
        return score_for_class




