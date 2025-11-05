import json
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
import pandas as pd
from .data_handler import DataHandler
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import BaseSampler
from optuna.trial import TrialState
from optuna.distributions import CategoricalDistribution
import itertools
from typing import Dict, Any, Optional
from .create_optuna_study import create_optuna_study

class MemoryEfficientGridSampler(BaseSampler):
    """Grid search deterministica memory-efficient per Optuna."""
    
    def __init__(self, search_space: Dict[str, list]):
        self.search_space = search_space
        self._param_iter = None
        self._search_space = self._convert_to_optuna_space(search_space)
        
    def _convert_to_optuna_space(self, search_space: Dict[str, list]):
        """Converte lo spazio di ricerca in formato Optuna."""
        return {
            param_name: CategoricalDistribution(values)
            for param_name, values in search_space.items()
        }
        
    def reseed_rng(self) -> None:
        self._param_iter = None
        
    def infer_relative_search_space(self, study, trial):
        return self._search_space
        
    def sample_relative(self, study, trial, search_space):
        """Restituisce una combinazione alla volta"""
        if self._param_iter is None:
            self._param_iter = self._generate_param_combinations()
        try:
            return next(self._param_iter)
        except StopIteration:
            return None  # ✅ Importante: None segnala fine ricerca
            
    def sample_independent(self, study, trial, param_name, param_distribution):
        # Non usato nel grid: restituisce il primo valore per compatibilità
        return param_distribution.choices[0]
            
    def _generate_param_combinations(self):
        """Generatore che produce combinazioni nativamente tipizzate"""
        keys = list(self.search_space.keys())
        value_lists = [self.search_space[key] for key in keys]
        for values in itertools.product(*value_lists):
            params = {
                k: (bool(v) if isinstance(v, np.bool_) else
                    int(v) if isinstance(v, np.integer) else
                    float(v) if isinstance(v, np.floating) else v)
                for k, v in zip(keys, values)
            }
            yield params


class CustomBestParamCalculator:
    def __init__(self, models:dict, metrics:dict, label_mapping:dict, cv=5, searcher_class=None):
        self.models = models
        self.metrics = metrics
        self.cv = cv
        self.searcher_class = searcher_class if searcher_class else GridSearchCV
        self.best_params_ = None   
        self.label_mapping = label_mapping
        self.scaler = StandardScaler()
        
    def make_metrics_by_labels(self, avg='binary'):
        scorers = {}
        for name_score, obj_score in self.metrics.items():
            for label, index in self.label_mapping.items():
                avg = "macro"
                if name_score not in ['accuracy']:
                    scorer_key = f"{name_score}_{label}"
                    scorers[scorer_key] = make_scorer(custom_scorer(index, obj_score, avg))
        return scorers
    
    def make_metrics(self, avg='binary'):
        scorers = {}
        for name_score, obj_score in self.metrics.items():
            if name_score not in ['accuracy']:
                scorers[name_score] = make_scorer(obj_score, average=avg, zero_division=0)
            else:
                scorers[name_score] = make_scorer(obj_score)
        return scorers

    def calculate_total_combinations(self, param_grid):
        """Calcola il numero totale senza generare combinazioni"""
        total = 1
        for param_name, param_values in param_grid.items():
            if hasattr(param_values, '__len__') and not hasattr(param_values, 'rvs'):
                total *= len(param_values)
            else:
                # Per distribuzioni, stima 10 valori
                total *= 10
        return total

    def convert_to_native_types(self,param_grid):
        """Converte tutti i valori numpy a Python native types"""
        converted_grid = {}
        
        for param_name, values in param_grid.items():
            if hasattr(values, 'tolist'):  # Se è array numpy
                converted_grid[param_name] = values.tolist()
            elif isinstance(values, (list, tuple)):
                # Converte ogni elemento nell'array
                converted_list = []
                for value in values:
                    if hasattr(value, 'item'):  # numpy scalar
                        converted_list.append(value.item())
                    else:
                        converted_list.append(value)
                converted_grid[param_name] = converted_list
            else:
                converted_grid[param_name] = values
        
        return converted_grid

    def create_optuna_objective(self, model, X, y, scorer, param_grid=None):
        """
        Crea una funzione objective che riceve lo scorer già pronto.
        Se viene passato un param_grid, Optuna genera i parametri in base ai valori presenti.
        """
        def objective(trial):
            try:
                # 🔹 Se non viene passato param_grid, usa direttamente i parametri del trial
                if param_grid is None:
                    params = trial.params
                else:
                    params = {}
                    for name, values in param_grid.items():
                        # Rimuovi valori None o non validi
                        if values is None or len(values) == 0:
                            continue

                        first_val = values[0]

                        # Se i valori sono numerici, scegli un intervallo continuo
                        if all(isinstance(v, (int, np.integer)) for v in values) and not all(isinstance(v, bool) for v in values):
                            low, high = int(min(values)), int(max(values))
                            params[name] = trial.suggest_int(name, low, high)

                        elif all(isinstance(v, (float, np.floating)) for v in values):
                            low, high = float(min(values)), float(max(values))
                            params[name] = trial.suggest_float(name, low, high)

                        else:
                            params[name] = trial.suggest_categorical(name, values)




                print(f"Testing parameters: {params}")
                # 🔹 Clona il modello per evitare side effects
                current_model = clone(model)
                current_model.set_params(**params)

                # 🔹 Cross-validation con lo scorer fornito
                scores = cross_val_score(
                    current_model, X, y,
                    cv=self.cv,
                    scoring=scorer
                )

                score = float(np.mean(scores))
                print(f"Score: {score:.4f} for params: {params}")

                return score

            except Exception as e:
                print(f"Error in trial with params {params}: {e}")
                return -1.0

        return objective


    def best_param_calculator(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        avg: str = 'binary',
        by_target_label: bool | None = None,
        searcher_class = GridSearchCV,
        searcher_kwargs: dict | None = None,
        brute_force: bool = False
    ):
        if by_target_label:
            self.scorers = self.make_metrics_by_labels(avg=avg)
        else:
            self.scorers = self.make_metrics(avg=avg)

        if searcher_kwargs is None:
            searcher_kwargs = {}

        perf = {}
        
        for model, param_grid in self.models.items():
            model_name = f"{model}".split("(")[0]
            if model_name not in perf:
                perf[model_name] = {}
    
            print(f"Testing model: {model} with {searcher_class.__name__}")

            # SE E' OPTUNA
                    # SE E' OPTUNA
            if searcher_class.__name__ == 'OptunaStudy':
                param_grid = self.convert_to_native_types(param_grid)
                
                # Calcola il numero totale di combinazioni
                total_trials = self.calculate_total_combinations(param_grid)
                print(f"  Total combinations to test: {total_trials}")
                
                for scorer_name in self.scorers.keys():
                    print(f"  Optimizing for scorer: {scorer_name}")
                    
                    scorer = self.scorers[scorer_name]  # Prendi lo scorer dal dizionario
                    
                    objective_func = self.create_optuna_objective(
                        model,
                        self.scale_data(X),
                        y,
                        scorer,
                        param_grid=param_grid 
                    )

                    # Filtra solo i parametri che sono liste (per grid search vero)
                    grid_params = {
                        k: v for k, v in param_grid.items() 
                        if hasattr(v, '__len__') and not hasattr(v, 'rvs')
                    }
                    
                    if brute_force == True:
                        print("Running brute-force grid search with MemoryEfficientGridSampler")
                        sampler = MemoryEfficientGridSampler(grid_params)
                        n_trials = total_trials
                        study = optuna.create_study(direction="maximize", sampler=sampler)
                    else:
                        print(" Running TPE-based optimization with pruning")
                        study = create_optuna_study(
                            direction="maximize",
                            storage_path="optuna_results.db",
                            sampler_type="tpe",
                            seed=42
                        )
                        n_trials = 100 
                    
                    
                    try:
                        study.optimize(
                            objective_func,
                            n_trials=n_trials,
                            callbacks=[save_best_callback],
                            show_progress_bar=True
                        )
                        
                        best_params = study.best_params
                        best_score = study.best_value
                        
                        print(f"\tBest parameters for {scorer_name}: {best_params}")
                        print(f"\tBest score for {scorer_name}: {best_score}")
                        
                        perf[model_name][scorer_name] = {
                            json.dumps({
                                k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                                for k, v in best_params.items()
                            }): best_score
                        }
                        
                    except Exception as e:
                        print(f" Error during optimization: {e}")
                        perf[model_name][scorer_name] = {json.dumps({}): -1.0}
                
                # SE E' SCIKIT-LEARN (mantieni compatibilità)
            else:
                cv_strategy = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

                if searcher_class == RandomizedSearchCV:
                    searcher = searcher_class(
                        estimator=model,
                        param_distributions=param_grid,
                        scoring=self.scorers,
                        refit=list(self.scorers.keys())[0],
                        cv=cv_strategy,
                        verbose=1,
                        **searcher_kwargs
                    )
                else:
                    searcher = searcher_class(
                        estimator=model,
                        param_grid=param_grid,         
                        scoring='f1' if searcher_class == HalvingGridSearchCV else self.scorers,
                        refit=list(self.scorers.keys())[0],
                        cv=cv_strategy,
                        verbose=1,
                        **searcher_kwargs
                    )

                    if searcher_class == RandomizedSearchCV:
                        searcher.param_distributions = param_grid

                    searcher.fit(self.scale_data(X), y)
                    best_params = searcher.best_params_

                for scorer_name in self.scorers.keys():
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
        return self.scaler.fit_transform(X)

class OptunaStudy:
    def __init__(self, *args, **kwargs):
        pass

def custom_scorer(target_class, metric, avg='macro'):
    def score_for_class(y_true, y_pred):
        if metric == accuracy_score:
            return metric(y_true, y_pred)
        else:
            return metric(y_true, y_pred, labels=[target_class], average=avg, zero_division=0)
    return score_for_class

def save_best_callback(study, trial):
    if study.best_trial.number == trial.number:
        print(f"💾 Saving best trial #{trial.number} with value {trial.value:.4f}")
        with open("best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=4)
