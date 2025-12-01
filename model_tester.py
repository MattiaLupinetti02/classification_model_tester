import pandas as pd
import re
from typing import Dict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV, cross_val_score
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score,f1_score
from functools import partial
import json
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import re
#from sklearn.ensemble import StackingClassifier, StackingRegressor
#from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.pipeline import Pipeline
from .data_handler import DataHandler
from .custom_best_param_calulator import CustomBestParamCalculator
from .visualizer import Visualizer

from typing import Dict, List, Optional

from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.discriminant_analysis import *
try:
    from xgboost import *
except ImportError:
    pass


class ModelTester:
    modelList = None
    metrics = None
    target = None
    #pipelines = []
    performances = dict()
    ensambleModelList = None
    boolean_columns = None 
    def __init__(self, modelList:Dict, metrics:Dict, data:pd.DataFrame, target,ensambleModelList:List = None, resamplingMethods = None,n_jobs = 1):
        if modelList is None:
            print("model list is empty")
        if data is None:
            print("data is none")
            return
        if metrics is None:
            print("metrics list is empty")
        if target not in data.columns:
            print("target column does not exist")
            return
        self.n_jobs = n_jobs
        self.target = target
        self.performances = {}
        self.performances["overall_base_dt"] = {}
        self.performances["specific_base_dt"] = {}
        self.modelList = modelList
        self.metrics = metrics
        self.y = data[target]
        
        
        self.data_handler = DataHandler(data,target)
        data.drop(target, axis=1, inplace=True)
        self.ensambleModelList = ensambleModelList
        if self.modelList is not None and self.ensambleModelList is not None and not any([value is None for value in self.modelList.values()]):
            self.ensamble_hyperpar, self.ensambleModelList = self.initialize_ensamble_models(self.modelList,self.ensambleModelList)
        self.data_handler.encode_target()
        self.data_handler.encode_features()         
        self.data_handler.concat_encoded_dataset()    
        self.visualizer = Visualizer(self.data_handler.encoded_data,self.data_handler.encode_target,self.target,self.data_handler.numeric_features,self.data_handler.categorical_features,self.data_handler.boolean_columns,self.modelList,self.data_handler.resampled_data_dict)
        if resamplingMethods:
            self.initialize_performance(resampling_methods=resamplingMethods)
            self.data_handler.dataResampler(resamplingMethods)

        self.initialize_performance()
        
    

    def initialize_ensamble_models(self,base_models:Dict, ensambleModelList:List): 
            keys = [str(m).split("(")[0] for m in base_models.keys()]
            estimators = [(k,m) for k,m in zip(keys,list(base_models.keys()))]
            ensamble_hyperpar = {}
            #make hyperpar
            for model, hyperpar_list in zip(keys,base_models.values()):
                    for hyperpar_name,values in hyperpar_list.items():
                        ensamble_hyperpar[str(model)+"__" + str(hyperpar_name)] = values
            #set base_models
            for ensamble in ensambleModelList:
                if isinstance(ensamble, (StackingClassifier, StackingRegressor)) or (isinstance(ensamble,(VotingClassifier, VotingRegressor)) and getattr(ensamble,"voting",None)=="soft"):
                    ensamble.set_params(estimators=self.get_soft_voting_ready_models(estimators))
                else:
                    ensamble.set_params(estimators=estimators)   
            return ensamble_hyperpar, ensambleModelList 
    def initialize_performance(self,resampling_methods=None):
        if resampling_methods is not None:
            for res_method in resampling_methods:
                self.performances[f"overall_{res_method}".split('(')[0]+"_dt"] = {}
                self.performances[f"specific_{res_method}".split('(')[0]+"_dt"] = {}
                for m in self.modelList:
                    self.performances[f"overall_{res_method}".split('(')[0]+"_dt"][f'{m}'.split('(')[0]] = {}
                    self.performances[f"specific_{res_method}".split('(')[0]+"_dt"][f'{m}'.split('(')[0]] = {}
                    for name, metric in self.metrics.items():
                        for label, index in self.data_handler.label_mapping.items():    
                            self.performances[f"specific_{res_method}".split('(')[0]+"_dt"][f'{m}'.split('(')[0]][f'{name}_{label}'] = None
                        self.performances[f"overall_{res_method}".split('(')[0]+"_dt"][f'{m}'.split('(')[0]][f'{name}'] = None
        elif self.modelList is not None:
            for m in self.modelList:
                self.performances["specific_base_dt"][f'{m}'.split('(')[0]] = {}
                self.performances["overall_base_dt"][f'{m}'.split('(')[0]] = {}
                for name, metric in self.metrics.items():
                    for label, index in self.data_handler.label_mapping.items():    
                        self.performances["specific_base_dt"][f'{m}'.split('(')[0]][f'{name}_{label}'] = None
                    self.performances["overall_base_dt"][f'{m}'.split('(')[0]][f'{name}'] = None
        return self.performances

    def get_soft_voting_ready_models(self,models):
        """
        Filtra e restituisce i modelli compatibili con soft voting / stacking.
        
        Args:
            models (list): Lista di tuple (nome, modello)
        
        Returns:
            list: Lista di tuple (nome, modello) idonei
        """
        ready_models = []
        for name, model in models:
            if hasattr(model, "predict_proba"):
                ready_models.append((name, model))
            else:
                print(f"{name} does not support voting or the sticking")
        return ready_models



    def scale_data(self,to_scale):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(to_scale)
        return X_scaled
       

    def make_dataframe_performances(self):
        # Creiamo una lista vuota per raccogliere tutte le informazioni da visualizzare

        # aggiungere un if di controllo iniziale per evitare che si inizi a creare un dataframe di valori None
        exp_types = []
        metrics = []
        hyperparams = []
        performances = []
        model_names = []
        
        for exp_type, model_metric_hyperpar_perf in self.performances.items():
            if model_metric_hyperpar_perf is not None:
                for model, metric_hyperpar_perf in model_metric_hyperpar_perf.items():
                    if metric_hyperpar_perf is not None:    
                        for metric, hyperpar_perf in metric_hyperpar_perf.items():
                            if hyperpar_perf is not None:
                                for hyperpar,perf in hyperpar_perf.items():
                                        if perf is not None:
                                            exp_type = re.split(r'\(', f'{exp_type}')[0]
                                            exp_types.append(exp_type)
                                            model_name = re.split(r'\(', f'{model}')[0]  # Estrai solo il nome del modello
                                            model_names.append(model_name)
                                            metrics.append(metric)
                                            hyperparams.append(hyperpar)
                                            performances.append(perf)
           # Creiamo un DataFrame con le informazioni raccolte
        data_to_plot = pd.DataFrame({
            "Experiment":exp_types,
            "Model": model_names,
            "Metric/Class": metrics,
            "Hyperparameters": hyperparams,
            "Performance": performances
        })
        
        return data_to_plot
    def prepare_model_to_test(
        self,
        exp_type,  # aggiungi tipo se lo conosci (es: str)
        dt_type,   # aggiungi tipo se lo conosci (es: str) 
        metric,
        model=None,  # aggiungi tipo se lo conosci (es: Optional[object])
        cv: int = 10,
        performance_dataset: Optional[pd.DataFrame] = None
        ) -> None:  # aggiungi return type
        
        if performance_dataset is not None:
            if performance_dataset.columns.tolist() != ['Experiment', 'Model', 'Metric/Class', 'Hyperparameters', 'Performance']:
                print("Performance dataset columns do not match expected format")
                return
            elif performance_dataset.empty:
                print("Performance dataset is empty")
                return
            
            elif performance_dataset['Performance'].isnull().all():
                print("No calculated performances to implement")
                return
            
            to_implement = performance_dataset
        else:
            to_implement = self.make_dataframe_performances()
        to_implement = to_implement[to_implement['Performance'].notnull()]
        pattern = f"^{exp_type}_.*{dt_type}_dt$"
        if model is None:
            to_implement = to_implement[(to_implement['Metric/Class'] == metric)&(to_implement['Experiment'].str.contains(pattern))]
        else:    
            if not isinstance(model,(List)):
                model = [model]
            model = [re.split(r'\(', f'{m}')[0] for m in model]
            to_implement = to_implement[(to_implement['Metric/Class'] == metric)&(to_implement['Model'].isin(model)) & (to_implement['Experiment'].str.contains(pattern))]
        return to_implement
    def implement_models(
        self,
        cv: int = 10,
        avg: Optional[str] = 'binary',
        metric: Optional[str] = 'accuracy',
        ensamble_by_base_models: bool = False,
        specific: bool = False,
        by_label: bool = False,
        resampled_data: bool = False,
        base_models: Optional[List] = None,
        performance_dataset: Optional[pd.DataFrame] = None
    ) -> None:
        if specific == True:
                exp_type = 'specific'
        else:
            exp_type = 'overall' 
        dataset = {}
        if resampled_data == True:
            dataset = self.data_handler.resampled_data_dict
        else:
            dataset['base'] = self.data_handler.encoded_data
            dataset['base'][self.target] = self.data_handler.y_encoded
        
        for k,data in dataset.items():
            print(f"\t Performance on {k} dataset ")
            model_to_implement = self.prepare_model_to_test(exp_type,k,metric,model=base_models,performance_dataset=performance_dataset)
            if model_to_implement.shape[0] == 0:
                print(f'\t\t\t The performance dataset has not {exp_type} experiments. You passed specific={specific} try with {not specific}')
                return
            self.implement_calculated_models(data,model_to_implement,by_label=by_label,ensamble_by_base_models=ensamble_by_base_models,avg=avg,cv=cv)

    def implement_calculated_models(
        self,
        dataset,
        model_to_implement: pd.DataFrame,
        avg: Optional[str] = 'binary',
        by_label: bool = False,
        ensamble_by_base_models: bool = False,
        cv:int=10        
    ) -> None:
           
        models = [globals()[model_name]() for model_name in model_to_implement['Model']]
        CBPC = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,n_jobs=self.n_jobs)
        if ensamble_by_base_models == True:
            base_models = []
        hyperparameters = model_to_implement['Hyperparameters']
        #metrics = model_to_implement['Metric/Class'].tolist()
        print(model_to_implement)
        for m in models:
            if ensamble_by_base_models:
                print(f"Adding model {type(m).__name__}".split('(')[0]+" to the ensamble base models list")
            else:
                model_name = f"{models}".split('(')[0] if models is not None else model_to_implement['Model']
                print(f"\tCalculating cross val score for model: {model_name}")
            hyperparameters = model_to_implement.loc[model_to_implement['Model']==f'{m}'.split('(')[0],'Hyperparameters']
            metrics = model_to_implement.loc[model_to_implement['Model']==f'{m}'.split('(')[0],'Metric/Class'].tolist()
            print(f"hyperparameters of model {type(m).__name__}".split('(')[0]+f" \n {hyperparameters}")
            i = 0                    
            for hp in hyperparameters:
                print(f'\t\t\t\t Optimized for the metric {metrics[i]}')
                hp_dict = json.loads(hp.replace("'", "\""))
                m.set_params(**hp_dict)
                if ensamble_by_base_models == True:
                    base_models.append((f'{m}'.split('(')[0],m))
                else:
                    CBPC.validation_model_CV(m, dataset.drop(self.target,axis=1), dataset[self.target],avg=avg,by_label=by_label,cv=cv)
                i= i +1
        if ensamble_by_base_models == True:
            self.implement_ensemble_by_base_models(base_models,dataset,CBPC,by_label=by_label,avg=avg,cv=cv)
            base_models = []

    def implement_ensemble_by_base_models(
        self,
        base_models: List,
        dataset,
        model_validator: CustomBestParamCalculator,
        by_label: bool = False,
        avg: Optional[str] = 'binary',
        cv:int=10
    ) -> None:
        for ens_m in self.ensambleModelList:
            ens_m.set_params(estimators=base_models)
            model_validator.validation_model_CV(ens_m, dataset.drop(self.target,axis=1), dataset[self.target],avg=avg,by_label=by_label,cv=cv)
        
    def best_param_calculator(self,cv = 10, avg='macro',searcher_class=GridSearchCV):
        CBPC = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,searcher_class=searcher_class,n_jobs=self.n_jobs)
        perf = CBPC.best_param_calculator(self.data_handler.encoded_data, self.data_handler.y_encoded,avg,searcher_class=searcher_class)
        
        for model,p in perf.items():
            self.performances['overall_base_dt'][f'{model}'] = p

    def best_param_calculator_by_label(self,cv = 10, avg='macro',searcher_class=GridSearchCV):
        
        CBPC_bylabel = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,searcher_class=searcher_class,n_jobs=self.n_jobs)
        perf = CBPC_bylabel.best_param_calculator(self.data_handler.encoded_data, self.data_handler.y_encoded,avg,by_target_label=True,searcher_class=searcher_class)
        
        for model,p in perf.items():
            self.performances['specific_base_dt'] = perf
        
        #self.best_param_calculator(cv,avg,True,searcher_class=searcher_class)

    def best_param_calculator_ensamble_by_label(self,cv=10, avg='macro', searcher_class=GridSearchCV):
        if self.ensambleModelList is None:
            print("No Ensamble Models")
            return
        copy_models = self.modelList.copy()
        self.modelList = {ens:None for ens in self.ensambleModelList}
        
        for k in self.modelList.keys():
            self.modelList[k] = self.ensamble_hyperpar
        
        CBPC_ensamblebylabel = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,searcher_class=searcher_class,n_jobs=self.n_jobs)
        perf = CBPC_ensamblebylabel.best_param_calculator(self.data_handler.encoded_data,self.data_handler.y_encoded,avg,by_target_label=True,searcher_class=searcher_class)
        
        self.performances['specific_base_dt'] = perf
       
        self.modelList = copy_models

    def best_param_calculator_ensamble(self,avg='macro',cv=10,searcher_class=GridSearchCV):
        if self.ensambleModelList is None:
            print("No Ensamble Models")
            return 
        copy_models = self.modelList.copy()
        self.modelList = {ens:None for ens in self.ensambleModelList}

        for k in self.modelList.keys():
            self.modelList[k] = self.ensamble_hyperpar
        
        CBPC_ensamble = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,searcher_class=searcher_class,n_jobs=self.n_jobs)
        perf = CBPC_ensamble.best_param_calculator(self.data_handler.encoded_data,self.data_handler.y_encoded, avg, by_target_label=True,searcher_class=searcher_class)
        for model,p in perf.items():
            self.performances['overall_base_dt'][f'{model}'] = p
        
        #self.best_param_calculator(cv,'macro',searcher_class=searcher_class)
        self.modelList = copy_models

    def best_param_calculator_ensamble_from_augmented_data(self,avg='macro',cv=10,searcher_class=GridSearchCV):
        if self.data_handler.encoded_data is None or self.data_handler.y_encoded is None:
            print("No augmented data")
            return 
        if self.ensambleModelList is None:
            print("No Ensamble Models")
            return 
        copy_models = self.modelList.copy()
        self.modelList = {ens:None for ens in self.ensambleModelList}
        for k in self.modelList.keys():
            self.modelList[k] = self.ensamble_hyperpar
        CBPC_ensamble_augmented_data = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,searcher_class=searcher_class,n_jobs=self.n_jobs)
        
        for m in self.data_handler.resampled_data_dict:
                print(f"for augmented data with {m}")
                X = self.data_handler.resampled_data_dict[m].drop(self.target,axis=1)
                y = self.data_handler.resampled_data_dict[m][self.target]
                perf = CBPC_ensamble_augmented_data.best_param_calculator(X,y,avg,searcher_class=searcher_class)
                for model,p in perf.items():
                    self.performances[f'overall_{m}_dt'][f'{model}'] = p
                #self.best_param_calculator(cv,"macro",resampled_data_x=data_x,searcher_class=searcher_class, resampled_data_y=data_y,resampler=m)
        self.modelList = copy_models


    def best_param_calculator_ensamble_from_augmented_data_by_label(self,cv=10,avg='macro', searcher_class=GridSearchCV):
        if self.data_handler.encoded_data is None or self.data_handler.y_encoded is None:
            print("No augmented data")
            return 
        if self.ensambleModelList is None:
            print("No Ensamble Models")
            return 
        copy_models = self.modelList.copy()
        self.modelList = {ens:None for ens in self.ensambleModelList}
        for k in self.modelList.keys():
            self.modelList[k] = self.ensamble_hyperpar
        CBPC_ensamble_augmented_data_bylabel = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,searcher_class=searcher_class,n_jobs=self.n_jobs)
        for m in self.data_handler.resampled_data_dict:
                print(f"for augmented data with {m}")
                #print(data.drop(self.target,axis=1))
                #print(data[self.target])
                X = self.data_handler.resampled_data_dict[m].drop(self.target,axis=1)
                y = self.data_handler.resampled_data_dict[m][self.target]
                perf = CBPC_ensamble_augmented_data_bylabel.best_param_calculator(X,y,avg,by_target_label=True,searcher_class=searcher_class)
                for model,p in perf.items():
                    self.performances[f'specific_{m}_dt'][f'{model}'] = p
        self.modelList = copy_models
    
    def best_param_calculator_from_augmented_data(self,cv = 10,avg='macro',searcher_class=GridSearchCV):
        if self.data_handler.encoded_data is None or self.data_handler.y_encoded is None:
            return "No augmented data"
        CBPC_augmented_data = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,searcher_class=searcher_class,n_jobs=self.n_jobs)        
        for m in self.data_handler.resampled_data_dict:
                print(f"for augmented data with {m}")
                X = self.data_handler.resampled_data_dict[m].drop(self.target,axis=1)
                y = self.data_handler.resampled_data_dict[m][self.target]
                perf = CBPC_augmented_data.best_param_calculator(X,y,avg,searcher_class=searcher_class)
                for model,p in perf.items():
                    self.performances[f'overall_{m}_dt'][f'{model}'] = p
    def best_param_calculator_from_augmented_data_by_label(self,cv = 10,avg='macro',searcher_class=GridSearchCV):
        if self.data_handler.encoded_data is None or self.data_handler.y_encoded is None:
            return "No augmented data"
        CBPC_augmented_data_bylabel = CustomBestParamCalculator(self.modelList,self.metrics,self.data_handler.get_label_mapping(),cv=cv,searcher_class=searcher_class,n_jobs=self.n_jobs)        

        for m in self.data_handler.resampled_data_dict:
            X = self.data_handler.resampled_data_dict[m].drop(self.target,axis=1)
            y = self.data_handler.resampled_data_dict[m][self.target]
            perf = CBPC_augmented_data_bylabel.best_param_calculator(X,y,avg,by_target_label=True,searcher_class=searcher_class)
            for model,p in perf.items():
                    self.performances[f'specific_{m}_dt'][f'{model}'] = p

    def plot_pie_chart(self, resampled= True, save=False):
        if resampled and self.data_handler.resampled_data_dict:
            for method, data in self.data_handler.resampled_data_dict.items():
                print(f"Plotting pie chart for resampled data with method: {method}")
                y_resampled = data[self.target]
                self.visualizer.plot_pie_chart(y_resampled, self.target, save=save)
        else:
            self.visualizer.plot_pie_chart(self.y, self.target, save=save)
    
    def plot_boxplots(self, resampled= False, save=False):
        if resampled and self.data_handler.resampled_data_dict:
            for method, data in self.data_handler.resampled_data_dict.items():
                print(f"Plotting boxplots for resampled data with method: {method}")
                self.visualizer.plot_boxplots(data, save=save)
        else:
            print("Plotting boxplots for original data")
            self.visualizer.plot_boxplots(self.data_handler.encoded_data, save=save)
    
    def plot_numeric_distribution(self, resampled= True, save=False):
        if resampled and self.data_handler.resampled_data_dict:
            for method, data in self.data_handler.resampled_data_dict.items():
                print(f"Plotting numeric distribution for resampled data with method: {method}")
                self.visualizer.plot_numeric_distribution(self.data_handler.original_data, save=save)
        else:
            self.visualizer.plot_numeric_distribution(self.data_handler.original_data, save=save)
        
    def plot_correlation_matrix(self, resampled= True, save=False):
        if resampled and self.data_handler.resampled_data_dict:
            for method, data in self.data_handler.resampled_data_dict.items():
                print(f"Plotting correlation matrix for resampled data with method: {method}")
                self.visualizer.plot_correlation_matrix(self.data_handler.original_data, save=save)
        else:
            self.visualizer.plot_correlation_matrix(self.data_handler.original_data, save=save)
    
    def plot_binary_distribution(self, resampled= False, save=False):
        if resampled and self.data_handler.resampled_data_dict:
            for method, data in self.data_handler.resampled_data_dict.items():
                print(f"Plotting binary distribution for resampled data with method: {method}")
                self.visualizer.plot_binary_distribution(data, save=save)
        else:
            self.visualizer.plot_binary_distribution(self.data_handler.original_data, save=save)


    def learning_curves(self,resampled = False):
        for model,params in self.modelList.items():
                # Disegna le learing curve
            
            if resampled:
                for method, data in self.data_handler.resampled_data_dict.items():
                    print(f"Learning curve for model: {model} with {method}")
                    self.visualizer.plot_learning_curve(model, data.drop(self.target,axis = 1), data[self.target],name_method=method)
            else:
                self.visualizer.plot_learning_curve(model, self.data_handler.encoded_data, self.data_handler.y_encoded)

    def validation_curves(self, cv=5, resampled=False,save=False):
        for model, params in self.modelList.items():
            model_name = re.split(r'\(', f'{model}')[0]  # Estrai solo il nome del modello
            
            # Determiniamo il numero totale di grafici per il modello corrente
            n = len(params)  # Numero di parametri del modello
            if n == 0:
                print(f"Model {model_name} has no parameters to validate.")
                continue  # Salta questo modello
    
            cols = 2  # Numero di colonne
            rows = (n + cols - 1) // cols  # Calcola il numero di righe
            
            # Creiamo una figura per il modello corrente
            fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
            axes = axes.flatten()  # Flattiamo per accedere facilmente agli assi
            
            plot_index = 0
            for param_name, param_values in params.items():
                print(f"Validation curve for model: {model_name} with parameter: {param_name}")
                
                # Imposta il titolo dell'asse corrente
                axes[plot_index].set_title(model_name)  
    
                # Passa l'asse corrente alla funzione per disegnare il grafico
                if resampled == False:
                    self.visualizer.plot_validation_curve(
                        model, self.scale_data(self.data_handler.encoded_data), self.data_handler.y_encoded,
                        param_name, param_values, cv, ax=axes[plot_index])
                else:
                    for method,data in self.data_handler.resampled_data_dict.items():
                        self.visualizer.plot_validation_curve(
                            model, self.scale_data(data.drop(self.target,axis=1)), data[self.target],
                            param_name, param_values, cv, ax=axes[plot_index],dt_name=method)
                
                plot_index += 1
    
            # Rimuovi eventuali assi vuoti nella griglia
            for i in range(plot_index, len(axes)):
                fig.delaxes(axes[i])
    
            # Salva l'immagine per il modello corrente
            if resampled and save == True:
                for method in self.data_handler.resampled_data_dict.keys():
                    resampling_method_name = method.split('(')[0]
                    
                    fig.savefig(f'validation_curves_{model_name}_on_augmented_data_{resampling_method_name}.png', dpi=300, bbox_inches='tight')
            elif save == True:
                fig.savefig(f'validation_curves_{model_name}.png', dpi=300, bbox_inches='tight')
            
                
            print(f"Saved validation curves for {model_name} as validation_curves_{model_name}.png")
            
            plt.tight_layout()  # Migliora la spaziatura
            plt.show()  # Mostra la figura
            plt.close(fig)  # Chiudi la figura per liberare memoria

