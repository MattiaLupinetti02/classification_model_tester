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
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.pipeline import Pipeline
class ModelTester:
    modelList = None
    metrics = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    y = None
    original_data = None
    numeric_features = None
    categorical_features = None
    target = None
    encoded_data = None
    y_encoded = None
    df_overall_performance = None
    df_specific_performance = None
    pipelines = []
    performances = dict()
    resampled_data = None
    boolean_columns = None 
    def __init__(self, modelList:Dict, metrics:Dict, data:pd.DataFrame, target,ensambleModelList:List = None, resamplingMethods = None):
        if modelList is None:
            print("model list is empty")
        if data is None:
            print("data is none")
            return
        if metrics is None:
            print("metrics list is empty")
    
        self.performances = {}
        self.performances["overall"] = {}
        self.performances["specific"] = {}
        self.modelList = modelList
        self.metrics = metrics
        self.original_data = data
        #print(self.original_data["6MWT"].unique())
        self.categorical_features = {i: col for i, col in enumerate(data.columns) if col != target and data[col].apply(lambda x: isinstance(x, (str))).all()}
        self.boolean_columns = data.select_dtypes(include=bool)
        if target not in data.columns:
            print("target column does not exist")
            return
        if self.modelList is not None and ensambleModelList is not None:
            self.ensambleModelList = ensambleModelList
            keys = [str(m).split("(")[0] for m in self.modelList.keys()]
            #self.estimators = dict(zip(keys,list(self.modelList.keys())))
            self.estimators = [(k,m) for k,m in zip(keys,list(self.modelList.keys()))]
            self.ensamble_hyperpar = {}
            for model, hyperpar_list in zip(keys,self.modelList.values()):
                    for hyperpar_name,values in hyperpar_list.items():
                        self.ensamble_hyperpar[str(model)+"__" + str(hyperpar_name)] = values
            for ensamble in self.ensambleModelList:
                if isinstance(ensamble, (StackingClassifier, StackingRegressor)) or (isinstance(ensamble,(VotingClassifier, VotingRegressor)) and getattr(ensamble,"voting",None)=="soft"):
                    ensamble.set_params(estimators=self.get_soft_voting_ready_models(self.estimators))
                else:
                    ensamble.set_params(estimators=self.estimators)
                    
        self.target = target
        self.y = data[target]
        data.drop(target, axis=1, inplace=True)
        
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        
        # Codifica le colonne categoriche
        self.encoded_data = encoder.fit_transform(data[self.categorical_features.values()])
        
    
        # Crea la mappatura delle colonne trasformate per ogni feature
        self.mapping = {}
        for feature in self.categorical_features.values():
            transformed_columns = [col for col in encoder.get_feature_names_out() if col.startswith(feature)]
            self.mapping[feature] = transformed_columns
    
        # Concatena i dati trasformati
        self.encoded_data = self.concat_encoded_dataset(self.mapping, self.categorical_features, self.encoded_data)
        
        # Codifica il target
        label_encoder = LabelEncoder()  
        self.y_encoded = label_encoder.fit_transform(self.y)
    
        self.label_mapping = {label: index for index, label in enumerate(label_encoder.classes_)}
    
        if resamplingMethods:
            self.resampled_data = self.dataResampler(self.encoded_data, self.y_encoded, resamplingMethods, self.target)

        self.numeric_features = self.numeric_features.drop(columns=self.boolean_columns.columns, errors='ignore')

        # Trasforma le colonne booleane in valori booleani
        for col in self.boolean_columns.columns:
            self.encoded_data[col] = self.encoded_data[col].astype(bool)

        # Converte le colonne numeriche in float
        for col in self.numeric_features.columns:
            self.encoded_data[col] = self.encoded_data[col].astype(float)


        #print(self.encoded_data.shape[0])
        #print(len(self.y_encoded))
        # Suddividi il dataset in train e test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.encoded_data, self.y_encoded, test_size=0.2, random_state=42)
    
        # Prepara le performance dei modelli
        if self.modelList is not None:
            for m in self.modelList:
                self.performances["specific"][f'{m}'.split('(')[0]] = {}
                self.performances["overall"][f'{m}'.split('(')[0]] = {}
                for name, metric in self.metrics.items():
                    for label, index in self.label_mapping.items():    
                        self.performances["specific"][f'{m}'.split('(')[0]][f'{name}_{label}'] = None
                    self.performances["overall"][f'{m}'.split('(')[0]][f'{name}'] = None

      
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

    def dataResampler(self,X,y, methods,target):
        data = dict()  
        for m in methods:
                X,y = m.fit_resample(X, y)
                X[target] = y
                data.update({f"{m}".split("(")[0]:X})
        for method in methods:
            self.performances[f"overall_{method}".split('(')[0]] = {}
            self.performances[f"specific_{method}".split('(')[0]] = {}
            for m in self.modelList:
                self.performances[f"overall_{method}".split('(')[0]][f'{m}'.split('(')[0]] = {}
                self.performances[f"specific_{method}".split('(')[0]][f'{m}'.split('(')[0]] = {}
                for name, metric in self.metrics.items():
                    for label, index in self.label_mapping.items():    
                        self.performances[f"specific_{method}".split('(')[0]][f'{m}'.split('(')[0]][f'{name}_{label}'] = None
                    self.performances[f"overall_{method}".split('(')[0]][f'{m}'.split('(')[0]][f'{name}'] = None

        #print(data)
    
        for d in data.values():
            for k,v in self.mapping.items():
                for c in v:
                    d[c] = (d[c] >= 0.5).astype(int)
            for c in self.boolean_columns:
                d[c] = (d[c] >= 0.5).astype(int)
        
        # Assicurati che i tipi di dato rimangano coerenti
        for d in data.values():
            # Converti le colonne numeriche al tipo originale
            for col in self.numeric_features.columns:
                if col in d.columns:  # Verifica che la colonna esista nel dataset
                    d[col] = pd.to_numeric(d[col], errors='coerce')
            
        return data

    def concat_encoded_dataset(self, map: Dict, cat_f, encoded_data):
        data = pd.DataFrame()     
        
        self.numeric_features = self.original_data.copy().select_dtypes(exclude=['object'])
        #print("features numeriche ")
        #print(self.numeric_features)
        new_cols = []
        for i in map.keys():
            new_cols.extend(map[i])
        encoded_data = pd.DataFrame(encoded_data, columns = new_cols)
        i = 0
        cols = self.original_data.copy().columns
        #print("num_cols"+ str(self.numeric_features.transpose().shape[0]))
        for index in cols:
            
            if index in map.keys():
                columns_to_concat = list(map[index])                
                data = pd.concat([data,encoded_data[columns_to_concat]], axis=1)
            else:
                #print("col " + str(i))
                
                data = pd.concat([data,self.numeric_features.transpose().iloc[i]],axis=1)
                i+=1        
        return data

    

    def _save_plot(self, filename):
            """ Salva il grafico con un nome specifico """
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Grafico salvato come: {filename}")

    def plot_pie_chart(self, save=False):
        """ Disegna un diagramma a torta per una colonna categorica """
        plt.figure(figsize=(6,6))
        num_classes = len(self.y.value_counts())
        colors = sns.color_palette("husl", num_classes) 
        self.y.value_counts().plot.pie(autopct='%1.1f%%', colors=colors)
        plt.title(f'Distribuzione di {self.target}')
        plt.ylabel("")

        if save:
            filename = f"pie_chart_{self.target}_original_data.png"
            self._save_plot(filename)
        plt.show()

    def plot_pie_resampled(self, save=False):
        """ Disegna un diagramma a torta per i dati resampled """
        if self.resampled_data:
            for index, data in self.resampled_data.items():
                print(f"Resampler algorithm: {index}")
                y = data[self.target]
                plt.figure(figsize=(6,6))
                y.value_counts().plot.pie(autopct='%1.1f%%', colors=['green', 'red', 'yellow'])
                plt.title(f'Distribuzione di {self.target}')
                plt.ylabel("")

                if save:
                    filename = f"pie_chart_{self.target}_augmented_data{index}.png"
                    self._save_plot(filename)
                plt.show()

    def plot_numeric_distribution(self, save=False):
        """ Disegna istogrammi per le colonne numeriche """
        self.numeric_features.hist(figsize=(12, 8), bins=15, edgecolor='black')
        plt.suptitle("Distribuzione delle variabili numeriche")

        if save:
            filename = f"numeric_distribution_original_data.png"
            self._save_plot(filename)
        plt.show()

    def plot_numeric_distribution_resampled(self, save=False):
        """ Disegna istogrammi per le variabili numeriche nei dati resampled """
        if self.resampled_data:
            for index, data in self.resampled_data.items():
                data_to_plot = data.select_dtypes(exclude=['object'])
                data_to_plot.hist(figsize=(12, 8), bins=15, edgecolor='black')
                plt.suptitle(f"Distribuzione delle variabili numeriche - Algoritmo {index}")

                if save:
                    filename = f"numeric_distribution_augmented_data.png"
                    self._save_plot(filename)
                plt.show()

    def plot_boxplots(self, save=False):
        """ Disegna boxplot per le colonne numeriche """
        plt.figure(figsize=(12, 6))
        self.numeric_features.boxplot()
        plt.title("Distribuzione delle variabili numeriche con boxplot")

        if save:
            filename = f"boxplot_original_data.png"
            self._save_plot(filename)
        plt.show()

    def plot_boxplots_resampled(self, save=False):
        """ Disegna boxplot per le colonne numeriche nei dati resampled """
        if self.resampled_data:
            for index, data in self.resampled_data.items():
                data_to_plot = data.select_dtypes(exclude=['object'])
                plt.figure(figsize=(12, 6))
                data_to_plot.boxplot()
                plt.title(f"Distribuzione delle variabili numeriche con boxplot - Algoritmo {index}")

                if save:
                    filename = f"boxplot_augmented_data.png"
                    self._save_plot(filename)
                plt.show()

    def plot_correlation_matrix(self, save=False):
        """ Disegna una heatmap delle correlazioni tra variabili numeriche """
        plt.figure(figsize=(8,6))
        corr_matrix = self.numeric_features.corr().fillna(0)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Matrice di correlazione tra variabili numeriche")

        if save:
            filename = f"correlation_matrix_original_data.png"
            self._save_plot(filename)
        plt.show()

    def plot_correlation_matrix_resampled(self, save=False):
        """ Disegna una heatmap delle correlazioni tra variabili numeriche nei dati resampled """
        if self.resampled_data:
            for index, data in self.resampled_data.items():
                numeric_data = data.select_dtypes(exclude=['object']).apply(pd.to_numeric, errors='coerce')
                if numeric_data.shape[1] < 2:
                    print(f"Saltata la matrice di correlazione per {index}: non abbastanza variabili numeriche.")
                    continue
                plt.figure(figsize=(8,6))
                sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                plt.title(f"Matrice di correlazione - Algoritmo {index}")

                if save:
                    filename = f"correlation_matrix_augmented_data.png"
                    self._save_plot(filename)
                plt.show()

    def plot_binary_distribution(self, save=False):
        """ Disegna conteggi per le variabili binarie """
        binary_cols = self.boolean_columns.columns
        fig, axes = plt.subplots(1, len(binary_cols), figsize=(15, 4))
        for i, col in enumerate(binary_cols):
            sns.countplot(x=self.original_data[col], ax=axes[i], palette="pastel")
            axes[i].set_title(f"Distribuzione di {col}")
            axes[i].set_xlabel("")

        plt.tight_layout()
        if save:
            filename = f"binary_distribution_original_data.png"
            self._save_plot(filename)
        plt.show()

    def plot_binary_distribution_resampled(self, save=False):
        """ Disegna conteggi per le variabili binarie nei dati resampled """
        if self.resampled_data:
            binary_cols = self.boolean_columns.columns
            fig, axes = plt.subplots(1, len(binary_cols), figsize=(15, 4))
            for index, data in self.resampled_data.items():
                for i, col in enumerate(binary_cols):
                    sns.countplot(x=data[col], ax=axes[i], palette="pastel")
                    axes[i].set_title(f"Distribuzione di {col}")
                    axes[i].set_xlabel("")

                plt.tight_layout()
                if save:
                    filename = f"binary_distribution_augmented_data.png"
                    self._save_plot(filename)
                plt.show()
        
        

    def plot_learning_curve(self,model, X, y, cv=5, n_jobs=-1,name_method=None):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Validation score')
        plt.ylabel('Score')
        plt.xlabel('Training Size')
        model_name = f'{model}'.split('(')[0]
        if name_method:
            
            plt.title(f'Learning Curve {model_name} con {name_method}')
        else:
            plt.title(f'Learning Curve {model_name}' )
        plt.legend()
        plt.grid()
        import re

        
        if name_method:
            plt.savefig(f'Learning Curve {model_name} con {name_method}.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'learning_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        #plt.savefig(f'learning_curve_{model}.png', dpi=300, bbox_inches='tight')  # Salva il grafico come immagine PNG
        plt.show()
    
    def plot_validation_curve(self, model, X, y, param_name, param_range, cv=5, ax=None):
        train_scores, test_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range, cv=cv
        )
        model_name = re.split(r'\(', f'{model}')[0]  # Estrai solo il nome del modello

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        # Disegna il grafico sull'asse specificato
        ax.plot(param_range, train_mean, label='Training score', color='blue')
        ax.plot(param_range, test_mean, label='Validation score', color='orange')
        ax.set_ylabel('Score')
        ax.set_xlabel(param_name)
        
        ax.set_title(f'Validation Curve: {model_name} - {param_name}')
        ax.legend()
        ax.grid()

    def scale_data(self,to_scale):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(to_scale)
        return X_scaled
        
    def learning_curves(self,augmented_data_X=None,augmented_data_y=None,name_method=None):
        for model,params in self.modelList.items():
                # Disegna le learing curve
            print(f"Learning curve for model: {model} with {name_method}")
            if augmented_data_X is None and augmented_data_y is None:
                self.plot_learning_curve(model, self.encoded_data, self.y_encoded)
            else:
                self.plot_learning_curve(model, augmented_data_X, augmented_data_y,name_method=name_method)
    def checkPipelineAugmentation(self,augmentationAlgoritms:Dict,cv=5):
        
        self.pipelines = {}
        for model,hyperpars in self.modelList.items():
            self.pipelines[f"{model}"] = {}
            pipe = self.pipelines[f"{model}"]
            for name, method in augmentationAlgoritms.items():
                p = Pipeline([
                    ('sampling', method),
                    ('classifier', model)
                ])
                d = {}
                for metric_name,metric in self.metrics.items():
                    if metric_name != "accuracy":
                        if metric_name == "f1-score":
                            metric_name = "f1_macro"
                        else:
                            metric_name +="_macro"
                    
                    scores = cross_val_score(p, X, y, cv=cv, scoring=metric_name)
                    d[metric_name] = scores.mean()
                    print(f"{model}")
                    print(f"\t {name}: {metric_name} = {scores.mean():.4f}")
                pipe[name] = d
    def learning_curves_on_augmented_data(self):
        if self.resampled_data:
            for name_method, data in self.resampled_data.items():
                print(f'resampled data with {name_method}')
                augmented_data_y = data[self.target]
                augmented_data_X = data.drop(self.target,axis=1)
                self.learning_curves(augmented_data_X=augmented_data_X,augmented_data_y=augmented_data_y,name_method=name_method)
        else:
            print("no resampled data")

    def validation_curves(self, cv=5, augmented_data_X=None, augmented_data_y=None,resampling_method_name=None):
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
                if augmented_data_X is None and augmented_data_y is None:
                    self.plot_validation_curve(
                        model, self.scale_data(self.encoded_data), self.y_encoded,
                        param_name, param_values, cv, ax=axes[plot_index])
                else:
                    self.plot_validation_curve(
                        model, self.scale_data(augmented_data_X), augmented_data_y,
                        param_name, param_values, cv, ax=axes[plot_index])
                
                plot_index += 1
    
            # Rimuovi eventuali assi vuoti nella griglia
            for i in range(plot_index, len(axes)):
                fig.delaxes(axes[i])
    
            # Salva l'immagine per il modello corrente
            if augmented_data_X is not None and augmented_data_y is not None and resampling_method_name is not None:
                fig.savefig(f'validation_curves_{model_name}_on_augmented_data_{resampling_method_name}.png', dpi=300, bbox_inches='tight')
            else:
                fig.savefig(f'validation_curves_{model_name}.png', dpi=300, bbox_inches='tight')
            
                
            print(f"Saved validation curves for {model_name} as validation_curves_{model_name}.png")
            
            plt.tight_layout()  # Migliora la spaziatura
            plt.show()  # Mostra la figura
            plt.close(fig)  # Chiudi la figura per liberare memoria


    def validation_curves_on_augmented_data(self):
        if self.resampled_data:
            for method,data in self.resampled_data.items():
                print(f"validation_curves for data resampled with {method} method")
                augmented_data_y = data[self.target]
                augmented_data_X = data.drop(self.target,axis=1)
                self.validation_curves(augmented_data_X=augmented_data_X,augmented_data_y=augmented_data_y,resampling_method_name= re.split(r'\(', f'{method}')[0])
        else:
            print("no resampled data")


    def make_dataframe_performances(self,data):
        # Creiamo una lista vuota per raccogliere tutte le informazioni da visualizzare

        # aggiungere un if di controllo iniziale per evitare che si inizi a creare un dataframe di valori None
        exp_types = []
        metrics = []
        hyperparams = []
        performances = []
        model_names = []
        
        for exp_type, model_metric_hyperpar_perf in data.items():
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
    

    def plot_model_performance_histograms(self,data,save=False, name = None):
    
        data_to_plot = self.make_dataframe_performances(data)
        # Otteniamo un elenco unico dei modelli da visualizzare come sottotitoli
        unique_models = data_to_plot["Model"].unique()
        
        # Determiniamo la griglia di subplots in base al numero di modelli
        n_models = len(unique_models)
        n_cols = 3  # Impostiamo il numero di colonne per il layout della matrice
        n_rows = (n_models + n_cols - 1) // n_cols  # Calcoliamo il numero di righe
    
        # Creiamo la figura e la griglia di subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
        axes = axes.flatten()  # Appiattiamo gli assi per un accesso più facile
    
        # Iteriamo sui modelli e creiamo un grafico per ogni modello
        for i, model in enumerate(unique_models):
            ax = axes[i]
            # Filtriamo i dati per il modello corrente
            model_data = data_to_plot[data_to_plot["Model"] == model]
            
            # Creiamo il barplot per il modello corrente, separando le metriche
            sns.barplot(x="Metric/Class",y="Performance",hue="Metric/Class",data=model_data,ax=ax,dodge=True ) # Dividiamo le barre per iperparametro
           
            ax.get_legend().remove()
            ax.set_title(f'Performance for {model}')
            ax.set_ylabel('Performance')
            ax.set_xlabel('Metric/Class')
            ax.tick_params(axis='x', rotation=45)
    
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])
    
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1), ncol=2, title="Metric/Class")
        if save:
            fig.savefig(name+'.png', dpi=300, bbox_inches='tight')  # Salva come file PNG con alta risoluzione

        plt.tight_layout()
        plt.show()
            
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
        cv=5,
        avg='binary',
        by_target_label=None,
        resampled_data_x=None,
        resampled_data_y=None,
        resampler=None,
        searcher_class=GridSearchCV,  
        searcher_kwargs=None          
    ):
        scorers = {}

        """
        if self.modelList is not None:
            for m in self.modelList:
                self.performances["specific"][f'{m}'.split('(')[0]] = {}
                self.performances["overall"][f'{m}'.split('(')[0]] = {}
                for name, metric in self.metrics.items():
                    for label, index in self.label_mapping.items():    
                        self.performances["specific"][f'{m}'.split('(')[0]][f'{name}_{label}'] = None
                    self.performances["overall"][f'{m}'.split('(')[0]][f'{name}'] = None
        """
        # Decidi il dataset
        if resampled_data_y is not None and resampled_data_x is not None and resampler is not None:
            y = resampled_data_y
            X = resampled_data_x
        else:
            X = self.encoded_data
            y = self.y_encoded

        # Configurazione scorers
        if by_target_label:
            scorers = self.make_metrics_by_labels(avg=avg)
        else:
            scorers = self.make_metrics(avg=avg)

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Parametri extra (se vuoti, inizializza a dict)
        if searcher_kwargs is None:
            searcher_kwargs = {}

        # Loop sui modelli
        for model, params in self.modelList.items():
            print(f"Testing model: {model} with {searcher_class.__name__}")

            # Inizializza il "searcher" scelto
            if searcher_class == RandomizedSearchCV:
                searcher = searcher_class(
                    estimator=model,
                    param_distributions=params,   # ✅ corretto per RandomizedSearchCV
                    scoring=scorers,
                    refit=list(scorers.keys())[0],
                    cv=cv_strategy,
                    verbose=1,
                    **searcher_kwargs
                )

            else:
                searcher = searcher_class(
                    estimator=model,
                    param_grid=params,            # ✅ corretto per GridSearchCV / HalvingGridSearchCV
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
            perf = {}
            for scorer_name in scorers.keys():
                best_params = searcher.best_params_
                best_score = searcher.cv_results_[f'mean_test_{scorer_name}'][searcher.best_index_]
                print(f"\tBest parameters for {scorer_name}: {best_params}")
                print(f"\tBest score for {scorer_name}: {best_score}")
                perf[scorer_name] = {
                    json.dumps({
                        k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                        for k, v in best_params.items()
                    }): best_score
                }

            # Salvataggio performance
            if by_target_label:
                if resampled_data_y is not None and resampled_data_x is not None and resampler is not None:
                    self.performances[f'specific_{resampler}'.split('(')[0]][f'{model}'.split('(')[0]] = perf
                else:
                    self.performances['specific'][f'{model}'.split('(')[0]] = perf
            else:
                if resampled_data_y is not None and resampled_data_x is not None and resampler is not None:
                    self.performances[f'overall_{resampler}'.split('(')[0]][f'{model}'.split('(')[0]] = perf
                else:
                    self.performances['overall'][f'{model}'.split('(')[0]] = perf

    def best_param_calculator_by_label(self,cv = 10,searcher_class=GridSearchCV):
        """
        for k in self.label_mapping.keys():
            self.best_param_calculator(cv,"macro",k)
        """
        self.best_param_calculator(cv,"macro",True,searcher_class=searcher_class)
    def best_param_calculator_ensamble_by_label(self,cv=10,searcher_class=GridSearchCV):
        if self.ensambleModelList is None:
            return "No Ensamble Models"
        copy_models = self.modelList.copy()
        self.modelList = {ens:None for ens in self.ensambleModelList}
        for k in self.modelList.keys():
            self.modelList[k] = self.ensamble_hyperpar
        self.best_param_calculator(cv,'macro',True,searcher_class=searcher_class)
        self.modelList = copy_models
    def best_param_calculator_ensamble(self,cv=10,searcher_class=GridSearchCV):
        if self.ensambleModelList is None:
            return "No Ensamble Models"
        copy_models = self.modelList.copy()
        self.modelList = {ens:None for ens in self.ensambleModelList}
        for k in self.modelList.keys():
            self.modelList[k] = self.ensamble_hyperpar
        self.best_param_calculator(cv,'macro',searcher_class=searcher_class)
        self.modelList = copy_models
    def best_param_calculator_ensamble_from_augmented_data(self,cv=10,searcher_class=GridSearchCV):
        if self.resampled_data is None:
            return "No augmented data"
        if self.ensambleModelList is None:
            return "No Ensamble Models"
        copy_models = self.modelList.copy()
        self.modelList = {ens:None for ens in self.ensambleModelList}
        for k in self.modelList.keys():
            self.modelList[k] = self.ensamble_hyperpar
        for m,data in self.resampled_data.items():
                print(f"for augmented data with {m}")
                data_y = data[self.target]
                data_x =  data.drop(self.target,axis=1)
                self.best_param_calculator(cv,"macro",resampled_data_x=data_x,searcher_class=searcher_class, resampled_data_y=data_y,resampler=m)
        self.modelList = copy_models
    def best_param_calculator_ensamble_from_augmented_data_by_label(self,cv=10, searcher_class=GridSearchCV):
        if self.resampled_data is None:
            return "No augmented data"
        if self.ensambleModelList is None:
            return "No Ensamble Models"
        copy_models = self.modelList.copy()
        self.modelList = {ens:None for ens in self.ensambleModelList}
        for k in self.modelList.keys():
            self.modelList[k] = self.ensamble_hyperpar
        for m,data in self.resampled_data.items():
                print(f"for augmented data with {m}")
                #print(data.drop(self.target,axis=1))
                #print(data[self.target])
                data_y = data[self.target]
                data_X =  data.drop(self.target,axis=1)
                print(data_X)
                self.best_param_calculator(cv,"macro",by_target_label=True,searcher_class=searcher_class,resampled_data_x=data_X ,resampled_data_y=data_y,resampler=m)
        self.modelList = copy_models
        
    def best_param_calculator_from_augmented_data(self,cv = 10,searcher_class=GridSearchCV):
        if self.resampled_data is None:
            return "No augmented data"
        for m,data in self.resampled_data.items():
                print(f"for augmented data with {m}")
                data_y = data[self.target]
                data_X =  data.drop(self.target,axis=1)
                self.best_param_calculator(cv,"macro",resampled_data_x=data_X,searcher_class=searcher_class, resampled_data_y=data_y,resampler=m)
    def best_param_calculator_from_augmented_data_by_label(self,cv = 10):
        if self.resampled_data is None:
            return "No augmented data"
        for m,data in self.resampled_data.items():
            data_y = data[self.target]
            data_x = data.drop(self.target,axis=1)
            self.best_param_calculator(cv,"macro",by_target_label=True,resampled_data_x=data_x, resampled_data_y=data_y,resampler=m)
def custom_scorer(target_class,metric,avg = 'macro'):
        
        def score_for_class(y_true, y_pred):
            if metric == accuracy_score:
                return metric(y_true, y_pred)
            else:
                return metric(y_true, y_pred, labels=[target_class], average=avg, zero_division=0)
        return score_for_class


