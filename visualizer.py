import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import learning_curve, validation_curve


class Visualizer:
    def __init__(self, data: pd.DataFrame, target: str,numeric_features, categorical_features,boolean_features, resampled_data: dict | None = None):
        self.original_data = data
        self.target = target
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.boolean_columns = boolean_features
        if resampled_data:
            print("resampled data provided to visualizer")
            self.resampled_data = resampled_data
        

    def _save_plot(self, filename):
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Grafico salvato come: {filename}")

    def plot_pie_chart(self, y, target, save=False):
        plt.figure(figsize=(6,6))
        colors = sns.color_palette("husl", len(y.value_counts()))
        y.value_counts().plot.pie(autopct='%1.1f%%', colors=colors)
        plt.title(f'Distribuzione di {target}')
        plt.ylabel("")
        if save:
            self._save_plot(f"pie_chart_{target}_original_data.png")
        plt.show()

    def plot_boxplots(self, numeric_features, save=False):
        plt.figure(figsize=(12, 6))
        numeric_features.boxplot()
        plt.title("Distribuzione delle variabili numeriche con boxplot")
        if save:
            self._save_plot("boxplot_original_data.png")
        plt.show()

    def _save_plot(self, filename):
        """ Salva il grafico con un nome specifico """
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Grafico salvato come: {filename}")



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


        
    def learning_curves(self,augmented_data_X=None,augmented_data_y=None,name_method=None):
        for model,params in self.modelList.items():
                # Disegna le learing curve
            print(f"Learning curve for model: {model} with {name_method}")
            if augmented_data_X is None and augmented_data_y is None:
                self.plot_learning_curve(model, self.encoded_data, self.y_encoded)
            else:
                self.plot_learning_curve(model, augmented_data_X, augmented_data_y,name_method=name_method)

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
