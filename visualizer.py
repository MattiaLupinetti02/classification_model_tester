import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import learning_curve, validation_curve
import random

class Visualizer:
    def __init__(self, data: pd.DataFrame, y_data:pd.Series, target: str,numeric_features, categorical_features,boolean_features, model_list,resampled_data: dict | None = None):
        self.original_data = data
        self.y_data = y_data
        self.target = target
        self.numeric_features = numeric_features
        self.modelList = model_list
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

    def plot_boxplots(self, data:pd.DataFrame, save=False):
        
        plt.figure(figsize=(12, 6))
        numeric_features = self.get_numeric_columns(data)
        numeric_features.boxplot()
        plt.title("Distribuzione delle variabili numeriche con boxplot")
        if save:
            self._save_plot("boxplot_original_data.png")
        plt.show()

    def _save_plot(self, filename):
        """ Salva il grafico con un nome specifico """
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Grafico salvato come: {filename}")

    def plot_numeric_distribution(self,data:pd.DataFrame ,save=False):
        """ Disegna istogrammi per le variabili numeriche nei dati resampled """
        
        data_to_plot = self.get_numeric_columns(data)
        data_to_plot.hist(figsize=(12, 8), bins=15, edgecolor='black')
        plt.suptitle(f"Distribuzione delle variabili numeriche")

        if save:
            filename = f"numeric_distribution_augmented_data.png"
            self._save_plot(filename)
        plt.show()

    def plot_boxplots(self,data:pd.DataFrame, save=False):
        """ Disegna boxplot per le colonne numeriche """
        plt.figure(figsize=(12, 6))
        data_to_plot = self.get_numeric_columns(data)
        data_to_plot.boxplot()
        plt.title("Distribuzione delle variabili numeriche con boxplot")

        if save:
            filename = f"boxplot_original_data.png"
            self._save_plot(filename)
        plt.show()



    def plot_correlation_matrix(self, data:pd.DataFrame,save=False):
        """ Disegna una heatmap delle correlazioni tra variabili numeriche """
        plt.figure(figsize=(6,4))
        corr_matrix = self.get_numeric_columns(data)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Matrice di correlazione tra variabili numeriche")

        if save:
            filename = f"correlation_matrix_original_data.png"
            self._save_plot(filename)
        plt.show()

    def plot_correlation_matrix_2(self, data:pd.DataFrame, save=False):
        """ Disegna una heatmap delle correlazioni tra variabili numeriche """
        corr_matrix = self.get_numeric_columns(data)
        num_vars = corr_matrix.shape[0]
        
        # Calcola dimensione figura in base al numero di variabili
        fig_size = max(8, num_vars * 0.8)
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        
        # Adatta dimensione annotazione
        annot = True if num_vars <= 15 else False
        annot_size = max(6, 12 - num_vars * 0.3)
        
        sns.heatmap(corr_matrix, annot=annot, cmap="coolwarm", fmt=".2f",
                    linewidths=0.5, square=True,
                    annot_kws={"size": annot_size} if annot else None,
                    cbar_kws={"shrink": 0.7})
        
        plt.title(f"Matrice di correlazione ({num_vars} variabili)", fontsize=14)
        plt.tight_layout()

        if save:
            filename = f"correlation_matrix_original_data.png"
            self._save_plot(filename)
        plt.show()

    def plot_binary_distribution(self,data:pd.DataFrame,title="binary columns distribution",figsize=(12,8), save=False):
        """ Disegna conteggi per le variabili binarie """

        boolean_cols = self.get_boolean_columns(data).columns.tolist()
        df = self.get_boolean_columns(data)
        
        # Calcola il layout dei subplot

        n_cols = min(3, len(boolean_cols))
        n_rows = max(1, (n_cols + 2) // 3)  # Calcolo robusto
        
        # Crea la figura e i subplot
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Appiattisce l'array di assi se necessario
        if n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes] if n_cols == 1 else axes
        
        # Crea un plot per ogni colonna booleana
        for i, col in enumerate(boolean_cols):
            if i < len(axes):
                ax = axes[i]
                
                # Calcola la distribuzione dei valori
                value_counts = df[col].value_counts()
                
                # Crea il bar plot
                bars = ax.bar(value_counts.index, value_counts.values, 
                            color='skyblue', alpha=0.7, edgecolor='black')
                
                # Aggiungi i valori sulle barre
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                # Personalizza il subplot
                ax.set_title(f'Distribuzione: {col}', fontweight='bold')
                ax.set_xlabel('Valori')
                ax.set_ylabel('Conteggio')
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['False', 'True'])
                ax.grid(axis='y', alpha=0.3)
                
                # Calcola percentuali
                total = len(df[col])
                true_percentage = (df[col].sum() / total) * 100
                false_percentage = 100 - true_percentage
                
                # Aggiungi testo con le percentuali
                ax.text(0.5, 0.95, f'False: {false_percentage:.1f}%', 
                    transform=ax.transAxes, ha='center', va='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.text(0.5, 0.85, f'True: {true_percentage:.1f}%', 
                    transform=ax.transAxes, ha='center', va='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Nasconde gli assi non utilizzati
        for j in range(len(boolean_cols), len(axes)):
            axes[j].set_visible(False)
        
        # Aggiusta il layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        # Stampa statistiche riassuntive
        print(f"\nStatistiche riassuntive per {len(boolean_cols)} colonne booleane:")
        for col in boolean_cols:
            true_count = df[col].sum()
            false_count = len(df) - true_count
            true_perc = (true_count / len(df)) * 100
            print(f"  {col}: True={true_count} ({true_perc:.1f}%), False={false_count}")



        

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
    
    def plot_validation_curve(self, model, X, y, param_name, param_range, cv=5, ax=None,dt_name=None):
        train_scores, test_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range, cv=cv
        )
        model_name = re.split(r'\(', f'{model}')[0]  # Estrai solo il nome del modello

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        # Disegna il grafico sull'asse specificato
        tr_score = f'Training score {dt_name}' if dt_name is not None else 'Training Score'
        vl_score = f'Validation score {dt_name}' if dt_name is not None else 'Validation Score'
        ax.plot(param_range, train_mean, label=tr_score, color=f"#{random.randint(0, 0xFFFFFF):06x}")
        ax.plot(param_range, test_mean, label=vl_score, color=f"#{random.randint(0, 0xFFFFFF):06x}")
        ax.set_ylabel('Score')
        ax.set_xlabel(param_name)
        
        ax.set_title(f'Validation Curve: {model_name} - {param_name}')
        ax.legend()
        ax.grid()



    def get_numeric_columns(self, df):
        """
        Restituisce le colonne numeriche del DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        return df[numeric_cols]
    
    def get_categorical_columns(self,df):
        """
        Restituisce le colonne categoriche del DataFrame
        Include: object, category, string
        """
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        return df[categorical_cols]
    
    def get_boolean_columns(self,df):
        """
        Restituisce le colonne booleane del DataFrame
        """
        boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
        return df[boolean_cols]