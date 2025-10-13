import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from visualizer import Visualizer

class DataHandler:
    """
    Gestisce tutte le operazioni di preprocessing dati:
    - codifica categoriale (OneHotEncoder)
    - codifica del target (LabelEncoder)
    - concatenazione dataset codificato
    - scaling
    - train/test split
    - resampling (SMOTE, ADASYN, ecc.)
    """

    def __init__(self, data: pd.DataFrame, target: str):
        self.original_data = data.copy()
        self.target = target
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.label_encoder = LabelEncoder()
        self.resampled_data_dict = None
        
        # Rilevamento automatico tipi di feature
        self.categorical_features = {
            i: col for i, col in enumerate(data.columns)
            if col != target and data[col].apply(lambda x: isinstance(x, str)).all()
        }
        self.boolean_columns = data.select_dtypes(include=bool)
        self.numeric_features = data.select_dtypes(exclude=['object'])

        

        # Placeholder per dati codificati
        self.encoded_data = None
        self.y_encoded = None
        self.mapping = {}

    
    def encode_features(self) -> pd.DataFrame:
        """Applica OneHotEncoding alle feature categoriali."""
        encoded_array = self.encoder.fit_transform(self.original_data[self.categorical_features.values()])

        # Crea la mappatura delle colonne trasformate per ogni feature
        for feature in self.categorical_features.values():
            transformed_columns = [
                col for col in self.encoder.get_feature_names_out() if col.startswith(feature)
            ]
            self.mapping[feature] = transformed_columns

        encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out())
        self.encoded_data = encoded_df
        return encoded_df

    def encode_target(self) -> np.ndarray:
        """Codifica il target usando LabelEncoder."""
        y = self.original_data[self.target]
        self.y_encoded = self.label_encoder.fit_transform(y)
        self.label_mapping = {label: i for i, label in enumerate(self.label_encoder.classes_)}
        return self.y_encoded

    def concat_encoded_dataset(self) -> pd.DataFrame:
        """
        Ricompone il dataset originale combinando feature numeriche e categoriali codificate.
        Mantenuto compatibile con la versione originale di ModelTester.
        """
        if self.mapping is None or self.encoded_data is None or self.categorical_features is None:
            print("Error: original data is not encoded yet. Run encode_features() first.")
            return
        data = pd.DataFrame()
        numeric_features = self.original_data.copy().select_dtypes(exclude=['object'])
        new_cols = [col for sublist in self.mapping.values() for col in sublist]
        encoded_df = pd.DataFrame(self.encoded_data, columns=new_cols)

        for col in self.original_data.columns:
            if col in self.categorical_features.values():
                data = pd.concat([data, encoded_df[self.mapping[col]]], axis=1)
            elif col != self.target:
                data = pd.concat([data, numeric_features[[col]]], axis=1)
        self.encoded_data = data
        return data




    def split_data(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        """Suddivide in train/test set."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    # ---------------------------------------------------------
    #  RESAMPLING
    # ---------------------------------------------------------

    def dataResampler(
        self,
        methods: list,
        X: pd.DataFrame = None,
        y: np.ndarray = None,
        target_name: str = None,
        mapping: dict = None,
        boolean_columns: list = None,
        numeric_features: pd.DataFrame = None
    ) -> dict:
        """
        Applica una o più tecniche di resampling (SMOTE, ADASYN, ecc.) ai dati codificati.

        Args:
            X (pd.DataFrame): feature codificate
            y (np.ndarray): target codificato
            methods (list): lista di metodi di resampling da applicare
            target_name (str): nome della colonna target
            mapping (dict): mappatura delle feature categoriali → colonne codificate
            boolean_columns (list): colonne booleane (da riconvertire)
            numeric_features (pd.DataFrame): feature numeriche originali (per ripristino tipo)

        Returns:
            dict[str, pd.DataFrame]: dataset resampled per ogni metodo
        """
        data_dict = {}
        if X is None and y is None or target_name is None:
            X = self.encoded_data
            y = self.y_encoded
        if mapping is None:
            mapping = self.mapping
        if boolean_columns is None:
            boolean_columns = self.boolean_columns
        if numeric_features is None:
            numeric_features = self.numeric_features
        if target_name is None:
            target_name = self.target
        # Applica ciascun metodo di resampling
        for m in methods:
            X_res, y_res = m.fit_resample(X, y)
            X_res[target_name] = y_res
            data_dict[m.__class__.__name__] = X_res

        # Post-elaborazione: mantenere tipi coerenti
        for name, df in data_dict.items():
            # Ripristina colonne categoriali come binarie
            if mapping:
                for cat, cols in mapping.items():
                    for col in cols:
                        if col in df.columns:
                            df[col] = (df[col] >= 0.5).astype(int)

            # Colonne booleane
            if boolean_columns is not None:
                for col in boolean_columns:
                    if col in df.columns:
                        df[col] = (df[col] >= 0.5).astype(int)

            # Ripristina tipi numerici
            if numeric_features is not None:
                for col in numeric_features.columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        self.resampled_data_dict = data_dict
        return data_dict

    # ---------------------------------------------------------
    #  UTILITÀ VARIE
    # ---------------------------------------------------------

    def get_label_mapping(self):
        """Restituisce la mappa label → indice del target codificato."""
        return self.label_mapping

    def get_encoded_data(self):
        """Restituisce il dataset codificato (feature + target)."""
        return self.encoded_data, self.y_encoded
