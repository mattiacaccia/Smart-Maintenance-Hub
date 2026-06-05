import pandas as pd
import numpy as np
import sklearn.preprocessing as pp

class DataPreprocessor:
    def __init__(self):
        self.scaler = pp.MinMaxScaler()
        self.selected_features = None  # Per memorizzare quali sensori non sono costanti

    def load_and_clean(self, file_path, is_train=True):
        df = pd.read_csv(file_path, sep='\s+', header=None)
        names = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
        df.columns = names
        
        if is_train:
            # Identifichiamo le colonne non costanti
            self.selected_features = df.columns[df.std() > 0].tolist()
        
        # Filtriamo il dataframe con le colonne selezionate
        return df[self.selected_features]

    def add_rul(self, df, max_rul=125):
        # Clipping: Piecewise Linear RUL
        real_rul = df.groupby('unit')['cycle'].transform('max') - df['cycle']
        df = df.copy() 
        df['RUL'] = np.minimum(real_rul, max_rul)
        return df

    def scale_and_smooth(self, df, is_train=True, window_size=15):
        cols = df.columns.difference(['unit', 'cycle', 'RUL'])
        
        if is_train:
            df[cols] = self.scaler.fit_transform(df[cols])
        else:
            df[cols] = self.scaler.transform(df[cols])

        # Feature Engineering: Medie mobili per pulire il segnale
        for col in cols:
            df[f'{col}_avg'] = df.groupby('unit')[col].transform(
                lambda x: x.rolling(window=window_size).mean()
            )
        
        return df.dropna()

    def full_preprocessing(self, file_path, is_train=True):
        """Esegue l'intera pipeline in un unico comando"""
        df = self.load_and_clean(file_path, is_train)
        if is_train:
            df = self.add_rul(df)
        df = self.scale_and_smooth(df, is_train)
        return df