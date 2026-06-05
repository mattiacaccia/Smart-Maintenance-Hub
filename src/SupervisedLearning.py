from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

class SupervisedLearning:
    def __init__(self, threshold=30):

        self.threshold = threshold

        # Modello di Classificazione (Decisionale)
        self.classifier = LogisticRegression(max_iter=1000 , random_state=42)

        # Modello regolarizzato per evitare overfitting e migliorare la generalizzazione.
        self.regressor = RandomForestRegressor( 
            n_estimators=100, 
            max_depth=10, 
            min_samples_leaf=10, 
            max_features='sqrt', 
            random_state=42,
            n_jobs=-1 
        )

    def train_and_validate(self, df_train, feature_names):
        self.feature_names = feature_names
        X = df_train[feature_names]
        y_reg = df_train['RUL']
        
        # Creazione target binario per la classificazione: 1 se RUL <= soglia, 0 altrimenti
        y_class = (y_reg <= self.threshold).astype(int)

        # 1. Validazione Regressore (MAE/RMSE)
        scores_reg = cross_validate(self.regressor, X, y_reg, cv=5, scoring={'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'})

        # 2. Validazione Classificatore (Accuracy/F1)
        scores_class = cross_validate(self.classifier, X, y_class, cv=5, scoring={'acc': 'accuracy', 'f1': 'f1'})

        # Addestramento finale
        self.regressor.fit(X, y_reg)
        self.classifier.fit(X, y_class)


        return {
            'reg_mae': -scores_reg['test_mae'].mean(),
            'reg_rmse': -scores_reg['test_rmse'].mean(),
            'class_acc': scores_class['test_acc'].mean(),
            'class_f1': scores_class['test_f1'].mean()
        }

    def evaluate_on_test(self, df_test, rul_truth_path):
        # Valutazione finale sul test set ufficiale (NASA protocol).
        # Prendiamo solo l'ultimo stato noto per ogni motore (100 motori totali)
        last_cycles = df_test.groupby('unit').last().reset_index()
        y_true_reg = pd.read_csv(rul_truth_path, header=None).values.flatten()
        y_true_class = (y_true_reg <= self.threshold).astype(int)
        X_test = last_cycles[self.feature_names]

        y_pred_reg = self.regressor.predict(X_test)
        y_pred_class = self.classifier.predict(X_test)
        y_prob_class = self.classifier.predict_proba(X_test)[:, 1] # Probabilità per la Bayesiana

        metrics = {
            'mae': mean_absolute_error(y_true_reg, y_pred_reg),
            'rmse': root_mean_squared_error(y_true_reg, y_pred_reg),
            'accuracy': accuracy_score(y_true_class, y_pred_class),
            'f1': f1_score(y_true_class, y_pred_class)
        }
        return metrics,y_pred_reg, y_pred_class, y_prob_class

    def predict_single_instance(self, data):
        # Metodo per il Reasoning: predice la RUL per un singolo motore.
        return self.regressor.predict(data)[0]