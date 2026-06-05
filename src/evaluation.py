import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, confusion_matrix, roc_curve, auc, accuracy_score)
from sklearn.model_selection import learning_curve, cross_validate 
import os
from Data_preprocessing import DataPreprocessor

def run_full_evaluation():
    prep = DataPreprocessor()
    threshold = 30

    print("Caricamento dati...")
    if not os.path.exists('train_FD001.txt'):
        print("ERRORE: File train_FD001.txt non trovato!")
        return

    # 1. Preprocessing
    train_df = prep.full_preprocessing('train_FD001.txt', is_train=True)
    features = train_df.columns.difference(['unit', 'cycle', 'RUL'])
    test_df = prep.full_preprocessing('test_FD001.txt', is_train=False)
    y_test_ground_truth = pd.read_csv('RUL_FD001.txt', header=None).values.flatten()
    X_test_final = test_df.groupby('unit').last()[features]

    # 2. Modello Regressione (Random Forest) 
    print("Addestramento e Validazione Statistica Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,            
        min_samples_leaf=10,     
        max_features='sqrt',     
        random_state=42, 
        n_jobs=-1                
    )
    
    # Validazione statistica (Cross-Validation)
    cv_results_rf_mae = cross_validate(rf, train_df[features], train_df['RUL'], cv=5, scoring='neg_mean_absolute_error')
    mae_cv_mean = -cv_results_rf_mae['test_score'].mean()
    mae_cv_std = cv_results_rf_mae['test_score'].std()

    cv_results_rf_rmse = cross_validate(rf, train_df[features], train_df['RUL'], cv=5, scoring='neg_root_mean_squared_error')
    rmse_cv_mean = -cv_results_rf_rmse['test_score'].mean()
    rmse_cv_std = cv_results_rf_rmse['test_score'].std()

    rf.fit(train_df[features], train_df['RUL'])
    y_pred_rf = rf.predict(X_test_final)

    # 3. Modello Classificazione (Logistic Regression)
    print("Addestramento e Validazione Statistica Logistic Regression...")
    y_train_bin = (train_df['RUL'] <= threshold).astype(int)
    y_test_bin = (y_test_ground_truth <= threshold).astype(int)
    lr = LogisticRegression(max_iter=1000)
    
    # Validazione statistica per la classificazione
    cv_results_lr = cross_validate(lr, train_df[features], y_train_bin, cv=5, scoring=['accuracy', 'f1'])
    acc_cv_mean = cv_results_lr['test_accuracy'].mean()
    acc_cv_std = cv_results_lr['test_accuracy'].std()

    lr.fit(train_df[features], y_train_bin)
    y_pred_lr = lr.predict(X_test_final)
    y_prob_lr = lr.predict_proba(X_test_final)[:, 1]

    # REPORT STATISTICO
    print("\n" + "="*60)
    print("DATI : MEDIE E DEV. STD)")
    print("="*60)
    print(f"RF Regression MAE (CV):      {mae_cv_mean:.4f} ± {mae_cv_std:.4f}")
    print(f"RF Regression RMSE (CV):     {rmse_cv_mean:.4f} ± {rmse_cv_std:.4f}")
    print(f"LR Classification Acc (CV): {acc_cv_mean:.4f} ± {acc_cv_std:.4f}")
    print(f"MAE Finale su Test Set:     {mean_absolute_error(y_test_ground_truth, y_pred_rf):.2f}")
    print(f"RMSE Finale su Test Set:    {root_mean_squared_error(y_test_ground_truth, y_pred_rf):.2f}")
    print(f"LR Accuracy su Test Set:    {accuracy_score(y_test_bin, y_pred_lr):.4f}")
    print("="*60 + "\n")

    # GENERAZIONE GRAFICI 
    print("Generazione grafici in corso...")

    # Grafico 1: Reale vs Predetto
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test_ground_truth, y_pred_rf, alpha=0.5, color='royalblue')
    plt.plot([0, 150], [0, 150], '--r', label="Predizione Ideale")
    plt.title("Random Forest: RUL Reale vs Predetta")
    plt.xlabel("RUL Reale (Cicli effettivi)")
    plt.ylabel("RUL Predetta (Cicli)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Grafico 2: Matrice di Confusione
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test_bin, y_pred_lr), annot=True, fmt='d', cmap='Greens')
    plt.title("Matrice di Confusione (Logistic Regression)")
    plt.xlabel("Stato Predetto (0: Sicuro, 1: Allarme)")
    plt.ylabel("Stato Reale (0: Sicuro, 1: Allarme)")
    plt.show()

    # Grafico 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_test_bin, y_prob_lr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title("Curva ROC (Logistic Regression)")
    plt.xlabel("Tasso Falsi Positivi (FP)")
    plt.ylabel("Tasso Veri Positivi (TP)")
    plt.legend(loc="lower right")
    plt.show()

    # Grafico 4: Learning Curve
    print("Calcolo Random Forest Learning Curve...")
    train_sizes, train_scores, test_scores = learning_curve(
        rf, train_df[features], train_df['RUL'], 
        cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), 
        scoring='neg_mean_absolute_error'
    )
    train_errors = -np.mean(train_scores, axis=1)
    test_errors = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors, 'o-', color="r", label="Errore Training")
    plt.plot(train_sizes, test_errors, 'o-', color="g", label="Errore Validazione")
    plt.title("Curva di Apprendimento (Random Forest)")
    plt.xlabel("Campioni di addestramento")
    plt.ylabel("Errore (MAE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Grafico 5: Correlazione
    plt.figure(figsize=(12, 10))
    full_corr = train_df[features.tolist() + ['RUL']].corr()
    sns.heatmap(full_corr, annot=False, cmap='coolwarm', center=0)
    plt.title("Matrice di Correlazione (Features)")
    plt.show()

    # Grafico 6: Boxplot Normalizzazione
    sample_sensors = features[:15] #prende i primi 15 sensori più importanti
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=train_df[sample_sensors], showfliers=False, palette="viridis", linewidth=1.5)
    plt.xticks(rotation=45)
    plt.title("Distribuzione Features Normalizzate")
    plt.xlabel("Sensori e Medie Mobili (Top 15)", fontsize=12)
    plt.ylabel("Valore Normalizzato [0, 1]", fontsize=12)
    plt.show()

    # Grafico 7: FEATURE IMPORTANCE
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-10:] # Prende i 10 sensori più importanti

    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Sensori Critici (Analisi della Degradazione)')
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importanza Relativa')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    # Grafico 8: Distribuzione degli errori reale-predetto
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test_ground_truth - y_pred_rf, bins=30, color='skyblue', label='Errore', kde=True)
    plt.title("Distribuzione degli Errori (Reale - Predetto)")
    plt.xlabel("Errore (Cicli)")
    plt.ylabel("Densità")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    #Grafico 9: Learning Curve Logistic Regression
    print("Calcolo Logistic Regression Learning Curve...")
    train_sizes_lr, train_scores_lr, test_scores_lr = learning_curve(
        lr, train_df[features], y_train_bin,
        cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='neg_mean_absolute_error'
    )
    train_errors_lr = -np.mean(train_scores_lr, axis=1)
    test_errors_lr = -np.mean(test_scores_lr, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_lr, train_errors_lr, 'o-', color="r", label="Errore Training")
    plt.plot(train_sizes_lr, test_errors_lr, 'o-', color="g", label="Errore Validazione")
    plt.title("Curva di Apprendimento (Logistic Regression)")
    plt.xlabel("Campioni di addestramento")
    plt.ylabel("Errore (MAE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_full_evaluation()