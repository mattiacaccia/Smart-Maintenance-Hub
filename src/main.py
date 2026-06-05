from SupervisedLearning import SupervisedLearning
from Data_preprocessing import DataPreprocessor
from CSP import CSP
from SearchingSolutions import FactorySearch
from Knowledge_graph import FactoryKG
from Reasoning import ReasoningModule
from Belief_network import UncertaintyModule

def main():
    # 1. Inizializzazione moduli
    ml_module = SupervisedLearning()
    dp_module = DataPreprocessor()
    csp_module = CSP()
    search_module = FactorySearch()
    kg_module = FactoryKG()
    reasoning = ReasoningModule(pl_file="kb_maintenance.pl")

    # 2. Addestramento modello ML con Validazione Statistica ibrida
    print("Caricamento dati e addestramento modelli (Regressione + Classificazione)...")
    df_train = dp_module.full_preprocessing('train_FD001.txt', is_train=True)
    feature_names = df_train.drop(['unit', 'cycle', 'RUL'], axis=1).columns.tolist()
    
    results = ml_module.train_and_validate(df_train, feature_names)

    print("\n" + "="*55)
    print("VALIDAZIONE STATISTICA IBRIDA")
    print(f"[REG] Random Forest    -> MAE: {results['reg_mae']:.2f}, RMSE: {results['reg_rmse']:.2f}")
    print(f"[CLS] Logistic Regress -> Acc: {results['class_acc']:.2%}, F1: {results['class_f1']:.2%}")
    print("="*55)

    # 2.1: Valutazione finale su Test Set 
    print("\n" + "="*55)
    print(" VERIFICA FINALE SUL TEST SET")
    print("="*55)

    # Carichiamo il test set e il file delle verità (Ground Truth)
    df_test = dp_module.full_preprocessing('test_FD001.txt', is_train=False)
    path_ground_truth = 'RUL_FD001.txt' # Assicurati che il file sia nella stessa cartella

    # Utilizziamo il metodo del modulo ML per valutare la performance
    metrics_test, _, _, _ = ml_module.evaluate_on_test(df_test, path_ground_truth)

    print(f"Performance Definitiva su 100 Motori di Test:")
    print(f"-> MAE Finale: {metrics_test['mae']:.2f} cicli")
    print(f"-> RMSE Finale: {metrics_test['rmse']:.2f} cicli")
    print(f"-> Accuracy Finale: {metrics_test['accuracy']:.2%}")
    print(f"-> F1 Finale: {metrics_test['f1']:.2%}")
    print("="*55 + "\n")

    # 3. Simulazione caso d'uso: dove la un
    print("SIMULAZIONE CASO D'USO: ANALISI PREDITTIVA E INTEGRAZIONE MODELLI")
    unit_id = 2
    component = "oil_pump"

    #  CONTROLLO COMPONENTE-UNITA' PRE-ML  ---
    # Trasformiamo l'ID per il controllo Prolog (engine_1, engine_2)
    engine_id = f"engine_{unit_id}"

    # Interroghiamo il modulo di ragionamento (o direttamente il Prolog)
    # per sapere se il componente appartiene all'unità
    is_valid = bool(list(reasoning.prolog.query(f"part_of('{component}', '{engine_id}')")))

    if not is_valid:
        print(f"ERRORE DI VALIDAZIONE ONTOLOGICA:")
        print(f"Il componente '{component}' non è presente nella unit '{unit_id}'.")
        print("Il calcolo della RUL e l'analisi ML sono stati interrotti per incoerenza dati.")
        return # ESCI DAL PROGRAMMA PRIMA DEL ML

    # Se il controllo passa, allora procedi con il Machine Learning
    print(f"Validazione superata. Avvio analisi ML per {component}...")

    # 4. Inizializzazione modulo incertezza (Rete Bayesiana)
    belief_net = UncertaintyModule(kg_module, component)

    # 5. Recupero dati per l'inferenza
    engine_data = df_train[df_train['unit'] == unit_id].drop(['unit', 'cycle', 'RUL'], axis=1).iloc[[-1]]
    engine_data = engine_data[feature_names]

    # 6. Predizione RUL e Classificazione Allarme
    predicted_rul = ml_module.regressor.predict(engine_data)[0]
    print(f"\n[ML] Predizione RUL per unit {unit_id}, {component}: {predicted_rul:.2f} cicli")
    pred_alert = ml_module.classifier.predict(engine_data)[0]
    prob_alert = ml_module.classifier.predict_proba(engine_data)[0][1]

    status = "ALLARME" if pred_alert == 1 else "SICURO"
    print(f"\n[ML SYSTEM] Risultato Integrato:")
    print(f" > RUL Predetta: {predicted_rul:.2f} cicli")
    print(f" > Stato Classificato: {status} (Confidenza: {prob_alert:.2%})")

    if (predicted_rul <= 30 and pred_alert == 1) or (predicted_rul > 30 and pred_alert == 0):
        print(" > Coerenza Modelli: ALTA (Entrambi i modelli concordano sullo stato)")
    else:
        print(" > Coerenza Modelli: SOSPETTA (Possibile anomalia nei sensori)")

    # 7. Reasoning: Utilizzo della KB Prolog
    # Passiamo la RUL predetta per ottenere i requisiti logici
    print("\n" + "="*55)
    print(" REASONING - ESTRATTO REQUISITI LOGICI")
    print("="*55)
    req_unit_id = f"engine_{unit_id}"
    reqs = reasoning.get_requirements(unit_id=req_unit_id, part_id=component, current_rul=int(predicted_rul))
    print(f"[REASONING] Requisiti estratti dalla KB: {reqs}")

    # 8. Knowledge Graph: Recupero contesto spaziale
    print("\n" + "="*55)
    print("KG - RECUPERO CONTEX")
    print("="*55)
    kg_context = kg_module.get_task_context(component)
    area_attuale = kg_context.get('area', 'Area_B')
    print(f"[KG] Contesto recuperato: {area_attuale}")

    # 9. Rete Bayesiana: Gestione Incertezza
    print("\n" + "="*55)
    print("GESTIONE INCERTEZZA (BAYES)")
    print("="*55)
    ml_alert = predicted_rul < 30 # Soglia di allerta
    sensor_load_value = float(engine_data['s2'].values[0]) if 's2' in engine_data.columns else 0.5
    failure_prob = belief_net.get_real_failure_probability(ml_alert, sensor_load_value)
    print(f"[BAYES] Probabilità reale di guasto combinata: {failure_prob:.2%}")

    '''  # 9.1 Rete Bayesiana: calcolo probabilità reale di guasto su 2 casi estremi

    print("\n" + "="*45)
    print("GESTIONE INCERTEZZA (BAYES) MA SU 2 CASI ESTREMI")
    print("="*45)
    
    ml_alert = True 
    
    # Caso A: Carico Basso (Alta confidenza)
    prob_basso = belief_net.get_real_failure_probability(ml_alert, sensor_load_value=0.2)
    print(f"[CASO A] Carico Basso (0.2) -> Prob. Reale: {prob_basso:.2%}")

    # Caso B: Carico Alto (Sospetto Falso Positivo)
    prob_alto = belief_net.get_real_failure_probability(ml_alert, sensor_load_value=0.9)
    print(f"[CASO B] Carico Alto (0.9)  -> Prob. Reale: {prob_alto:.2%}")
    print("="*45 + "\n")
    '''

    # 10. CSP - Risoluzione Assegnazione Compiti
    print("\n" + "="*55)
    print("CSP - ASSEGNAZIONE TECNICI")
    print("="*55)

    # Colleghiamo il livello richiesto dal Reasoning Prolog al CSP
    req_level = 'Senior' if reqs.get('is_senior', False) else 'Junior'
    
    csp_module.Tasks = [
        {'id': f'Task_{component}', 'area': area_attuale, 'priority': 'Medium', 'required_level': req_level}
    ]

    assignments = csp_module.solve()

    if assignments:
        print(f"{'ID TASK':<25} | {'TECNICO':<10} | {'LIVELLO'}")
        print("-" * 55)
        for t_id, tech_id in assignments.items():
            t_info = next(t for t in csp_module.technicians if t['id'] == tech_id)
            print(f"{t_id:<25} | {tech_id:<10} | {t_info['level']}")
    else:
        print("⚠ Nessuna soluzione CSP valida trovata per i vincoli attuali.")

    # 11. Pianificazione Percorso A*
    print("\n" + "="*55)
    print("PIANIFICAZIONE PERCORSO OTTIMO A*")
    print("="*55)

    # I parametri di navigazione derivano direttamente dal ragionamento logico
    n_warehouse = reqs.get('needs_warehouse', True) 
    n_test = reqs.get('needs_test', False)

    percorso, costo = search_module.get_full_path(
        start='Enter',
        goal=area_attuale,
        needs_warehouse=n_warehouse,
        needs_test=n_test
    )

    if percorso:
        print(f"Configurazione: Magazzino={n_warehouse}, Test Area={n_test}")
        print(f"Percorso pianificato: {' -> '.join(percorso)}")
        print(f"Costo stimato (tempo): {costo:.2f} min")
    else:
        print(f"ERRORE: Nessun percorso disponibile per raggiungere {area_attuale}")
    
    print("="*55 + "\n")

if __name__ == "__main__":
    main()



